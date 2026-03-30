import cv2
import numpy as np
import pybullet as p
import time
import traceback
from simulation_setup import setup_simulation

# --- TUNABLE PARAMETERS ---
IMG_WIDTH, IMG_HEIGHT = 320, 240
TARGET_VELOCITY = 4.0   
MAX_STEER = 0.6         

# LANE CONSTANTS
LANE_LEFT_Y = 0.42      
LANE_RIGHT_Y = -0.42    
LANE_TOLERANCE = 0.1    # Accuracy needed to consider lane change "done"
LANE_CHANGE_COOLDOWN_DIST = 5.5 # Meters to travel before reactivating detection

# CONTROL CONSTANTS
SPRING_KP = 1.2         # Proportional gain (The "Spring")
DAMPING_KD = 1        # Derivative gain (The "Shock Absorber" using Yaw)
OBSTACLE_THRESHOLD = 2.6 # The "Alarm Bell" magnitude
# --------------------------

def custom_lucas_kanade(img1, img2, points, win_size=15):
    """Manual Lucas-Kanade implementation."""
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
    It = img2.astype(np.float64) - img1.astype(np.float64)
    half_w = win_size // 2
    new_pts, status = [], []
    for pt in points:
        x, y = int(pt[0][0]), int(pt[0][1])
        if y-half_w < 0 or y+half_w+1 >= IMG_HEIGHT or x-half_w < 0 or x+half_w+1 >= IMG_WIDTH:
            new_pts.append([x, y]); status.append(0); continue
        ix = Ix[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        iy = Iy[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        it = It[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        A = np.vstack((ix, iy)).T
        b = -it.reshape(-1, 1)
        ATA = A.T @ A
        if np.linalg.det(ATA) > 0.01:
            flow = np.linalg.inv(ATA) @ (A.T @ b)
            new_pts.append([x + flow[0][0], y + flow[1][0]])
            status.append(1)
        else:
            new_pts.append([x, y]); status.append(0)
    return np.array(new_pts).reshape(-1, 1, 2), np.array(status)

def get_robust_camera_image(car_id):
    """Captures and formats the camera feed."""
    pos, orn = p.getBasePositionAndOrientation(car_id)
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    cam_pos = pos + rot.dot(np.array([0.4, 0.0, 0.3]))
    target = cam_pos + rot.dot(np.array([1.0, 0.0, 0.0]))
    view_mat = p.computeViewMatrix(cam_pos, target, rot.dot(np.array([0.0, 0.0, 1.0])))
    proj_mat = p.computeProjectionMatrixFOV(60, IMG_WIDTH/IMG_HEIGHT, 0.1, 100.0)
    _, _, rgba, _, _ = p.getCameraImage(IMG_WIDTH, IMG_HEIGHT, view_mat, proj_mat)
    frame = np.reshape(rgba, (IMG_HEIGHT, IMG_WIDTH, 4)).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY), cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

def main():
    try:
        # Initialize simulation
        car_id, steer_joints, motor_joints = setup_simulation(gui=True)
        old_gray, _ = get_robust_camera_image(car_id)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=60, qualityLevel=0.2, minDistance=7)
        
        # --- INITIAL STATE ---
        curr_pos, _ = p.getBasePositionAndOrientation(car_id)
        # Start by targeting the lane we spawned in
        target_y = LANE_LEFT_Y if curr_pos[1] > 0 else LANE_RIGHT_Y
        is_changing_lane = False
        
        # Tracks X position of the last triggered change
        last_trigger_x = -LANE_CHANGE_COOLDOWN_DIST 

        while True:
            # 1. Constant Forward Drive
            for j in motor_joints:
                p.setJointMotorControl2(car_id, j, p.VELOCITY_CONTROL, targetVelocity=TARGET_VELOCITY)
            
            gray, bgr = get_robust_camera_image(car_id)
            curr_pos, curr_orn = p.getBasePositionAndOrientation(car_id)
            curr_y = curr_pos[1]
            yaw = p.getEulerFromQuaternion(curr_orn)[2]

            # 2. Check if the current maneuver is finished
            if is_changing_lane and abs(curr_y - target_y) < LANE_TOLERANCE:
                is_changing_lane = False
                print(f"Lane change complete. Now holding {target_y}")

            # 3. Spatial Cooldown Gate
            # Uses X-coordinate because the road runs along the X-axis
            dist_since_last = curr_pos[0] - last_trigger_x
            can_detect = not is_changing_lane and dist_since_last > LANE_CHANGE_COOLDOWN_DIST

            # 4. Obstacle Detection (The Alarm Bell)
            if can_detect and p0 is not None:
                p1, status = custom_lucas_kanade(old_gray, gray, p0)
                good_new = p1[status == 1]
                good_old = p0[status == 1]
                
                max_mag = 0.0
                for i in range(len(good_new)):
                    x, y = good_new[i].ravel()
                    mag = np.linalg.norm(good_new[i] - good_old[i])
                    
                    # Only look for obstacles in the center path
                    if (IMG_WIDTH*0.3) < x < (IMG_WIDTH*0.7) and y > IMG_HEIGHT*0.5:
                        if mag > max_mag: max_mag = mag

                # 5. Trigger Lane Switch: Chase the FARTHER lane
                if max_mag > OBSTACLE_THRESHOLD:
                    dist_to_left = abs(curr_y - LANE_LEFT_Y)
                    dist_to_right = abs(curr_y - LANE_RIGHT_Y)
                    
                    # Target the lane that is mathematically further away
                    target_y = LANE_LEFT_Y if dist_to_left > dist_to_right else LANE_RIGHT_Y
                    
                    # Lock states inside the trigger block
                    is_changing_lane = True
                    last_trigger_x = curr_pos[0] 
                    print(f"Obstacle! Switching to target: {target_y}")

                # Refresh optical flow points
                if len(good_new) < 20:
                    p0 = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=60, qualityLevel=0.2, minDistance=7)
                else:
                    p0 = good_new.reshape(-1, 1, 2)

            # 6. The Virtual Spring (PD Control)
            # Uses proportional gain for lane tracking and yaw damping for stability.
            steering_error = (curr_y - target_y)
            f_spring = (steering_error * SPRING_KP) + (yaw * DAMPING_KD)
            
            # Apply steering (negated for URDF orientation)
            steer_angle = np.clip(-f_spring, -MAX_STEER, MAX_STEER)
            for j in steer_joints:
                p.setJointMotorControl2(car_id, j, p.POSITION_CONTROL, targetPosition=steer_angle)

            # View Simulation
            p.resetDebugVisualizerCamera(4.0, 0, -35, curr_pos)
            old_gray = gray.copy()
            cv2.imshow("Optical Flow View", bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            p.stepSimulation()
            time.sleep(1/60.0)

    except Exception:
        traceback.print_exc()
    finally:
        p.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()