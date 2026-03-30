import sys
import os
import cv2
import numpy as np
import time

# Add Subtask 1 paths for visualizer
sys.path.append(os.path.abspath(r"..\Subtask 1\python"))
from odometry_visualizer import TrajectoryVisualizer

# K given in the task description
K = np.array([
    [517.3, 0.0,   318.6],
    [0.0,   516.5, 255.3],
    [0.0,   0.0,   1.0]
], dtype=np.float64)

def pnp_dlt(pts2d, pts3d):
    """
    DLT PnP solver.
    pts2d: Nx2
    pts3d: Nx3
    Returns the 3x4 projection matrix P
    """
    N = pts2d.shape[0]
    A = np.zeros((2*N, 12), dtype=np.float64)
    for i in range(N):
        x, y = pts2d[i]
        X, Y, Z = pts3d[i]
        A[2*i]   = [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y]

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    return P

def decompose_P_with_K(P, K_mat):
    """
    Decompose P matrix into R and t using given K
    P ~ K [R | t]
    [R | t] ~ K_inv * P
    """
    K_inv = np.linalg.inv(K_mat)
    M = K_inv @ P
    
    # Left 3x3 block is scaled Rotation matrix
    M33 = M[:, :3]
    
    # Ensure R is a proper rotation matrix using SVD
    U, _, Vt = np.linalg.svd(M33)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    
    # Recover scale factor
    scale = np.sum(M33 * R) / 3.0
    
    # Scale translation component
    t = (M[:, 3] / scale).reshape(3, 1)
    
    return R, t

def calculate_reprojection_errors(P, pts2d, pts3d):
    """
    Given P and points, return array of projection errors for each point.
    """
    pts3d_h = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    proj = (P @ pts3d_h.T).T
    proj = proj[:, :2] / proj[:, 2:]
    errors = np.linalg.norm(proj - pts2d, axis=1)
    return errors

def pnp_ransac(pts2d, pts3d, K_mat, threshold=4.0, max_iters=500):
    """
    RANSAC wrapper for PnP DLT.
    Returns: best_R, best_t, best_inliers_idx
    """
    N = pts2d.shape[0]
    if N < 6:
        return None, None, []
        
    best_inlier_count = 0
    best_P = None
    best_inliers = []
    
    for _ in range(max_iters):
        # Pick 6 random points
        sample_indices = np.random.choice(N, 6, replace=False)
        pts2_sample = pts2d[sample_indices]
        pts3_sample = pts3d[sample_indices]
        
        P = pnp_dlt(pts2_sample, pts3_sample)
        
        # Guard against zero/singular projection matrices
        if np.sum(np.abs(P)) < 1e-6:
            continue
            
        errors = calculate_reprojection_errors(P, pts2d, pts3d)
        
        inliers = np.where(errors < threshold)[0]
        inlier_count = len(inliers)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_P = P
            best_inliers = inliers
            
            # Stop early if very confident
            if inlier_count > 0.9 * N:
                break
                
    if best_inlier_count < 6:
        return None, None, []
        
    # Refit with all inliers using Least Squares
    best_P = pnp_dlt(pts2d[best_inliers], pts3d[best_inliers])
    
    R, t = decompose_P_with_K(best_P, K_mat)
    
    # Fix front/back ambiguity (check Z>0 for most points)
    test_X_cam = (R @ pts3d[best_inliers[0]].reshape(3, 1) + t)
    if test_X_cam[2, 0] < 0:
        R = -R
        t = -t
        
    return R, t, best_inliers

def extract_features(frame_gray):
    # Shi-Tomasi corners
    p = cv2.goodFeaturesToTrack(frame_gray, maxCorners=2000, qualityLevel=0.01, minDistance=10)
    if p is not None:
        return p.reshape(-1, 2)
    return np.empty((0, 2))

def main():
    video_path = "dataset_video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
        
    visualizer = TrajectoryVisualizer()
    
    # State keeping
    prev_gray = None
    prev_pts2d = None
    active_pts3d = None
    
    frame_idx = 0
    
    R_current = np.eye(3)
    t_current = np.zeros((3, 1))
    
    # Max frames
    max_frames_to_test = 80
    
    # Initiation structures
    init_frame_gray = None
    init_pts2d = None
    initialized = False
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx > max_frames_to_test:
            break
            
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if not initialized:
            if init_frame_gray is None:
                init_frame_gray = current_gray
                init_pts2d = extract_features(current_gray)
                print(f"[{frame_idx}] Found {len(init_pts2d)} initial features.")
            else:
                # Track initial features to current frame
                curr_pts2d, status, _ = cv2.calcOpticalFlowPyrLK(init_frame_gray, current_gray, init_pts2d, None)
                
                good_old = init_pts2d[status[:, 0] == 1]
                good_new = curr_pts2d[status[:, 0] == 1]
                
                # Check mean displacement
                disparity = np.mean(np.linalg.norm(good_new - good_old, axis=1))
                
                # If disparity is enough, compute E matrix and 3D points
                if disparity > 20.0 or frame_idx == 10:
                    print(f"[{frame_idx}] Disparity {disparity:.2f} reached. Initializing 3D map.")
                    
                    E, mask = cv2.findEssentialMat(good_old, good_new, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                    _, R_est, t_est, mask_pose = cv2.recoverPose(E, good_old, good_new, K, mask=mask)
                    
                    # keep only inliers
                    inlier_idx = (mask_pose.ravel() > 0)
                    pts1_in = good_old[inlier_idx]
                    pts2_in = good_new[inlier_idx]
                    
                    # Triangulate
                    P1 = (K @ np.hstack((np.eye(3), np.zeros((3, 1))))).astype(np.float64)
                    P2 = (K @ np.hstack((R_est, t_est))).astype(np.float64)
                    
                    pts1_in_t = np.ascontiguousarray(pts1_in.T, dtype=np.float64)
                    pts2_in_t = np.ascontiguousarray(pts2_in.T, dtype=np.float64)
                    print("P1 shape:", P1.shape, "dtype:", P1.dtype)
                    print("pts1_in_t shape:", pts1_in_t.shape, "dtype:", pts1_in_t.dtype)
                    pts4d = cv2.triangulatePoints(P1, P2, pts1_in_t, pts2_in_t)
                    pts3d = (pts4d[:3, :] / pts4d[3, :]).T
                    
                    # Filter behind camera
                    z_cam1 = pts3d[:, 2]
                    z_cam2 = (R_est @ pts3d.T + t_est)[2, :]
                    valid_depth = (z_cam1 > 0) & (z_cam2 > 0)
                    
                    active_pts3d = pts3d[valid_depth]
                    prev_pts2d = pts2_in[valid_depth]
                    prev_gray = current_gray
                    
                    R_current = R_est
                    t_current = t_est
                    
                    C = -R_current.T @ t_current
                    visualizer.add_pose(C)
                    print(f"Initialized with {len(active_pts3d)} 3D points.")
                    initialized = True
        else:
            # Continuous Tracking (3D-to-2D Pipeline)
            curr_pts2d, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_pts2d, None)
            
            tracked_mask = (status[:, 0] == 1)
            p_curr = curr_pts2d[tracked_mask]
            pts_3d_tracked = active_pts3d[tracked_mask]
            
            # PnP DLT + RANSAC
            R_curr, t_curr, inliers = pnp_ransac(p_curr, pts_3d_tracked, K, threshold=5.0)
            
            if R_curr is not None:
                R_current = R_curr
                t_current = t_curr
                
                # Keep active map clean to only include inliers
                active_pts3d = pts_3d_tracked[inliers]
                prev_pts2d = p_curr[inliers]
                prev_gray = current_gray
                
                # Add Camera Pose C = -R^T t
                C = -R_current.T @ t_current
                visualizer.add_pose(C)
                visualizer.visualize()
                
                print(f"[{frame_idx}] Tracked {len(inliers)} inliers / {len(p_curr)} points. Pose: {C.ravel()}")
            else:
                print(f"[{frame_idx}] PnP Failed - Too few features or Tracking Lost.")
                prev_gray = current_gray
                
            if len(active_pts3d) < 50:
                print(f"[{frame_idx}] Tracked features very low. Re-initializing...")
                init_frame_gray = current_gray
                init_pts2d = extract_features(current_gray)
                initialized = False
                
        frame_idx += 1
        
    cap.release()
    print("Done. Close the plot window to exit.")
    
    # Import matplotlib to keep window open
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.savefig('trajectory_output.png')
    print("Saved trajectory_output.png")
    
if __name__ == '__main__':
    main()
