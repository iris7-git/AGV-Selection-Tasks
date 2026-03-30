import cv2
import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

# 1. Load the two temple images and the points from data/some_corresp.npz
im1 = cv2.imread('../data/im1.png', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('../data/im2.png', cv2.IMREAD_GRAYSCALE)
corresp = np.load('../data/some_corresp.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']

# 2. Run eight_point to compute F
M = max(im1.shape[0], im1.shape[1])
F = sub.eight_point(pts1, pts2, M)
print(f"Computed Fundamental Matrix F:\n{F}")

# 3. Load points in image 1 from data/temple_coords.npz
temple_coords = np.load('../data/temple_coords.npz')
pts1_temple = temple_coords['pts1']

# 4. Run epipolar_correspondences to get points in image 2
pts2_temple = sub.epipolar_correspondences(im1, im2, F, pts1_temple)

# 5. Compute the camera projection matrix P1
intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']
E = sub.essential_matrix(F, K1, K2)
print(f"Computed Essential Matrix E:\n{E}")

P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))

# 6. Use camera2 to get 4 camera projection matrices P2
# Note: P2 candidates returned by camera2 need to be pre-multiplied by K2
M2s = hlp.camera2(E)
P2_candidates = [K2 @ M2s[:, :, i] for i in range(4)]

# 7. Run triangulate using the projection matrices
best_pts3d = None
best_P2 = None
best_M2_idx = 0
max_positive_depths = -1

for i in range(4):
    P2 = P2_candidates[i]
    pts3d = sub.triangulate(P1, pts1_temple, P2, pts2_temple)
    
    # 8. Figure out the correct P2
    # Check depth in both cameras
    # Camera 1 depth is just Z coordinate (since P1=[I|0])
    depths1 = pts3d[:, 2]
    
    # Camera 2 depth: proj = M2 * X (where M2 = [R|t])
    M2 = M2s[:, :, i]
    pts3d_hom = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    depths2 = (M2 @ pts3d_hom.T)[2, :]
    
    # Count how many points are in front of BOTH cameras
    num_positive = np.sum((depths1 > 0) & (depths2 > 0))
    print(f"Candidate {i}: {num_positive}/{len(pts3d)} points in front of cameras")
    
    if num_positive > max_positive_depths:
        max_positive_depths = num_positive
        best_pts3d = pts3d
        best_P2 = P2
        best_M2_idx = i

print(f"Selected Candidate {best_M2_idx} as correct P2.")

# 9. Scatter plot the correct 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(best_pts3d[:, 0], best_pts3d[:, 1], best_pts3d[:, 2], c='r', marker='.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Reconstruction of Temple')
# Orient the plot so it looks upright
ax.view_init(elev=-90, azim=-90)

plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
R1 = np.eye(3)
t1 = np.zeros((3, 1))

best_M2 = M2s[:, :, best_M2_idx]
R2 = best_M2[:, :3]
t2 = best_M2[:, 3:]

np.savez('../data/extrinsics.npz', R1=R1, t1=t1, R2=R2, t2=t2)
print("Saved extrinsic parameters to ../data/extrinsics.npz")
