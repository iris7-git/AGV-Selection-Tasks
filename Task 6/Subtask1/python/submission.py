"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import numpy.linalg as la
import scipy.linalg
import helper as hlp


"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # Normalization matrix
    T = np.array([[1/M, 0,   0],
                  [0,   1/M, 0],
                  [0,   0,   1]], dtype=np.float64)

    # Normalize points
    pts1_n = pts1 / M   # Nx2
    pts2_n = pts2 / M   # Nx2

    N = pts1_n.shape[0]

    # Build the DLT matrix A (N x 9)
    # Each row: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    x1, y1 = pts1_n[:, 0], pts1_n[:, 1]
    x2, y2 = pts2_n[:, 0], pts2_n[:, 1]

    A = np.column_stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1,      y1,      np.ones(N)
    ])

    # Solve: take last row of V^T (smallest singular value)
    _, _, Vt = la.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint: zero out smallest singular value
    U, S, Vt2 = la.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt2

    # Optionally refine F (before unscaling, using normalized points)
    F = hlp.refineF(F, pts1_n, pts2_n)

    # Enforce rank-2 again after refinement
    U, S, Vt2 = la.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt2

    # Unscale: F_unnorm = T^T @ F_norm @ T
    F = T.T @ F @ T

    return F


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    H2, W2 = im2.shape[:2]
    win = 5   # half-window size
    pts2 = np.zeros_like(pts1, dtype=np.float64)

    # Convert images to float for comparison
    im1f = im1.astype(np.float64)
    im2f = im2.astype(np.float64)

    for i, (x1, y1) in enumerate(pts1):
        x1, y1 = int(round(x1)), int(round(y1))

        # Compute epipolar line l = F @ [x1, y1, 1]^T
        l = F @ np.array([x1, y1, 1.0])
        a, b, c = l  # ax + by + c = 0

        # Extract patch from im1 (handle borders)
        y1c = np.clip(y1, win, H2 - win - 1)
        x1c = np.clip(x1, win, W2 - win - 1)
        patch1 = im1f[y1c - win: y1c + win + 1,
                      x1c - win: x1c + win + 1]

        best_score = np.inf
        best_x2, best_y2 = x1, y1   # fallback

        # Search along the epipolar line across all valid x in image 2
        # For each x, compute y from line equation: y = -(a*x + c) / b
        if abs(b) >= abs(a):
            # Iterate over x
            xs = np.arange(win, W2 - win)
            ys = (-(a * xs + c) / b).astype(int)
        else:
            # Iterate over y
            ys = np.arange(win, H2 - win)
            xs = (-(b * ys + c) / a).astype(int)

        for x2, y2 in zip(xs, ys):
            # Bounds check
            if (win <= x2 < W2 - win) and (win <= y2 < H2 - win):
                patch2 = im2f[y2 - win: y2 + win + 1,
                              x2 - win: x2 + win + 1]
                if patch1.shape == patch2.shape:
                    # SSD similarity
                    score = np.sum((patch1 - patch2) ** 2)
                    if score < best_score:
                        best_score = score
                        best_x2, best_y2 = x2, y2

        pts2[i] = [best_x2, best_y2]

    return pts2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # E = K2^T @ F @ K1
    E = K2.T @ F @ K1
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    N = pts1.shape[0]
    pts3d = np.zeros((N, 3))

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        # Build 4x4 linear system using the DLT cross-product formulation
        # x × (P @ X) = 0  =>  two independent equations per view
        A = np.array([
            y1 * P1[2] - P1[1],
            P1[0] - x1 * P1[2],
            y2 * P2[2] - P2[1],
            P2[0] - x2 * P2[2],
        ])

        _, _, Vt = la.svd(A)
        X = Vt[-1]              # homogeneous 3D point
        pts3d[i] = X[:3] / X[3]  # dehomogenize

    return pts3d


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # Optical centers: c = -R^T @ t
    c1 = -R1.T @ t1   # 3x1
    c2 = -R2.T @ t2   # 3x1

    # New x-axis: baseline direction
    r1 = (c1 - c2).ravel()
    r1 = r1 / la.norm(r1)

    # New y-axis: orthogonal to r1, using old z-axis of cam1
    # New z-axis: same as old camera 1 z-axis (row 2 of R1)
    r3 = R1[2, :]
    r2 = np.cross(r3, r1)
    r2 = r2 / la.norm(r2)

    # Recompute r3 to ensure orthogonality
    r3 = np.cross(r1, r2)
    r3 = r3 / la.norm(r3)

    # New shared rotation matrix
    R = np.stack([r1, r2, r3], axis=0)   # 3x3

    # New shared intrinsics: average focal length, keep K1's principal point
    # (use K1 for both as a common choice)
    K1p = K1.copy()
    K2p = K1.copy()   # same K for both so horizontal scanlines match

    # New extrinsics
    R1p = R.copy()
    R2p = R.copy()

    t1p = -R @ c1   # 3x1
    t2p = -R @ c2   # 3x1

    # Rectification homographies
    # M = K_new @ R_new @ inv(K_old @ R_old)
    M1 = (K1p @ R1p) @ la.inv(K1 @ R1)
    M2 = (K2p @ R2p) @ la.inv(K2 @ R2)

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    H, W = im1.shape
    dispM = np.zeros((H, W), dtype=np.float64)
    half = win_size // 2

    # Pad both images to handle borders
    im1p = np.pad(im1, half, mode='edge').astype(np.float64)
    im2p = np.pad(im2, half, mode='edge').astype(np.float64)

    # For each disparity d compute SSD across all pixels with vectorized ops
    best_ssd = np.full((H, W), np.inf)

    for d in range(max_disp + 1):
        # Shift im2 to the right by d (im1 pixel x matches im2 pixel x-d)
        ssd = np.zeros((H, W), dtype=np.float64)
        for dy in range(win_size):
            for dx in range(win_size):
                row1 = im1p[dy:dy + H, dx:dx + W]
                # shift im2 patch by disparity d along x
                x_start = max(0, dx - d)
                col_offset = dx - d
                if col_offset < 0:
                    row2 = im2p[dy:dy + H, 0:W]
                else:
                    row2 = im2p[dy:dy + H, col_offset:col_offset + W]
                ssd += (row1 - row2) ** 2

        mask = ssd < best_ssd
        best_ssd[mask] = ssd[mask]
        dispM[mask] = d

    return dispM


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # Optical centers
    c1 = (-R1.T @ t1).ravel()
    c2 = (-R2.T @ t2).ravel()

    # Baseline
    b = la.norm(c1 - c2)

    # Focal length (from K1, assume both share same after rectification)
    f = K1[0, 0]

    # Depth = b * f / disparity; avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        depthM = np.where(dispM > 0, b * f / dispM, 0.0)

    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    N = x.shape[0]

    # Build the 2N x 12 DLT matrix A
    A = []
    for i in range(N):
        xi, yi = x[i]
        Xi, Yi, Zi = X[i]
        # Row for u equation:  [X^T 0^T -u*X^T]
        A.append([Xi, Yi, Zi, 1,  0,  0,  0, 0, -xi*Xi, -xi*Yi, -xi*Zi, -xi])
        # Row for v equation:  [0^T X^T -v*X^T]
        A.append([ 0,  0,  0, 0, Xi, Yi, Zi, 1, -yi*Xi, -yi*Yi, -yi*Zi, -yi])

    A = np.array(A, dtype=np.float64)

    # Solve via SVD: last row of V^T
    _, _, Vt = la.svd(A)
    P = Vt[-1].reshape(3, 4)

    return P


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # RQ decomposition of the left 3x3 submatrix
    K, R = scipy.linalg.rq(P[:, :3])

    # Ensure K has positive diagonal (canonical form)
    D = np.sign(np.diag(K))
    D[D == 0] = 1
    T = np.diag(D)
    K = K @ T
    R = T @ R
    
    # If determinant of R is negative, the projection matrix P was scaled by -1
    # We must negate R and t to ensure R is a valid rotation matrix
    if la.det(R) < 0:
        R = -R
        P = -P

    # Translation: t = K^-1 @ P[:,3]
    # We must compute this BEFORE normalizing K so that the scale factor in P cancels out with K
    t = (la.inv(K) @ P[:, 3]).reshape(3, 1)

    # Normalize so K[2,2] = 1
    K = K / K[2, 2]

    return K, R, t
