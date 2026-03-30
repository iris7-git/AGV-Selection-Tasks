import cv2
import numpy as np
import helper as hlp
import submission as sub
import matplotlib.pyplot as plt
import numpy.linalg as la

def main():
    print("--- 3D Reconstruction Write-up Data Generation ---")
    
    # Load data
    im1_c = cv2.imread('../data/im1.png')
    im2_c = cv2.imread('../data/im2.png')
    
    # Needs to be RGB for matplotlib display
    im1_c = cv2.cvtColor(im1_c, cv2.COLOR_BGR2RGB)
    im2_c = cv2.cvtColor(im2_c, cv2.COLOR_BGR2RGB)
    
    im1 = cv2.cvtColor(im1_c, cv2.COLOR_RGB2GRAY)
    im2 = cv2.cvtColor(im2_c, cv2.COLOR_RGB2GRAY)
    
    corresp = np.load('../data/some_corresp.npz')
    pts1 = corresp['pts1']
    pts2 = corresp['pts2']
    
    # 2.1 Eight Point Algorithm -> Recover F
    M = max(im1.shape[0], im1.shape[1])
    F = sub.eight_point(pts1, pts2, M)
    print("\n[Section 2.1] Recovered Fundamental Matrix F:")
    print(F)
    
    # 2.3 Essential Matrix
    intrinsics = np.load('../data/intrinsics.npz')
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = sub.essential_matrix(F, K1, K2)
    print("\n[Section 2.3] Estimated Essential Matrix E:")
    print(E)

    print("\n-> Launching displayEpipolarF GUI for Section 2.1 screenshots...")
    hlp.displayEpipolarF(im1_c, im2_c, F)
    # The user has to close the matplotlib figure to continue
    plt.show()

    # 2.2 Epipolar Correspondences
    print("\n-> Launching epipolarMatchGUI for Section 2.2 screenshots...")
    print("Click on easy points like corners/dots in the left image to see the matching.")
    # We use the F already computed
    hlp.epipolarMatchGUI(im1_c, im2_c, F)
    plt.show()

    # 2.4 Triangulation Reprojection Error
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    M2s = hlp.camera2(E)
    
    best_P2 = None
    best_pts3d = None
    max_positive = -1
    best_i = -1
    
    # Find the correct P2 to compute reprojection error properly
    for i in range(4):
        P2_candidate = K2 @ M2s[:, :, i]
        pts3d_candidate = sub.triangulate(P1, pts1, P2_candidate, pts2)
        
        depths1 = pts3d_candidate[:, 2]
        pts3d_hom = np.hstack((pts3d_candidate, np.ones((pts3d_candidate.shape[0], 1))))
        depths2 = (M2s[:, :, i] @ pts3d_hom.T)[2, :]
        
        num_positive = np.sum((depths1 > 0) & (depths2 > 0))
        if num_positive > max_positive:
            max_positive = num_positive
            best_P2 = P2_candidate
            best_pts3d = pts3d_candidate
            best_i = i

    # Now compute reprojection error on pts1 from some_corresp.npz
    # Project best_pts3d back to image 1 using P1
    pts3d_hom = np.hstack((best_pts3d, np.ones((best_pts3d.shape[0], 1))))
    proj_pts1 = (P1 @ pts3d_hom.T).T
    proj_pts1 = proj_pts1[:, :2] / proj_pts1[:, 2:]  # Dehomogenize
    
    reproj_error = np.mean(la.norm(pts1 - proj_pts1, axis=1))
    
    print(f"\n[Section 2.4] Reprojection Error on some_corresp.npz using best P2 (Candidate {best_i}):")
    print(f"{reproj_error:.4f} pixels")
    if reproj_error < 2.0:
        print("(Success! Error is less than 2 pixels as required)")
        
    print("\nAll data extracted. You can copy these values and screenshots for your write-up!")

if __name__ == '__main__':
    main()
