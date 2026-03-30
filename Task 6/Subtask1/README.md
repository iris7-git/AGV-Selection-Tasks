# 3D Reconstruction and Depth Estimation

This repository implements a full sparse and dense 3D Reconstruction pipeline from stereo images (temple dataset) along with Camera Pose estimation logic.

### Prerequisites & Dependencies
- Python 3+
- `numpy`, `scipy`, `matplotlib`, `opencv-python`, `scikit-image`

---

## Final Execution Order & Expected Outputs

To successfully run through the assignment sections and verify the outputs, navigate into the `/python` directory and execute the following scripts in order:

### 1. `generate_writeup_data.py` (Write-Up Report Helper)
- **What it does:** Runs through Sections 2.1 to 2.4, generating the required text outputs and launching the visual GUIs.
- **Expected Output:**
  - Prints the 3x3 **Fundamental Matrix (F)**.
  - Opens the `displayEpipolarF` GUI. *You must click around and then close the window to continue.*
  - Opens the `epipolarMatchGUI` window. *You must click features and close the window to continue.*
  - Prints the 3x3 **Essential Matrix (E)**.
  - Prints the **Reprojection Error** showing it optimally triangulated below 2.0 pixels.

### 2. `test_pose.py` & `test_params.py` (Validation Tests)
- **What they do:** Verifies the underlying `estimate_pose` (DLT projection matrix generation) and `estimate_params` (RQ decomposition for Instrinsics/Extrinsics extraction) algorithms using randomly generated 3D points.
- **Expected Output:** Only console logs showing minimal mathematical reconstruction errors (on the order of `1e-12` for clear points). No visual output.

### 3. `test_temple_coords.py` (Sparse Reconstruction)
- **What it does:** This is the core script for **Part 1**, implementing Section 2.5 of the assignment. It strings together Fundamental matrix extraction, Epipolar correspondences along image 2 epipolar lines, Triangulation, and Positive-Depth candidate mathematical sorting.
- **Expected Output:**
  - A 3D interactive plotting window showing a scatter cloud corresponding to the shape of the temple.
  - **Crucial Side Effect:** It saves the correct translation/rotation extrinsics to `../data/extrinsics.npz` (needed for Part 2). *You must close the plot window for this save to occur.*

### 4. `test_rectify.py` (Stereo Rectification)
- **What it does:** Rotates the camera coordinate frames iteratively so the epipoles are propelled to infinity, locking both images perfectly across horizontal scan lines to significantly speed up depth sliding windows.
- **Expected Output:**
  - A GUI window showing the two images merged side-by-side with purely horizontal matching epipolar lines.
  - **Crucial Side Effect:** Saves the warping homography matrices to `../data/rectify.npz`.

### 5. `test_depth.py` (Dense Reconstruction)
- **What it does:** Uses a sliding window logic (SSD) horizontally across the rectified images to resolve Dense disparity mapping per-pixel, transforming the pixel disparity displacement into a physical depth map utilizing the recovered physical baseline and focal lengths.
- **Expected Output:** A matplotlib window displaying two matrices:
  - The calculated Disparity Map.
  - The final physical Depth Map output.

---

## Running inside Docker

If you prefer to run this in an isolated environment, a `Dockerfile` has been explicitly configured to bundle the complex GUI capabilities logic (`python3-tk`, `libgl1`, etc.) required by Matplotlib and OpenCV.

**Build the image:**
```bash
docker build -t 3d-reconstruction .
```

*Note: Since the scripts rely heavily on `plt.show()` to display interactive GUIs, running GUI applications seamlessly from a Docker container fundamentally requires [X11 Forwarding](https://wiki.ros.org/docker/Tutorials/GUI) mapped to your local host machine.*
