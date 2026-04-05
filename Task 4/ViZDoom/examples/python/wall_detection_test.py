import cv2
import numpy as np


# ── Exact map colors (provided) ───────────────────────────────────────────────
# Background : RGBA(111, 87, 67)  →  BGR=(67,  87, 111)
# Wall border: RGBA( 71, 49, 32)  →  BGR=(32,  49,  71)
# Gray lines : RGBA(135,132,130)  →  BGR=(130,132, 135)
COLOR_WALL = (71,  49,  32)   # RGB
COLOR_GRAY = (135, 132, 130)  # RGB
COLOR_BG   = (111,  87,  67)  # RGB
COLOR_TOL  = 25               # ± tolerance for color matching


def load_map(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    return img


def color_mask(img, rgb_target, tol=COLOR_TOL):
    """Return binary mask for pixels within ±tol of rgb_target."""
    r, g, b = rgb_target
    lower = np.array([max(0,   b-tol), max(0,   g-tol), max(0,   r-tol)], dtype=np.uint8)
    upper = np.array([min(255, b+tol), min(255, g+tol), min(255, r+tol)], dtype=np.uint8)
    return cv2.inRange(img, lower, upper)


def build_occupancy_grid(img):
    """
    Build wall occupancy grid using exact wall border color.
    1 = wall/obstacle, 0 = free space.
    Walls are dilated by 1px to ensure no path clips a wall edge.
    Gray lines are replaced with background color before processing
    so they contribute zero wall pixels.
    """
    # 1. Mask gray lines → replace with background so they are invisible to wall detector
    gray_px   = color_mask(img, COLOR_GRAY)
    bg_color_bgr = [COLOR_BG[2], COLOR_BG[1], COLOR_BG[0]]   # BGR
    cleaned   = img.copy()
    cleaned[gray_px > 0] = bg_color_bgr

    # 2. Detect wall pixels by exact color on the cleaned image
    wall_px   = color_mask(cleaned, COLOR_WALL)

    # 3. Dilate walls by 1px to close hairline gaps
    kernel    = np.ones((3, 3), np.uint8)
    wall_full = cv2.dilate(wall_px, kernel, iterations=1)

    occ_grid  = (wall_full > 0).astype(np.uint8)
    return occ_grid, gray_px, wall_px


def find_player_and_dest(img):
    """
    Player = white dot  (largest white cluster, ignoring image border).
    Dest   = blue dot   (only a few pixels, use centroid).
    Returns pixel (x, y) for each.
    """
    h, w = img.shape[:2]
    BORDER = 5   # ignore outermost pixels (top border artifact is white)

    # ── White dot (player) ─────────────────────────────────────────────────
    white_mask = cv2.inRange(img,
                             np.array([200, 200, 200]),
                             np.array([255, 255, 255]))
    white_mask[:BORDER, :]  = 0   # kill top-border artifact
    white_mask[-BORDER:, :] = 0
    white_mask[:, :BORDER]  = 0
    white_mask[:, -BORDER:] = 0

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(white_mask)
    if num_labels > 1:
        largest    = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        player_pos = (int(centroids[largest][0]), int(centroids[largest][1]))
    else:
        player_pos = None

    # ── Blue dot (destination) ────────────────────────────────────────────
    blue_mask = cv2.inRange(img,
                            np.array([100,   0,   0]),
                            np.array([255,  80,  80]))
    M = cv2.moments(blue_mask)
    dest_pos = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] > 0 else None

    return player_pos, dest_pos


def visualize(img, gray_px, wall_px, occ_grid, player_pos, dest_pos):
    """
    2×2 panel:
      Original  |  Gray lines detected
      Wall mask |  Final occupancy grid (with P and D dots)
    """
    h, w = img.shape[:2]

    def label(panel, text):
        out = panel.copy()
        cv2.putText(out, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        return out

    def to_bgr(mask, color):
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[mask > 0] = color
        return base

    def draw_dots(panel):
        vis = panel.copy()
        if player_pos:
            cv2.circle(vis, player_pos, 7, (255, 255, 255), -1)
            cv2.putText(vis, "P", (player_pos[0]+8, player_pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if dest_pos:
            cv2.circle(vis, dest_pos, 7, (255, 100, 0), -1)
            cv2.putText(vis, "D", (dest_pos[0]+8, dest_pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
        return vis

    p1 = label(draw_dots(img.copy()),                       "Original Map")
    p2 = label(to_bgr(gray_px,  [0, 255, 255]),             "Gray Lines Detected")
    p3 = label(to_bgr(wall_px,  [0, 80, 200]),              "Wall Color Mask")
    p4 = label(draw_dots(to_bgr(occ_grid*255, [0,200,80])), "Final Occupancy Grid")

    return np.vstack([np.hstack([p1, p2]),
                      np.hstack([p3, p4])])


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    IMAGE_PATH = "map_full.png"

    print("[INFO] Loading map...")
    img = load_map(IMAGE_PATH)
    print(f"  Size: {img.shape[1]}x{img.shape[0]} px")

    print("[INFO] Building occupancy grid...")
    occ_grid, gray_px, wall_px = build_occupancy_grid(img)
    print(f"  Gray pixels masked : {(gray_px > 0).sum()}")
    print(f"  Wall pixels (raw)  : {(wall_px > 0).sum()}")
    print(f"  Wall coverage      : {occ_grid.mean()*100:.2f}%")

    print("[INFO] Locating player (white dot) and destination (blue dot)...")
    player_pos, dest_pos = find_player_and_dest(img)
    print(f"  Player (pixel)     : {player_pos}")
    print(f"  Destination (pixel): {dest_pos}")

    print("[INFO] Rendering visualization...")
    vis = visualize(img, gray_px, wall_px, occ_grid, player_pos, dest_pos)
    cv2.imshow("Wall Detection Final  |  Press any key to exit", vis)
    cv2.imwrite("wall_detection_final.png", vis)
    print("[INFO] Saved: wall_detection_final.png")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ── Save for RRT step ──────────────────────────────────────────────────
    np.save("occupancy_grid.npy", occ_grid)
    np.save("player_pos.npy",     np.array(player_pos))
    np.save("dest_pos.npy",       np.array(dest_pos))
    print("[INFO] Saved: occupancy_grid.npy | player_pos.npy | dest_pos.npy")
    print("[INFO] ✓ Ready for Step 2: RRT path planning!")