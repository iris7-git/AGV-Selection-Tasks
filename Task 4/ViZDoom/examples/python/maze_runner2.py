from argparse import ArgumentParser
import os
from random import choice
import vizdoom as vzd
import numpy as np
import cv2
import math
import random

DEFAULT_CONFIG = "../../scenarios/level1.cfg"

# ═══════════════════════════════════════════════════════════════════════════════
#  OCCUPANCY GRID BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def color_mask_rgb(img, rgb, tol):
    r, g, b = rgb
    lo = np.array([max(0,b-tol), max(0,g-tol), max(0,r-tol)], dtype=np.uint8)
    hi = np.array([min(255,b+tol), min(255,g+tol), min(255,r+tol)], dtype=np.uint8)
    return cv2.inRange(img, lo, hi)


def build_occupancy_grid(automap, player_pixel):
    """
    Correct occupancy grid:
      1. Detect dark wall pixels RGB≈(52,29,8)
      2. Dilate 1px to thicken lines
      3. Flood-fill from player pixel → only connected navigable space is free
    Returns occ (H×W uint8): 0=free, 1=obstacle
    """
    H, W = automap.shape[:2]
    kernel = np.ones((3, 3), np.uint8)

    # Mask gray sector lines → replace with background
    gray_mask = color_mask_rgb(automap, (135, 132, 130), 25)
    cleaned   = automap.copy()
    cleaned[gray_mask > 0] = [67, 87, 111]

    # Detect actual dark wall pixels
    wall_px = color_mask_rgb(cleaned, (52, 29, 8), 30)
    wall_d  = cv2.dilate(wall_px, kernel, iterations=1)

    # Find free seed near player
    px, py = int(round(player_pixel[0])), int(round(player_pixel[1]))
    if wall_d[py, px] > 0:
        found = False
        for r in range(1, 15):
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    nx, ny = px+dx, py+dy
                    if 0<=nx<W and 0<=ny<H and wall_d[ny,nx]==0:
                        px, py = nx, ny; found = True; break
                if found: break
            if found: break

    # Flood fill from player → marks navigable region
    flood = wall_d.copy()
    cv2.floodFill(flood, None, (px, py), 128)

    occ = np.ones((H, W), dtype=np.uint8)
    occ[flood == 128] = 0
    return occ


def find_player_pixel(automap):
    """White dot = player. Ignores top-border artifact."""
    white_mask = cv2.inRange(automap,
                             np.array([200, 200, 200]),
                             np.array([255, 255, 255]))
    white_mask[:5,:]  = 0; white_mask[-5:,:] = 0
    white_mask[:,:5]  = 0; white_mask[:,-5:] = 0
    n, _, stats, cents = cv2.connectedComponentsWithStats(white_mask)
    if n > 1:
        lg = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (int(cents[lg][0]), int(cents[lg][1]))
    return None


def find_dest_pixel(automap):
    """Blue dot = destination."""
    blue_mask = cv2.inRange(automap,
                            np.array([100,  0,  0]),
                            np.array([255, 80, 80]))
    M = cv2.moments(blue_mask)
    if M["m00"] > 0:
        return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  RRT PATH PLANNING
# ═══════════════════════════════════════════════════════════════════════════════

class RRTNode:
    __slots__ = ['x','y','parent']
    def __init__(self, x, y): self.x=x; self.y=y; self.parent=None


def _collision_free(occ, x0, y0, x1, y1):
    W, H = occ.shape[1], occ.shape[0]
    d = math.hypot(x1-x0, y1-y0)
    if d == 0: return True
    steps = max(int(d)+1, 2)
    ts = np.linspace(0, 1, steps)
    xs = np.clip(np.round(x0+ts*(x1-x0)).astype(int), 0, W-1)
    ys = np.clip(np.round(y0+ts*(y1-y0)).astype(int), 0, H-1)
    return bool(np.all(occ[ys, xs] == 0))


def plan_rrt(occ, start, goal,
             max_iter=20000, step=6, goal_radius=8, goal_bias=0.15):
    H, W = occ.shape
    free_ys, free_xs = np.where(occ == 0)
    if len(free_xs) == 0: return None, []

    sn = RRTNode(*start); gn = RRTNode(*goal)
    nodes = [sn]
    xs_a  = np.array([float(start[0])])
    ys_a  = np.array([float(start[1])])

    random.seed(42)
    for i in range(max_iter):
        if random.random() < goal_bias:
            rx, ry = float(goal[0]), float(goal[1])
        else:
            idx = random.randint(0, len(free_xs)-1)
            rx, ry = float(free_xs[idx]), float(free_ys[idx])

        ni   = int(np.argmin((xs_a-rx)**2 + (ys_a-ry)**2))
        near = nodes[ni]
        d    = math.hypot(rx-near.x, ry-near.y)
        if d < 1e-9: continue
        if d < step: nx, ny = rx, ry
        else:
            a = math.atan2(ry-near.y, rx-near.x)
            nx = near.x + step*math.cos(a)
            ny = near.y + step*math.sin(a)

        xi, yi = int(round(nx)), int(round(ny))
        if not(0<=xi<W and 0<=yi<H) or occ[yi,xi]!=0: continue
        if not _collision_free(occ, near.x, near.y, nx, ny): continue

        nn = RRTNode(nx, ny); nn.parent = near
        nodes.append(nn)
        xs_a = np.append(xs_a, nx)
        ys_a = np.append(ys_a, ny)

        if math.hypot(nn.x-goal[0], nn.y-goal[1]) < goal_radius:
            if _collision_free(occ, nn.x, nn.y, goal[0], goal[1]):
                gn.parent = nn; nodes.append(gn)
                print(f"[RRT] Goal reached at iter {i+1}, tree={len(nodes)}")
                path = []
                n = gn
                while n: path.append((int(round(n.x)), int(round(n.y)))); n=n.parent
                return _smooth(occ, list(reversed(path))), nodes

    print(f"[RRT] Failed after {max_iter} iters"); return None, nodes


def _smooth(occ, path, passes=5):
    for _ in range(passes):
        i=0; s=[path[0]]
        while i < len(path)-1:
            j = len(path)-1
            while j > i+1:
                if _collision_free(occ, path[i][0],path[i][1],
                                        path[j][0],path[j][1]): break
                j -= 1
            s.append(path[j]); i=j
        path = s
    return path


def visualize_path(automap, occ, path, nodes=None):
    vis = automap.copy()
    vis[occ==1] = (vis[occ==1]*0.3).astype(np.uint8)
    if nodes:
        for n in nodes:
            if n.parent:
                cv2.line(vis,(int(n.parent.x),int(n.parent.y)),
                              (int(n.x),int(n.y)),(40,40,40),1)
    if path:
        for i in range(len(path)-1):
            cv2.line(vis, path[i], path[i+1], (0,230,80), 2)
        for pt in path:
            cv2.circle(vis, pt, 4, (0,255,255), -1)
    return vis


# ═══════════════════════════════════════════════════════════════════════════════
#  COORDINATE CALIBRATION  (pixel ↔ game world)
# ═══════════════════════════════════════════════════════════════════════════════

class CoordCalibrator:
    """
    Learns pixel↔world transform from (pixel_pos, world_pos) pairs.
    Two samples are enough; more improves accuracy.
    """
    def __init__(self):
        self.samples = []
        self.scale = self.ox = self.oy = None

    def add_sample(self, pixel_pos, world_pos):
        # Avoid duplicate samples
        if self.samples and self.samples[-1][0] == pixel_pos:
            return
        self.samples.append((pixel_pos, world_pos))
        if len(self.samples) >= 2:
            self._fit()

    def _fit(self):
        # Use first and latest sample for scale
        (p0,w0),(p1,w1) = self.samples[0], self.samples[-1]
        dpix = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
        dwor = math.hypot(w1[0]-w0[0], w1[1]-w0[1])
        if dpix < 1.0: return
        self.scale = dwor / dpix
        # wx = ox + px*scale ;  wy = oy - py*scale  (image Y flipped vs world Y)
        self.ox = w0[0] - p0[0]*self.scale
        self.oy = w0[1] + p0[1]*self.scale

    def is_ready(self):
        return self.scale is not None

    def pixel_to_world(self, px, py):
        if not self.is_ready(): return None
        return (self.ox + px*self.scale,
                self.oy - py*self.scale)

    def path_to_world(self, pixel_path):
        if not self.is_ready(): return None
        return [self.pixel_to_world(px,py) for px,py in pixel_path]


# ═══════════════════════════════════════════════════════════════════════════════
#  NAVIGATION CONTROLLER
#  Two-phase per waypoint:
#    PHASE 1 — ROTATE in place until aligned (no forward movement → no overshoot)
#    PHASE 2 — MOVE FORWARD (tiny speed, small turn corrections)
#  Deadlock detection: if position barely changes for N steps → back up & retry
# ═══════════════════════════════════════════════════════════════════════════════

# ── Tunable constants ─────────────────────────────────────────────────────────
TURN_DELTA_COARSE  = 3        # deg/tick rotating in place
ALIGN_ENTER        = 5.0      # deg error: stop rotating, start moving forward
ALIGN_EXIT         = 15.0     # deg error: stop moving, go back to rotating
                              # Near ±180° flips: biased to turn right to avoid oscillation
ARRIVAL_RADIUS     = 35.0     # game-units: waypoint considered reached
DEADLOCK_STEPS     = 150       # ticks window for deadlock detection
DEADLOCK_MOVE_THR  = 8.0      # min game-units in window to NOT be stuck
BACKUP_STEPS       = 25       # ticks of pure backward movement to escape
# ─────────────────────────────────────────────────────────────────────────────

class NavState:
    ROTATE  = "ROTATE"
    FORWARD = "FORWARD"
    BACKUP  = "BACKUP"
    DONE    = "DONE"


class NavigationController:
    def __init__(self, world_waypoints):
        self.waypoints        = world_waypoints
        self.wp_idx           = 1
        self.state            = NavState.ROTATE
        self.backup_ctr       = 0
        self.dl_pos_history   = []
        self.committed_dir    = None   # +1=right, -1=left, None=not yet chosen

    def get_action(self, pos_x, pos_y, angle_deg):
        """
        CRITICAL RULE: never send non-zero TURN_DELTA and MOVE_FORWARD=True
        in the same tick — that causes the engine shake/wobble.

        ROTATE state  → pure turn tick:    [F,F,F,F, delta]  (no movement)
        FORWARD state → pure forward tick: [F,F,T,F, 0]      (no turn delta)
        BACKUP state  → pure back tick:    [F,F,F,T, 0]
        """
        if self.state == NavState.DONE:
            return [False, False, False, False, 0]

        self._update_deadlock(pos_x, pos_y)

        # Deadlock check
        if self.state != NavState.BACKUP and self._is_deadlocked():
            print(f"[NAV] Deadlock at ({pos_x:.0f},{pos_y:.0f}) → backing up")
            self.state      = NavState.BACKUP
            self.backup_ctr = BACKUP_STEPS
            self.dl_pos_history.clear()

        if self.state == NavState.BACKUP:
            return self._do_backup()

        # Current waypoint
        tx, ty = self.waypoints[self.wp_idx]
        dist   = math.hypot(tx - pos_x, ty - pos_y)

        # Arrival
        if dist < ARRIVAL_RADIUS:
            self.wp_idx += 1
            if self.wp_idx >= len(self.waypoints):
                print("[NAV] Destination reached!")
                self.state = NavState.DONE
                return [False, False, False, False, 0]
            print(f"[NAV] wp {self.wp_idx-1} reached → heading to {self.wp_idx}/{len(self.waypoints)-1}")
            self.state         = NavState.ROTATE
            self.committed_dir = None   # reset direction for new waypoint
            tx, ty = self.waypoints[self.wp_idx]

        desired = math.degrees(math.atan2(ty - pos_y, tx - pos_x))
        err     = _angle_diff(desired, angle_deg)

        # ── ROTATE: pure turn, zero movement ─────────────────────────────
        if self.state == NavState.ROTATE:
            if abs(err) <= ALIGN_ENTER:
                self.state         = NavState.FORWARD
                self.committed_dir = None   # release lock when aligned
                # fall through to FORWARD
            else:
                # Commit to a direction once and LOCK IT until aligned.
                # Never re-evaluate mid-turn — this prevents oscillation at boundaries.
                if self.committed_dir is None:
                    self.committed_dir = 1 if err >= 0 else -1
                delta = self.committed_dir * TURN_DELTA_COARSE
                return [False, False, False, False, delta]

        # ── FORWARD: pure move, zero turn delta ───────────────────────────
        if abs(err) > ALIGN_EXIT:
            self.state         = NavState.ROTATE
            self.committed_dir = None   # reset so direction re-evaluated fresh
            delta = int(math.copysign(TURN_DELTA_COARSE, err))
            return [False, False, False, False, delta]

        return [False, False, True, False, 0]

    def is_done(self):
        return self.state == NavState.DONE

    def current_wp_idx(self):
        return self.wp_idx

    def _update_deadlock(self, px, py):
        self.dl_pos_history.append((px, py))
        if len(self.dl_pos_history) > DEADLOCK_STEPS:
            self.dl_pos_history.pop(0)

    def _is_deadlocked(self):
        if len(self.dl_pos_history) < DEADLOCK_STEPS:
            return False
        p0, p1 = self.dl_pos_history[0], self.dl_pos_history[-1]
        return math.hypot(p1[0]-p0[0], p1[1]-p0[1]) < DEADLOCK_MOVE_THR

    def _do_backup(self):
        self.backup_ctr -= 1
        if self.backup_ctr <= 0:
            print("[NAV] Backup done → re-aligning")
            self.state = NavState.ROTATE
        return [False, False, False, True, 0]   # pure backward, no turn


def _angle_diff(desired, current):
    """Shortest-path error in [-180, 180]. Near ±180 bias right."""
    d = (desired - current) % 360
    if d > 180:
        d -= 360
    if d < -175:      # ambiguous flip → bias right
        d = abs(d)
    return d



# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    """
    ############################################################################################################################################################
    These are pre-set configurations for level1 and level2 of the task, please dont change them
    ############################################################################################################################################################
    """

    parser = ArgumentParser("ViZDoom maze runner with RRT path planning.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario.")
    args = parser.parse_args()

    game = vzd.DoomGame()
    game.load_config(args.config)
    game.set_screen_format(vzd.ScreenFormat.BGR24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)
    game.add_game_args("+viz_am_center 1")

    """
    ##############################################################################################################################################################
    Feel free to change anything after this
    ##############################################################################################################################################################
    """

    game.set_mode(vzd.Mode.PLAYER)
    game.set_available_buttons([
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_RIGHT,
        vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.TURN_LEFT_RIGHT_DELTA,
    ])
    game.set_available_game_variables([
        vzd.GameVariable.POSITION_X,
        vzd.GameVariable.POSITION_Y,
        vzd.GameVariable.ANGLE,
    ])
    game.init()

    episodes   = 10
    sleep_time = 0.028   # keep this; effective speed is controlled by TURN/MOVE params

    for ep in range(episodes):
        print(f"\n{'='*60}\nEpisode {ep+1}\n{'='*60}")
        game.new_episode()

        # ════════════════════════════════════════════════════════════════
        #  STAGE 1 — PATH PLANNING  (no movement, game is paused in PLAYER mode
        #             but we just don't call make_action yet)
        # ════════════════════════════════════════════════════════════════
        print("[STAGE 1] Path planning...")

        state     = game.get_state()
        automap   = state.automap_buffer
        game_vars = state.game_variables
        pos_x, pos_y, angle = game_vars[0], game_vars[1], game_vars[2]

        print(f"  World pos=({pos_x:.1f},{pos_y:.1f})  angle={angle:.1f}°")

        player_px = find_player_pixel(automap)
        dest_px   = find_dest_pixel(automap)
        print(f"  Player pixel={player_px}  Dest pixel={dest_px}")

        if player_px is None or dest_px is None:
            print("  [WARN] Could not detect dots — skipping episode")
            continue

        # Build occupancy grid
        occ = build_occupancy_grid(automap, player_px)
        print(f"  Navigable pixels: {(occ==0).sum()}")

        # RRT
        pixel_path, rrt_nodes = plan_rrt(occ, player_px, dest_px)
        if pixel_path is None:
            print("  [WARN] RRT failed — skipping episode")
            continue
        print(f"  Path: {len(pixel_path)} waypoints → {pixel_path}")

        # Show planned path (freeze display until key or short timeout)
        vis = visualize_path(automap, occ, pixel_path, rrt_nodes)
        cv2.circle(vis, player_px, 8, (255,255,255), -1)
        cv2.circle(vis, dest_px,   8, (255,100, 0),  -1)
        cv2.putText(vis,"STAGE 1: Path planned — press any key to start movement",
                    (5,15),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,255,255),1)
        cv2.imshow("ViZDoom Map Buffer", vis)
        cv2.imwrite(f"path_ep{ep+1}.png", vis)
        print("[STAGE 1] Done. Press any key in the map window to begin movement.")
        cv2.waitKey(0)   # wait for keypress before moving

        # ════════════════════════════════════════════════════════════════
        #  STAGE 2 — MOVEMENT
        #  Calibrate pixel↔world coords, then navigate waypoint by waypoint.
        # ════════════════════════════════════════════════════════════════
        print("[STAGE 2] Navigation starting...")

        # ── Calibration: nudge forward a few ticks to get 2 pixel samples ──
        # We cannot derive scale from one point alone.
        # Strategy: move forward for CALIB_NUDGE_STEPS ticks, collect
        # a second (pixel, world) pair, then stop and plan.
        CALIB_NUDGE_STEPS = 20
        calib = CoordCalibrator()
        calib.add_sample(player_px, (pos_x, pos_y))

        print("  [CALIB] Nudging forward to calibrate pixel↔world scale...")
        for _ in range(CALIB_NUDGE_STEPS):
            if game.is_episode_finished(): break
            game.make_action([False, False, True, False, 0])  # move forward only
            s2 = game.get_state()
            if s2 is None: break
            gv2 = s2.game_variables
            pp2 = find_player_pixel(s2.automap_buffer)
            if pp2 is not None:
                calib.add_sample(pp2, (gv2[0], gv2[1]))

        if not calib.is_ready():
            print("  [WARN] Calibration failed (player may be blocked) — skipping")
            continue

        print(f"  [CALIB] Done — scale={calib.scale:.4f}  "
              f"offset=({calib.ox:.1f},{calib.oy:.1f})")

        # Convert pixel path → world coords and build navigator
        world_path = calib.path_to_world(pixel_path)
        nav        = NavigationController(world_path)

        # Re-read state after nudge (position has changed slightly)
        state     = game.get_state()
        automap   = state.automap_buffer
        game_vars = state.game_variables
        pos_x, pos_y, angle = game_vars[0], game_vars[1], game_vars[2]
        step_count = 0

        while not game.is_episode_finished():

            state     = game.get_state()
            automap   = state.automap_buffer
            game_vars = state.game_variables
            pos_x, pos_y, angle = game_vars[0], game_vars[1], game_vars[2]

            cur_px = find_player_pixel(automap)

            # ── Choose action ─────────────────────────────────────────
            action = nav.get_action(pos_x, pos_y, angle)
            if nav.is_done():
                print("[STAGE 2] Agent reached destination — ending episode.")
                break

            game.make_action(action)

            # ── HUD ───────────────────────────────────────────────────
            if automap is not None:
                hud = automap.copy()
                # Darken non-navigable
                hud[occ==1] = (hud[occ==1]*0.3).astype(np.uint8)
                # Draw planned path
                for i in range(len(pixel_path)-1):
                    cv2.line(hud, pixel_path[i], pixel_path[i+1], (0,200,60), 1)
                for pt in pixel_path:
                    cv2.circle(hud, pt, 3, (0,255,255), -1)
                # Current player dot
                if cur_px:
                    cv2.circle(hud, cur_px, 5, (255,255,255), -1)
                # Destination
                if dest_px:
                    cv2.circle(hud, dest_px, 5, (255,100,0), -1)
                # Status text
                cv2.putText(hud,
                    f"STAGE 2 | step={step_count} "
                    f"wp={nav.current_wp_idx()}/{len(pixel_path)-1} "
                    f"state={nav.state} pos=({pos_x:.0f},{pos_y:.0f}) ang={angle:.0f}",
                    (5,12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,255,255), 1)

                cv2.imshow("ViZDoom Map Buffer", hud)

            cv2.waitKey(int(sleep_time * 1000))
            step_count += 1

            if step_count % 200 == 0:
                print(f"  step={step_count}  pos=({pos_x:.0f},{pos_y:.0f})"
                      f"  ang={angle:.0f}°  wp={nav.current_wp_idx()}"
                      f"  state={nav.state}")

        print(f"Episode {ep+1} finished — {step_count} steps.")

    cv2.destroyAllWindows()
    game.close()