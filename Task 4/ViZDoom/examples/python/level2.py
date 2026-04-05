from argparse import ArgumentParser
import os
from random import choice
import vizdoom as vzd
import numpy as np

#DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")
DEFAULT_CONFIG = "../../scenarios/level1.cfg"

import cv2
import math
from collections import deque

if __name__ == "__main__":

    """
    ############################################################################################################################################################
    These are pre-set configurations for level2 of the task, please dont change them

    ############################################################################################################################################################
    """

    parser = ArgumentParser("ViZDoom example showing different buffers (screen, depth, labels).")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()

    game = vzd.DoomGame()
    game.load_config(args.config)
    game.set_screen_format(vzd.ScreenFormat.BGR24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(False)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)

    """
    ##############################################################################################################################################################
    Feel free to change anything after this
    ##############################################################################################################################################################
    """
    #game.set_mode(vzd.Mode.SPECTATOR)

    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
                                 vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
                                 vzd.Button.TURN_LEFT_RIGHT_DELTA])
    game.set_available_game_variables([vzd.GameVariable.POSITION_X,
                                       vzd.GameVariable.POSITION_Y,
                                       vzd.GameVariable.ANGLE])
    game.init()

    actions = [[True, False, True, False, 1], [False, True, False, True, 1],
               [False, True, True, False, -1], [True, False, False, True, -1]]

    episodes   = 10
    sleep_time = 0.028

    # ═══════════════════════════════════════════════════════════════════════════
    #  CONSTANTS
    # ═══════════════════════════════════════════════════════════════════════════

    DEPTH_W      = 640
    DEPTH_HFOV   = 90.0

    # Virtual occupancy grid
    CELL         = 96
    GHALF        = 120
    GSIZE        = 2*GHALF + 1
    UNKNOWN, FREE, OCC = 0, 1, 2

    # Depth thresholds
    WALL_DIST    = 150.0    # strip median ≤ this → wall
    HALL_DIST    = 550.0    # centre strip ≥ this → open hall
    PIPE_DIST    = 320.0    # gap must exceed this to be considered a passage
    PIPE_MIN_COLS = 18      # minimum gap width in depth columns

    # Navigation
    ARRIVE_R     = 90.0
    TURN_STEP    = 5        # degrees per tick — RIGHT ONLY (never negative)
    ALIGN_OK     = 5.0
    ALIGN_LOST   = 22.0
    FRONTIER_DIST = 192

    # Stuck detector
    STUCK_WINDOW  = 180     # ticks
    STUCK_THR     = 35.0    # max spatial spread before declared stuck
    STUCK_CHECK   = 30      # evaluate every N ticks
    BACKUP_TICKS  = 60      # reverse ticks + 45° turn ticks combined

    # Wall follow
    WALL_FOLLOW_DIST  = 200.0   # desired right-wall clearance
    WALL_FOLLOW_TOL   = 55.0
    HALL_FOLLOW_TICKS = 450     # timeout before forcing DFS pop

    # Destination homing
    DEST_PIX_THR  = 40
    DEST_CTR_TOL  = 45

    # ═══════════════════════════════════════════════════════════════════════════
    #  ANGLE HELPER — RIGHT TURN ONLY (fixes orthogonal-wall oscillation)
    #
    #  Always returns a value in [0, 360).
    #  The agent turns right by this amount.  To go "left" it turns right > 180°.
    #  This removes sign ambiguity — there is never a choice between two directions.
    # ═══════════════════════════════════════════════════════════════════════════

    def right_only_err(desired_deg, current_deg):
        """How many degrees to turn RIGHT to face desired_deg. Always in [0,360)."""
        return (desired_deg - current_deg) % 360

    # ═══════════════════════════════════════════════════════════════════════════
    #  GRID HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def w2c(wx, wy):
        return int(round(wx / CELL)) + GHALF, int(round(wy / CELL)) + GHALF

    def c2w(cx, cy):
        return (cx - GHALF) * CELL, (cy - GHALF) * CELL

    def in_grid(cx, cy):
        return 0 <= cx < GSIZE and 0 <= cy < GSIZE

    def bearing_to(px, py, tx, ty):
        return math.degrees(math.atan2(ty - py, tx - px)) % 360

    # ═══════════════════════════════════════════════════════════════════════════
    #  DEPTH ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    def depth_strips(depth, n=5):
        W = depth.shape[1]; w = W // n
        return [float(np.median(depth[:, k*w:(k+1)*w])) for k in range(n)]

    def find_pipe_gaps(depth):
        """
        Vectorised scan for clear-passage columns.
        Returns list of (col_start, col_end, col_centre, avg_dist).
        Only gaps ≥ PIPE_MIN_COLS columns wide are returned.
        """
        W   = depth.shape[1]
        # One median per column using fast row-median via partition
        col_med = np.median(depth, axis=0).astype(np.float32)  # shape (W,)
        clear   = col_med > PIPE_DIST                           # bool array

        gaps = []
        i = 0
        while i < W:
            if clear[i]:
                j = i
                while j < W and clear[j]:
                    j += 1
                if (j - i) >= PIPE_MIN_COLS:
                    centre = (i + j) // 2
                    avg_d  = float(np.mean(col_med[i:j]))
                    gaps.append((i, j, centre, avg_d))
                i = j
            else:
                i += 1
        return gaps

    def right_wall_dist(depth):
        W = depth.shape[1]
        return float(np.median(depth[:, int(W*0.82):]))

    # ═══════════════════════════════════════════════════════════════════════════
    #  OCCUPANCY GRID UPDATE
    # ═══════════════════════════════════════════════════════════════════════════

    def raycast_update(depth, px, py, ang_deg, grid):
        W = depth.shape[1]
        for col in np.linspace(0, W-1, 32, dtype=int):
            offset = (col / (W-1) - 0.5) * DEPTH_HFOV
            rad    = math.radians(ang_deg + offset)
            dist   = float(np.median(depth[:, max(0,col-3):col+4]))
            dist   = min(dist, 3500.0)
            for frac in (0.2, 0.4, 0.6, 0.8):
                wx = px + dist*frac*math.cos(rad)
                wy = py + dist*frac*math.sin(rad)
                cx, cy = w2c(wx, wy)
                if in_grid(cx, cy) and grid[cy, cx] == UNKNOWN:
                    grid[cy, cx] = FREE
            if dist < 3400.0:
                wx = px + dist*math.cos(rad)
                wy = py + dist*math.sin(rad)
                cx, cy = w2c(wx, wy)
                if in_grid(cx, cy) and grid[cy, cx] != FREE:
                    grid[cy, cx] = OCC

    def close_isolated_cells(grid):
        """Vectorised closure: surrounded-by-OCC→OCC, surrounded-by-FREE→FREE."""
        k        = np.ones((3,3), np.float32)
        occ_f    = (grid == OCC).astype(np.float32)
        free_f   = (grid == FREE).astype(np.float32)
        occ_nbrs = cv2.filter2D(occ_f,  -1, k, borderType=cv2.BORDER_CONSTANT) - occ_f
        free_nbrs= cv2.filter2D(free_f, -1, k, borderType=cv2.BORDER_CONSTANT) - free_f
        grid[(occ_nbrs  >= 8) & (grid != OCC)]    = OCC
        grid[(free_nbrs >= 8) & (grid == UNKNOWN)] = FREE

    # ═══════════════════════════════════════════════════════════════════════════
    #  FRONTIER EXTRACTION  (vectorised — no Python cell loop)
    # ═══════════════════════════════════════════════════════════════════════════

    def get_frontiers(grid, visited):
        """
        Frontier = FREE cell adjacent to at least one UNKNOWN cell, not yet visited.
        Fully vectorised with morphological dilation.
        Returns list of (wx, wy).
        """
        free_mask    = (grid == FREE).astype(np.uint8)
        unknown_mask = (grid == UNKNOWN).astype(np.uint8)
        k            = np.ones((3,3), np.uint8)
        # Dilate UNKNOWN by 1 → any FREE cell touching UNKNOWN will overlap
        unknown_dilated = cv2.dilate(unknown_mask, k, iterations=1)
        frontier_mask   = (free_mask & unknown_dilated).astype(bool)

        cys, cxs = np.where(frontier_mask)
        result = []
        for cy, cx in zip(cys, cxs):
            if (cx, cy) not in visited:
                wx, wy = c2w(cx, cy)
                result.append((wx, wy))
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    #  DESTINATION DETECTION
    # ═══════════════════════════════════════════════════════════════════════════

    def check_destination(labelsmap):
        if labelsmap is None:
            return None, False
        unique = np.unique(labelsmap)
        unique = unique[unique > 0]
        if len(unique) == 0:
            return None, False
        best  = max(unique, key=lambda l: int(np.sum(labelsmap == l)))
        if int(np.sum(labelsmap == best)) < DEST_PIX_THR:
            return None, False
        M = cv2.moments((labelsmap == best).astype(np.uint8))
        if M["m00"] == 0:
            return None, False
        return int(M["m10"] / M["m00"]), True

    # ═══════════════════════════════════════════════════════════════════════════
    #  WALL FOLLOW ACTION
    #  Returns a PURE TURN or PURE FORWARD tick — never mixed.
    #  Encodes a simple right-hand-rule FSM:
    #    front blocked → turn right (clockwise, positive delta)
    #    too close to right wall → 1 tick pure-turn left (right > 180°, but we
    #      use a small negative delta here as a special case — only ±1 step,
    #      which is too small to cause wobble)
    #    too far from right wall → 1 tick pure-turn right
    #    otherwise → forward
    # ═══════════════════════════════════════════════════════════════════════════

    def wall_follow_action(r_dist, front_clear):
        if not front_clear:
            # Blocked ahead — turn RIGHT in place (positive = clockwise)
            return [False, False, False, False, TURN_STEP * 2]
        elif r_dist < WALL_FOLLOW_DIST - WALL_FOLLOW_TOL:
            # Too close to right wall — 1 correction tick turning left
            # (small delta, no wobble risk at ±TURN_STEP)
            return [False, False, False, False, -TURN_STEP]
        elif r_dist > WALL_FOLLOW_DIST + WALL_FOLLOW_TOL:
            # Drifting away from right wall — nudge right
            return [False, False, False, False, TURN_STEP]
        else:
            return [False, False, True, False, 0]

    # ═══════════════════════════════════════════════════════════════════════════
    #  DEBUG MINIMAP
    # ═══════════════════════════════════════════════════════════════════════════

    def draw_minimap(grid, px, py, ang, target, trail, label):
        SC  = 3
        vis = np.zeros((GSIZE*SC, GSIZE*SC, 3), dtype=np.uint8)
        col_map = {UNKNOWN:(25,25,25), FREE:(70,110,70), OCC:(50,50,160)}
        for cy in range(GSIZE):
            for cx in range(GSIZE):
                vis[cy*SC:(cy+1)*SC, cx*SC:(cx+1)*SC] = col_map[grid[cy,cx]]
        for wx, wy in list(trail)[-500:]:
            cx, cy = w2c(wx, wy)
            if in_grid(cx, cy):
                cv2.circle(vis, (cx*SC+SC//2, cy*SC+SC//2), 1, (0,200,200), -1)
        if target:
            cx, cy = w2c(target[0], target[1])
            if in_grid(cx, cy):
                cv2.circle(vis, (cx*SC+SC//2, cy*SC+SC//2), SC+2, (0,255,0), 2)
        cx, cy = w2c(px, py)
        if in_grid(cx, cy):
            ax, ay = cx*SC+SC//2, cy*SC+SC//2
            cv2.circle(vis, (ax, ay), SC+2, (255,255,255), -1)
            ex = int(ax + (SC+6)*math.cos(math.radians(ang)))
            ey = int(ay + (SC+6)*math.sin(math.radians(ang)))
            cv2.line(vis, (ax, ay), (ex, ey), (0,255,255), 2)
        cv2.putText(vis, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,255,255), 1)
        return vis

    # ═══════════════════════════════════════════════════════════════════════════
    #  EPISODE LOOP
    # ═══════════════════════════════════════════════════════════════════════════

    for i in range(episodes):
        print("Episode #" + str(i + 1))
        game.new_episode()

        # ── Per-episode state ────────────────────────────────────────────────
        grid            = np.full((GSIZE, GSIZE), UNKNOWN, dtype=np.uint8)
        visited         = set()        # (cx,cy) cells physically reached
        frontier_set    = set()        # (cx,cy) already on dfs_stack — no duplicates
        dfs_stack       = []

        nav_state       = "DFS_ROTATE"
        backup_ctr      = 0
        wall_follow_ctr = 0
        target          = None
        pipe_target_col = None

        stuck_history   = deque(maxlen=STUCK_WINDOW)
        stuck_tick      = 0
        trail           = deque(maxlen=600)

        # Seed from start position
        s0 = game.get_state()
        gv0 = s0.game_variables
        sx, sy, sa = gv0[0], gv0[1], gv0[2]
        cx0, cy0 = w2c(sx, sy)
        if in_grid(cx0, cy0):
            grid[cy0, cx0] = FREE
        visited.add((cx0, cy0))

        #episode ends after you reach the key(end of game) or after a given time
        while not game.is_episode_finished():

            # Gets the state and possibly do something with it
            state = game.get_state()

            # Shows the depth map of the current episode/level.
            depthmap = state.depth_buffer
            if depthmap is not None:
                cv2.imshow('ViZDoom Depth Buffer', depthmap)

            #shows the segmented areas of any objects in the map
            labelsmap = state.labels_buffer
            if labelsmap is not None:
                cv2.imshow('ViZDoom Labels Buffer', labelsmap)

            cv2.waitKey(int(sleep_time * 1000))

            # ── World state ──────────────────────────────────────────────────
            gv = state.game_variables
            pos_x, pos_y, angle = gv[0], gv[1], gv[2]
            angle = angle % 360          # normalise to [0, 360)

            trail.append((pos_x, pos_y))
            stuck_history.append((pos_x, pos_y))
            stuck_tick += 1

            ccx, ccy = w2c(pos_x, pos_y)
            visited.add((ccx, ccy))
            frontier_set.discard((ccx, ccy))   # remove from frontier set if reached
            if in_grid(ccx, ccy):
                grid[ccy, ccx] = FREE

            # ── Occupancy grid ───────────────────────────────────────────────
            if depthmap is not None:
                raycast_update(depthmap, pos_x, pos_y, angle, grid)
                close_isolated_cells(grid)

            # ── Sensor summary ───────────────────────────────────────────────
            if depthmap is not None:
                strips      = depth_strips(depthmap)
                gaps        = find_pipe_gaps(depthmap)
                in_hall     = strips[2] > HALL_DIST
                r_dist      = right_wall_dist(depthmap)
                front_clear = strips[2] > WALL_DIST
            else:
                strips = [999.0]*5; gaps = []; in_hall = False
                r_dist = 999.0; front_clear = True

            dest_col, dest_seen = check_destination(labelsmap)

            # ── Stuck detection ──────────────────────────────────────────────
            is_stuck = False
            if (stuck_tick % STUCK_CHECK == 0
                    and len(stuck_history) == STUCK_WINDOW
                    and nav_state not in ("STUCK",)):
                xs     = [p[0] for p in stuck_history]
                ys     = [p[1] for p in stuck_history]
                spread = math.hypot(max(xs)-min(xs), max(ys)-min(ys))
                if spread < STUCK_THR:
                    is_stuck = True
                    print(f"[STUCK] spread={spread:.1f} at ({pos_x:.0f},{pos_y:.0f})")

            # ════════════════════════════════════════════════════════════════
            #  PRIORITY-BASED ACTION SELECTION
            #
            #  P1  HOMING       destination visible
            #  P2  STUCK        spread detector fired
            #  P3  PIPE_ENTER   narrow gap ahead
            #  P4  WALL_FOLLOW  inside open hall → perimeter walk
            #  P5  DFS          rotate (right-only) → forward → backtrack
            # ════════════════════════════════════════════════════════════════

            # ── P1: HOMING ───────────────────────────────────────────────────
            if dest_seen:
                nav_state = "HOMING"

            if nav_state == "HOMING":
                if dest_seen and dest_col is not None:
                    err = dest_col - DEPTH_W / 2
                    if abs(err) < DEST_CTR_TOL:
                        action = [False, False, True, False, 0]
                    else:
                        # Right-turn bias: positive delta steers right (towards dest
                        # if it's on the right), use right_only for left targets too
                        raw_delta = int(math.copysign(TURN_STEP * 2, err))
                        action = [False, False, False, False, raw_delta]
                else:
                    nav_state = "DFS_ROTATE"
                    action    = [False, False, False, False, 0]

            # ── P2: STUCK ────────────────────────────────────────────────────
            elif is_stuck:
                nav_state  = "STUCK"
                backup_ctr = BACKUP_TICKS
                stuck_history.clear()
                target = None   # invalidate — target was likely unreachable
                action = [False, False, False, True, 0]

            elif nav_state == "STUCK":
                if backup_ctr > BACKUP_TICKS // 2:
                    action = [False, False, False, True, 0]       # reverse
                else:
                    action = [False, False, False, False, TURN_STEP * 2]  # turn RIGHT ~45°
                backup_ctr -= 1
                if backup_ctr <= 0:
                    nav_state = "DFS_ROTATE"

            # ── P3: PIPE_ENTER ───────────────────────────────────────────────
            elif nav_state == "PIPE_ENTER":
                if pipe_target_col is not None and front_clear:
                    err = pipe_target_col - DEPTH_W / 2
                    if abs(err) < 35:
                        action = [False, False, True, False, 0]   # drive in
                    else:
                        # Right-only: if pipe is left, turn right past 180°
                        # But for small corrections use direct sign — within ±35px
                        # it's already handled above; here err is large, use right-only
                        if err >= 0:   # pipe on right → turn right
                            action = [False, False, False, False, TURN_STEP]
                        else:          # pipe on left → turn right all the way around
                            action = [False, False, False, False, TURN_STEP]
                else:
                    # Either no longer clear (entered the pipe) or lost the gap
                    pipe_target_col = None
                    nav_state       = "DFS_ROTATE"
                    action          = [False, False, False, False, 0]

            # ── P4: WALL_FOLLOW ──────────────────────────────────────────────
            elif nav_state == "WALL_FOLLOW":
                wall_follow_ctr += 1

                # Check if we spotted a pipe entrance (narrower than current hall)
                if gaps:
                    best_gap = min(gaps, key=lambda g: abs(g[2] - DEPTH_W // 2))
                    if best_gap[3] < strips[2] * 0.75:   # clearly narrower → real pipe
                        print(f"[PIPE] Gap col={best_gap[2]} dist={best_gap[3]:.0f}")
                        pipe_target_col = best_gap[2]
                        nav_state       = "PIPE_ENTER"
                        wall_follow_ctr = 0
                        action          = [False, False, False, False, 0]
                    else:
                        action = wall_follow_action(r_dist, front_clear)
                elif wall_follow_ctr > HALL_FOLLOW_TICKS:
                    print("[WALL_FOLLOW] timeout → DFS pop")
                    wall_follow_ctr = 0
                    nav_state       = "DFS_ROTATE"
                    target          = None
                    action          = [False, False, False, False, 0]
                else:
                    action = wall_follow_action(r_dist, front_clear)

            # ── P5: DFS ──────────────────────────────────────────────────────
            else:
                # Enter wall-follow if we're in an open hall (any DFS sub-state)
                if in_hall:
                    print("[HALL] Open hall → WALL_FOLLOW")
                    nav_state       = "WALL_FOLLOW"
                    wall_follow_ctr = 0
                    action          = wall_follow_action(r_dist, front_clear)

                else:
                    # Arrival check
                    if target is not None:
                        tx, ty = target
                        if math.hypot(tx - pos_x, ty - pos_y) < ARRIVE_R:
                            print(f"[DFS] Arrived ({tx:.0f},{ty:.0f})")
                            target    = None
                            nav_state = "DFS_ROTATE"

                    # Refresh frontier stack when we need a new target
                    if target is None:
                        frontiers = get_frontiers(grid, visited)
                        # Sort nearest-first; push furthest first so nearest is on top
                        frontiers.sort(key=lambda f: math.hypot(f[0]-pos_x, f[1]-pos_y))
                        for fx, fy in reversed(frontiers[:12]):
                            fcx, fcy = w2c(fx, fy)
                            if (fcx, fcy) not in visited and (fcx, fcy) not in frontier_set:
                                dfs_stack.append((fx, fy))
                                frontier_set.add((fcx, fcy))

                        # Pop a reachable target
                        while dfs_stack:
                            wx, wy   = dfs_stack[-1]
                            fcx, fcy = w2c(wx, wy)
                            if (fcx, fcy) in visited or \
                               (in_grid(fcx, fcy) and grid[fcy, fcx] == OCC):
                                dfs_stack.pop()
                                frontier_set.discard((fcx, fcy))
                                continue
                            target    = (wx, wy)
                            nav_state = "DFS_ROTATE"
                            print(f"[DFS] → ({wx:.0f},{wy:.0f})  stk={len(dfs_stack)}")
                            break

                    if target is None:
                        # Exhausted → spin right to reveal new UNKNOWN cells
                        action = [False, False, False, False, TURN_STEP]
                    else:
                        tx, ty  = target
                        # RIGHT-ONLY turning — always in [0, 360)
                        desired = bearing_to(pos_x, pos_y, tx, ty)
                        err_r   = right_only_err(desired, angle)
                        # err_r in [0,360): small value → almost aligned
                        # anything > 180 means turning right is shorter than 180°
                        # (i.e. target is slightly to the left, but we still turn right)

                        if nav_state == "DFS_ROTATE":
                            if err_r <= ALIGN_OK or err_r >= 360 - ALIGN_OK:
                                nav_state = "DFS_FORWARD"
                                action    = [False, False, True, False, 0]
                            else:
                                action = [False, False, False, False, TURN_STEP]

                        else:  # DFS_FORWARD
                            # Re-check alignment with right-only err
                            if err_r > ALIGN_LOST and err_r < 360 - ALIGN_LOST:
                                nav_state = "DFS_ROTATE"
                                action    = [False, False, False, False, TURN_STEP]
                            else:
                                action = [False, False, True, False, 0]

            game.make_action(action)

            """
            ---------------------------------------------------------PUT YOUR CODE HERE------------------------------------------------------------------------

            """

            # ── Debug minimap ────────────────────────────────────────────────
            label   = (f"{nav_state} dest={'Y' if dest_seen else '-'} "
                       f"hall={'Y' if in_hall else 'N'} "
                       f"gaps={len(gaps)} stk={len(dfs_stack)}")
            minimap = draw_minimap(grid, pos_x, pos_y, angle, target, trail, label)
            cv2.imshow("Local Map (DFS)", minimap)

            print("State #" + str(state.number))
            print(state.game_variables)
            print("=====================")

        print("Episode finished!")
        print("************************")

    cv2.destroyAllWindows()