import numpy as np
import matplotlib.pyplot as plt


class TrajectoryVisualizer:
    def __init__(self):
        self.trajectory = []
        self.reference_trajectory = None

        # Interactive mode ON
        plt.ion()

        # Create ONE figure only
        self.fig = plt.figure("Odometry Viewer", figsize=(12, 5))
        self.fig.clf()  # clear if exists

        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax2d = self.fig.add_subplot(122)

    # --------------------------------
    # Add pose
    # --------------------------------
    def add_pose(self, pose):
        pose = np.array(pose).reshape(3)
        self.trajectory.append(pose)

    # --------------------------------
    # Visualize (single window only)
    # --------------------------------
    def visualize(self):

        self.ax3d.cla()
        self.ax2d.cla()

        # Estimated trajectory (Blue)
        if len(self.trajectory) > 0:
            traj = np.array(self.trajectory)
            self.ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                           color="blue", linewidth=2, label="Estimated")
            self.ax2d.plot(traj[:, 0], traj[:, 2],
                           color="blue", linewidth=2)

        self.ax3d.set_title("3D Trajectory")
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.legend()

        self.ax2d.set_title("Top-Down View (X-Z)")
        self.ax2d.set_xlabel("X")
        self.ax2d.set_ylabel("Z")
        self.ax2d.axis("equal")
        self.ax2d.grid(True)

        # Update same window
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)