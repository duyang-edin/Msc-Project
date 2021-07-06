import numpy as np
from numpy import cos, sin, arccos, arctan2, sqrt
import matplotlib.pyplot as plt

(target_x, target_y) = (-1, -1)

class TwoLinkArm:
    """
    (joint0)——link0——(joint1)——link1——[joint2]
    """

    def __init__(self, _joint_angles=[0, 0]):
        # the first joint is in the origin 0,0
        self.p0 = np.array([0, 0])
        self.update_joints(_joint_angles)

    def update_joints(self, _joint_angles):
        self.joint_angles = _joint_angles
        self.forward_kinematics()

    def forward_kinematics(self):
        """
        The structure of arms:
        (joint0)——link0——(joint1)——link1
        """

        # link0
        # joint0: first link's angle
        joint0 = self.joint_angles[0];
        R0 = [cos(joint0), sin(joint0)];

        self.p1 = self.p0 + R0 * 1

        # link1
        # q1: angel between link0 and link1
        joint1 = self.joint_angles[1]
        R1 = [cos(joint0 + joint1), sin(joint0 + joint1)];
        self.p2 = self.p1 + R1 * 1

    def plot(self):
        plt.cla() # clear plot
        # position of three points
        x = [self.p0[0], self.p1[0], self.p2[0]]
        y = [self.p0[1], self.p1[1], self.p2[1]]
        # link0——link1
        plt.plot(x, y, c="black", zorder=1)
        plt.scatter(x, y, c="blue", zorder=2)
        # target point
        global target_x, target_y
        plt.scatter(target_x, target_y, c='red', marker='*')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

    def inverse_kinematic(self, x, y):

        q1 = arccos((x ** 2 + y ** 2 - 2) / 2 )
        q0 = arctan2(y, x) - arctan2( sin(q1), cos(q1) + 1)

        return [q0, q1]

    def animation(self, x, y):
        R = self.inverse_kinematic(x, y)
        # 1 second show 15 times
        duration_time_seconds = 1
        actions_num = 15
        angles_per_action = (np.array(R) - np.array(self.joint_angles)) / actions_num
        plt.ion()  # open interactive mode
        for action_i in range(actions_num):
            joint_angles = np.array(self.joint_angles) + angles_per_action
            self.update_joints(joint_angles)
            self.plot()
            dt = duration_time_seconds / actions_num
            plt.pause(dt)
        print("Got it, click another position.")

    def mouse_posi(self, event):
        """
        Mouse click event
        Record the position of the mouse in the coordinate system (x, y)
        """
        global target_x, target_y
        if event.xdata == None or event.ydata == None:
            print("Choose a point")
            return
        target_x = event.xdata
        target_y = event.ydata
        if pow(target_x.tolist(),2)+pow(target_y.tolist(),2) > 4:
            print("No solution, try another position")
        else:
            self.animation(target_x, target_y)

if __name__ == "__main__":
    fig = plt.figure()
    arm_robot = TwoLinkArm()
    arm_robot.animation(target_x, target_y)
    fig.canvas.mpl_connect("button_press_event", arm_robot.mouse_posi); plt.ioff() # close interactive mode
    plt.show()
