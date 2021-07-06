import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.optimize
import matplotlib.pyplot as plt

class Arm:

    def __init__(self, angle=None, default_angle=None):
        """
        Structure: [joint0, link0, joint1, link1, ... , joint n, link n].
        angle : the initial joint angles of the arm
        default_angle : the default joint configuration
        L : arm length set to 1
        """

        # initial joint angles
        self.angle = [0.1, 0.1, 0.1] if angle is None else angle
        # default arm positions
        self.default_angle = [1.4, 2.3, 2.1] if default_angle is None else default_angle
        # constraints on link orientation. eg. angle of joint 1 is limited to [0,pi]
        self.max_angles = [np.pi, np.pi, np.pi]
        self.min_angles = [0, -np.pi , -np.pi]

    def forward_kinematics(self, angle=None):
        """
        input : joint angles
        output: end-effector coordinates [x,y]
            x: horizontal axis
            y: vertical axis
        """
        if angle is None:
            angle = self.angle

        x = np.cos(angle[0]) + \
            np.cos(angle[0]+angle[1]) + \
            np.cos(np.sum(angle))

        y = np.sin(angle[0]) + \
            np.sin(angle[0]+angle[1]) + \
            np.sin(np.sum(angle))

        return [x, y]

    def inv_kin_gurobipy(self, xy):
        """
        Input the desired (x,y) position
        return the joint angles

        minimize the movement of each joint
        constraint is to match (x,y)
        """

        try:
            # Create a new model
            m = gp.Model("IK")

            # Create variables x,y,z refer to angle0, angle1, angle2
            angle0 = m.addVar(vtype=GRB.BINARY, name="angle0")
            angle0_cos = m.addVar(vtype=GRB.BINARY)
            angle0_sin = m.addVar(vtype=GRB.BINARY)

            angle1 = m.addVar(vtype=GRB.BINARY, name="angle1")
            angle1_cos = m.addVar(vtype=GRB.BINARY)
            angle1_sin = m.addVar(vtype=GRB.BINARY)
            angle1_sum = m.addVar(vtype=GRB.BINARY)

            angle2 = m.addVar(vtype=GRB.BINARY, name="angle2")
            angle2_cos = m.addVar(vtype=GRB.BINARY)
            angle2_sin = m.addVar(vtype=GRB.BINARY)
            angle2_sum = m.addVar(vtype=GRB.BINARY)

            # Set objective, minimize the euclidean distance through joint space to the default arm configuration.
            m.setObjective(((angle0 - self.default_angle[0]) * (angle0 - self.default_angle[0]) + (angle1 - self.default_angle[1]) * (
                        angle1 - self.default_angle[1]) + (angle2 - self.default_angle[2]) * (angle2 - self.default_angle[2])),
                           GRB.MINIMIZE)

            """
            let difference between current and desired x position = 0
            """
            m.addGenConstrCos(angle0, angle0_cos);
            m.addConstr(angle1 + angle0 == angle1_sum);
            m.addGenConstrCos(angle1_sum, angle1_cos);
            m.addConstr(angle2 + angle1 + angle0 == angle2_sum);
            m.addGenConstrCos(angle2_sum, angle2_cos);
            m.addConstr((angle0_cos + angle1_cos + angle2_cos) - xy[0] == 0.00)

            """
            let difference between current and desired y position = 0
            """
            m.addGenConstrSin(angle0, angle0_sin);
            m.addGenConstrSin(angle1_sum, angle1_sin);
            m.addGenConstrSin(angle2_sum, angle2_sin);
            m.addConstr((angle0_sin + angle1_sin + angle2_sin) - xy[1] == 0.00);

            # Optimize model
            m.optimize()

            # print('Obj: %g' % m.objVal)
            res_angle = [m.getVarByName("angle0").x, m.getVarByName("angle1").x, m.getVarByName("angle2").x]

            return res_angle
        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

        except AttributeError as e:
            print('Attribute error: ', e)

    def inv_kin_scipy(self, xy):
        """
        Input the desired (x,y) position
        return the joint angles

        minimize the movement of each joint
        constraint is to match (x,y)
        """

        def distance_to_default(angle, *args):
            """
            Objective function to minimize. Calculate the euclidean distance through joint space to the default arm configuration.
            """
            return np.sqrt( np.sum([(angle_i - default_angle_i)**2 for angle_i, default_angle_i in zip(angle, self.default_angle)]) )

        def x_constraint(angle, xy):
            # let difference between current and desired x position = 0
            x = (np.cos(angle[0]) + np.cos(angle[0]+angle[1]) +np.cos(np.sum(angle))) - xy[0]
            return x

        def y_constraint(angle, xy):
            # let difference between current and desired y position = 0
            y = (np.sin(angle[0]) + np.sin(angle[0]+angle[1]) + np.sin(np.sum(angle))) - xy[1]
            return y

        def joint_limits_upper_constraint(angle, xy):
            """
            max_angles - angle >= 0
            """
            return self.max_angles - angle

        def joint_limits_lower_constraint(angle, xy):
            """
            angle - self.min_angles >=0
            """
            return angle - self.min_angles

        return scipy.optimize.fmin_slsqp(
            func=distance_to_default,
            x0=self.angle, # initial guess, find the rest around the guess, paper don't use this method
            eqcons=[x_constraint,
                    y_constraint],
            ieqcons=[joint_limits_upper_constraint,
                     joint_limits_lower_constraint],
            args=(xy,),
            iprint=0)

    def animation(self, angle):
        R = angle;
        # 1 second show 15 times
        duration_time_seconds = 1
        actions_num = 15
        angles_per_action = (np.array(R) - np.array(self.default_angle)) / actions_num
        joint_angles=self.default_angle
        plt.ion()  # open interactive mode
        for action_i in range(actions_num):
            joint_angles = joint_angles + angles_per_action
            # print(joint_angles)
            self.plot(joint_angles)
            dt = duration_time_seconds / actions_num
            plt.pause(dt)
        print("Got it, click another position.")

    def plot(self, joint_angles):
        plt.cla() # clear plot
        # position of three points
        x = [0, np.cos( joint_angles[0] ), np.cos( joint_angles[0] ) + np.cos( joint_angles[0]+joint_angles[1] ), np.cos( joint_angles[0] ) + np.cos( joint_angles[0]+joint_angles[1] ) + np.cos( joint_angles[0]+joint_angles[1]+joint_angles[2]) ]
        y = [0, np.sin( joint_angles[0] ), np.sin( joint_angles[0] ) + np.sin( joint_angles[0]+joint_angles[1] ), np.sin( joint_angles[0] ) + np.sin( joint_angles[0]+joint_angles[1] ) + np.sin( joint_angles[0]+joint_angles[1]+joint_angles[2]) ]
        # link0——link1
        plt.plot(x, y, c="black", zorder=1)
        plt.scatter(x, y, c="blue", zorder=2)
        # target point
        plt.scatter(3, 0, c='red', marker='*')
        plt.xlim(-2, 3.5)
        plt.ylim(-1, 3)


def main():

    arm = Arm();

    tolerance = 0.0000000000001
    count = 0
    total_error = 0

    xy = [3, 0]

    # Inverse_Kinematics function, get the optimal joint angles
    print('-------------gurobipy-------------------')
    angle = arm.inv_kin_gurobipy(xy)
    print()
    print('-------------scipy.optimize-------------')
    angle = arm.inv_kin_scipy(xy)

    # Find the (x,y) position of the hand given these angles
    actual_xy = arm.forward_kinematics(angle)
    # calculate the root squared error
    error = np.sqrt(np.sum((np.array(xy) - np.array(actual_xy))**2))
    # total the error
    total_error += np.nan_to_num(error)
    # if the error was high, print out more information
    if np.sum(error) > tolerance:
        print('-------------------------')
        print('Initial joint angles', arm.angle)
        print('Final joint angles: ', angle)
        print('Desired hand position: ', xy)
        print('Actual hand position: ', actual_xy)
        print('Error: ', error)
        print('-------------------------')
    count += 1
    print('\n---------Results---------')
    print('Final joint angles: ', angle)
    print('Desired hand position: ', xy)
    print('Actual hand position: ', actual_xy)
    print('-------------------------')

    fig = plt.figure()
    arm.animation(angle)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()

