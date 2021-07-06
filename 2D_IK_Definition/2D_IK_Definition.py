import math
import matplotlib.pyplot as plt

def Forward_Kinematics(Alpha,Beta):
    L1=1
    L2=1
    Alpha= Alpha  /180  * math.pi
    Beta = Beta /180  * math.pi
    # print(Alpha,Beta)
    # 0.5235987755982988 -1.0471975511965976
    x = L1*math.cos(Alpha)+L2*math.cos(Beta);
    y = L1*math.sin(Alpha)+L2*math.sin(Beta);
    print('position of end',x,y)
    # 1.3660254037844388 -0.36602540378443865

    m1=[0,L1*math.cos(Alpha)]
    n1=[0,L1*math.sin(Alpha)]
    plt.plot(m1,n1,'-',label='arm 1')

    m2=[L1*math.cos(Alpha),L1*math.cos(Alpha)+L2*math.cos(Beta)]
    n2=[L1*math.sin(Alpha),L1*math.sin(Alpha)+L2*math.sin(Beta)]
    plt.plot(m2,n2,'-',label='arm 2')

    plt.text(-0.05,-0.05,"original")
    plt.text(x,y,'end')
    plt.title('Forward Kinematics')
    plt.legend()
    plt.show()
    plt.savefig('/Users/duyang/Desktop/yangfan.png')

def Inverse_Kinematics(x,y):
    L1 = 1
    L2 = 1

    # x = 1.3660254037844388
    # y = -0.36602540378443865

    B = math.acos((-(L1 * L1 - x * x - y * y - L2 * L2) / 2 * L2) / math.sqrt(x * x + y * y)) - math.acos(
        x / math.sqrt(x * x + y * y))
    A = math.acos((x - L2 * math.cos(B)) / L1)

    Alpha = -A
    Beta =  B
    print(Alpha, Beta)
    # 0.5235987755982988 -1.0471975511965976

    m1 = [0, L1 * math.cos(Alpha)]
    n1 = [0, L1 * math.sin(Alpha)]
    plt.plot(m1, n1, '-', label='arm 1')

    a=L1 * math.cos(Alpha);
    b=L1*math.cos(Alpha) + L2*math.cos(Beta)
    m2 = [a, b];
    n2 = [L1 * math.sin(Alpha), L1*math.sin(Alpha) + L2*math.sin(Beta)]
    plt.plot(m2, n2, '-', label='arm 2')

    plt.text(-0.05, -0.05, "original")
    plt.text(x, y, 'end')

    plt.title('Inverse Kinematics')

    plt.legend()
    plt.show()




if __name__ == '__main__':

    # input angle of each part of the arm, output is the end
    Forward_Kinematics(30,-60)

    # input the position of end, output is the angle
    Inverse_Kinematics(1.3,-0.3)




