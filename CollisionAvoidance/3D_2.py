import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator

# builds a N sided polygon approximation of a circle for MIP.

def circle(N):
    x = cvx.Variable()
    y = cvx.Variable()
    l = cvx.Variable(N)

    segment = cvx.Variable(N,boolean=True)

    angles = np.linspace(0, 2*np.pi, N, endpoint=False)  #divide 2pi
    xs = np.cos(angles)
    ys = np.sin(angles)

    constraints = []
    constraints = [x == l*xs, y == l*ys]
    constraints = constraints + [cvx.sum(l) == 1, l <= 1, 0 <= l]
    constraints = constraints +  [cvx.sum(segment) == 1] # only one variable can be nonzero

    constraints = constraints +  [l[N-1] <= segment[N-1] + segment[0]]
    for i in range(N-1):
        constraints = constraints + [l[i] <= segment[i] + segment[i+1]]
    return x, y, constraints

def main():
    # build a 2d rotation matrix using circle
    def R(N):
        constraints = []
        c, s, constraint = circle(N) # get cosines and sines from a circle
        constraints += constraint
        r = cvx.Variable((2,2)) # build rotation matrix
        constraints += [r[0,0] == c, r[0,1] == s]
        constraints += [r[1,0] == -s, r[1,1] == c]
        return r, constraints
        # np.array([[c , s], [-s, c]])

    link_lengths = [1,1,1,1]
    pivots = []
    Rs = []
    Rs_z=[];
    N = len(link_lengths)
    Len=20;

    constraints = []
    origin = np.array([0,0,0])
    p1 = origin

    for l in link_lengths:

        R1, c= R(Len)
        constraints = constraints + c
        p2 = cvx.Variable(3);

        constraints = constraints + [p2[0] == p1[0] + R1[0,0] * l ];
        constraints = constraints + [p2[1] == p1[1] + R1[1,0] * l ];

        R2, c2 = R(Len);
        constraints = constraints + c2;
        constraints = constraints + [p2[0] == p1[0] + R2[0,0] * l ];
        constraints = constraints + [p2[2] == p1[2] + R2[1,0] * l ];

        # R3, c3 = R(Len);
        # constraints = constraints + c3;
        # constraints = constraints + [p2[1] == p1[0] + R3[0,0] * l ];
        # constraints = constraints + [p2[2] == p1[2] + R3[1,0] * l ];

        # mid = (p1 + p2)/2;
        # Setvalue=1000;
        # iii = cvx.Variable(6,boolean=True);
        # constraints = constraints + [mid[0] <= 0.7 + iii[0] * Setvalue];
        # constraints = constraints + [mid[0] + iii[1] * Setvalue >= 1.1 ];
        # constraints = constraints + [mid[1] + iii[2] * Setvalue >= 1.1 ];
        # constraints = constraints + [mid[1] <= 0.7 + iii[3] * Setvalue ];
        # constraints = constraints + [mid[2] <= 0.7 + iii[4] * Setvalue ];
        # constraints = constraints + [mid[2] + iii[5] * Setvalue >= 1.1 ];
        # constraints = constraints + [cvx.sum(iii) <= 5, iii <= 1, iii >= 0];

        random = (p1 + p2)/N;
        Setvalue = 1000;
        iiii = cvx.Variable(4,boolean=True);
        constraints = constraints + [random[0] <= 0.7 + iiii[0] * Setvalue];
        constraints = constraints + [random[0] + iiii[1] * Setvalue >= 1.1 ];
        constraints = constraints + [random[1] + iiii[2] * Setvalue >= 1.1 ];
        constraints = constraints + [random[1] <= 0.7 + iiii[3] * Setvalue ];
        constraints = constraints + [random[2] <= 0.7 + iiii[2] * Setvalue];
        constraints = constraints + [random[2] + iiii[3] * Setvalue >= 1.1  ];
        constraints = constraints + [cvx.sum(iiii) <= N-1, iiii <= 1, iiii >= 0];


        p1 = p2;

        # Setvalue = 1000;
        # ii = cvx.Variable(6,boolean=True);
        # constraints = constraints + [p2[0] <= 0.7 + ii[0] * Setvalue];
        # constraints = constraints + [p2[0] + ii[1] * Setvalue >= 1.1 ];
        # constraints = constraints + [p2[1] + ii[2] * Setvalue >= 1.1 ];
        # constraints = constraints + [p2[1] <= 0.7 + ii[3] * Setvalue ];
        # constraints = constraints + [p2[2] <= 0.7 + ii[4] * Setvalue];
        # constraints = constraints + [p2[2] + ii[5] * Setvalue >= 1.1  ];
        # constraints = constraints + [cvx.sum(ii) <= 5, ii <= 1, ii >= 0];

        pivots.append(p2)
        Rs.append(R1)
        Rs_z.append(R2)

    end_position = np.array([2,2,1])
    constraints = constraints + [p2 == end_position]

    objective = cvx.Maximize(1);
    prob = cvx.Problem(objective, constraints);


    res = prob.solve(solver=cvx.GLPK_MI, verbose=True);
    print(list(map(lambda r: r.value, pivots)));

    xa=[0];
    ya=[0];
    za=[0];
    for i in list(map(lambda r: r.value, pivots)):
        # i=i.tolist();
        xa.append(i[0]);
        ya.append(i[1]);
        za.append(i[2]);
    print(xa,ya,za)

    def plot_3D(x, y, z):
        fig = plt.figure(1);
        fig.clf();
        ax = Axes3D(fig);
        ax.plot(x, y, z, linewidth=10);

        ax.set_xlim3d([-0.15, 2.1]);
        ax.set_ylim3d([-0.15, 2.1]);
        ax.set_zlim3d([-0.15, 1.1]);

        def plot_opaque_cube(x=0.7, y=0.7, z=0.7, dx=0.4, dy=0.4, dz=0.4):
            xx = np.linspace(x, x + dx, 2)
            yy = np.linspace(y, y + dy, 2)
            zz = np.linspace(z, z + dz, 2)

            xx2, yy2 = np.meshgrid(xx, yy)

            ax.plot_surface(xx2, yy2, np.full_like(xx2, z))
            ax.plot_surface(xx2, yy2, np.full_like(xx2, z + dz))

            yy2, zz2 = np.meshgrid(yy, zz)
            ax.plot_surface(np.full_like(yy2, x), yy2, zz2)
            ax.plot_surface(np.full_like(yy2, x + dx), yy2, zz2)

            xx2, zz2 = np.meshgrid(xx, zz)
            ax.plot_surface(xx2, np.full_like(yy2, y), zz2)
            ax.plot_surface(xx2, np.full_like(yy2, y + dy), zz2)

        plot_opaque_cube();

        ax.text(-0.1, -0.1, -0.1, "Origin", color='black');
        ax.text(2, 2, 1, "End", color='black');

        for i in range(1, len(x) - 2):
            ax.text(x[i], y[i], z[i], "Joint", color='black');

        ax.text(0.95, 0.95, 1.15, "Obstacle", color='black');

        plt.draw();
        plt.show();

    plot_3D(xa,ya,za);

main()
