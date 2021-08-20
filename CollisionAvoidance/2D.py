import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# builds a N sided polygon approximation of a circle for MIP. It is the union of the segments making up the polygon
# might also be useful to directly encode arcs. for joint constraint limits.

def circle(N):
    x = cvx.Variable()
    y = cvx.Variable()
    l = cvx.Variable(N)

    segment = cvx.Variable(N,boolean=True) #segment indicator variables, relaxing the boolean constraint gives the convex hull of the polygon

    angles = np.linspace(0, 2*np.pi, N, endpoint=False)  #divide 2pi
    xs = np.cos(angles)
    ys = np.sin(angles)

    constraints = []
    constraints = [x == l*xs, y == l*ys]
    constraints = constraints + [cvx.sum(l) == 1, l <= 1, 0 <= l]
    constraints = constraints +  [cvx.sum(segment) == 1] # only one indicator variable can be nonzero

    constraints = constraints +  [l[N-1] <= segment[N-1] + segment[0]] #special wrap around case
    for i in range(N-1):
        constraints = constraints + [l[i] <= segment[i] + segment[i+1]] # interpolation variables suppressed
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
    N = len(link_lengths)
    Len=30;

    constraints = []
    origin = np.array([0,0])
    p1 = origin

    for l in link_lengths:
        R1, c= R(Len)
        constraints = constraints + c

        p2 = cvx.Variable(2)

        # constraints = constraints + [p2[0] == p1[0] + R1[0,0] * l ];
        # constraints = constraints + [p2[1] == p1[1] + R1[1,0] * l ];

        constraints = constraints + [p2 == p1 + R1 * np.array([l,0]) ];


        mid = (p1 + p2)/2;
        Setvalue=1000;
        iii = cvx.Variable(N,boolean=True);
        constraints = constraints + [mid[0] <= 0.7 + iii[0] * Setvalue];
        constraints = constraints + [mid[0] + iii[1] * Setvalue >= 1.1 ];
        constraints = constraints + [mid[1] + iii[2] * Setvalue >= 1.1 ];
        constraints = constraints + [mid[1] <= 0.7 + iii[3] * Setvalue ];
        constraints = constraints + [cvx.sum(iii) <= N-1, iii <= 1, iii >= 0];

        random = (p1 + p2)/Len;
        Setvalue=1000;
        iii = cvx.Variable(N,boolean=True);
        constraints = constraints + [random[0] <= 0.7 + iii[0] * Setvalue];
        constraints = constraints + [random[0] + iii[1] * Setvalue >= 1.1 ];
        constraints = constraints + [random[1] + iii[2] * Setvalue >= 1.1 ];
        constraints = constraints + [random[1] <= 0.7 + iii[3] * Setvalue ];
        constraints = constraints + [cvx.sum(iii) <= N-1, iii <= 1, iii >= 0];


        p1 = p2;

        ii = cvx.Variable(N,boolean=True);
        constraints = constraints + [p2[0] <= 0.7 + ii[0] * Setvalue];
        constraints = constraints + [p2[0] + ii[1] * Setvalue >= 1.1 ];
        constraints = constraints + [p2[1] + ii[2] * Setvalue >= 1.1 ];
        constraints = constraints + [p2[1] <= 0.7 + ii[3] * Setvalue ];
        constraints = constraints + [cvx.sum(ii) <= N-1, ii <= 1, ii >= 0];

        pivots.append(p2)
        Rs.append(R1)

    end_position = np.array([2,2])
    constraints = constraints + [p2 == end_position]

    objective = cvx.Maximize(1)
    prob = cvx.Problem(objective, constraints)


    res = prob.solve(solver=cvx.GLPK_MI, verbose=True)
    print(list(map(lambda r: r.value, Rs)))

    p1 = origin
    for l, r in zip(link_lengths, Rs):
        p2 = p1 + r.value@np.array([l,0])
        plt.plot([p1[0],p2[0]], [p1[1],p2[1]], marker='o'),
        p1 = p2

    x_major_locator = MultipleLocator(0.5)
    y_major_locator = MultipleLocator(0.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-1, 3)
    plt.ylim(-0.5, 3)
    plt.plot([0.7,0.7],[0.7,1.1],'b-');
    plt.plot([0.7,1.1],[1.1,1.1],'b-');
    plt.plot([1.1,1.1],[1.1,0.7],'b-');
    plt.plot([1.1,0.7],[0.7,0.7],'b-');

    plt.text(1.1, 0.85, 'Obstacle')
    plt.text(0, 0, 'Origin')
    plt.text(0, 0, 'Origin')
    plt.text(2, 2, 'End')
    plt.show()


main()