from sympy import symbols, solve
from sympy.physics.mechanics import *

Vector.simp = False # To increase the computation speed

# Define the configuration variables and their derivatives
q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)
#q = dynamicsymbols('q:'+str(5))     # I won't be using the first element q0(t)

# Define the generalized speeds
#u = dynamicsymbols('u:'+str(5))
u1, u2, u3, u4 = dynamicsymbols('u1 u2 u3 u4')
u1d, u2d, u3d, u4d = dynamicsymbols('u1 u2 u3 u4', 1)

# Define the link lengths
l0, l1, l2, l3, l4 = symbols('l0 l1 l2 l3 l4')
#l = symbols('l:'+str(5))

# Define the masses
mA, mB, mC, mD = symbols('mA mB mC mD')

# Define external forces
Fx, Fy = symbols('Fx Fy')

# Define the gravitational constant and the time
g, t = symbols('g t')

# Define the Newtonian reference frame
N = ReferenceFrame('N')

# Define the orientation of the frames with respect to the Newtonian frame
A = N.orientnew('A', 'Body', [0, 0, q1], '123')
B = N.orientnew('B', 'Body', [0, 0, q2], '123')
C = N.orientnew('C', 'Body', [0, 0, q3], '123')
D = N.orientnew('D', 'Body', [0, 0, q4], '123')

# Set angular velocities of frames with the u's
A.set_ang_vel(N, u1 * N.z)
B.set_ang_vel(N, u2 * N.z)
C.set_ang_vel(N, u3 * N.z)
D.set_ang_vel(N, u4 * N.z)

# Define the points
O1 = Point('O1')
P = O1.locatenew('P', l1*A.x)
Q = P.locatenew('Q', l2*B.x)
R = Q.locatenew('R', -l3*C.x)
O2 = R.locatenew('O2', -l4*D.x)

Ao = O1.locatenew('Ao', l1/2*A.x)
Bo = P.locatenew('Bo', l2/2*B.x)
Co = R.locatenew('Co', l3/2*C.x)
Do = O2.locatenew('Do', l4/2*D.x)

# Set the velocity of the fixed points
O1.set_vel(N, 0)
O2.set_vel(N, 0)

# Define the unconstrained velocities of the points
P.v2pt_theory(O1, N, A)
Q.v2pt_theory(P, N, B)
R.v2pt_theory(Q, N, C)

Ao.v2pt_theory(O1, N, A)
Bo.v2pt_theory(P, N, B)
Co.v2pt_theory(Q, N, C)
Do.v2pt_theory(R, N, D)

# Define the kinematic differential equations
kd = [q1d-u1, q2d-u2, q3d-u3, q4d-u4]

# Define the configuration level constraints
zero = O2.pos_from(O1) + l0 * N.x
conlist_coor = [zero & N.x, zero & N.y]

# Define the velocity level constraints
dzero = time_derivative(zero, N)
conlist_speed = [dzero & N.x, dzero & N.y]

# Define the inertia dyads
IA11, IA22, IA33, IA12, IA23, IA31 = symbols('IA11 IA22 IA33 IA12 IA23 IA31')
IB11, IB22, IB33, IB12, IB23, IB31 = symbols('IB11 IB22 IB33 IB12 IB23 IB31')
IC11, IC22, IC33, IC12, IC23, IC31 = symbols('IC11 IC22 IC33 IC12 IC23 IC31')
ID11, ID22, ID33, ID12, ID23, ID31 = symbols('ID11 ID22 ID33 ID12 ID23 ID31')
inertiaA = (inertia(A, IA11, IA22, IA33, IA12, IA23, IA31), Ao)
inertiaB = (inertia(B, IB11, IB22, IB33, IB12, IB23, IB31), Bo)
inertiaC = (inertia(C, IC11, IC22, IC33, IC12, IC23, IC31), Co)
inertiaD = (inertia(D, ID11, ID22, ID33, ID12, ID23, ID31), Do)

# Define the bodies
bodyA = RigidBody('bodyA', Ao, A, mA, inertiaA)
bodyB = RigidBody('bodyB', Bo, B, mB, inertiaB)
bodyC = RigidBody('bodyC', Co, C, mC, inertiaC)
bodyD = RigidBody('bodyD', Do, D, mD, inertiaD)

# Define the force list
FL = [(Ao, -mA*g*N.y), (Bo, -mB*g*N.y), (Co, -mC*g*N.y), (Do, -mD*g*N.y), (Q, Fx*N.x+Fy*N.y)]

# Define the body list
BL = [bodyA, bodyB, bodyC, bodyD]

# Use Kane's method to solve for the Equations on Motion of the system
KM = KanesMethod(N,
                 q_ind=[q1, q4],
                 u_ind=[u1, u4],
                 kd_eqs=kd,
                 q_dependent=[q2, q3],
                 configuration_constraints=conlist_coor,
                 u_dependent=[u2, u3],
                 velocity_constraints=conlist_speed)

(fr, frstar) = KM.kanes_equations(FL, BL)
kanezero = fr + frstar
