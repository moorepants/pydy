#!/usr/bin/env python

import sympy as sm
import sympy.physics.mechanics as me


def decompose_fstar(fstar, ind_gen_speeds, dep_gen_speeds=None,
                    time_symbol=None):
    """Decomposes the generalized inertial forces, Kane's F*, into the linear
    coefficient matrices corresponding to the indepdendent and dependent
    generalized speeds and the remaining vector that of terms that are not
    coffecients.

    F* = A_F*I * uI' + A_F*D * uD' + B_F*

    Parameters
    ==========
    fstar : Matrix, shape(p, 1)
        Column matrix of expressions representing Fr* in Kane's equations. Fr*
        should be a function of (u', u, q, t) only.
    ind_gen_speeds : Sequence[Function]
        A sequence of arbitrary functions of time that represent the
        independent generalized speeds.
    dep_gen_speeds : Sequence[Function]
        A sequence of arbitrary functions of time that represent the
        dependent generalized speeds.

    Returns
    =======
    B_Fstar : Matrix, shape(p, 1)
    A_FstarI : Matrix, shape(p, p)
        Linear coefficients of the indepedent generalized accelerations.
    A_FstarD : Matrix, shape(p, m)
        Linear coefficients of the depedent generalized accelerations.

    """

    if time_symbol is None:
        t = me.dynamicsymbols._t
    else:
        t = time_symbol

    uI_dot = sm.Matrix(ind_gen_speeds).diff(t)
    A_FstarI = fstar.jacobian(uI_dot)

    udot_repl = {udot: 0 for udot in uI_dot}

    if dep_gen_speeds is not None:
        uD_dot = sm.Matrix(dep_gen_speeds).diff(t)
        A_FstarD = fstar.jacobian(uD_dot)
        udot_repl.update({udot: 0 for udot in uD_dot})
        B_Fstar = fstar.xreplace(udot_repl)
        return B_Fstar, A_FstarI, A_FstarD
    else:
        B_Fstar = fstar.xreplace(udot_repl)
        return B_Fstar, A_FstarI


def decompose_nonholonomic(G, ind_gen_speeds, dep_gen_speeds):
    """Decomposes the nonholonomic constraints into linear coefficient matrices
    and the remainder.

    The nonholonmic constraint equations G are linear in all of the generalized
    speeds and can be decomposed as::

       G(u, q, t) = A_GuI(q, t)*uI(t) + A_GuD(q, t)*uD(t) + B_G(q, t) = 0

    Parameters
    ==========
    G : Sequence[Expr], len(m)
        Column matrix of expressions that equate to zero.
    ind_gen_speeds : Sequence[Function], len(p)
        Ordered sequence of arbitrary functions of time that represent the
        independent generalized speeds.
    dep_gen_speeds : Sequence[Function], len(m)
        Ordered sequence of arbitrary functions of time that represent the
        dependent generalized speeds.

    """

    G = sm.Matrix(G)

    u_repl = {u: 0 for u in ind_gen_speeds}
    u_repl.update({u: 0 for u in dep_gen_speeds})

    A_GuI = G.jacobian(ind_gen_speeds)
    A_GuD = G.jacobian(dep_gen_speeds)
    B_G = G.xreplace(u_repl)

    return A_GuI, A_GuD, B_G


def formulate_mass_matrix_form(Frstar, Fr, uI, uD, kin_diff_map, nonholonomic,
                               time_symbol=None):

    if time_symbol is None:
        t = me.dynamicsymbols._t
    else:
        t = time_symbol

    print('Taking the time derivative of the nonholomic constraints.')
    A_GuI, A_GuD, B_G = decompose_nonholonomic(nonholonomic, uI, uD)
    A_GuI_dot = A_GuI.diff(t).xreplace(kin_diff_map)
    A_GuD_dot = A_GuD.diff(t).xreplace(kin_diff_map)
    B_G_dot = B_G.diff(t).xreplace(kin_diff_map)

    print('Decomposing F*')
    B_Fstar, A_FstarI, A_FstarD = decompose_fstar(Frstar, uI, uD)

    print('B_F*')
    print(list(sm.ordered(me.find_dynamicsymbols(B_Fstar))))
    print('A_F*I')
    print(list(sm.ordered(me.find_dynamicsymbols(A_FstarI))))
    print('A_F*D')
    print(list(sm.ordered(me.find_dynamicsymbols(A_FstarD))))

    print('Calculate the inverse of A_GuD')
    A_GuD_inv = inv_of_3_by_3(A_GuD)

    uI = sm.Matrix(uI)
    uD = sm.Matrix(uD)

    print("Matrix multiply to get M and F from M*u' = F")
    A = A_FstarD * A_GuD_inv
    M = -A*A_GuI + A_FstarI
    F = A*(A_GuD_dot*uD + A_GuI_dot*uI + B_G_dot) - B_Fstar - Fr

    return M, F


def generalized_active_forces(newtonian_frame, generalized_speeds, bodies,
                              loads, dependent_generalized_speeds=None):
    """Returns a column matrix containing p expressions for generalized active
    forces. p is the number of degrees of freedom of the system.

    Parameters
    ==========
    newtonian_frame : ReferenceFrame
        Newtonian reference frame in which the Kane's equations are being
        calculated with respect to.
    generalized_speeds : Sequence[Function], len(p)
        Arbitrary functions of time that represent the independent generalized
        speeds.
    bodies : Iterable[Union[Particle, RigidBody]]
        All of the particles and rigid bodies that make up the system S.
    loads : Mapping[Union[Point, Frame], Vector]
        Mapping of points to their resultant applied force vector and reference
        frames to their resultant applied torque vector.
    dependent_generalized_speeds : Mapping
        Mapping of generalized speed functions of time to either expressions or
        arbitrary functions of the independent speeds.

    Returns
    =======
    Fr : Matrix, shape(p, 1)
        A matrix containing the p expressions for the generalized active
        forces.

    Notes
    =====

    It is expected that the velocities and accelerations for all particles and
    rigid bodies are expressed in terms of the independent generalized speeds.

    """

    Fr = []

    for ur in generalized_speeds:
        rth_generalized_force = sm.S(0)
        for body in bodies:
            contribution = sm.S(0)
            if isinstance(body, me.Particle):
                point = body.point
            if isinstance(body, me.RigidBody):
                point = body.masscenter
                frame = body.frame
                ang_vel = frame.ang_vel_in(newtonian_frame)
                if dependent_generalized_speeds is not None:
                    ang_vel = ang_vel.subs(dependent_generalized_speeds)
                par_ang_vel = ang_vel.diff(ur, newtonian_frame)
                try:
                    torque = loads[frame]
                except KeyError:  # no contribution from this torque
                    pass
                else:
                    contribution += par_ang_vel.dot(torque)
            lin_vel = point.vel(newtonian_frame)
            if dependent_generalized_speeds is not None:
                lin_vel = lin_vel.subs(dependent_generalized_speeds)
            par_lin_vel = lin_vel.diff(ur, newtonian_frame)
            try:
                force = loads[point]
            except KeyError:  # no contribution from this force
                pass
            else:
                contribution += par_lin_vel.dot(force)
            rth_generalized_force += contribution
        Fr.append(rth_generalized_force)

    return sm.Matrix(Fr)


def generalized_inertia_forces(newtonian_frame, generalized_speeds,
                               kinematical_differential_eqs, bodies,
                               dependent_generalized_speeds=None):
    """Returns a column matrix containing p expressions for generalized
    inertial forces. p is the number of degrees of freedom of the system.

    Parameters
    ==========
    newtonian_frame : ReferenceFrame
        Newtonian reference frame in which the Kane's equations are being
        calculated with respect to.
    generalized_speeds : Sequence[Function], len(p)
        Arbitrary functions of time that represent the independent generalized
        speeds.
    kinematical_differential_eqs : Mappping
        A mapping of time derivatives of generalized coordinates to expressions
        that are functions of the generalized coordinates and genearlized
        speeds.
    bodies : Iterable[Union[Particle, RigidBody]]
        All of the particles and rigid bodies that make up the system S.
    dependent_generalized_speeds : Mapping
        Mapping of generalized speed functions of time to either expressions or
        arbitrary functions of the independent speeds.

    Returns
    =======
    Frstar : Matrix, shape(p, 1)
        A matrix containing the p expressions for the generalized inertial
        forces.

    Notes
    =====

    It is expected that the velocities and accelerations for all particles and
    rigid bodies are expressed in terms of the independent generalized speeds.

    """

    Frstar = []

    for ur in generalized_speeds:
        rth_generalized_force = sm.S(0)
        for body in bodies:
            contribution = sm.S(0)
            if isinstance(body, me.Particle):
                point = body.point
            if isinstance(body, me.RigidBody):
                point = body.masscenter
                frame = body.frame
                inertia = body.central_inertia
                ang_vel = frame.ang_vel_in(newtonian_frame)
                if dependent_generalized_speeds is not None:
                    ang_vel = ang_vel.subs(dependent_generalized_speeds)
                ang_acc = frame.ang_acc_in(newtonian_frame)
                # eliminate qdots
                ang_acc = ang_acc.subs(kinematical_differential_eqs)
                par_ang_vel = ang_vel.diff(ur, newtonian_frame)
                inertial_torque = (ang_acc.dot(inertia) +
                                   ang_vel.cross(inertia.dot(ang_vel)))
                contribution += par_ang_vel.dot(-inertial_torque)
            mass = body.mass
            lin_acc = point.acc(newtonian_frame)
            # eliminate qdots
            lin_acc = lin_acc.subs(kinematical_differential_eqs)
            lin_vel = point.vel(newtonian_frame)
            if dependent_generalized_speeds is not None:
                lin_vel = lin_vel.subs(dependent_generalized_speeds)
            par_lin_vel = lin_vel.diff(ur, newtonian_frame)
            contribution += par_lin_vel.dot(-mass*lin_acc)
            rth_generalized_force += contribution
        Frstar.append(rth_generalized_force)

    return sm.Matrix(Frstar)


def inv_of_3_by_3(matrix):
    """Returns the inverse of a 3x3 matrix. The matrix must not be singular.

    Parameters
    ==========
    matrix : Matrix, shape(3, 3)

    Returns
    =======
    invA : Matrix, shape(3, 3)

    """

    a, b, c, d, e, f, g, h, k = matrix

    det = a*(e*k - f*h) - b*(d*k - f*g) + c*(d*h - e*g)

    invA = sm.zeros(3, 3)

    # Using the equations defined above, calculate the inverse matrix entries.
    invA[0, 0] = (e*k - f*h) / det
    invA[0, 1] = -(b*k - c*h) / det
    invA[0, 2] = - (b*f - c*e) / det
    invA[1, 0] = -(d*k - f*g) / det
    invA[1, 1] = (a*k - c*g) / det
    invA[1, 2] = -(a*f - c*d) / det
    invA[2, 0] = (d*h - e*g) / det
    invA[2, 1] = -(a*h - b*g) / det
    invA[2, 2] = (a*e - b*d) / det

    return invA


def replace_dep_speeds_chain_rule_derivatives(expr, u_I, u_D):

    first_repl = {}
    second_repl = {}
    for u_D_i in u_D:
        u_D_i_str = str(u_D_i)[:-3]  # remove "(t)"
        second_repl[sm.Function(u_D_i_str)(*u_I)] = u_D_i
        for u_I_i in u_I:
            u_I_i_str = str(u_I_i)[:-3]  # remove "(t)"
            first_repl[sm.Function(u_D_i_str)(*u_I).diff(u_I_i)] = \
                me.dynamicsymbols('d{}d{}'.format(u_D_i_str, u_I_i_str))

    print(first_repl)
    print(second_repl)

    return expr.xreplace(first_repl).xreplace(second_repl)
