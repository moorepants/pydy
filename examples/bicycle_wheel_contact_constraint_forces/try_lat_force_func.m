p.d1 = 0.9534570696121849;
p.d2 = 0.2676445084476887;
p.d3 = 0.03207142672761929;
p.g = 9.81;
p.ic11 = 7.178169776497895;
p.ic22 = 11.0;
p.ic31 = 3.8225535938357873;
p.ic33 = 4.821830223502103;
p.id11 = 0.0603;
p.id22 = 0.12;
p.ie11 = 0.05841337700152972;
p.ie22 = 0.06;
p.ie31 = 0.009119225261946298;
p.ie33 = 0.007586622998470264;
p.if11 = 0.1405;
p.if22 = 0.28;
p.l1 = 0.4707271515135145;
p.l2 = -0.47792881146460797;
p.l3 = -0.00597083392418685;
p.l4 = -0.3699518200282974;
p.mc = 85.0;
p.md = 2.0;
p.me = 4.0;
p.mf = 3.0;
p.rf = 0.35;
p.rr = 0.3;

% steady turn taken from Table 2 in Basu-Mandal 2007 row 2
roll_angle = 1.9178291654;
steer_angle = 0.4049333918;
rear_wheel_spin_rate = 10.3899258905;
rear_wheel_traversal_radius = 2.2588798195;

q = [roll_angle, steer_angle];
u = [0.0, rear_wheel_spin_rate, 0.0];
up = [0.0, 0.0, 0.0];

[Ff, Fr] = lateral_tire_forces(q, u, up, p)