%% Plot Trajectory evolution
close all
cd('/home/lex/catkin_ws/src/youbot/youbot_navigation/data')
q = load('trajectory.txt');
qstar = repmat([0.82452343,  0.59753333,  0.07282408], [length(q), 1]);
time_scale = 0.3;

X = time_scale* linspace(1, length(q), length(q));
p = plot(X, [q(:,1), q(:,2), q(:,3)]); hold on
%p2 = plot(X, [qstar(:,1), qstar(:,2), qstar(:,2)]);

title("ILQG Trajectory Evolution")

p(1).LineWidth = 8;
p(1).MarkerSize = 10;

p(2).LineWidth = 8;
p(2).MarkerSize = 10;

p(3).LineWidth = 8;
p(3).MarkerSize = 10;

% p2(1).LineWidth = 5;
% p2(1).LineStyle = '--';
% p2(1).MarkerSize = 10;
% 
% p2(2).LineWidth = 5;
% p2(2).LineStyle = '--';
% p2(2).MarkerSize = 10;
% 
% p2(3).LineWidth = 5;
% p2(3).LineStyle = '--';
% p2(3).MarkerSize = 10;

lgd = legend({'x_I', 'y_I', '\theta_I', 'x_I^*', 'y_I^*', '\theta_I^*'}, 'location', 'southeast', ...
'FontSize',20, 'FontWeight', 'bold')
%lgd2 = legend({'x_I^*', 'y_I^*', '\theta_I^*'}, 'location', 'southeast', ...
%'FontSize',20, 'FontWeight', 'bold')

xlabel("Time(seconds)", 'FontSize',20, 'FontWeight', 'bold')
ylabel("State", 'FontSize',20, 'FontWeight', 'bold')

%savefig(p, '/home/lex/catkin_ws/src/youbot/youbot_navigation/data/trajectory.fig')

grid on

%% Verifying equations
clc; close all; clear all
syms r l a t pd1 pd2 pd3 pd4 xd yd td real

Zetad = [xd yd td]';
rotmat = [-sin(t) cos(t) 0; ...
           cos(t), sin(t) 0; ...
           0 , 0, 1];
Phid = [pd1, pd2, pd3, pd4]';  

J = [sqrt(2)/2, sqrt(2)/2, l*sin(pi/4-a); ...
    -sqrt(2)/2, sqrt(2)/2, l*sin(pi/4-a); ...
    -sqrt(2)/2, -sqrt(2)/2, l*sin(pi/4-a); ...
    sqrt(2)/2, -sqrt(2)/2, l*sin(pi/4-a)];

Phid_mat = sqrt(2) * J * rotmat;
Phid_eq = -(1/r) * Phid_mat * Zetad

 
Phid = -sqrt(2)/r * J * Zetad

%% Define the Lagrangian
syms Ib t td mb xid yid d2 d1 mw a ...
    mw l I1 I2 I3 I4 r 
L = 0.5 * Ib * td^2 ...
    + 0.5 * mb * ( (-xid*sin(t) + yid*cos(t) + td*d2)^2  ...
                  + (xid*cos(t) + yid*sin(t) - td*d1)^2  ) ...
     + 0.5 * mw * ( ( xid * (sin(t)- cos(t)) - yid * (cos(t) + sin(t)) - sqrt(2)*l*td * sin(pi/4-a))^2 ...
                  + (-xid * (sin(t)+ cos(t)) + yid * (cos(t) - sin(t)) - sqrt(2)*l*td * sin(pi/4-a))^2 ...
                  + ( xid * (-sin(t)+ cos(t)) + yid * (cos(t) + sin(t)) - sqrt(2)*l*td * sin(pi/4-a))^2 ...
                  + (xid * (sin(t)+ cos(t)) + yid * (-cos(t) + sin(t)) - sqrt(2)*l*td * sin(pi/4-a))^2 ) ...
     + 1/(2*r^2) * I1 * + (xid * (sin(t)- cos(t)) - yid * (cos(t) + sin(t)) - sqrt(2)*l*td * sin(pi/4-a))^2 ...
     + 1/(2*r^2) * I2 * + (-xid * (sin(t)+ cos(t)) + yid * (cos(t) - sin(t)) - sqrt(2)*l*td * sin(pi/4-a))^2 ...
     + 1/(2*r^2) * I3 * + (xid * (-sin(t)+ cos(t)) + yid * (cos(t) + sin(t)) - sqrt(2)*l*td * sin(pi/4-a))^2 ...
     + 1/(2*r^2) * I4 * + (xid * (sin(t)+ cos(t)) + yid * (-cos(t) + sin(t)) - sqrt(2)*l*td * sin(pi/4-a))^2;
% define the torques
syms t1 t2 t3 t4 f phi1d phi2d phi3d phi4d 
left_mult = [(t1 - r* sign(phi1d)*f), (t2 - r* sign(phi2d)*f), (t3 - r* sign(phi3d)*f), (t4 - r* sign(phi4d)*f)];

F1 = left_mult * Phid_mat(1:end,1);
F2 = left_mult * Phid_mat(1:end,2);
F3 = left_mult * Phid_mat(1:end,3);

 eq_xid = F1 - diff(L, xid) * xid - diff(L, xid);
 eq_yid = F2 - diff(L, yid) * yid - diff(L, yid);
 eq_td  = F3 - diff(L, td)  * td  - diff(L, td);
 
 dynamics_eq = [eq_xid; eq_yid; eq_td];
 