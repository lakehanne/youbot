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