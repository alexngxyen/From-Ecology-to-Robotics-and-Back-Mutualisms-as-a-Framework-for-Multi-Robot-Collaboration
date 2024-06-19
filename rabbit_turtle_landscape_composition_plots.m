%% Ecobotic Landscape Composition Plots
%
% Description: Plot the task fecundity, longevity, and fitness metrics for
% a varying landscape composition -- between uniform and clumped -- for
% both the rabbit and turtle
%
% Date: March 19, 2024
% Author: Alex Nguyen

clearvars -except N_actual
close all
clc

%% Individual 
% Task Fecundity [tasks/s]
T_rabbit_individual = [0.0350, 0.0318, 0.0326, 0.0306]';
T_turtle_individual = [0.0266, 0.0311, 0.0294, 0.0310]';

% Longevity [s]
L_rabbit_individual = [12262, 11245, 9369, 9507]';
L_turtle_individual = [8436, 10878, 11424, 10868]';

% Fitness [tasks]
F_rabbit_individual = [429, 357, 305, 291]';
F_turtle_individual = [224, 339, 336, 337]';

%% Mutualism
% Task Fecundity [tasks/s]
TF_rabbit_mutualism  = [0.0193, 0.0234, 0.0281, 0.0334]';
TF_turtle_mutualism  = [0.0255, 0.0285, 0.0345, 0.0422]';

% Longevity [s]
L_rabbit_mutualism   = [5732, 8169, 11653, 13103]';
L_turtle_mutualism   = [7728, 10905, 14466, 28660]';

% Fitness [tasks]
FIT_rabbit_mutualism = [111, 191, 327, 438]';
FIT_turtle_mutualism = [197, 311, 499, 1210]';

%% Plot
% Landscape Composition
x = 1:4;
a = x(1); b = x(end);

% Rabbit
figure;
subplot(311)
hold on;
plot(x, T_rabbit_individual, 'linewidth', 3)
plot(x, TF_rabbit_mutualism, 'linewidth', 3)
hold off;
ylabel('Task Fecundity (tasks/s)');
set(gca, 'xtick', [a, b], 'xticklabel', {'Uniform'; 'Clumped'})
legend('Individual', 'Mutualism', 'location', 'northwest');
subplot(312)
hold on;
plot(x, L_rabbit_individual, 'linewidth', 3)
plot(x, L_rabbit_mutualism, 'linewidth', 3)
hold off;
ylabel('Longevity (s)');
set(gca, 'xtick', [a, b], 'xticklabel', {'Uniform'; 'Clumped'})
yticks = get(gca, 'ytick');
yticklabels = arrayfun(@(x) sprintf('%d', x), yticks, 'UniformOutput', false);
set(gca, 'yticklabels', yticklabels)
legend('Individual', 'Mutualism', 'location', 'northwest');
subplot(313)
hold on;
plot(x, F_rabbit_individual, 'linewidth', 3)
plot(x, FIT_rabbit_mutualism, 'linewidth', 3)
hold off;
ylabel('Fitness (tasks)');
xlabel('Landscape Composition')
set(gca, 'xtick', [a, b], 'xticklabel', {'Uniform'; 'Clumped'})
legend('Individual', 'Mutualism', 'location', 'northwest');
sgtitle('Rabbit')

% Turtle
figure;
subplot(311)
hold on;
plot(x, T_turtle_individual, 'linewidth', 3)
plot(x, TF_turtle_mutualism, 'linewidth', 3)
hold off;
ylabel('Task Fecundity (tasks/s)');
set(gca, 'xtick', [a, b], 'xticklabel', {'Uniform'; 'Clumped'})
legend('Individual', 'Mutualism', 'location', 'northwest');
subplot(312)
hold on;
plot(x, L_turtle_individual, 'linewidth', 3)
plot(x, L_turtle_mutualism, 'linewidth', 3)
hold off;
ylabel('Longevity (s)');
set(gca, 'xtick', [a, b], 'xticklabel', {'Uniform'; 'Clumped'})
yticks = get(gca, 'ytick');
yticklabels = arrayfun(@(x) sprintf('%d', x), yticks, 'UniformOutput', false);
set(gca, 'yticklabels', yticklabels)
legend('Individual', 'Mutualism', 'location', 'northwest');
subplot(313)
hold on;
plot(x, F_turtle_individual, 'linewidth', 3)
plot(x, FIT_turtle_mutualism, 'linewidth', 3)
hold off;
ylabel('Fitness (tasks)');
xlabel('Landscape Composition')
set(gca, 'xtick', [a, b], 'xticklabel', {'Uniform'; 'Clumped'})
legend('Individual', 'Mutualism', 'location', 'northwest');
sgtitle('Turtle')
