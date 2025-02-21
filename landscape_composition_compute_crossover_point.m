%% Landscape Composition Computation of Crossover Point
%
% Description: Compute the crossover point of benefit in task fecundity, 
% longevity, and fitness as a function of the landscape composition --
% varying between high-variability and low-variability patterns -- for both 
% the ``rabbit'' and ``turtle''.
%
% Date: February 21, 2025
% Author: Alex Nguyen

clearvars -except N_actual
close all
clc

%% Individual 
% Task Fecundity [tasks/s]
TF_rabbit_individual = [0.0350, 0.0318, 0.0325, 0.0306 ; ...
                        0.0310, 0.0311, 0.0321, 0.0248];
TF_turtle_individual = [0.0266, 0.0311, 0.0294, 0.0310 ; ...
                        0.0316, 0.0282, 0.0302, 0.0283];

% Longevity [s]
L_rabbit_individual = [12262, 11245,  9354, 9507; ...
                        9234,  8779, 11484, 4804];
L_turtle_individual = [ 8436, 10878, 11424, 10868 ; ...
                       10589,  9438,  9185, 9395];
                                     
% Fitness [tasks]
FIT_rabbit_individual = [429, 357, 304, 291 ; ...
                         286, 291, 369, 119];
FIT_turtle_individual = [224, 339, 336, 337 ; ...
                         335, 266, 277, 266];
                
%% Mutualism
% Task Fecundity [tasks/s]
TF_rabbit_mutualism  = [0.0193, 0.0234, 0.0281, 0.0334 ; ...
                        0.0216, 0.0193, 0.0256, 0.0257];
TF_turtle_mutualism  = [0.0255, 0.0285, 0.0345, 0.0422 ; ...
                        0.0267, 0.0225, 0.0270, 0.0299];

% Longevity [s]
L_rabbit_mutualism   = [5732, 8169, 11653, 13103 ; ...
                        5485, 6715,  8132,  9912];
L_turtle_mutualism   = [7728, 10905, 14466, 28660 ; ...
                        6990,  8208,  7815, 10784];

% Fitness [tasks]
FIT_rabbit_mutualism = [111, 191, 327, 438 ; ...
                        119, 129, 208, 255];
FIT_turtle_mutualism = [197, 311, 499, 1210 ; ...
                        187, 185, 211,  323];

%% Plot 
% Landscape Composition
% x = 1:4;
x = linspace(1, 0, length(TF_rabbit_individual));
a = x(1); b = x(end);

% Rabbit
figure;
subplot(311)
hold on;
plot(x, TF_rabbit_individual(1, :), 'linewidth', 3)
plot(x, TF_rabbit_mutualism(1, :), 'linewidth', 3)
% plot(x, TF_rabbit_individual(2, :), 'linewidth', 3)
% plot(x, TF_rabbit_mutualism(2, :), 'linewidth', 3)
hold off;
ylabel('Task Fecundity (tasks/s)', 'fontsize', 10);
% set(gca, 'xtick', [a, b], 'xticklabel', {'High-Variability'; 'Low-Variability'})
% set(gca, 'xtick', [a, b], 'xticklabel', {''; ''})
legend('Individual', 'Mutualism', 'location', 'southeast');
subplot(312)
hold on;
plot(x, L_rabbit_individual(1, :), 'linewidth', 3)
plot(x, L_rabbit_mutualism(1, :), 'linewidth', 3)
% plot(x, L_rabbit_individual(2, :), 'linewidth', 3)
% plot(x, L_rabbit_mutualism(2, :), 'linewidth', 3)
hold off;
ylabel('Longevity (s)', 'fontsize', 10);
% set(gca, 'xtick', [a, b], 'xticklabel', {'High-Variability'; 'Low-Variability'})
% set(gca, 'xtick', [a, b], 'xticklabel', {''; ''})
yticks = get(gca, 'ytick');
yticklabels = arrayfun(@(x) sprintf('%d', x), yticks, 'UniformOutput', false);
set(gca, 'yticklabels', yticklabels)
legend('Individual', 'Mutualism', 'location', 'southeast');
subplot(313)
hold on;
plot(x, FIT_rabbit_individual(1, :), 'linewidth', 3)
plot(x, FIT_rabbit_mutualism(1, :), 'linewidth', 3)
% plot(x, FIT_rabbit_individual(2, :), 'linewidth', 3)
% plot(x, FIT_rabbit_mutualism(2, :), 'linewidth', 3)
hold off;
ylabel('Fitness (tasks)', 'fontsize', 10);
xlabel('Landscape Composition', 'fontsize', 10)
% set(gca, 'xtick', [a, b], 'xticklabel', {'High-Variability'; 'Low-Variability'}, 'fontsize', 10);
legend('Individual', 'Mutualism', 'location', 'southeast');
sgtitle('"Rabbit"', 'fontsize', 16);

% Turtle
figure;
subplot(311)
hold on;
plot(x, TF_turtle_individual(1, :), 'color', '#0072BD', 'linewidth', 3)
plot(x, TF_turtle_mutualism(1, :), 'color', '#D95319', 'linewidth', 3)
% plot(x, TF_turtle_individual(2, :), 'color', '#0072BD', 'linewidth', 3)
% plot(x, TF_turtle_mutualism(2, :), 'color', '#D95319', 'linewidth', 3)
hold off;
ylabel('Task Fecundity (tasks/s)', 'fontsize', 10);
% set(gca, 'xtick', [a, b], 'xticklabel', {'High-Variability'; 'Low-Variability'})
% set(gca, 'xtick', [a, b], 'xticklabel', {''; ''})
legend('Individual', 'Mutualism', 'location', 'northwest');
subplot(312)
hold on;
plot(x, L_turtle_individual(1, :), 'linewidth', 3)
plot(x, L_turtle_mutualism(1, :), 'linewidth', 3)
% plot(x, L_turtle_individual(2, :), 'linewidth', 3)
% plot(x, L_turtle_mutualism(2, :), 'linewidth', 3)
hold off;
ylabel('Longevity (s)', 'fontsize', 10);
% set(gca, 'xtick', [a, b], 'xticklabel', {'High-Variability'; 'Low-Variability'})
% set(gca, 'xtick', [a, b], 'xticklabel', {''; ''})
yticks = get(gca, 'ytick');
yticklabels = arrayfun(@(x) sprintf('%d', x), yticks, 'UniformOutput', false);
set(gca, 'yticklabels', yticklabels)
legend('Individual', 'Mutualism', 'location', 'northwest');
subplot(313)
hold on;
plot(x, FIT_turtle_individual(1, :), 'linewidth', 3)
plot(x, FIT_turtle_mutualism(1, :), 'linewidth', 3)
% plot(x, FIT_turtle_individual(2, :), 'linewidth', 3)
% plot(x, FIT_turtle_mutualism(2, :), 'linewidth', 3)
hold off;
ylabel('Fitness (tasks)', 'fontsize', 10);
xlabel('Landscape Composition', 'fontsize', 10);
% set(gca, 'xtick', [a, b], 'xticklabel', {'High-Variability'; 'Low-Variability'}, 'fontsize', 10);
legend('Individual', 'Mutualism', 'location', 'northwest');
sgtitle('"Turtle"', 'fontsize', 16);

%% Nominal Crossover Points
% Task Fecundity
cp_rabbit_TF = mean([0.041, 0.13]);
cp_turtle_TF = mean([0.11, 0.55]);

sigcp_rabbit_TF = std([0.041, 0.13]);
sigcp_turtle_TF = std([0.11, 0.55]);

% Longevity 
cp_rabbit_L = mean([0.20, 0.48]);
cp_turtle_L = mean([0.17, 0.68]);

sigcp_rabbit_L = std([0.20, 0.48]);
sigcp_turtle_L = std([0.17, 0.68]);

% Robot Fitness
cp_rabbit_FIT = mean([0.15, 0.37]);
cp_turtle_FIT = mean([0.15, 0.62]);

sigcp_rabbit_FIT = std([0.15, 0.37]);
sigcp_turtle_FIT = std([0.15, 0.62]);
