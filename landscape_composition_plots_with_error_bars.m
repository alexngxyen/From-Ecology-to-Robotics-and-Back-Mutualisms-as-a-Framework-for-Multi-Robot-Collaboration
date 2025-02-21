%% Landscape Composition Plots With Error Bars
%
% Description: Plot the task fecundity, longevity, and fitness metrics for
% a varying landscape composition -- between high-variability and low-
% variability patterns -- for both the ``rabbit'' and ``turtle''.
%
% Date: February 21, 2025
% Author: Alex Nguyen

clearvars -except N_actual
close all
clc

%% Individual 
% Task Fecundity [tasks/s]
TF_rabbit_individual = mean([0.0350, 0.0318, 0.0325, 0.0306 ; ...
                             0.0310, 0.0311, 0.0321, 0.0248], 1);
TF_turtle_individual = mean([0.0266, 0.0311, 0.0294, 0.0310 ; ...
                             0.0316, 0.0282, 0.0302, 0.0283], 1);

sigTF_rabbit_individual = std([0.0350, 0.0318, 0.0325, 0.0306 ; ...
                               0.0310, 0.0311, 0.0321, 0.0248], 1);
sigTF_turtle_individual = std([0.0266, 0.0311, 0.0294, 0.0310 ; ...
                               0.0316, 0.0282, 0.0302, 0.0283], 1);

% Longevity [s]
L_rabbit_individual = mean([12262, 11245,  9354, 9507; ...
                             9234,  8779, 11484, 4804], 1);
L_turtle_individual = mean([ 8436, 10878, 11424, 10868 ; ...
                            10589,  9438,  9185, 9395], 1);
                   
sigL_rabbit_individual = std([12262, 11245,  9354, 9507; ...
                               9234,  8779, 11484, 4804], 1);
sigL_turtle_individual = std([ 8436, 10878, 11424, 10868 ; ...
                              10589,  9438,  9185, 9395], 1);
                   
% Fitness [tasks]
FIT_rabbit_individual = mean([429, 357, 304, 291 ; ...
                              286, 291, 369, 119], 1);
FIT_turtle_individual = mean([224, 339, 336, 337 ; ...
                              335, 266, 277, 266], 1);
                   
sigFIT_rabbit_individual = std([429, 357, 304, 291 ; ...
                                286, 291, 369, 119], 1);
sigFIT_turtle_individual = std([224, 339, 336, 337 ; ...
                                335, 266, 277, 266], 1);

%% Mutualism
% Task Fecundity [tasks/s]
TF_rabbit_mutualism  = mean([0.0193, 0.0234, 0.0281, 0.0334 ; ...
                             0.0216, 0.0193, 0.0256, 0.0257], 1);
TF_turtle_mutualism  = mean([0.0255, 0.0285, 0.0345, 0.0422 ; ...
                             0.0267, 0.0225, 0.0270, 0.0299], 1);
                    
sigTF_rabbit_mutualism  = std([0.0193, 0.0234, 0.0281, 0.0334 ; ...
                               0.0216, 0.0193, 0.0256, 0.0257], 1);
sigTF_turtle_mutualism  = std([0.0255, 0.0285, 0.0345, 0.0422 ; ...
                               0.0267, 0.0225, 0.0270, 0.0299], 1);

% Longevity [s]
L_rabbit_mutualism   = mean([5732, 8169, 11653, 13103 ; ...
                             5485, 6715,  8132,  9912], 1);
L_turtle_mutualism   = mean([7728, 10905, 14466, 28660 ; ...
                             6990,  8208,  7815, 10784], 1);
                    
sigL_rabbit_mutualism   = std([5732, 8169, 11653, 13103 ; ...
                               5485, 6715,  8132,  9912], 1);
sigL_turtle_mutualism   = std([7728, 10905, 14466, 28660 ; ...
                               6990,  8208,  7815, 10784], 1);

% Fitness [tasks]
FIT_rabbit_mutualism = mean([111, 191, 327, 438 ; ...
                             119, 129, 208, 255], 1);
FIT_turtle_mutualism = mean([197, 311, 499, 1210 ; ...
                             187, 185, 211,  323], 1);
               
sigFIT_rabbit_mutualism = std([111, 191, 327, 438 ; ...
                               119, 129, 208, 255], 1);
sigFIT_turtle_mutualism = std([197, 311, 499, 1210 ; ...
                               187, 185, 211,  323], 1);

%% Plot (With Error Bars)
% Landscape Composition
x = 1:4;
a = x(1); b = x(end);

% Rabbit
figure;
subplot(311)
hold on;
% plot(x, TF_rabbit_individual, 'linewidth', 3)
% plot(x, TF_rabbit_mutualism, 'linewidth', 3)
errorbar(x, flip(TF_rabbit_individual), flip(sigTF_rabbit_individual), 'linewidth', 3)
errorbar(x, flip(TF_rabbit_mutualism), flip(sigTF_rabbit_mutualism), 'linewidth', 3)
hold off;
ylabel('Task Fecundity (tasks/s)', 'fontsize', 10);
% set(gca, 'xtick', [a, b], 'xticklabel', {'High-Variability'; 'Low-Variability'})
set(gca, 'xtick', [a, b], 'xticklabel', {''; ''})
% legend('Individual', 'Mutualism', 'location', 'best');
legend('Individual', 'Mutualism', 'location', 'northeast');
subplot(312)
hold on;
% plot(x, L_rabbit_individual, 'linewidth', 3)
% plot(x, L_rabbit_mutualism, 'linewidth', 3)
errorbar(x, flip(L_rabbit_individual), flip(sigL_rabbit_individual), 'linewidth', 3)
errorbar(x, flip(L_rabbit_mutualism), flip(sigL_rabbit_mutualism), 'linewidth', 3)
hold off;
ylabel('Longevity (s)', 'fontsize', 10);
% set(gca, 'xtick', [a, b], 'xticklabel', {'High-Variability'; 'Low-Variability'})
set(gca, 'xtick', [a, b], 'xticklabel', {''; ''})
yticks = get(gca, 'ytick');
yticklabels = arrayfun(@(x) sprintf('%d', x), yticks, 'UniformOutput', false);
set(gca, 'yticklabels', yticklabels)
% legend('Individual', 'Mutualism', 'location', 'best');
legend('Individual', 'Mutualism', 'location', 'northeast');
subplot(313)
hold on;
% plot(x, FIT_rabbit_individual, 'linewidth', 3)
% plot(x, FIT_rabbit_mutualism, 'linewidth', 3)
errorbar(x, flip(FIT_rabbit_individual), flip(sigFIT_rabbit_individual), 'linewidth', 3)
errorbar(x, flip(FIT_rabbit_mutualism), flip(sigFIT_rabbit_mutualism), 'linewidth', 3)
hold off;
ylabel('Fitness (tasks)', 'fontsize', 10);
xlabel('Landscape Composition', 'fontsize', 10)
% set(gca, 'xtick', [a, b], 'xticklabel', {'High-Variability'; 'Low-Variability'}, 'fontsize', 10);
set(gca, 'xtick', [a, b], 'xticklabel', {'0 (Low-Variability)'; '1 (High-Variability)'}, 'fontsize', 10);
% legend('Individual', 'Mutualism', 'location', 'best');
legend('Individual', 'Mutualism', 'location', 'northeast');
sgtitle('"Rabbit"', 'fontsize', 16);

% Turtle
figure;
subplot(311)
hold on;
% plot(x, TF_turtle_individual, 'linewidth', 3)
% plot(x, TF_turtle_mutualism, 'linewidth', 3)
errorbar(x, flip(TF_turtle_individual), flip(sigTF_turtle_individual), 'linewidth', 3)
errorbar(x, flip(TF_turtle_mutualism), flip(sigTF_turtle_mutualism), 'linewidth', 3)
hold off;
ylabel('Task Fecundity (tasks/s)', 'fontsize', 10);
% set(gca, 'xtick', [a, b], 'xticklabel', {'High-Variability'; 'Low-Variability'})
set(gca, 'xtick', [a, b], 'xticklabel', {''; ''})
% legend('Individual', 'Mutualism', 'location', 'best');
legend('Individual', 'Mutualism', 'location', 'northeast');
subplot(312)
hold on;
% plot(x, L_turtle_individual, 'linewidth', 3)
% plot(x, L_turtle_mutualism, 'linewidth', 3)
errorbar(x, flip(L_turtle_individual), flip(sigL_turtle_individual), 'linewidth', 3)
errorbar(x, flip(L_turtle_mutualism), flip(sigL_turtle_mutualism), 'linewidth', 3)
hold off;
ylabel('Longevity (s)', 'fontsize', 10);
% set(gca, 'xtick', [a, b], 'xticklabel', {'High-Variability'; 'Low-Variability'})
set(gca, 'xtick', [a, b], 'xticklabel', {''; ''})
yticks = get(gca, 'ytick');
yticklabels = arrayfun(@(x) sprintf('%d', x), yticks, 'UniformOutput', false);
set(gca, 'yticklabels', yticklabels)
% legend('Individual', 'Mutualism', 'location', 'best');
legend('Individual', 'Mutualism', 'location', 'northeast');
subplot(313)
hold on;
% plot(x, FIT_turtle_individual, 'linewidth', 3)
% plot(x, FIT_turtle_mutualism, 'linewidth', 3)
errorbar(x, flip(FIT_turtle_individual), flip(sigFIT_turtle_individual), 'linewidth', 3)
errorbar(x, flip(FIT_turtle_mutualism), flip(sigFIT_turtle_mutualism), 'linewidth', 3)
hold off;
ylabel('Fitness (tasks)', 'fontsize', 10);
xlabel('Landscape Composition', 'fontsize', 10);
% set(gca, 'xtick', [a, b], 'xticklabel', {'High-Variability'; 'Low-Variability'}, 'fontsize', 10);
set(gca, 'xtick', [a, b], 'xticklabel', {'0 (Low-Variability)'; '1 (High-Variability)'}, 'fontsize', 10);
% legend('Individual', 'Mutualism', 'location', 'best');
legend('Individual', 'Mutualism', 'location', 'northeast');
sgtitle('"Turtle"', 'fontsize', 16);
