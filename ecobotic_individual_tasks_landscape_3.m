%% Ecobotics Individual Task Completion (Landscape 3 - Clumped)
%
% Description: This demo is for two amphibious robots completing their
% respective tasks individually, i.e., no mutualism, in an environment with
% a clumped pixel distribution. The robots traverse the shortest 
% path, i.e., straight line.
%
% Date: March 14, 2024
% Author: Alex Nguyen

clearvars -except N_actual
close all
clc

%% Robotarium Demo Setup
N_actual = 2;
% initialize;

% Flags
flag_IC = 0;                  % 0: pre-defined ICs; 1: randomized ICs
flag_FC = 0;                  % 0: pre-defined FCs; 1: randomized FCs 
flag_iteration_timer = 1;     % 0: do not show iteration/timer; 1: show iteration/timer
flag_time_type = 0;           % 0: estimated run-time; 1: machine run-time (tic toc)
flag_color = 0;               % 0: pre-defined color scheme; 1: randomized color scheme
flag_robot_position = 0;      % 0: do not show 2d position values; 1: show 2d position values
flag_robot_trajectory = 1;    % 0: do not show robot trajectory; 1: show robot trajectory
flag_mp4_video = 0;           % 0: do not save animation; 1: save animation 
flag_save_workspace = 0;      % 0: do not save workspace data; 1: save workspace data

% Setup Demo
N = N_actual;                                % Number of Robots
T = 0.033;                                   % Robotarium sampling time

if flag_FC == 1                              % generate final conditions
    final_goal_points = generate_initial_conditions(N, 'Width',  3.1, 'Height', 1.9, 'Spacing', 0.5);
elseif flag_FC == 0
    final_goal_points = [-1.35, -1.35; ...
                         -0.75,  0.75; ...
                             0,    0];
end

if flag_IC == 1                              % generate initial conditions
    initial_positions = generate_initial_conditions(N, 'Width', 3.1, 'Height', 1.9, 'Spacing', 0.1);
elseif flag_IC == 0                         
    initial_positions = [ 1.15, 1.44 ; ...
                          0.30, 0.30 ; ...
                             0,    0];
    initial_positions(3, :) = [atan2(final_goal_points(2, 1) + 0.2, final_goal_points(1, 1) - 1.3), ...
                               atan2(final_goal_points(2, 2) - 0.2, final_goal_points(1, 2) - 1.3)];
end

r = Robotarium('NumberOfRobots', N, 'ShowFigure', true, 'InitialConditions', initial_positions);

% Tool needed for the demo to work
vel_mag_limit = 0.15;
safety_radius = 0.12;
si_to_uni_dyn = create_si_to_uni_dynamics('LinearVelocityGain', 0.8, 'AngularVelocityLimit', pi);

% Motor Speeds [rad/s]
gamma      = 0.95;              % reduced speed during collaboration

v_r_land   = 12;                % rabbit
v_rt_land  = gamma*v_r_land;
v_r_water  = 5; 
v_rt_water = gamma*v_r_water;

v_t_water  = 9; %8;                 % turtle
v_tr_water = gamma*v_t_water;
v_t_land   = 4; %10/3;
v_tr_land  = gamma*v_t_land;

% We'll make the rotation error huge so that the initialization checker
% doesn't care about it
args = {'PositionError', 1e-2, 'RotationError', 50};
init_checker = create_is_initialized(args{:});
controller = create_waypoint_controller(args{:});

% Get initial location data for while loop condition.
x=r.get_poses();

% Plotting Initialization
% Color Vector for Plotting
% Note the Robotarium MATLAB instance runs in a docker container which will 
% produce the same rng value every time unless seeded by the user.
if flag_color == 0
    CM = [1 0 0 ; ...  % red
          0 1 0];      % green
elseif flag_color == 1
    CM = rand(N,3);%{'.-k','.-b','.-r','.-g','.-m','.-y','.-c'};
end

% Plot Uniform Pixel Distribution
n_grid = 9600;                                        % Grid resolution 
x_img = linspace(-1.6, 1.6, 1.6*n_grid);
y_img = linspace(-1.0, 1.0, n_grid);
[qx, qy] = meshgrid(x_img, y_img);
% figure; plot(qx, qy, '.b');
rows       = 2;
columns    = 3;
domainMask = -1*[zeros(n_grid/rows, (1.6/columns)*n_grid), ones(n_grid/rows, (1.6/columns)*n_grid), zeros(n_grid/rows, (1.6/columns)*n_grid); ...
                 ones(n_grid/rows, (1.6/columns)*n_grid), zeros(n_grid/rows, (1.6/columns)*n_grid), ones(n_grid/rows, (1.6/columns)*n_grid)]; 

% Energetic Cost Values for Robot 1 (Rabbit)
c_land(1) = 1;                         % [J/kg/m]
c_water(1) = 10;                       % [J/kg/m]

% Energetic Cost Values for Robot 2 (Turtle)
c_land(2) = 10;                        % [J/kg/m]
c_water(2) = 1;                        % [J/kg/m]

% Display domain image
custom_colormap = [0, 0, 1;        % Blue
                   0.7, 0.4, 0.1]; % Brown
domain_img = image((x_img), (y_img), domainMask,'CDataMapping','scaled');
alpha(domain_img, 0.55);
colormap(custom_colormap)
uistack(domain_img, 'bottom');

% Robot Trajectories
if flag_robot_trajectory == 1
    % Trajectory Colors
    traj_color = [0.6350 0.0780 0.1840 ; ...
                  0.4660 0.6740 0.1880];

    % Preallocate X and Y Coordinates
    x_recorded = []; 
    y_recorded = [];

    % Save Initial Poses
    x_recorded = [x_recorded, x(1,:)'];
    y_recorded = [y_recorded, x(2,:)'];

    % Trajectory Settings
    for n = 1:N
        trajectory(n) = plot(x_recorded(n,:),y_recorded(n,:));
        trajectory(n).LineWidth = 4;
        trajectory(n).Color = traj_color(n, :);
    end
end

%Marker, font, and line sizes
marker_size_goal = determine_marker_size(r, 0.20);
marker_size_robot = determine_robot_marker_size(r);
font_size = determine_font_size(r, 0.05);
line_width = 5;
rc = {'Rabbit', 'Turtle'};

if flag_mp4_video == 1
    % Create a VideoWriter object
    video_filename = 'ecobotic_individual_tasks_uniform.mp4';
    frame_rate = 30;  % Set the desired frame rate
    video = VideoWriter(video_filename, 'MPEG-4');
    video.FrameRate = frame_rate;
    
    open(video);  % Open the video writer
    
    % Set the figure size for better resolution
    figure_width = 1280;  % Set the desired figure width in pixels
    figure_height = 720;  % Set the desired figure height in pixels
    set(gcf, 'Position', [100, 100, figure_width, figure_height]);
end

start_time = tic; %The start time to compute time elapsed.
count = 0;        %Time counter

for i = 1:N
    % Initialize additional information plot here. Note the order of
    % plotting matters, objects that are plotted later will appear over
    % object plotted previously.
    
    % Text for robot identification
%     robot_caption = sprintf('%s (Robot %d)', rc{i}, i);
    robot_caption = sprintf('%s', rc{i});

    % Text with robot position information
    robot_details = sprintf('X-Pos: %d \nY-Pos: %d', x(1,i), x(2,i));

    % Text with goal identification
    goal_caption = sprintf('G%d', i);

    % Plot colored square for goal location.
    d(i) = plot(final_goal_points(1,i), final_goal_points(2,i),'s','MarkerSize',marker_size_goal,'LineWidth',line_width,'Color',[0.4660 0.6740 0.1880]);
   
    % Plot the goal identification text inside the goal location
    goal_labels{i} = text(final_goal_points(1,i)-0.0625, final_goal_points(2,i), goal_caption, 'FontSize', font_size, 'FontWeight', 'bold');
   
    % Plot colored circles showing robot location.
    g(i) = plot(x(1,i),x(2,i),'o','MarkerSize', marker_size_robot,'LineWidth',5,'Color',CM(i,:));
   
    % Plot the robot label text 
    robot_labels{i} = text(500, 500, robot_caption, 'FontSize', font_size, 'FontWeight', 'bold');

    % Plot the robot position information text
    robot_details_text{i} = text(500, 500, robot_details, 'FontSize', font_size, 'FontWeight', 'bold'); 
    
end

if flag_iteration_timer == 1
    % Plot the iteration and time in the lower left. Note when run on your 
    % computer, this time is based on your computers simulation time. For a
    % better approximation of experiment time on the Robotarium when running
    % this simulation on your computer, multiply iteration by 0.033. 
%     iteration_caption = sprintf('Iteration %d', 0);
    description = sprintf('Individual Task Execution');
    environment = sprintf('Uniform Pixel Distribution');

    if flag_time_type == 0
        time_caption = sprintf('Total Time Elapsed %0.2f', count*T);
    elseif flag_time_type == 1
        time_caption = sprintf('Total Time Elapsed %0.2f sec', toc(start_time));
    end
    
%     iteration_label = text(0.25, -0.775, iteration_caption, 'FontSize', font_size, 'Color', 'w', 'FontWeight', 'bold');
    description_label = text(0.25, -0.775, description, 'FontSize', font_size, 'Color', 'w', 'FontWeight', 'bold');
%     environment_label = text(0.25, -0.775, environment, 'FontSize', font_size, 'Color', 'w', 'FontWeight', 'bold');
    time_label = text(0.25, -0.9, time_caption, 'FontSize', font_size, 'Color', 'w', 'FontWeight', 'bold');

%     uistack([iteration_label], 'top'); % Iteration label is on top.
    uistack([description_label], 'top'); % Demo label is above iteration label
%     uistack([environment_label], 'top'); % Description label is above iteration label
    uistack([time_label], 'top'); % Time label is above demo caption label.
end

% We can change the order of plotting priority, we will plot goals on the 
% bottom and have the iteration/time caption on top.
uistack([goal_labels{:}], 'top'); % Goal labels are at the very bottom, arrows are now only above the goal labels.
uistack(d, 'top');% Goal squares are at the very bottom, goal labels are above the squares and goal arrows are above those.

%% Robotarium Demo
r.step();

% Preallocate Cost Function
% robotCost = zeros(N, iter);
robotCost = [];
t_r = inf; t_t = inf;
J_r = []; J_t = [];

% Check if all robots have reached their goal location.
while(~init_checker(x, final_goal_points))
    % Update Counter 
    count = count + 1;

    % Update Plotting Information and Locations
    for q = 1:N
        g(q).XData = x(1,q);
        g(q).YData = x(2,q);
  
%         robot_labels{q}.Position = x(1:2, q) + [-0.4; 0.25];
        robot_labels{q}.Position = x(1:2, q) + [-0.15; 0.20];

        if flag_robot_position == 1
            robot_details = sprintf('X-Pos: %0.2f \nY-Pos: %0.2f', x(1,q), x(2,q));
            robot_details_text{q}.String = robot_details;
            robot_details_text{q}.Position = x(1:2, q) - [0.2;0.25];
        end
    end

    % Robot Poses
    x = r.get_poses();

    % Compute Robot Cost
    robotCost_n = zeros(n, 1);
    for n = 1:N
        % Indices
        idx_j = round((x(1, n) - r.boundaries(1)) / (r.boundaries(2) - r.boundaries(1)) * (1.6*n_grid - 1)) + 1;
        idx_i = round((1 - (x(2, n) - r.boundaries(3)) / (r.boundaries(4) - r.boundaries(3))) * (n_grid - 1)) + 1; 

        % Assign Cost Value
        if domainMask(idx_i, idx_j) == 0
            robotCost_n(n) = c_water(n);
        elseif domainMask(idx_i, idx_j) == -1
            robotCost_n(n) = c_land(n);
        end
    end
    robotCost = [robotCost, robotCost_n];

    % Control Law
    dxi = final_goal_points(1:2, :) - x(1:2, :);  % proportional controller (SI dynamics)
    unom = dxi;
    norm_ = vecnorm(dxi, 2, 1);                   % limit velocity magnitude (SI dynamics)
    dxi(:, norm_ > vel_mag_limit) = (vel_mag_limit./norm_(norm_ > vel_mag_limit)).*dxi(:, norm_ > vel_mag_limit);
    dxu = si_to_uni_dyn(dxi, x);            % convert SI to UNI dynamics
    
    % Set Velocity Limit and Barrier Certificate for Rabbit and Turtle
    idx_rj = round((x(1, 1) - r.boundaries(1)) / (r.boundaries(2) - r.boundaries(1)) * (1.6*n_grid - 1)) + 1;  
    idx_ri = round((1 - (x(2, 1) - r.boundaries(3)) / (r.boundaries(4) - r.boundaries(3))) * (n_grid - 1)) + 1;   
    idx_tj = round((x(1, 2) - r.boundaries(1)) / (r.boundaries(2) - r.boundaries(1)) * (1.6*n_grid - 1)) + 1;  
    idx_ti = round((1 - (x(2, 2) - r.boundaries(3)) / (r.boundaries(4) - r.boundaries(3))) * (n_grid - 1)) + 1; 
    
    if domainMask(idx_ri, idx_rj) == 0 && domainMask(idx_ti, idx_tj) == 0
        % Both Water (Alone)
        unicycle_barrier_certificate = create_uni_barrier_certificate_with_boundary_modified('SafetyRadius', safety_radius, 'WheelVelocityLimit', [v_r_water, v_t_water]);
    elseif domainMask(idx_ri, idx_rj) == -1 && domainMask(idx_ti, idx_tj) == -1
        % Both Land (Alone)
        unicycle_barrier_certificate = create_uni_barrier_certificate_with_boundary_modified('SafetyRadius', safety_radius, 'WheelVelocityLimit', [v_r_land, v_t_land]);
     elseif domainMask(idx_ri, idx_rj) == -1 && domainMask(idx_ti, idx_tj) == 0
        % Turtle Water and Rabbit Land (Alone)
        unicycle_barrier_certificate = create_uni_barrier_certificate_with_boundary_modified('SafetyRadius', safety_radius, 'WheelVelocityLimit', [v_r_land, v_t_water]);
     elseif domainMask(idx_ri, idx_rj) == 0 && domainMask(idx_ti, idx_tj) == -1
        % Turtle Land and Rabbit Water (Alone)
        unicycle_barrier_certificate = create_uni_barrier_certificate_with_boundary_modified('SafetyRadius', safety_radius, 'WheelVelocityLimit', [v_r_water, v_t_land]);
    end
    dxu = unicycle_barrier_certificate(dxu, x);  
    
    % Set Velocities
    r.set_velocities(1:N, dxu);
    r.step();  
    
    % Robots Reached Goal Points?
    if norm(x(1:2, 1) - final_goal_points(1:2, 1)) <= 1e-2 && count*T < t_r
        t_r = count*T;
    end
    if norm(x(1:2, 2) - final_goal_points(1:2, 2)) <= 1e-2 && count*T < t_t
        t_t = count*T;
    end

    % Objective Function
    J_r = [J_r; norm(unom(:, 1) - dxi(:, 1))^2];
    J_t = [J_t; norm(unom(:, 2) - dxi(:, 2))^2];
    
    if flag_iteration_timer == 1
        % Update Iteration and Time marker
%         iteration_caption = sprintf('Iteration %d', i);
        description = sprintf('Individual Task Execution');
        environment = sprintf('Uniform Pixel Distribution');

        if flag_time_type == 0
            time_caption = sprintf('Total Time Elapsed %0.2f', count*T);
        elseif flag_time_type == 1
            time_caption = sprintf('Total Time Elapsed %0.2f sec', toc(start_time));
        end
            
%         iteration_label.String = iteration_caption;
        description_label.String = description;
%         environment_label.String = environment;
        time_label.String = time_caption;
    end
    
    % Resize Marker Sizes (In case user changes simulated figure window
    % size, this is unnecessary in submission as the figure window 
    % does not change size).

    marker_size_goal = num2cell(ones(1,N)*determine_marker_size(r, 0.20));
    [d.MarkerSize] = marker_size_goal{:};
    marker_size_robot = num2cell(ones(1,N)*determine_robot_marker_size(r));
    [g.MarkerSize] = marker_size_robot{:};
    font_size = determine_font_size(r, 0.05);
%     iteration_label.FontSize = font_size;
    description_label.FontSize = font_size;
%     environment_label.FontSize = font_size;
    time_label.FontSize = font_size;
    
    for k = 1:N
        % Have to update font in loop for some conversion reasons.
        % Again this is unnecessary when submitting as the figure
        % window does not change size when deployed on the Robotarium.
        %     iteration_label = text(0.25, -0.8, iteration_caption, 'FontSize', font_size, 'Color', 'k', 'FontWeight', 'bold');
        robot_labels{k}.FontSize = font_size;
        goal_labels{k}.FontSize = font_size;
        robot_details_text{k}.FontSize = font_size;
    end

    if flag_robot_trajectory == 1
        % Save Robot Poses
        x_recorded = [x_recorded, x(1,:)'];
        y_recorded = [y_recorded, x(2,:)'];
    end

    if flag_mp4_video == 1
        % Capture the current figure as a frame and add it to the video
        frame = getframe(gcf);
        writeVideo(video, frame);
    end
    
end

% Demo Run Time
fprintf('Simulation Run Time = %4.2f\n\n', toc(start_time));

if flag_robot_trajectory == 1
    hold on
    for n = 1:N
        trajectory(n).XData = x_recorded(n, :);
        trajectory(n).YData = y_recorded(n, :);
    end
    hold off
end

pause(5);

if flag_mp4_video == 1
    close(video);  % Close the video writer
end

% Save Workspace
if flag_save_workspace == 1
    save('ecobotic_individual_tasks_uniform_logs.mat');
end

r.debug()

%% Energy Expenditure of Robots
% % Plot Resistance
% t = linspace(0, (count+1)*T, length(robotCost));
% figure; 
% plot(t, robotCost')
% xlabel('t (s)');
% ylabel('Resistance (J/kg/m)');
% legend('rabbit', 'turtle', 'location', 'best');

% Robot Mass (kg) [BEST GUESS]
m_r = 0.265;
m_t = 0.265;

% Position States (m)
pos_r = [x_recorded(1, :); ...
         y_recorded(1, :)];
pos_t = [x_recorded(2, :); ...
         y_recorded(2, :)];

% Compute Rabbit Energy Expenditure
r_idx_land  = find(robotCost(1, :) == 1); 
r_idx_water = find(robotCost(1, :) == 10);   

r_seg_start_land = find(diff(r_idx_land) > 1);
if ~isempty(r_seg_start_land)
    % preallocate 
    r_dist_land = 0;

    % compute distance traveled on land (all but last segment)
    start_idx = 1;
    for i = 1:numel(r_seg_start_land)
        % end index
        end_idx = r_seg_start_land(i);

        % trajectory distance
        r_dist_land = r_dist_land + sum(sqrt(sum(diff(pos_r(:, r_idx_land(start_idx:end_idx)), [], 2).^2)));
        
        % start index
        start_idx = end_idx + 1;
    end
    % trajectory distance (last segment)
    r_dist_land = r_dist_land + sum(sqrt(sum(diff(pos_r(:, r_idx_land(start_idx:end)), [], 2).^2)));

else
    % compute distance traveled on land (single segment)
    r_dist_land = sum(sqrt(sum(diff(pos_r(:, r_idx_land), [], 2).^2)));
end

r_seg_start_water  = find(diff(r_idx_water) > 1);
if ~isempty(r_seg_start_water)
    % preallocate 
    r_dist_water = 0;

    % compute distance traveled in water (multiple segments)
    start_idx = 1;
    for i = 1:numel(r_seg_start_water)
        % end index
        end_idx = r_seg_start_water(i);

        % trajectory distance
        r_dist_water = r_dist_water + sum(sqrt(sum(diff(pos_r(:, r_idx_water(start_idx:end_idx)), [], 2).^2)));
        
        % start index
        start_idx = end_idx + 1;
    end
    % trajectory distance (last segment)
    r_dist_water = r_dist_water + sum(sqrt(sum(diff(pos_r(:, r_idx_water(start_idx:end)), [], 2).^2)));

else
    % compute distance traveled in water (single segment)
    r_dist_water = sum(sqrt(sum(diff(pos_r(:, r_idx_water), [], 2).^2)));
end

E_r = m_r*(r_dist_land*c_land(1) + r_dist_water*c_water(1));

% Compute Turtle Energy Expenditure
t_idx_land  = find(robotCost(2, :) == 10); 
t_idx_water = find(robotCost(2, :) == 1); 

t_seg_start_land = find(diff(t_idx_land) > 1);
if ~isempty(t_seg_start_land)
    % preallocate 
    t_dist_land = 0;

    % compute distance traveled on land (all but last segment)
    start_idx = 1;
    for i = 1:numel(t_seg_start_land)
        % end index
        end_idx = t_seg_start_land(i);

        % trajectory distance
        t_dist_land = t_dist_land + sum(sqrt(sum(diff(pos_t(:, t_idx_land(start_idx:end_idx)), [], 2).^2)));
        
        % start index
        start_idx = end_idx + 1;
    end
    % trajectory distance (last segment)
    t_dist_land = t_dist_land + sum(sqrt(sum(diff(pos_t(:, t_idx_land(start_idx:end)), [], 2).^2)));

else
    % compute distance traveled on land (single segment)
    t_dist_land = sum(sqrt(sum(diff(pos_t(:, t_idx_land), [], 2).^2)));
end

t_seg_start_water  = find(diff(t_idx_water) > 1);
if ~isempty(t_seg_start_water)
    % preallocate 
    t_dist_water = 0;

    % compute distance traveled in water (multiple segments)
    start_idx = 1;
    for i = 1:numel(t_seg_start_water)
        % end index
        end_idx = t_seg_start_water(i);

        % trajectory distance
        t_dist_water = t_dist_water + sum(sqrt(sum(diff(pos_t(:, t_idx_water(start_idx:end_idx)), [], 2).^2)));
        
        % start index
        start_idx = end_idx + 1;
    end
    % trajectory distance (last segment)
    t_dist_water = t_dist_water + sum(sqrt(sum(diff(pos_t(:, t_idx_water(start_idx:end)), [], 2).^2)));

else
    % compute distance traveled in water (single segment)
    t_dist_water = sum(sqrt(sum(diff(pos_t(:, t_idx_water), [], 2).^2)));
end

E_t = m_t*(t_dist_land*c_land(2) + t_dist_water*c_water(2));

% Total Energy
E = E_r + E_t;

%% Compute Task Fecundity, Longevity, and Fitness
% Battery Life [Joules]
E0 = 37800;

% Task Fecundity
T_r = 1/t_r;
T_t = 1/t_t;

% Longevity
L_r = E0/E_r;
L_t = E0/E_t;

% Fitness
F_r = L_r*T_r;
F_t = L_t*T_t;

% Print Results
fprintf('\nIndividual Tasks (Landscape 3 - Clumped)\n');
fprintf('\tRabbit: Task Fecundity = %4.4f tasks/s; Longevity = %4.0f s; and Fitness = %4.0f tasks\n', T_r, L_r, F_r);
fprintf('\tTurtle: Task Fecundity = %4.4f tasks/s; Longevity = %4.0f s; and Fitness = %4.0f tasks\n', T_t, L_t, F_t);

%% Helper Functions

% Marker Size Helper Function to scale size of markers for robots with figure window
% Input: robotarium class instance
function marker_size = determine_robot_marker_size(robotarium_instance)

% Get the size of the robotarium figure window in pixels
curunits = get(robotarium_instance.figure_handle, 'Units');
set(robotarium_instance.figure_handle, 'Units', 'Pixels');
cursize = get(robotarium_instance.figure_handle, 'Position');
set(robotarium_instance.figure_handle, 'Units', curunits);

% Determine the ratio of the robot size to the x-axis (the axis are
% normalized so you could do this with y and figure height as well).
robot_ratio = (robotarium_instance.robot_diameter + 0.03)/...
    (robotarium_instance.boundaries(2) - robotarium_instance.boundaries(1));

% Determine the marker size in points so it fits the window. cursize(3) is
% the width of the figure window in pixels. (the axis are
% normalized so you could do this with y and figure height as well).
marker_size = cursize(3) * robot_ratio;

end

% Marker Size Helper Function to scale size with figure window
% Input: robotarium instance, desired size of the marker in meters
function marker_size = determine_marker_size(robotarium_instance, marker_size_meters)

% Get the size of the robotarium figure window in pixels
curunits = get(robotarium_instance.figure_handle, 'Units');
set(robotarium_instance.figure_handle, 'Units', 'Pixels');
cursize = get(robotarium_instance.figure_handle, 'Position');
set(robotarium_instance.figure_handle, 'Units', curunits);

% Determine the ratio of the robot size to the x-axis (the axis are
% normalized so you could do this with y and figure height as well).
marker_ratio = (marker_size_meters)/(robotarium_instance.boundaries(2) -...
    robotarium_instance.boundaries(1));

% Determine the marker size in points so it fits the window. cursize(3) is
% the width of the figure window in pixels. (the axis are
% normalized so you could do this with y and figure height as well).
marker_size = cursize(3) * marker_ratio;

end

% Font Size Helper Function to scale size with figure window
% Input: robotarium instance, desired height of the font in meters
function font_size = determine_font_size(robotarium_instance, font_height_meters)

% Get the size of the robotarium figure window in point units
curunits = get(robotarium_instance.figure_handle, 'Units');
set(robotarium_instance.figure_handle, 'Units', 'Pixels');
cursize = get(robotarium_instance.figure_handle, 'Position');
set(robotarium_instance.figure_handle, 'Units', curunits);

% Determine the ratio of the font height to the y-axis
font_ratio = (font_height_meters)/(robotarium_instance.boundaries(4) -...
    robotarium_instance.boundaries(3));

% Determine the font size in points so it fits the window. cursize(4) is
% the hight of the figure window in points.
font_size = cursize(4) * font_ratio;

end