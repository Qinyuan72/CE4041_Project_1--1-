% Step 1: Load the data from CSV files
xx = readmatrix('custom_xx_values.csv'); % Load X-coordinate meshgrid
yy = readmatrix('custom_yy_values.csv'); % Load Y-coordinate meshgrid
Z = readmatrix('custom_Z_values.csv');   % Load Z-values (decision boundary or predictions)

% Step 2: Create the 3D Surface Plot
figure;
surf(xx, yy, Z); % Plot the surface
colormap('jet'); % Use 'jet' colormap for better visual contrast
shading interp;  % Smooth shading for a clean surface
colorbar;        % Add a color bar to indicate Z values
xlabel('Feature 1 (X-axis)'); % Label X-axis
ylabel('Feature 2 (Y-axis)'); % Label Y-axis
zlabel('Prediction Boundary (Z-axis)'); % Label Z-axis
title('3D Height Map of Decision Boundary'); % Add a title
grid on; % Enable grid for reference

% Fix axes for consistency
axis([-3 3 -3 3 min(Z(:)) max(Z(:))]);
daspect([1 1 0.5]); % Maintain consistent aspect ratio

% Set initial view angle
view(45, 30);

disp('Basic plot displayed. Starting yaw and pitch animations...');

% Step 3: Animate the Plot
videoFilename = 'Yaw_Pitch_Animation_Corrected.mp4'; % Output video file
gifFilename = 'Yaw_Pitch_Animation_Corrected.gif';  % Output GIF file
v = VideoWriter(videoFilename, 'MPEG-4'); % Create video writer
v.FrameRate = 10; % Set frame rate for smooth animation
open(v); % Open video writer

frames_per_yaw_cycle = 120; % Frames for one full yaw rotation (360°)
frames_per_pitch_cycle = 60; % Frames for one pitch oscillation (45°–90°)
total_cycles = 2; % Number of yaw and pitch cycles
total_frames = frames_per_yaw_cycle * total_cycles; % Total frames for animation

for frame = 1:total_frames
    % Yaw animation: Rotate around z-axis
    yaw_angle = 360 * mod(frame, frames_per_yaw_cycle) / frames_per_yaw_cycle; 
    
    % Pitch animation: Oscillate between 45° and 90°
    pitch_angle = 45 + 45 * sin(2 * pi * frame / frames_per_pitch_cycle); 
    
    % Update the view
    view(yaw_angle, pitch_angle);
    drawnow; % Update the figure

    % Capture and save the frame
    frame_data = getframe(gcf);
    writeVideo(v, frame_data);

    % Save the frame as GIF (first frame initializes the GIF)
    [imind, cm] = rgb2ind(frame2im(frame_data), 256);
    if frame == 1
        imwrite(imind, cm, gifFilename, 'gif', 'Loopcount', inf, 'DelayTime', 1 / v.FrameRate);
    else
        imwrite(imind, cm, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', 1 / v.FrameRate);
    end
end

% Close the video writer
close(v);

% Step 4: Save the Final Enhanced Plot
saveas(gcf, 'Yaw_Pitch_Corrected_Figure.jpg'); % Save as JPEG
saveas(gcf, 'Yaw_Pitch_Corrected_Figure.fig'); % Save as MATLAB Figure

disp(['Yaw and pitch animation completed and saved as ', videoFilename]);
disp(['GIF animation also saved as ', gifFilename]);
disp('Final plot saved as Yaw_Pitch_Corrected_Figure.jpg and Yaw_Pitch_Corrected_Figure.fig');
