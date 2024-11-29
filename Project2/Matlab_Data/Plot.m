% Step 1: Load the data from CSV files
xx = readmatrix('custom_xx_values.csv'); % Load X-coordinate meshgrid
yy = readmatrix('custom_yy_values.csv'); % Load Y-coordinate meshgrid
Z = readmatrix('custom_Z_values.csv');   % Load Z-values (decision boundary or predictions)

% Step 2: Create the 3D Surface Plot
figure; % Create a new figure
surf(xx, yy, Z, 'FaceColor', 'interp', 'EdgeColor', 'none'); % Smooth surface rendering

% Step 3: Apply Anti-Aliasing Fixes
% Increase rendering resolution
set(gcf, 'Renderer', 'opengl'); % Use OpenGL renderer for smoother rendering
set(gcf, 'GraphicsSmoothing', 'on'); % Enable graphics smoothing for anti-aliasing

% Step 4: Customize the Plot with Lighting and Shadows
colormap('jet');    % Use 'jet' colormap for vibrant colors
shading interp;     % Smooth shading for a clean surface
colorbar;           % Add a color bar to indicate Z values

% Add lighting to enhance the 3D effect
camlight('headlight');    % Add a light source from the camera view
lighting phong;           % Use Phong lighting model for smooth reflections
material shiny;           % Make the surface shiny for better highlights

% Step 5: Customize Axes and Labels
xlabel('Feature 1 (X-axis)', 'FontWeight', 'bold', 'FontSize', 12); % Label X-axis
ylabel('Feature 2 (Y-axis)', 'FontWeight', 'bold', 'FontSize', 12); % Label Y-axis
zlabel('Prediction Boundary (Z-axis)', 'FontWeight', 'bold', 'FontSize', 12); % Label Z-axis
title('3D Height Map of Decision Boundary', 'FontWeight', 'bold', 'FontSize', 14); % Add a title

% Step 6: Adjust Viewing Angle
view(45, 30); % Set viewing angle for better visualization
grid on;      % Enable grid for reference

% Step 7: Improve Axis and Perspective
axis tight;              % Fit the axes tightly around the data
ax = gca;                % Get current axes
ax.Projection = 'perspective'; % Set perspective projection for depth effect

% Step 8: Save the Enhanced Plot
saveas(gcf, 'Fixed_Aliasing_Plot.jpg'); % Save as JPEG
saveas(gcf, 'Fixed_Aliasing_Plot.fig'); % Save as MATLAB Figure

disp('Anti-aliasing fixes applied. Enhanced plot saved as Fixed_Aliasing_Plot.jpg and Fixed_Aliasing_Plot.fig');
