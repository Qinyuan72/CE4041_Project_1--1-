% Step 1: Load the data from CSV files
% Replace the file paths with the actual paths if needed
xx = readmatrix('custom_xx_values.csv'); % Load X-coordinate meshgrid
yy = readmatrix('custom_yy_values.csv'); % Load Y-coordinate meshgrid
Z = readmatrix('custom_Z_values.csv');   % Load Z-values (decision boundary or predictions)

% Step 2: Create the 3D Surface Plot
figure; % Create a new figure
surf(xx, yy, Z); % Plot the surface

% Step 3: Customize the Plot
colormap('jet');    % Use 'jet' colormap for better visual contrast
shading interp;     % Smooth shading for a clean surface
colorbar;           % Add a color bar to indicate Z values
xlabel('Feature 1 (X-axis)'); % Label X-axis
ylabel('Feature 2 (Y-axis)'); % Label Y-axis
zlabel('Prediction Boundary (Z-axis)'); % Label Z-axis
title('3D Height Map of Decision Boundary'); % Add a title

% Step 4: Adjust Viewing Angle
view(45, 30); % Set viewing angle for better visualization
grid on;      % Enable grid for reference
