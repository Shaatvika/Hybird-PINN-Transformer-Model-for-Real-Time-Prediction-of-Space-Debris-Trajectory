import numpy as np
import matplotlib.pyplot as plt

# Load the predictions
predictions = np.load('multi_step_predictions.npy')

# Check the structure of the loaded predictions (This will help you understand the shape of the array)
print(f"Shape of predictions: {predictions.shape}")  # Should be (N, 6) or similar, depending on how many features were predicted

# Extract Keplerian elements from predictions
mean_motion = predictions[:, 0]  # Orbits per day (first column)
eccentricity = predictions[:, 1]  # Orbital eccentricity (second column)
inclination = predictions[:, 2]  # Orbital inclination in degrees (third column)
RA_of_ASC_node = predictions[:, 3]  # Right Ascension of Ascending Node in degrees (fourth column)
argument_of_periapsis = predictions[:, 4]  # Argument of periapsis in degrees (fifth column)
mean_anomaly = predictions[:, 5]  # Mean anomaly in degrees (sixth column)

# Constants
mu = 398600  # Gravitational parameter for Earth (km^3/s^2)

# Convert orbital elements to position using Keplerian mechanics
def keplerian_to_cartesian(a, e, i, raan, omega, M):
    # Convert from degrees to radians
    i = np.radians(i)
    raan = np.radians(raan)
    omega = np.radians(omega)
    M = np.radians(M)

    # Solve Kepler's equation for eccentric anomaly (E)
    E = M
    for _ in range(100):
        E = M + e * np.sin(E)
    
    # True anomaly
    theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

    # Semi-major axis (a) is the distance from the focus to periapsis in the orbit
    r = a * (1 - e**2) / (1 + e * np.cos(theta))  # Radial distance

    # Orbital position in the orbital plane
    x_orb = r * np.cos(theta)
    y_orb = r * np.sin(theta)
    z_orb = 0  # In the orbital plane, z = 0

    # Rotation matrix for inclination, RAAN, and argument of periapsis
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)

    # Rotation matrix for the transformation from the orbital plane to the inertial frame
    R = np.array([
        [cos_raan * cos_omega - sin_raan * cos_i * sin_omega, -cos_raan * sin_omega - sin_raan * cos_i * cos_omega, sin_raan * sin_i],
        [sin_raan * cos_omega + cos_raan * cos_i * sin_omega, -sin_raan * sin_omega + cos_raan * cos_i * cos_omega, -cos_raan * sin_i],
        [sin_i * sin_omega, sin_i * cos_omega, cos_i]
    ])

    # Convert from the orbital plane to the inertial frame
    position = np.dot(R, np.array([x_orb, y_orb, z_orb]))

    return position

# Create empty lists to store the trajectory coordinates
X_positions = []
Y_positions = []
Z_positions = []

# Loop through each prediction and calculate the corresponding position
for i in range(len(mean_motion)):
    # For each prediction, convert the Keplerian elements to Cartesian coordinates
    # Mean motion (n) is related to the semi-major axis a by the formula:
    # n = sqrt(mu / a^3), so a = (mu / n^2)^(1/3)
    
    n = mean_motion[i]  # Mean motion (orbits per day)
    a = (mu / (n * 2 * np.pi / 86400)**2)**(1/3)  # Semi-major axis in km

    # Calculate the position for this set of orbital elements
    position = keplerian_to_cartesian(a, eccentricity[i], inclination[i], RA_of_ASC_node[i], argument_of_periapsis[i], mean_anomaly[i])

    # Store the position
    X_positions.append(position[0])
    Y_positions.append(position[1])
    Z_positions.append(position[2])

# Convert lists to numpy arrays for easier plotting
X_positions = np.array(X_positions)
Y_positions = np.array(Y_positions)
Z_positions = np.array(Z_positions)

# Plot the trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(X_positions, Y_positions, Z_positions, label='Orbital trajectory')

# Labels and title
ax.set_xlabel('X Position (km)')
ax.set_ylabel('Y Position (km)')
ax.set_zlabel('Z Position (km)')
ax.set_title('Predicted Orbital Trajectory')

# Show plot
plt.legend()
plt.show()
