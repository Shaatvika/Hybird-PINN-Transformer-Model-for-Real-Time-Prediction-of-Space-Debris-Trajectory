import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.data_processing import load_and_preprocess_data

# Load multi-step predictions
predictions = np.load('multi_step_predictions.npy')

# Load scaler to inverse transform
_, _, _, _, scaler, _, feature_names = load_and_preprocess_data("data/raw/tle_data.csv")
predictions_original = scaler.inverse_transform(predictions)

# Constants
mu = 398600.4418  # Earth's gravitational parameter, km^3/s^2
earth_radius = 6371  # km

# Keplerian to Cartesian conversion
def kepler_to_xyz(a, e, i, raan, argp, M):
    i = np.radians(i)
    raan = np.radians(raan)
    argp = np.radians(argp)
    M = np.radians(M)

    # Solve Kepler's equation for eccentric anomaly
    E = M
    for _ in range(10):
        E = E - (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
    
    # True anomaly
    nu = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    
    # Orbital radius
    r = a * (1 - e * np.cos(E))
    
    # Position in orbital plane
    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)
    z_p = 0
    
    # Rotation to inertial frame
    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_w, sin_w = np.cos(argp), np.sin(argp)
    cos_i, sin_i = np.cos(i), np.sin(i)
    
    R = np.array([
        [cos_O*cos_w - sin_O*sin_w*cos_i, -cos_O*sin_w - sin_O*cos_w*cos_i, sin_O*sin_i],
        [sin_O*cos_w + cos_O*sin_w*cos_i, -sin_O*sin_w + cos_O*cos_w*cos_i, -cos_O*sin_i],
        [sin_w*sin_i, cos_w*sin_i, cos_i]
    ])
    
    return R @ np.array([x_p, y_p, z_p])

# Initialize lists for the continuous trajectory
X_positions, Y_positions, Z_positions = [], [], []

# Compute trajectory over all prediction steps
for row in predictions_original:
    _, mean_motion, ecc, inc, raan, argp, M, *rest = row
    a = (mu / (2 * np.pi * mean_motion / 86400)**2) ** (1/3)  # Semi-major axis
    
    # Compute Cartesian position for this time step
    pos = kepler_to_xyz(a, ecc, inc, raan, argp, M)
    X_positions.append(pos[0])
    Y_positions.append(pos[1])
    Z_positions.append(pos[2])

# Convert to numpy arrays
X_positions = np.array(X_positions)
Y_positions = np.array(Y_positions)
Z_positions = np.array(Z_positions)

# Plot the trajectory
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
x = earth_radius * np.cos(u) * np.sin(v)
y = earth_radius * np.sin(u) * np.sin(v)
z = earth_radius * np.cos(v)
ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)

# Plot debris trajectory
ax.plot(X_positions, Y_positions, Z_positions, color='red', lw=2, label='Predicted Trajectory')

# Labels and title
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Predicted Multi-Step Orbital Trajectory')
ax.set_box_aspect([1,1,1])
ax.legend()
plt.tight_layout()
plt.show()
