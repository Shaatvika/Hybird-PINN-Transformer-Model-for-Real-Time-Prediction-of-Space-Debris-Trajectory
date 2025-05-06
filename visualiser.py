import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from utils.data_processing import load_and_preprocess_data

# Load predictions
predictions = np.load('multi_step_predictions.npy')

# Load scaler and inverse transform
_, _, _, _, scaler, _, feature_names = load_and_preprocess_data("data/raw/tle_data.csv")
predictions_original = scaler.inverse_transform(predictions)

# Extract orbital elements
mu = 398600.4418  # km^3/s^2

def kepler_to_xyz(a, e, i, raan, argp, M):
    i = np.radians(i)
    raan = np.radians(raan)
    argp = np.radians(argp)
    M = np.radians(M)
    E = M
    for _ in range(10):
        E = E - (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r = a * (1 - e * np.cos(E))
    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)
    z_p = 0
    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_w, sin_w = np.cos(argp), np.sin(argp)
    cos_i, sin_i = np.cos(i), np.sin(i)
    R = np.array([
        [cos_O*cos_w - sin_O*sin_w*cos_i, -cos_O*sin_w - sin_O*cos_w*cos_i, sin_O*sin_i],
        [sin_O*cos_w + cos_O*sin_w*cos_i, -sin_O*sin_w + cos_O*cos_w*cos_i, -cos_O*sin_i],
        [sin_w*sin_i, cos_w*sin_i, cos_i]
    ])
    return R @ np.array([x_p, y_p, z_p])

# Set up 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
x = 6371 * np.cos(u)*np.sin(v)
y = 6371 * np.sin(u)*np.sin(v)
z = 6371 * np.cos(v)
ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)

# Plot predicted orbits
for row in predictions_original:
    time, mean_motion, ecc, inc, raan, argp, M, _, _, _ = row
    a = (mu / (2 * np.pi * mean_motion / 86400)**2) ** (1/3)  # Semi-major axis
    orbit_points = []
    for M_deg in np.linspace(0, 360, 100):
        r_vec = kepler_to_xyz(a, ecc, inc, raan, argp, M_deg)
        orbit_points.append(r_vec)
    orbit_points = np.array(orbit_points)
    ax.plot(orbit_points[:,0], orbit_points[:,1], orbit_points[:,2], lw=1)

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title("3D Predicted Orbits of Space Debris")
ax.set_box_aspect([1,1,1])
plt.tight_layout()
plt.show()
