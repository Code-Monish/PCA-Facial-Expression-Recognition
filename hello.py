import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.decomposition import PCA

# Function to generate random data
def generate_data(num_tuples):
    data = []
    for _ in range(num_tuples):
        height = random.randint(40, 70)
        width = random.randint(10, min(55, height))  # Ensure width does not exceed height
        data.append((height, width))
    return data

# Generate a random set of 10 tuples
random_data = generate_data(140)

# Separate the data into two lists for PCA
heights = np.array([t[0] for t in random_data]).reshape(-1, 1)
widths = np.array([t[1] for t in random_data]).reshape(-1, 1)

# Combine the heights and widths into a single dataset
data = np.hstack((heights, widths))

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

# Create the scatter plot of the PCA results
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], color='red', marker='o', label='PCA Results')

# Function to calculate distance from point to line
def distance_from_line(point, m, b):
    x, y = point
    return abs(m * x - y + b) / np.sqrt(m**2 + 1)

# Function to calculate variance of distances
def variance_of_distances(m, b, points):
    distances = [distance_from_line(point, m, b) for point in points]
    return np.var(distances)

# Initialize parameters for the line
best_variance = float('inf')
best_m = 0
best_b = 0

# Rotate the line and find the best parameters
for angle in np.linspace(-np.pi / 2, np.pi / 2, 360):  # Rotate from -90 to 90 degrees
    m = np.tan(angle)  # Slope of the line
    b = random.uniform(-5, 5)  # Random intercept
    var = variance_of_distances(m, b, pca_result)
    
    if var < best_variance:
        best_variance = var
        best_m = m
        best_b = b

# Create a range of x values for the best line
x = np.linspace(-40, 40, 100)
y = best_m * x + best_b

# Plot the best line
plt.plot(x, y, label=f'Best Line: y = {best_m:.2f}x + {best_b:.2f}', color='blue')

# Adding titles and labels
plt.title('PCA Outcomes of Widths and Heights with Best SVM Line')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Show grid
plt.grid()

# Add a legend
plt.legend()

# Display the plot
plt.axis('equal')  # Equal scaling for x and y axes
plt.show()

# Print the best variance
print(f'Best variance of distances: {best_variance:.4f}')
print(f'Best line parameters: m = {best_m:.4f}, b = {best_b:.4f}')