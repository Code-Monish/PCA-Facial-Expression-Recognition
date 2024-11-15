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
random_data = generate_data(10)

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
plt.scatter(pca_result[:, 0], pca_result[:, 1], color='red', marker='x', label='PCA Results')

# Adding titles and labels
plt.title('PCA Outcomes of Widths and Heights')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Show grid
plt.grid()

# Add a legend
plt.legend()

# Display the plot
plt.axis('equal')  # Equal scaling for x and y axes
plt.show()

# Print explained variance
print(f'Explained variance by PC1: {pca.explained_variance_ratio_[0]:.2f}')
print(f'Explained variance by PC2: {pca.explained_variance_ratio_[1]:.2f}')