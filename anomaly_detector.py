import numpy as np          # For numerical operations, data generation
import pandas as pd         # For data manipulation, DataFrame
import sklearn.cluster as cluster  # For K-Means clustering
import matplotlib.pyplot as plt  # For plotting
from sklearn.cluster import KMeans # Correct import for KMeans
import matplotlib.pyplot as plt #For plotting


# Generate synthetic data
np.random.seed(42)  # For reproducibility
normal_traffic = np.random.normal(loc=500, scale=100, size=(200, 2))    #~500 bytes, normal traffic
anamolous_traffic = np.random.normal(loc=2000, scale=500, size=(20, 2))  #~2000 bytes, anomalous traffic0
data = np.vstack((normal_traffic, anamolous_traffic)) # Combine data
df = pd.DataFrame(data, columns=['packet_size', 'frequency']) # Create DataFrame
print("Frist 5 data points:\n", df.head()) # Display sample data

# Apply K-Means clustering, 2 clusters (normal, anomalous)
kmeans = KMeans(n_clusters=2, random_state=42)  #Sets of 2 Clusters
kmeans.fit(data)   #Trains the model
labels = kmeans.predict(data)  # Assigns clusters labels
centers = kmeans.cluster_centers_  # Get cluster centers
print("Cluster Centers:\n", centers)  # Display cluster centers
print("Frist 5 data points:\n", list(zip(data[:5], labels[:5]))) # Display sample data

#Detect anomalies based on distance to cluster centers
distances = np.min(np.linalg.norm(data[:, np.newaxis] - centers, axis=2), axis=1) # Calculate distances to nearest center
threshold = np.mean(distances) + 2 * np.std(distances)  # Define anomaly threshold
anomalies = data[distances > threshold]  # Identify anomalies
print (f"Threshold for anomalies: {threshold:.2f}") # Display threshold
print(f"Detected {len(anomalies)} anomalies.") # Display number of anomalies
print("Anomalous data points (packet_size, frequency):\n", anomalies)  # Display anomalies

#Visualizing clusters and anomalies

plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5, label='Data Points')  # Plots clusters
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', marker='x', s=100, label='Anomalies')  # Plots anomalies
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='o', s=200, label='Cluster Centers')  # Plots centers
plt.title('Network Traffic Clustering with Anomalies')  # Sets title
plt.xlabel('Packet Size (bytes)')  # X-axis label
plt.ylabel('Frequency (packets/min)')  # Y-axis label
plt.legend()  # Shows legend
plt.savefig('anomaly_plot.png')  # Saves plot for GitHub
plt.show()  # Displays plot