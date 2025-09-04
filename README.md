**Network Anomaly Detector**

**Overview**

This project implements a basic network anomaly detector using K-Means clustering to identify unusual network traffic patterns, such as potential cyber threats (e.g., DDoS attacks or data exfiltration). It uses synthetic data to simulate network traffic, clusters it into normal and anomalous groups, and visualizes the results.

**Features**:
Generates synthetic network traffic data (200 normal, 20 anomalous points).
Applies K-Means clustering to group data into two clusters.
Detects anomalies based on distance from cluster centers.
Visualizes clusters, anomalies, and centers in a scatter plot.

**Requirements**

Python 3.13.7
Libraries: numpy, pandas, scikit-learn, matplotlib

**Setup**

Clone the repository:git clone <repository-url>
cd network_anomaly_detector


Install dependencies:pip install numpy pandas scikit-learn matplotlib


Run the script:python anomaly_detector.py



**Files**

anomaly_detector.py: Main script for data generation, clustering, anomaly detection, and visualization.
anomaly_plot.png: Output scatter plot showing clusters and anomalies.
README.md: Project documentation.

**Usage**
Run anomaly_detector.py to:

Generate synthetic data (packet size, frequency).
Cluster data using K-Means (2 clusters).
Detect anomalies (points beyond mean + 2*std distance).
Display/save a scatter plot (anomaly_plot.png).

**Example Output:**

Console: DataFrame head, cluster centers, anomaly count, anomalous points.
Plot: Two clusters (yellow/purple), red ‘x’ anomalies, black dot centers.

