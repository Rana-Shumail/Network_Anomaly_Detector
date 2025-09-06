**Network Anomaly Detector**

**Overview**

This project implements a network anomaly detector using K-Means clustering to identify unusual network traffic patterns, such as potential cyber threats (e.g., DDoS attacks or data exfiltration). It now captures live network traffic from your machine, clusters it into normal and anomalous groups, and visualizes the results in a multi-panel dashboard.

**Features**

Captures live network packets (packet size and protocol).

Applies K-Means clustering to group traffic into normal and anomalous clusters.

Detects anomalies based on distance from cluster centers.

Generates a multi-panel dashboard including:

Protocol distribution summary (bar chart)

Cluster visualization (normal = green, anomalies = red)

Timeline of packet activity over time

Recent alerts table showing top anomalous packets

**Requirements**

Python 3.13.7

Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, scapy

**Files**

network_anomaly_dashboard.py: Main script for live packet capture, clustering, anomaly detection, and dashboard visualization.

network_dashboard.png: Output image of the multi-panel dashboard.

README.md: Project documentation.

**Usage**

Run network_anomaly_dashboard.py to:

Capture live network traffic (packet size and protocol).

Cluster captured data using K-Means (2 clusters).

Detect anomalies (points beyond mean + 2*std distance from cluster centers).

Display and save a dashboard showing protocol distribution, clusters, timeline, and recent alerts.

**Example Output**

Console:

DataFrame head, cluster centers, anomaly count, anomalous points.

Dashboard Plot:

Protocol distribution bar chart.

Cluster scatter plot (green = normal, red = anomalies, black = cluster centers).

Timeline line chart of packet sizes over time.

Recent alerts table with top anomalies.
