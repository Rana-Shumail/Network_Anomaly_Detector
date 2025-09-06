import numpy as np          # For numerical operations, data generation
import pandas as pd         # For data manipulation, DataFrame
import matplotlib.pyplot as plt  # For plotting
from sklearn.cluster import KMeans # Correct import for KMeans
from datetime import datetime, timedelta #For time operations
import seaborn as sns #For advanced plotting
from scapy.all import sniff #For packet capturing




# Capture live packets

data = []
packets = sniff(count=200)

for pkt in packets:
    size = len(pkt)
    proto = int(pkt.proto) if hasattr(pkt, "proto") else 0
    data.append([size, proto])  # 2D: (size, protocol)


data = np.array(data)

df = pd.DataFrame(data, columns=['packet_size','frequency'])
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



#Add Dashboard

timestamps = [datetime.now() + timedelta(minutes=i) for i in range(len(data))]

def random_ip():
    return f"{np.random.randint(1,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}"

ips = [random_ip() for _ in range(len(data))]
protocols = np.random.choice(['TCP','UDP','ICMP'], size=len(data), p=[0.6,0.3,0.1])


df['timestamp'] = timestamps
df['source_ip'] = ips
df['protocol'] = protocols
df['anomaly'] = distances > threshold
df['cluster'] = labels


# Traffic summary
protocol_counts = df['protocol'].value_counts()
total_packets = len(df)
unique_ips = df['source_ip'].nunique()


# Multi-panel dashboard
fig, axs = plt.subplots(2, 2, figsize=(14,10))
fig.suptitle("Network Anomaly Detection Dashboard", fontsize=16)

# Panel 1: Traffic Summary (Protocol counts)
sns.barplot(x=protocol_counts.index, y=protocol_counts.values, ax=axs[0,0])
axs[0,0].set_title(f"Protocol Distribution\nTotal Packets: {total_packets}, Unique IPs: {unique_ips}")
axs[0,0].set_ylabel("Count")

# Panel 2: Cluster Visualization
colors = ['green' if a==False else 'red' for a in df['anomaly']]
axs[0,1].scatter(df['packet_size'], df['frequency'], c=colors, alpha=0.6, s=50)
axs[0,1].scatter(centers[:,0], centers[:,1], c='black', s=200, marker='X', label='Cluster Centers')
axs[0,1].set_title("Clusters (Normal=Green, Anomaly=Red)")
axs[0,1].set_xlabel("Packet Size")
axs[0,1].set_ylabel("Frequency")
axs[0,1].legend()

# Panel 3: Timeline of Activity
sns.lineplot(data=df, x='timestamp', y='packet_size', hue='anomaly', palette={False:'green', True:'red'}, ax=axs[1,0])
axs[1,0].set_title("Timeline of Packet Sizes")
axs[1,0].set_xlabel("Time")
axs[1,0].set_ylabel("Packet Size")
axs[1,0].tick_params(axis='x', rotation=45)

# Panel 4: Recent Alerts (Top 10 anomalies)
recent_alerts = df[df['anomaly']].sort_values('timestamp', ascending=False).head(10)
axs[1,1].axis('off')  # Turn off axes for table
table_data = recent_alerts[['timestamp','source_ip','packet_size','frequency','protocol']]
axs[1,1].table(cellText=table_data.values,
               colLabels=table_data.columns,
               cellLoc='center',
               loc='center',
               bbox=[0, -0.15, 1, 1])
axs[1,1].set_title("Recent Alerts (Top 10)", pad=2)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("network_dashboard.png")
plt.show()
