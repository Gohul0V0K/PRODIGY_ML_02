import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data=pd.read_csv("ML_02.csv")

X=data[['Age','Annual Income (k$)','Spending Score (1-100)']]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

silhouette_scores=[]
for n_clusters in range(2,11):
    kmeans=KMeans(n_clusters=n_clusters,init='k-means++',random_state=42,n_init=10)
    cluster_labels=kmeans.fit_predict(X_scaled)
    silhouette_avg=silhouette_score(X_scaled,cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.plot(range(2,11),silhouette_scores,marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.show()
print("\n\n")

optimal_num_clusters=silhouette_scores.index(max(silhouette_scores))+2
kmeans=KMeans(n_clusters=optimal_num_clusters,init='k-means++',random_state=42,n_init=10)
kmeans.fit(X_scaled)
labels=kmeans.labels_

data['Cluster']=labels

plt.figure(figsize=(10,6))
for cluster in range(optimal_num_clusters):
    cluster_data=data[data['Cluster']==cluster]
    plt.scatter(cluster_data['Annual Income (k$)'],cluster_data['Spending Score (1-100)'],label=f'Cluster {cluster}')
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],s=300,c='red',label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-means Clustering of Customers')
plt.legend()
plt.show()

print("\n\n")

print(data[['CustomerID','Cluster']])
