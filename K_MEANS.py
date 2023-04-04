import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# K_Means
if __name__ == "__main__":
    N_CLUSTERS = 5
    MARKERS = ['*', 'v', '+', '^', 's', 'x']
    COLORS = ['r', 'g', 'm', 'c', 'y', 'b']
    AREAS = ['Caribbean', 'the Mediterranean and Baltic Sea', 'U.S. West Coast', 'European Atlantic coast',
             'U.S. East Coast']
    DATA_PATH = 'pos.xlsx'
    df = pd.read_excel(DATA_PATH)
    x = df.drop(['Make', 'Year1', 'Listing Price (USD)', 'Length1', 'Geographic Region', 'Country/Region/State'], axis=1)
    x_np = np.array(x)
    model = KMeans(n_clusters=N_CLUSTERS, random_state=0)
    model.fit(x)
    labels = model.labels_
    # print silhouette score
    print(silhouette_score(x, labels))
    # print cluster centers
    print(model.cluster_centers_)
    # draw picture
    plt.figure(figsize=(9, 6))
    plt.title("Regions selling boats in dataset", fontsize=22)
    plt.xlabel('Longitude', fontsize=18)
    plt.ylabel('Latitude', fontsize=18)
    for i in range(N_CLUSTERS):
        members = labels == i
        plt.scatter(x_np[members, 1], x_np[members, 0], marker=MARKERS[i], c=COLORS[i], label=AREAS[i])
    plt.grid()
    plt.legend()
    plt.show()
