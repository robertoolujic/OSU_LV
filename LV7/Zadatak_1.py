import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering

def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

for i in range(1, 6):
# generiranje podatkovnih primjera
    X = generate_data(500, i)

    # prikazi primjere u obliku dijagrama rasprsenja
    plt.figure()
    plt.scatter(X[:,0],X[:,1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('podatkovni primjeri')
    plt.show()

    #Podatci s i=1 su grupirani u 3 skupine

    plt.figure(figsize=(15,10))
    for k in range(1,7):
        # inicijalizacija algoritma K srednjih vrijednosti
        km = KMeans ( n_clusters =k, init ="random",
        n_init =5, random_state =0)
        # pokretanje grupiranja primjera
        km.fit(X)
        # dodijeljivanje grupe svakom primjeru
        labels = km.predict(X)
        plt.subplot(2,3,k)
        plt.scatter(X[:,0],X[:,1], c=labels, cmap="jet")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title("podatkovni primjeri s K={}".format(k))
    plt.show()

    #Primjetivo je kako povećanjem broja K se stvaraju specifične grupe koje se ne čine potrebnima

    #Za neke grupacije (npr. za flagc=4 i flagc=5) grupiranje nije isto kao u npr. Kneighbors, jer ovisi o lokaciji točke grupiranja a ne o obliku same grupe