import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

for n_slika in range(1,7):
    # ucitaj sliku
    img = Image.imread("imgs\\test_{}.jpg".format(n_slika))

    # prikazi originalnu sliku
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()
    
    print("Veličina originalne slike:")
    print(img.size)
    print("Količina boja originalne slike:")
    print(np.unique(img).size)
    #Slika ima 256 različitih boja (kao kombinacija R G B)
    inertia = []
    for k in range(1,10):
        km = KMeans ( n_clusters =k, init ="random",
                n_init =5, random_state =0)
        # pokretanje grupiranja primjera
        km.fit(img_array)

        inertia.append(km.inertia_)
        # dodijeljivanje grupe svakom primjeru
        labels = km.predict(img_array)
        labels_2d = labels.reshape(w,h)
        reconstructed = np.zeros((w,h,d))
        binary = np.zeros((w,h,d))
        label_idx=0
        for i in range(0,w):
            for j in range(0,h):
                reconstructed[i][j]=km.cluster_centers_[labels[label_idx]]
                label_idx+=1
        
        
        plt.subplot(3,3,k)
        plt.title("Rekonstruirana slika s {} boja".format(k))
        plt.imshow(reconstructed)
        plt.tight_layout()
        print("Veličina rekonstruirane slike {}:".format(k))
        print(reconstructed.size)
        print("Količina boja rekonstruirane slike {}:".format(k))
        print(np.unique(reconstructed).size)
    plt.show()
    for cluster_idx in range(k):
            binary_mask = (labels_2d==cluster_idx)
            plt.subplot(1, k, cluster_idx + 1)
            plt.title(f"Boja {cluster_idx}")
            plt.imshow(binary_mask, cmap='gray')
            plt.axis('off')
    plt.show()
    plt.plot(range(1,10),inertia, marker=".")
    plt.title("Ovisnost J o K")
    plt.show()