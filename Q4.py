import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.decomposition import PCA

def funct_pca(p_img):
    b = p_img[:, :, 0]
    g = p_img[:, :, 1]
    r = p_img[:, :, 2]

    pca_b = PCA(70)
    pca_g = PCA(70)
    pca_r = PCA(70)

    pca_img_b = pca_b.fit_transform(b)
    pca_img_g = pca_g.fit_transform(g)
    pca_img_r = pca_r.fit_transform(r)

    approx_b = pca_b.inverse_transform(pca_img_b)
    approx_g = pca_g.inverse_transform(pca_img_g)
    approx_r = pca_r.inverse_transform(pca_img_r)

    pca_img = np.stack([approx_b,
                        approx_g,
                        approx_r],
                        axis=2).astype("uint8")

    return pca_img

def Image_Reconstruction():
    imgs = os.listdir("./Q4_Image")

    for i in range(15):
        img = cv2.imread("./Q4_Image/" + imgs[i])
        img_2 = img[:,:,[2,1,0]]
        plt.subplot(4, 15, i+1)
        plt.imshow(img_2)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        if i == 0:
            plt.ylabel('original')

        pca_img = funct_pca(img_2)

        plt.subplot(4, 15, 15+i+1)
        plt.imshow(pca_img)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        if i == 0:
            plt.ylabel('reconstruction')

    for i in range(15, 30):
        img = cv2.imread("./Q4_Image/" + imgs[i])
        img_3 = img[:,:,[2,1,0]]
        plt.subplot(4, 15, 15+i+1)
        plt.imshow(img_3)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        if i == 15:
            plt.ylabel('original')

        pca_img = funct_pca(img_3)

        plt.subplot(4, 15, 15*2+i+1)
        plt.imshow(pca_img)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        if i == 15:
            plt.ylabel('reconstruction')
    
    plt.show()


def Computer_Reconstruction_Error():
    imgs = os.listdir("./Q4_Image")
    #re_list=[]
    for i in range(len(imgs)):
        img = cv2.imread("./Q4_Image/" + imgs[i], cv2.IMREAD_GRAYSCALE)
        total_loss=0
        pca = PCA(70)

        pca_img = pca.fit_transform(img)

        approx_img_a = pca.inverse_transform(pca_img).astype('uint8')
        approx_img = approx_img_a.astype('int32')

        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                total_loss += abs(img[i][j] - approx_img[i][j])
        #re_list.append(total_loss)
        print(total_loss)
    #print(re_list)

#Computer_Reconstruction_Error()
#Image_Reconstruction()
