from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
def dimensionality_reduction(matrix, method, orig_shape, target_dims, whiten=True):
    if method == "none" or method is None:
        return matrix
    if method == "pca":
        # Assume first is channels
        matrix = matrix.transpose((1,2,0)) 
        matrix = matrix.reshape(-1, orig_shape[0])
        pca = PCA(n_components=target_dims, whiten=whiten)      
        matrix = pca.fit_transform(matrix)
        matrix = matrix.transpose((1,0))
        matrix = matrix.reshape((target_dims,*orig_shape[1:]))
        # plt.imshow(matrix[0]) 
        # plt.show()
        return matrix
       
       
