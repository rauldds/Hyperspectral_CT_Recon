import numpy as np
from sklearn.decomposition import PCA
def dimensionality_reduction(matrix, method, orig_shape, target_dims, whiten=True):
    if method == "merge":
        group_size = orig_shape[0] // target_dims
        mean_values = []
        for i in range(target_dims):
            group_data = matrix[i * group_size : (i + 1) * group_size]
            mean_value = group_data.mean(axis=0)
            mean_values.append(mean_value)
        new_matrix = np.stack(mean_values)
        return new_matrix
    elif method == "pca":
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
    else:
        return matrix
       
       
