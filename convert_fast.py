import matplotlib.pyplot as plt
import decomposition_umap
import numpy as np


data = np.load('./dyn20231205.npz')
data = data['arr_0']

np.save('data_fast.npy',data)




