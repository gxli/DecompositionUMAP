import matplotlib.pyplot as plt
import decomposition_umap
import numpy as np

data, signal, anomaly =decomposition_umap.example.generate_fractal_with_gaussian()
plt.figure()
plt.imshow(data, cmap='gray')
plt.show()
embedding, decomposition, umap_obj = decomposition_umap.decomposition_umap(data)

datax = embedding[0]

datay = embedding[1]


plt.figure()
plt.scatter(datax.flatten(), datay.flatten(), s=1, c='blue', alpha=0.5)

# Save datax and datay as .npy files
np.save('data.npy',data)
np.save('signal.npy', signal)
np.save('anomaly.npy', anomaly)
np.save('umap_x.npy', datax)
np.save('umap_y.npy', datay)
