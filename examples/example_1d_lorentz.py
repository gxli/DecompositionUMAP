
import matplotlib.pyplot as plt
import decomposition_umap
import numpy as np

# Generate Lorenz system data
t,x,y,z = decomposition_umap.example.generate_lorenz_system()

# Perform decomposition and UMAP embedding
embedding, decomposition, umap_obj = decomposition_umap.decompose_and_embed(data=x,decomposition_method='amd')

embed_x = embedding[0]
embed_y = embedding[1]

# Save the data (as in your original script)
np.save('lorentz_data_x.npy', x)
np.save('lorentz_data_y.npy', y)
np.save('lorentz_data_z.npy', z)
np.save('lorentz_umap_x.npy', embed_x)
np.save('lorentz_umap_y.npy', embed_y)

# --- Add Plots ---

plt.figure(figsize=(12, 6)) # Create a figure with a good size for two subplots

# Left Plot: Original x against y
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st subplot
plt.plot(x, y, linewidth=0.5, alpha=0.7)
plt.title('Lorenz System: x vs. y')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

# Right Plot: UMAP_x against UMAP_y
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd subplot
# Using scatter for UMAP as points are often distinct and can reveal density
plt.scatter(embed_x, embed_y, s=5, alpha=0.7)
plt.title('UMAP Embedding: UMAP_x vs. UMAP_y')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.grid(True)

plt.tight_layout() # Adjust subplot parameters for a tight layout
plt.show() # Display the plots