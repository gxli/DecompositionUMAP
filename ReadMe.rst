Project Overview
=================

This project provides a Python implementation for performing dimensionality reduction on decomposed data using UMAP (Uniform Manifold Approximation and Projection). It supports multiple decomposition methods, including Constrained Diffusion Decomposition (CDD), Empirical Mode Decomposition (EMD), Multiscale Median Decomposition (MSM), and Adaptive Multiscale Decomposition (AMD). The `DecompositionUMAP` class and associated wrapper functions allow for flexible decomposition and embedding workflows, supporting direct decomposition, pre-computed decompositions, custom decomposition functions, and processing new data with a pre-trained UMAP model.

Installation
============

Requirements
------------

- Python 3.6+
- NumPy
- SciPy
- UMAP-learn
- PyEMD (optional, only required for EMD decomposition)

Install dependencies using pip:

.. code-block:: bash

    pip install numpy scipy umap-learn
    # Install PyEMD only if using EMD decomposition
    pip install emd-signal

Files
-----

- `main.py`: Contains the `DecompositionUMAP` class, `decompose_and_embed`, and `decompose_with_existing_model` functions for decomposition and UMAP embedding.
- `decomposition.py`: Implements decomposition methods (CDD, EMD, MSM, AMD).

Usage
=====

The project supports four main use cases. Below, synthetic data is used for demonstration, but you can load your own data (e.g., from `.npy` files or other formats supported by NumPy).

1. **Direct Decomposition**
----------------------------

Perform decomposition and UMAP embedding in one step using built-in decomposition methods: CDD, EMD, MSM, or AMD.

**Example**:

.. code-block:: python

    import numpy as np
    from main import decompose_and_embed

    # Generate synthetic 2D data
    data = np.random.rand(100, 100)

    # Perform AMD decomposition and UMAP embedding
    embed_map, decomposition, umap_model = decompose_and_embed(
        data=data,
        decomposition_method='amd',
        umap_n_neighbors=15,
        n_component=2,
        threshold=0.01,
        cdd_max_n=5,
        verbose=True
    )

    # Save results
    np.save('embed_map.npy', embed_map)
    np.save('decomposition.npy', decomposition)

**Supported Methods**:
- `cdd`: Constrained Diffusion Decomposition
- `emd`: Empirical Mode Decomposition (1D data only)
- `msm`: Multiscale Median Decomposition
- `amd`: Adaptive Multiscale Decomposition

2. **Pre-Computed Decompositions**
-----------------------------------

Use pre-computed decomposition results with a trained UMAP model or directly with `DecompositionUMAP`.

**Example**:

.. code-block:: python

    import numpy as np
    from main import DecompositionUMAP, decompose_with_existing_model
    from decomposition import adaptive_multiscale_decomposition

    # Generate synthetic data and compute decomposition
    data = np.random.rand(100, 100)
    decomposition, _ = adaptive_multiscale_decomposition(data, max_n=5)

    # Create UMAP embedding
    decomp_umap = DecompositionUMAP(
        decomposition=decomposition,
        original_data=data,
        umap_n_neighbors=15,
        n_component=2,
        threshold=0.01,
        verbose=True
    )
    decomp_umap.save_umap_model('umap_model.pkl')

    # Use pre-computed decomposition with existing model
    new_decomposition, _ = adaptive_multiscale_decomposition(data, max_n=5)
    embed_map, decomposition = decompose_with_existing_model(
        model_filename='umap_model.pkl',
        decomposition=new_decomposition,
        threshold=0.01
    )

3. **Use Supplied Decomposition Function**
--------------------------------------------

Provide a custom decomposition function to `decompose_and_embed` or `DecompositionUMAP`, which is stored and reused for new data.

**Example**:

.. code-block:: python

    import numpy as np
    from main import decompose_and_embed, DecompositionUMAP

    # Define custom decomposition function
    def custom_decomposition(data):
        return np.array([data / 2, data / 2])

    # Generate synthetic data
    data = np.random.rand(100, 100)

    # Perform decomposition and UMAP embedding
    embed_map, decomposition, umap_model = decompose_and_embed(
        data=data,
        decomposition_func=custom_decomposition,
        umap_n_neighbors=15,
        n_component=2,
        threshold=0.01,
        verbose=True
    )

    # Save model
    decomp_umap = DecompositionUMAP(
        decomposition=decomposition,
        original_data=data,
        decomposition_func=custom_decomposition
    )
    decomp_umap.save_umap_model('umap_model.pkl')

**Note**: The custom function must return a `numpy.ndarray` or a tuple where the first element is the decomposition array with shape `(n_components, *data_shape)`.

4. **Take New Data**
--------------------

Process new data using a pre-trained UMAP model, reusing the stored decomposition function or a specified method.

**Example**:

.. code-block:: python

    import numpy as np
    from main import decompose_with_existing_model

    # Generate new synthetic data
    new_data = np.random.rand(100, 100)

    # Use existing model with stored decomposition function (e.g., from AMD)
    embed_map, new_decomposition = decompose_with_existing_model(
        model_filename='umap_model.pkl',
        data=new_data,
        decomposition_method='amd',
        threshold=0.01,
        cdd_max_n=5,
        verbose=True
    )

    # Use existing model with custom decomposition function
    def custom_decomposition(data):
        return np.array([data / 2, data / 2])

    embed_map, new_decomposition = decompose_with_existing_model(
        model_filename='umap_model.pkl',
        data=new_data,
        decomposition_func=custom_decomposition,
        threshold=0.01
    )

Notes
=====

- **Decomposition Consistency**: When processing new data, `decompose_with_existing_model` prioritizes the decomposition function stored in `DecompositionUMAP` (if provided during initialization), ensuring consistency with the training phase.
- **Parameters**:
  - `threshold`: Masks data below the specified value with NaN.
  - `cdd_max_n`: Limits the number of components for CDD or AMD.
  - `emd_max_imf`: Limits the number of IMFs for EMD.
  - `msm_filter_sizes`: Controls MSM decomposition scales (use 'auto' or a tuple `(base_window, max_scale, spacing)`).
- **Dependencies**: Install `numpy`, `scipy`, and `umap-learn`. Install `emd-signal` only if using EMD decomposition.
- **Performance**: For large datasets, optimize decomposition functions (e.g., reduce `cdd_max_n` or adjust `msm_filter_sizes`).

License
=======

This project is licensed under the MIT License.