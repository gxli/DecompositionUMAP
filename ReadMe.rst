===============================================================================================
Decomposition-UMAP: A framework for pattern classification and anomaly detection
===============================================================================================

.. image:: https://img.shields.io/pypi/v/decomposition-umap.svg
        :target: https://pypi.python.org/pypi/decomposition-umap
        :alt: PyPI Version

.. image:: https://img.shields.io/travis/gxli/DecompositionUMAP.svg
        :target: https://travis-ci.org/gxli/DecompositionUMAP
        :alt: Build Status

Decomposition-UMAP
==================
Decomposition-UMAP is a general-purpose framework for pattern classification and anomaly detection. The methodology involves a two-stage process: first, the application of a multiscale decomposition technique, followed by a non-linear dimension reduction using the Uniform Manifold Approximation and Projection (UMAP) algorithm.

Abstract
--------

This software provides a structured implementation for analyzing numerical data by combining signal and image decomposition with manifold learning. The primary workflow involves decomposing an input dataset into a set of components, which serve as a high-dimensional feature vector for each point in the original data. Subsequently, the UMAP algorithm is employed to project these features into a lower-dimensional space. This process is designed to facilitate the analysis of data where features may be present across multiple scales or frequencies, enabling the separation of structured signals from noise.

Functionality
-------------

*   **Flexible API with Explicit Modes**: Provides a high-level API that supports single datasets, batch processing, multi-channel data, and pre-computed decompositions.
*   **Powerful Decomposition Techniques**: Includes interfaces for methods like Constrained Diffusion Decomposition (CDD) and Empirical Mode Decomposition (EMD).
*   **Full UMAP Control**: Allows for complete control over the UMAP algorithm's parameters via convenience arguments and a flexible dictionary (`umap_params`).
*   **Support for Custom Functions**: Users can supply their own decomposition functions for maximum extensibility.
*   **Serialization of Models**: Trained UMAP models can be saved using `pickle` and reloaded for consistent inference on new data.

Installation
------------

The required Python packages must be installed prior to use. It is recommended to use a virtual environment.

.. code-block:: bash

    pip install numpy umap-learn scipy matplotlib

The Python package can then be installed or integrated into a project. The decomposition functions (`cdd_decomposition`, etc.) are presumed to be located in a `multiscale_decomposition` module.

Usage
-----

The following examples demonstrate the core workflows using a synthetic 256x256 dataset composed of a Gaussian anomaly embedded in a fractal noise background.

1. Data Generation
~~~~~~~~~~~~~~~~~~

First, we generate the data. This function is assumed to be available in an `example` module within the library.

.. code-block:: python

    import numpy as np
    import src as decomposition_umap
    from src import example as du_example

    # Generate a dataset with a known anomaly
    data, signal, anomaly = du_example.generate_fractal_with_gaussian()

2. Running the Pipeline (Core Examples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Example A: Standard Mode (Built-in Decomposition)**

This is the most common use case for training a new model.

.. code-block:: python

    embed_map, decomposition, umap_model = decomposition_umap.decompose_and_embed(
        data=data,
        decomposition_method='cdd',
        decomposition_max_n=6,
        n_component=2,
        umap_n_neighbors=20
    )

    # Save the model for the inference example
    with open("fractal_umap_model.pkl", "wb") as f:
        pickle.dump(umap_model, f)

**Example B: Custom Decomposition Function (`decomposition_func=...`)**

Use this when you have your own method for separating features.

.. code-block:: python

    from scipy.ndimage import gaussian_filter

    def my_custom_decomposition(data):
        """A simple decomposition using Gaussian filters."""
        comp1 = gaussian_filter(data, sigma=3)
        comp2 = data - comp1
        return np.array([comp1, comp2])

    embed_map_custom, _, _ = decomposition_umap.decompose_and_embed(
        data=data,
        decomposition_func=my_custom_decomposition,
        n_component=2
    )

**Example C: Pre-computed Decomposition (`decomposition=...`)**

This is efficient if your decomposition is slow and you want to reuse it while testing UMAP parameters.

.. code-block:: python

    from src.multiscale_decomposition import cdd_decomposition

    # Manually run the decomposition first
    precomputed, _ = cdd_decomposition(data, max_n=6)

    embed_map_pre, _, _ = decomposition_umap.decompose_and_embed(
        decomposition=np.array(precomputed),
        n_component=2
    )

**Example D: Inference with a Pre-trained Model**

Use `decompose_with_existing_model` to apply a saved model to new data.

.. code-block:: python

    # Generate new data for inference
    new_data, _, _ = du_example.generate_fractal_with_gaussian(anomaly_center=(200, 200))

    # Apply the model saved from Example A
    new_embed_map, _ = decomposition_umap.decompose_with_existing_model(
        model_filename="fractal_umap_model.pkl",
        data=new_data,
        decomposition_method='cdd',
        decomposition_max_n=6
    )

3. Visualizing Results
~~~~~~~~~~~~~~~~~~~~~~

The UMAP embedding can effectively separate the anomaly from the background.

.. code-block:: python

    import matplotlib.pyplot as plt

    # --- Plot the UMAP embedding from Example A ---
    umap_x = embed_map[0].flatten()
    umap_y = embed_map[1].flatten()

    is_highlighted = anomaly.flatten() > data.flatten()

    plt.figure(figsize=(8, 8))
    plt.scatter(
        umap_x[~is_highlighted], umap_y[~is_highlighted],
        label='Background', alpha=0.1, s=10, color='gray'
    )
    plt.scatter(
        umap_x[is_highlighted], umap_y[is_highlighted],
        label='Highlighted Anomaly (Anomaly > Data)',
        alpha=0.8, s=15, color='red'
    )
    plt.title('UMAP Embedding with Anomaly Highlighted', fontsize=16)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.show()


API Reference
-------------

**`decompose_and_embed(...)`**

The primary function for **training** a new Decomposition-UMAP model. It intelligently handles multiple input modes for maximum flexibility.

*   **Operating Modes (provide exactly one)**:

    *   `data`: For a single raw dataset.

    *   `datasets`: For a batch of raw datasets.

    *   `data_multivariate`: For a multi-channel raw dataset.

    *   `decomposition`: For a single pre-computed decomposition.

*   **Key Parameters**:

    *   `decomposition_method` (`str`): The name of the built-in decomposition method (e.g., `'cdd'`).

    *   `decomposition_func` (`callable`): A user-provided decomposition function.

    *   `n_component` (`int`): The target dimension for the final UMAP embedding.

    *   `umap_n_neighbors` (`int`): Convenience argument for UMAP's `n_neighbors`.

    *   `low_memory` (`bool`): Convenience argument for UMAP's `low_memory` flag.

    *   `umap_params` (`dict`): For advanced control, a dictionary of arguments passed directly to the `umap.UMAP` constructor.

*   **Returns**: A tuple whose contents depend on the operating mode. For single dataset modes, it returns `(embed_map, decomposition, umap_model)`.


**`decompose_with_existing_model(...)`**

The primary function for **inference**. It applies a pre-trained UMAP model to new data, ensuring a consistent transformation. It does not require UMAP training parameters as they are loaded from the model file.

*   **Operating Modes (provide exactly one)**:

    *   `data`, `datasets`, `data_multivariate`, or `decomposition`.

*   **Key Parameters**:

    *   `model_filename` (`str`): Path to the pickled UMAP model file.

    *   `data` (`numpy.ndarray`): The new data array to transform.

    *   Decomposition parameters (`decomposition_method`, etc.) **must match** those used during model training.

*   **Returns**: A tuple whose contents depend on the operating mode. For single dataset modes, it returns `(embed_map, final_decomposition)`.


**`DecompositionUMAP` class**

The core engine that encapsulates the workflow state. It offers granular control and can be initialized with raw data (using `decomposition_method` or `decomposition_func`) or a pre-computed `decomposition`.

*   **Key Methods**:

    *   `save_umap_model(filename)`: Saves the trained model to a file.

    *   `load_umap_model(filename)`: Loads a serialized model from a file.

    *   `compute_new_embeddings(...)`: The core inference method that projects new data using the trained model.


Dependencies
------------

*   `numpy`
*   `umap-learn`
*   `scipy`
*   `matplotlib` (for running visualization examples)

Contributing
------------

Contributions to the source code are welcome. Please feel free to fork the repository, make changes, and submit a pull request. For bugs or feature requests, please open an issue on the repository's GitHub page.

License
-------

This software is distributed under the MIT License. Please refer to the `LICENSE` file for full details.

Contact
-------

**Author**: Guang-Xiang Li
**Email**: `guangxiangli@gmail.com`
**GitHub**: `https://github.com/gxli`