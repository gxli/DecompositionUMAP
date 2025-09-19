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

*   **Supported Decomposition Techniques**: Includes interfaces for several decomposition methods:

    *   Constrained Diffusion Decomposition (CDD)
    
    *   Adaptive Multiscale Decomposition (AMD)

    *   Multiscale Median Decomposition (MSM)
    
    *   Empirical Mode Decomposition (EMD)

*   **UMAP for Dimensionality Reduction**: Utilizes the `umap-learn` library to compute a low-dimensional embedding of the decomposed data, revealing the underlying data manifold.

*   **Flexible API**: Provides both high-level wrapper functions for ease of use and a core `DecompositionUMAP` class for more granular control over the workflow.


*   **Support for Custom Decomposition Functions**: Users can supply their own decomposition functions, provided they adhere to the specified interface.

*   **Training on Data Subsets**: The UMAP model can be trained on a specified fraction of the data, which is useful for managing memory and computational costs with large datasets.

*   **Serialization of Trained Models**: Trained UMAP models can be saved to disk using `pickle` and subsequently reloaded to transform new data, ensuring reproducible results.

Installation
------------

The required Python packages must be installed prior to use. It is recommended to use a virtual environment.

.. code-block:: bash

    pip install numpy umap-learn scipy matplotlib

The Python package can then be installed from the repository or integrated into a project. The decomposition functions (`cdd_decomposition`, etc.) are presumed to be located in a `multiscale_decomposition` module.

Usage
-----
The following examples demonstrate the workflow using a synthetic 256x256 dataset composed of Gaussian blobs (signal) embedded in a pink noise background.

Data Generation
~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    # Assuming the package is structured under 'src' or installed
    import src as decomposition_umap

    def generate_pink_noise(shape):
        """Generates pink noise with a 1/f power spectrum."""
        rows, cols = shape
        u, v = np.fft.fftfreq(rows), np.fft.fftfreq(cols)
        frequency_radius = np.sqrt(u[:, np.newaxis]**2 + v**2)
        frequency_radius[0, 0] = 1.0
        fft_white_noise = np.fft.fft2(np.random.randn(rows, cols))
        fft_pink_noise = fft_white_noise / frequency_radius
        pink_noise = np.real(np.fft.ifft2(fft_pink_noise))
        return (pink_noise - pink_noise.mean()) / pink_noise.std()

    def add_gaussian_blobs(data, centers, sigmas, amplitudes):
        """Adds Gaussian blobs to an existing data array."""
        rows, cols = data.shape
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        signal = np.zeros_like(data, dtype=float)
        for center, sigma, amp in zip(centers, sigmas, amplitudes):
            cx, cy, sx, sy = *center, *sigma
            signal += amp * np.exp(-(((x - cx)**2 / (2 * sx**2)) + ((y - cy)**2 / (2 * sy**2))))
        return data + signal, signal

    # Generate the 256x256 dataset
    shape = (256, 256)
    pink_noise = generate_pink_noise(shape)
    data, signal_blobs = add_gaussian_blobs(
        pink_noise,
        centers=[(60, 80), (160, 180), (100, 200)],
        sigmas=[(10, 10), (16, 8), (12, 12)],
        amplitudes=[3.0, 2.5, 2.0],
    )

Running the Pipeline
~~~~~~~~~~~~~~~~~~~~

There are three primary ways to execute the workflow.

**Example A: High-Level API (Recommended)**

.. code-block:: python

    embed_map, decomposition, umap_model = decomposition_umap.decompose_and_embed(
        data,
        decomposition_method='cdd',
        decomposition_max_n=6,
        n_component=2,
        verbose=True,
    )

**Example B: Using the `DecompositionUMAP` Class Directly**

.. code-block:: python

    decomposition_func = lambda d: decomposition_umap.multiscale_decomposition.cdd_decomposition(d, max_n=6)

    pipeline_instance = decomposition_umap.DecompositionUMAP(
        original_data=data,
        decomposition_func=decomposition_func,
        n_component=2,
        verbose=True,
    )

**Example C: Using a Pre-computed Decomposition**

.. code-block:: python

    precomputed_decomposition, _ = decomposition_umap.multiscale_decomposition.cdd_decomposition(data, max_n=6)

    pipeline_precomputed = decomposition_umap.DecompositionUMAP(
        decomposition=precomputed_decomposition,
        n_component=2,
        verbose=True,
    )

Saving and Visualizing Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    import matplotlib.pyplot as plt

    # Save the results to .npy files
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "decomposition.npy"), decomposition)
    np.save(os.path.join(output_dir, "embed_map.npy"), np.array(embed_map))
    
    # Example Visualization...

API Reference
-------------

**`decompose_and_embed(...)`**

Performs decomposition on raw data and trains a new UMAP model in a single step.

*   **Key Parameters**:

    *   `data` (`numpy.ndarray`): The input data array to be processed.

    *   `decomposition_method` (`str`): The name of the built-in decomposition method (e.g., `'cdd'`).

    *   `decomposition_func` (`callable`, optional): A user-provided custom function for decomposition.

    *   `decomposition_max_n` (`int`, optional): Controls the number of components for relevant methods.

    *   `n_component` (`int`): The target dimension for the final UMAP embedding.

    *   `norm_func` (`callable`, optional): Function to normalize feature vectors. Defaults to `None`.

*   **Returns**: A tuple `(embed_map, decomposition, umap_model)`.

**`decompose_with_existing_model(...)`**

Applies a pre-trained UMAP model to new data.

*   **Key Parameters**:

    *   `model_filename` (`str`): Path to the pickled UMAP model file.

    *   `data` (`numpy.ndarray`): The new raw data array to transform.

    *   Decomposition parameters **must match** those used during model training.
    
*   **Returns**: A tuple `(embed_map, final_decomposition)`.


**`DecompositionUMAP` class**

The core class that encapsulates the workflow state. It offers more granular control over the process.

*   **Initialization Options**:
    The class is initialized in one of two ways:

    1.  **With Raw Data for Training**: Provide ``original_data`` and a ``decomposition_func``.

        .. code-block:: python

            instance = DecompositionUMAP(
                original_data=my_data,
                decomposition_func=my_func,
            )

    2.  **With a Pre-computed Decomposition for Training**: Provide a ``decomposition`` that has already been computed.

        .. code-block:: python

            instance = DecompositionUMAP(
                decomposition=my_precomputed_decomposition,
            )

*   **Key Methods**:
    *   `save_umap_model(filename)`: Saves the trained model to a file.
    *   `load_umap_model(filename)`: Loads a serialized model from a file.
    *   `compute_new_embeddings(...)`: Projects new data using the trained model.

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