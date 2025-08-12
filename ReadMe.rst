======================================================
Multiscale Decomposition with UMAP Projection
======================================================

.. image:: https://img.shields.io/pypi/v/your-package-name.svg
        :target: https://pypi.python.org/pypi/your-package-name
.. image:: https://img.shields.io/travis/your-username/your-repo-name.svg
        :target: https://travis-ci.org/your-username/your-repo-name
.. image:: https://readthedocs.org/projects/your-package-name/badge/?version=latest
        :target: https://your-package-name.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

A Python module for performing dimensionality reduction on multi-dimensional data. The methodology involves a two-stage process: first, the application of a multiscale decomposition technique, followed by a non-linear dimension reduction using the Uniform Manifold Approximation and Projection (UMAP) algorithm.

Abstract
--------

This software provides a structured implementation for analyzing numerical data by combining signal and image decomposition with manifold learning. The primary workflow involves decomposing an input dataset into a set of components, which serve as a high-dimensional feature vector for each point in the original data. Subsequently, the UMAP algorithm is employed to project these features into a lower-dimensional space. This process is designed to facilitate the analysis of data where features may be present across multiple scales or frequencies.

Functionality
-------------

*   **Supported Decomposition Techniques**: Includes interfaces for several decomposition methods:
    *   Component Decomposition (CDD)
    *   Empirical Mode Decomposition (EMD)
    *   Multiscale Morphological Decomposition (MSM)
    *   Adaptive Multiscale Decomposition (AMD)
*   **Optional Hilbert Transform**: Provides an option to apply the Hilbert transform to each decomposition component to compute the analytical signal's amplitude. This is applicable for 1D or 2D component data.
*   **UMAP for Dimensionality Reduction**: Utilizes the UMAP algorithm to compute a low-dimensional embedding of the decomposed data.
*   **Support for Custom Decomposition Functions**: Users can supply their own decomposition functions, provided they adhere to the specified interface.
*   **Training on Data Subsets**: The UMAP model can be trained on a specified fraction of the data, which is useful for managing memory and computational costs with large datasets.
*   **Serialization of Trained Models**: Trained UMAP models can be saved to disk using `pickle` and subsequently reloaded to transform new data, ensuring reproducible results.

Installation
------------

The required Python packages must be installed prior to use:

.. code-block:: bash

    pip install numpy umap-learn scipy

The Python module (`decomposition_umap.py`) can then be integrated into a project. The decomposition functions (`cdd_decomposition`, `emd_decomposition`, etc.) are presumed to be located in a module named `multiscale_decomposition` within the same project structure.

Usage
-----

### Primary Usage: Decomposing and Embedding Data

The principal function `decompose_and_embed` executes the full workflow. The following example demonstrates its application to a 2D numpy array using Multiscale Morphological (MSM) decomposition.

.. code-block:: python

    import numpy as np
    import pickle
    from decomposition_umap import decompose_and_embed

    # 1. Define a 2D input dataset.
    data = np.random.rand(128, 128)

    # 2. Execute the decomposition and UMAP embedding.
    #    - embed_map: A list of numpy arrays, one for each UMAP component.
    #    - decomposition: The intermediate multiscale components from MSM.
    #    - umap_model: The trained umap.UMAP object.
    embed_map, decomposition, umap_model = decompose_and_embed(
        data,
        decomposition_method='msm',
        msm_filter_sizes='auto',
        n_component=2,
        verbose=True
    )

    # The returned 'embed_map' contains the low-dimensional representation.
    umap_component_1 = embed_map[0]
    umap_component_2 = embed_map[1]

    print(f"Dimensions of UMAP component 1: {umap_component_1.shape}")

    # 3. The trained model can be serialized for later use.
    with open('umap_model.pkl', 'wb') as f:
        pickle.dump(umap_model, f)


### Application of the Hilbert Transform

To base the UMAP embedding on the analytical amplitude of the components rather than their direct values, set the `use_hilbert_amplitude` parameter to `True`.

.. code-block:: python

    embed_map_hilbert, _, _ = decompose_and_embed(
        data,
        decomposition_method='msm',
        use_hilbert_amplitude=True,
        n_component=2,
        verbose=True
    )


### Applying a Trained Model to New Data

The `decompose_with_existing_model` function applies a previously trained model to new data. This ensures that the decomposition and projection are performed in a manner consistent with the original model training.

.. code-block:: python

    from decomposition_umap import decompose_with_existing_model

    # 1. Define a new dataset with dimensions compatible with the trained model.
    new_data = np.random.rand(128, 128)
    model_filename = 'umap_model.pkl'

    # 2. Apply the serialized model to the new data.
    #    The same decomposition method and parameters used for training
    #    must be specified to ensure a valid transformation.
    new_embed_map, new_decomposition = decompose_with_existing_model(
        model_filename=model_filename,
        data=new_data,
        decomposition_method='msm',
        msm_filter_sizes='auto',
        verbose=True
    )

    print(f"Dimensions of new UMAP component 1: {new_embed_map[0].shape}")


### Integration of Custom Decomposition Functions

Users may provide a custom function for the decomposition stage. The function must accept a `numpy.ndarray` as input and return a `numpy.ndarray` with the shape `(n_components, ...data_shape)`.

.. code-block:: python

    from scipy.ndimage import gaussian_filter

    def custom_decomposition(data):
        """A simple decomposition based on Gaussian filtering."""
        comp1 = gaussian_filter(data, sigma=2)
        comp2 = data - comp1
        return np.array([comp1, comp2])

    # Employ the custom function within the standard workflow.
    embed_map_custom, _, _ = decompose_and_embed(
        data,
        decomposition_func=custom_decomposition,
        n_component=2,
        verbose=True
    )

API Reference
-------------

**`decompose_and_embed(...)`**

Performs decomposition and trains a new UMAP model.

*   **Parameters**:
    *   `data` (`numpy.ndarray`): Input data array.
    *   `decomposition_method` (`str`): The name of the built-in decomposition method.
    *   `decomposition_func` (`callable`): A user-provided decomposition function.
    *   `use_hilbert_amplitude` (`bool`): If True, applies the Hilbert transform to components.
    *   `n_component` (`int`): The target dimension for the UMAP embedding.
    *   Other keyword arguments for UMAP and the selected decomposition method.
*   **Returns**: A tuple containing the list of UMAP component arrays, the decomposition result, and the trained `umap.UMAP` model.

**`decompose_with_existing_model(...)`**

Applies a pre-trained UMAP model to new data.

*   **Parameters**:
    *   `model_filename` (`str`): Path to the serialized UMAP model file.
    *   `data` (`numpy.ndarray`): The new data array to transform.
    *   Decomposition parameters must match those used during model training.
*   **Returns**: A tuple containing the list of new UMAP component arrays and the new decomposition result.

**`DecompositionUMAP` class**

A class that encapsulates the workflow state. The wrapper functions are the recommended interface, but this class can be instantiated for more granular control.

Dependencies
------------

*   `numpy`
*   `umap-learn`
*   `scipy`

Contributing
------------

Contributions to the source code are welcome. Please submit pull requests or open issues through the project's repository for consideration.

License
-------

This software is distributed under the MIT License. Please refer to the `LICENSE` file for full details.