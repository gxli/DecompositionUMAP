================================================================================
Decomposition-UMAP: A framework for pattern classification and anomaly detection
================================================================================

.. image:: https://img.shields.io/pypi/v/decomposition-umap.svg
   :target: https://pypi.org/project/decomposition-umap/0.1.0
   :alt: PyPI Version

.. .. image:: https://img.shields.io/travis/gxli/DecompositionUMAP.svg
..    :target: https://travis-ci.org/gxli/DecompositionUMAP
..    :alt: Build Status

.. image:: images/logo.png
   :alt: Project Logo
   :width: 200px
   :align: center

------------------------
Decomposition-UMAP
------------------------

.. image:: images/decomposition-umap_workflow.png
   :width: 100%
   :align: center
   :alt: Decomposition-UMAP workflow

Decomposition-UMAP is a general-purpose framework for pattern classification and anomaly detection. The methodology involves a two-stage process: first, the application of a multiscale decomposition technique, followed by a non-linear dimension reduction using the Uniform Manifold Approximation and Projection (UMAP) algorithm.

This software provides a structured implementation for analyzing numerical data by combining signal and image decomposition with manifold learning. The primary workflow involves decomposing an input dataset into a set of components, which serve as a high-dimensional feature vector for each point in the original data. Subsequently, the UMAP algorithm is employed to project these features into a lower-dimensional space. This process is designed to facilitate the analysis of data where features may be present across multiple scales or frequencies, enabling the separation of structured signals from noise.

.. Functionality
.. -------------
.. *   **Flexible API with Explicit Modes**: Provides a high-level API that
..     supports single datasets, single dataset with use-supplied decomposition function and pre-computed decompositions.
.. *   **Powerful Decomposition Techniques**: Includes interfaces for methods like
..     Constrained Diffusion Decomposition (cdd) and Empirical Mode Decomposition
..     (EMD), and Wavelet Decomposition (Wavelet).
.. *   **Full UMAP Control**: Allows for complete control over the UMAP algorithm's parameters via convenience arguments and a flexible dictionary (`umap_params`).
.. *   **Support for Custom Functions**: Users can supply their own decomposition functions for maximum extensibility.
.. *   **Serialization of Models**: Trained UMAP models can be saved using `pickle` and reloaded for consistent inference on new data.

Installation
------------

The required Python packages must be installed prior to use. It is recommended to use a virtual environment.

.. code-block:: bash

    pip install numpy umap-learn scipy matplotlib constrained-diffusion

and install 

Decomposition-UMAP via pip:

.. code-block:: bash
    
    pip install decomposition-umap==0.1.0

or clone the repository and install it manually:

.. code-block:: bash

    git clone https://github.com/gxli/DecompositionUMAP.git
    cd DecompositionUMAP
    pip install .



Usage
-----

The following examples demonstrate the core workflows using a synthetic 256x256 dataset composed of a Gaussian anomaly embedded in a fractal noise background.
Usage
-----

The following examples demonstrate the core workflows using a synthetic 256x256 dataset composed of a Gaussian anomaly embedded in a fractal noise background.

1. Data Generation
~~~~~~~~~~~~~~~~~~

First, we generate the data. This function is assumed to be available in an `example` module within the library. After installing your package, you can import it as shown below.

.. code-block:: python

    import numpy as np
    # Import the library and the example data generator
    import decomposition_umap
    from decomposition_umap import example as du_example

    # Generate a dataset with a known anomaly
    data, signal, anomaly = du_example.generate_fractal_with_gaussian()

2. Running the Pipeline (Core Examples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Example A: Standard Mode (Built-in Decomposition)**

This is the most common use case for training a new model.

.. code-block:: python

    import pickle

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

    from decomposition_umap.multiscale_decomposition import cdd_decomposition

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


4. Command-Line Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This package includes a convenient command-line tool, `decomposition-umap`, for quick analysis of FITS or NPY files. After installing the package, you can run it directly from your terminal.

By default, the tool saves the output files in the same directory as the input file, prefixed with the input file's name. You can optionally specify a different output directory.

**Usage:**
.. code-block:: text

    usage: decomposition-umap [-h] [-o OUTPUT_DIR] [-d DECOMPOSITION_LEVEL] [-n {2,3}]
                              [-m {cdd,emd}] [-p UMAP_PARAMS] [--no-verbose]
                              input_file

**Examples:**

1. **Basic Analysis (Default Output Path)**: Process a FITS file with default settings. The output files (e.g., `my_image_decomposition.npy`) will be saved in the same directory as `my_image.fits`.
.. code-block:: bash

    decomposition-umap path/to/my_image.fits

2. **Specifying an Output Directory**: Process a file and save the results into a specific folder named `analysis_results`.
.. code-block:: bash

    decomposition-umap path/to/my_image.fits -o analysis_results/

3. **3D Embedding and Custom Decomposition**: Process a NumPy file, use exactly 8 decomposition components, and create a 3D UMAP embedding.
.. code-block:: bash

    decomposition-umap my_data.npy -o results/ -d 8 -n 3

4. **Advanced UMAP Control**: Use the `--umap_params` flag to pass a JSON string of advanced parameters, such as enabling UMAP's `low_memory` mode.
.. code-block:: bash

    decomposition-umap large_image.fits -o results/ -d 10 -p '{"n_neighbors":
    50, "low_memory": true}'
    



API Reference
-------------


**`decompose_and_embed(...)`**

The primary function for **training** a new Decomposition-UMAP model. It intelligently handles multiple input modes for maximum flexibility.

*   **Operating Modes (provide exactly one)**:

    *   `data` (`numpy.ndarray`): For a single raw dataset.

    *   `datasets` (`list`): For a batch of raw datasets.

    *   `data_multivariate` (`numpy.ndarray`): For a multi-channel raw dataset.

    *   `decomposition` (`numpy.ndarray`): For a single pre-computed decomposition.

*   **Key Parameters**:


      *   `decomposition_method` (`str`): The name of the built-in decomposition method (e.g., `'cdd'`, `'emd'`, `'wavelet'`). Ignored if `decomposition` is provided.

    *   `decomposition_max_n` (`int`): The number of components to generate for relevant decomposition methods.

    *   `decomposition_func` (`callable`): A user-provided decomposition function, which overrides `decomposition_method`. Ignored if `decomposition` is provided.

    *   `n_component` (`int`): The target dimension for the final UMAP embedding.

    *   `norm_func` (`callable`): A function to normalize feature vectors before UMAP (e.g., `max_norm`).

    *   `threshold` (`float`): A value below which data points are masked and excluded from analysis.

    *   `umap_n_neighbors` (`int`): Convenience argument for UMAP's `n_neighbors`.

    *   `low_memory` (`bool`): Convenience argument for UMAP's `low_memory` flag.

    *   `umap_params` (`dict`): For advanced control, a dictionary of arguments passed directly to the `umap.UMAP` constructor (e.g., `{'min_dist': 0.0, 'metric': 'cosine'}`).

*   **Returns**: A tuple whose contents depend on the operating mode. For single dataset modes, it returns `(embed_map, decomposition, umap_model)`.

**`decompose_with_existing_model(...)`**

The primary function for **inference**. It applies a pre-trained UMAP model to new data, ensuring a consistent transformation.

*   **Operating Modes (provide exactly one)**:

    *   `data` (`numpy.ndarray`): For a single raw dataset.

    *   `datasets` (`list`): For a batch of raw datasets.

    *   `data_multivariate` (`numpy.ndarray`): For a multi-channel raw dataset.

    *   `decomposition` (`numpy.ndarray`): For a single pre-computed decomposition.

*   **Key Parameters**:

    *   `model_filename` (`str`): Path to the pickled UMAP model file.

    *   `data` (`numpy.ndarray`): The new data array to transform.

    *   `decomposition_method` & `decomposition_max_n`: These decomposition parameters **must match** those used during model training to ensure a valid transformation.

    *   `norm_func` (`callable`): The normalization function, which **must be consistent** with the one used during training.

*   **Returns**: A tuple whose contents depend on the operating mode. For single dataset modes, it returns `(embed_map, final_decomposition)`.


**`DecompositionUMAP` class**

The core engine that encapsulates the workflow state. It offers granular control over the process and can be initialized with raw data or a pre-computed `decomposition`. When an instance is created, it immediately runs the full decomposition (if needed) and UMAP training pipeline. The resulting model and data are stored as attributes.

*   **Initialization Options**:

    The class is initialized in one of three ways:

    1.  **With Raw Data & Built-in Method**: Provide ``original_data`` and use ``decomposition_method`` to specify a built-in function.

        .. code-block:: python

            # Initialize by providing raw data and a method name
            instance = DecompositionUMAP(
                original_data=data,
                decomposition_method='cdd',
                decomposition_max_n=6,
                n_component=2
            )
            # instance.umap_model is now a trained model.

    2.  **With Raw Data & Custom Function**: Provide ``original_data`` and your own ``decomposition_func``.

        .. code-block:: python

            from scipy.ndimage import gaussian_filter

            def my_custom_decomposition(data):
                comp1 = gaussian_filter(data, sigma=3)
                comp2 = data - comp1
                return np.array([comp1, comp2])

            # Initialize with the custom function
            instance = DecompositionUMAP(
                original_data=data,
                decomposition_func=my_custom_decomposition,
                n_component=2
            )

    3.  **With a Pre-computed Decomposition**: Provide a ``decomposition`` array directly. This skips the decomposition step.

        .. code-block:: python

            # Initialize by providing a pre-computed decomposition
            precomputed, _ = cdd_decomposition(data, max_n=6)
            instance = DecompositionUMAP(
                decomposition=np.array(precomputed),
                n_component=2
            )

*   **Key Methods**:

    -   ``save_umap_model(filename)``: Saves the trained ``umap.UMAP`` model instance to a file using Python's `pickle` serialization. This allows for model persistence and later use in inference.

        .. code-block:: python

            # After training (e.g., from the first example above)
            instance.save_umap_model("my_trained_model.pkl")

    -   ``load_umap_model(filename)``: Loads a serialized ``umap.UMAP`` model from a specified file path, replacing the current instance's model. This is useful for specific workflows where you might want to swap models within an existing instance.

        .. code-block:: python

            # Create a minimal instance and load a model into it
            inference_instance = DecompositionUMAP(decomposition=np.zeros((1, 1, 1)))
            inference_instance.load_umap_model("my_trained_model.pkl")

    -   ``compute_new_embeddings(...)``: The core inference method that projects new data using the instance's existing (trained or loaded) UMAP model. It takes either ``new_original_data`` (which it will decompose first) or a ``new_decomposition``.

        .. code-block:: python

            # Use the trained instance from the first example to transform new data
            new_data, _, _ = du_example.generate_fractal_with_gaussian()
            new_embedding = instance.compute_new_embeddings(
                new_original_data=new_data
            )



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
**Email**: `ligx.ngc7293@gmail.com`
**GitHub**: `https://github.com/gxli`