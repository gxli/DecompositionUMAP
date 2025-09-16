import numpy as np
import pickle
from scipy.signal import hilbert
# Handle UMAP import at module level
try:
    import umap
    assert umap.UMAP() is not None
except AttributeError:
    import umap.umap_ as umap
    assert umap.UMAP() is not None



from .embedding import umap_embedding
from .utils import max_norm
from .multiscale_decomposition import (cdd_decomposition, emd_decomposition,
                                       msm_decomposition, _auto_msm_filter_sizes,
                                       adaptive_multiscale_decomposition)
class DecompositionUMAP:
    """
    Manages a decomposition-then-UMAP dimensionality reduction workflow.

    This class is the core engine that orchestrates the entire process, from
    data decomposition to UMAP model training and inference. It holds the state
    of the pipeline, including the decomposition components, the trained model,
    and all configuration parameters.

    While this class can be used directly for fine-grained control, it is often
    more convenient to use the high-level wrapper functions `decompose_and_embed`
    for training and `decompose_with_existing_model` for inference.

    Args:
        original_data (numpy.ndarray, optional): The input raw data to be
            decomposed and embedded. Required if `decomposition` is not provided.
        decomposition (numpy.ndarray, optional): A pre-computed decomposition.
            Required if `original_data` is not provided. Must have shape
            (n_components, ...original_shape).
        decomposition_func (callable, optional): A function that takes `original_data`
            and returns its decomposition. Required if `original_data` is provided.
        umap_n_neighbors (int, optional): The `n_neighbors` parameter for UMAP,
            controlling the balance between local and global structure. Defaults to 30.
        umap_min_dist (float, optional): The `min_dist` parameter for UMAP,
            controlling how tightly points are clustered. Defaults to 0.0.
        n_component (int, optional): The number of dimensions for the UMAP
            embedding. Defaults to 2.
        threshold (float, optional): A value to mask low-signal regions in the
            `original_data` before analysis. Points below this value are
            excluded. Defaults to None.
        norm_func (callable, optional): A function to normalize feature vectors
            before passing them to UMAP. If `None`, no normalization is applied.
            Defaults to None.
        use_hilbert_amplitude (bool, optional): If True, computes the amplitude of
            the analytic signal for each decomposition component before UMAP.
            Defaults to False.
        train_fraction (float, optional): Fraction of data (0.0 to 1.0) to use
            for training the UMAP model. Defaults to None (use all data).
        train_mask (numpy.ndarray, optional): A boolean mask specifying the exact
            points to use for training. Overrides `train_fraction`. Defaults to None.
        verbose (bool, optional): If True, prints progress messages. Defaults to True.

    Attributes:
        decomposition (numpy.ndarray): The decomposition components of the input data.
        original_shape (tuple): The shape of the original input data.
        embed_map (list[np.ndarray]): The resulting list of UMAP embedding maps.
        umap_model (umap.UMAP): The trained UMAP reducer instance.
        decomposition_func (callable): The function used for decomposition.
        norm_func (callable): The function used for normalization.
        threshold (float): The threshold used for masking.
        use_hilbert_amplitude (bool): Flag indicating if Hilbert amplitude was used.
        verbose (bool): Verbosity flag.

    Raises:
        ValueError: If initialization parameters are provided incorrectly (e.g.,
            both or neither of `original_data` and `decomposition` are given, or
            if `decomposition_func` is missing when needed).
    """
    def __init__(self, original_data=None, decomposition=None, decomposition_func=None,
                 umap_n_neighbors=30, umap_min_dist=0.0, n_component=2, threshold=None,
                 norm_func=None, use_hilbert_amplitude=False, train_fraction=None,
                 train_mask=None, verbose=True):

        if not (original_data is None) ^ (decomposition is None):
            raise ValueError("Provide either 'original_data' (with 'decomposition_func') or a 'decomposition', but not both or neither.")

        if original_data is not None and not callable(decomposition_func):
            raise ValueError("If 'original_data' is provided, a callable 'decomposition_func' is required.")

        self.decomposition_func = decomposition_func
        self.norm_func = norm_func
        self.use_hilbert_amplitude = use_hilbert_amplitude
        self.threshold = threshold
        self.verbose = verbose
        effective_original_data = None

        if original_data is not None:
            if verbose: print("[Init] Decomposing original data...")
            decomp_result = self.decomposition_func(original_data)
            self.decomposition = np.array(decomp_result[0] if isinstance(decomp_result, tuple) else decomp_result)
            self.original_shape = original_data.shape
            effective_original_data = original_data
        else:
            if verbose: print("[Init] Using pre-computed decomposition.")
            self.decomposition = decomposition
            self.original_shape = decomposition.shape[1:]
            effective_original_data = np.empty(self.original_shape)

        if use_hilbert_amplitude:
            self.decomposition = self._apply_hilbert(self.decomposition)

        if verbose: print("[Init] Training UMAP model...")
        self.embed_map, self.umap_model = umap_embedding(
            self.decomposition, original_data=effective_original_data,
            umap_n_neighbors=umap_n_neighbors, umap_min_dist=umap_min_dist,
            n_component=n_component, threshold=threshold, norm_func=norm_func,
            train_fraction=train_fraction, train_mask=train_mask, verbose=verbose
        )

    def _apply_hilbert(self, decomposition_data):
        """
        Applies the Hilbert transform to each component and returns the amplitude.
        This is an internal helper method.

        Args:
            decomposition_data (numpy.ndarray): The data components.

        Returns:
            numpy.ndarray: The amplitude of the analytic signal for each component.

        Raises:
            ValueError: If the input data's shape has more than 3 dimensions.
        """
        if len(decomposition_data.shape) > 3:
            raise ValueError("Hilbert transform is only supported for 1D or 2D channel images.")
        if self.verbose: print("[Hilbert] Applying Hilbert transform and calculating amplitude.")
        return np.array([np.abs(hilbert(comp)) for comp in decomposition_data])

    def save_umap_model(self, filename):
        """
        Saves the trained UMAP model to a file using pickle.

        Args:
            filename (str): The path to the file where the model will be saved.

        Raises:
            ValueError: If the UMAP model has not been trained yet.
        """
        if self.umap_model is None: raise ValueError("UMAP model is not trained.")
        with open(filename, 'wb') as f: pickle.dump(self.umap_model, f)
        if self.verbose: print(f"[UMAP] Model saved to {filename}")

    def load_umap_model(self, filename):
        """
        Loads a UMAP model from a pickle file and sets it as the instance's model.

        Args:
            filename (str): The path to the file from which to load the model.

        Raises:
            ValueError: If the loaded object from the file is not a valid UMAP model.
            FileNotFoundError: If the filename does not exist.
        """
        with open(filename, 'rb') as f: loaded_model = pickle.load(f)
        if not isinstance(loaded_model, umap.UMAP):
            raise ValueError(f"File {filename} does not contain a valid UMAP model.")
        self.umap_model = loaded_model
        if self.verbose: print(f"[UMAP] Model loaded from {filename}")

    def compute_new_embeddings(self, new_original_data=None, new_decomposition=None):
        """
        Projects new data into the embedding space of the trained UMAP model.

        This is the primary inference method. It takes new data (either raw or
        pre-decomposed), applies the same preprocessing steps configured during
        initialization (e.g., Hilbert transform, normalization), and uses the
        trained model's `transform` method to generate the new embedding.

        Args:
            new_original_data (numpy.ndarray, optional): New raw data to decompose
                and project.
            new_decomposition (numpy.ndarray, optional): New pre-computed decomposition
                to project. Provide either this or `new_original_data`.

        Returns:
            list[np.ndarray]: A list of new embedding maps, one for each dimension,
            reshaped to match the new data's original shape.

        Raises:
            ValueError: If the UMAP model is not trained/loaded, if arguments
                are provided incorrectly, or if a decomposition function is
                unavailable when needed for `new_original_data`.
        """
        if self.umap_model is None: raise ValueError("UMAP model is not trained or loaded.")
        if not (new_original_data is None) ^ (new_decomposition is None):
            raise ValueError("Provide either 'new_original_data' or 'new_decomposition', but not both or neither.")

        final_decomposition, output_shape, effective_data = None, None, None

        if new_original_data is not None:
            if self.decomposition_func is None:
                raise ValueError("Cannot process 'new_original_data' without a 'decomposition_func'.")
            if self.verbose: print("[Inference] Decomposing new data...")
            decomp_result = self.decomposition_func(new_original_data)
            final_decomposition = np.array(decomp_result[0] if isinstance(decomp_result, tuple) else decomp_result)
            output_shape = new_original_data.shape
            effective_data = new_original_data
        else:
            if self.verbose: print("[Inference] Using new pre-computed decomposition.")
            final_decomposition = new_decomposition
            output_shape = new_decomposition.shape[1:]
            # Create a placeholder for thresholding compatibility
            effective_data = np.empty(output_shape)

        if self.use_hilbert_amplitude:
            final_decomposition = self._apply_hilbert(final_decomposition)

        if self.threshold is not None:
            if new_original_data is None:
                print("Warning: Threshold was set during training, but 'new_original_data' is not provided for inference. NaN masking will be skipped.")
            else:
                mask = np.ones(output_shape)
                mask[new_original_data < self.threshold] = np.nan
                final_decomposition = final_decomposition * mask

        # Reshape for UMAP and handle any NaN values from masking
        data_input = final_decomposition.reshape(final_decomposition.shape[0], -1)
        umap_data = data_input.T
        valid_indices = ~np.isnan(umap_data).any(axis=1)
        umap_data_filtered = umap_data[valid_indices]

        if self.norm_func is not None:
            umap_data_filtered = np.apply_along_axis(self.norm_func, 1, umap_data_filtered)

        if self.verbose: print(f"[Inference] Projecting {len(umap_data_filtered)} data points...")
        transformed_embedding = self.umap_model.transform(umap_data_filtered)

        # Reconstruct the full embedding map, inserting NaNs where data was invalid
        full_embedding = np.full((len(umap_data), self.umap_model.n_components), np.nan)
        full_embedding[valid_indices] = transformed_embedding

        return [full_embedding[:, i].reshape(output_shape) for i in range(full_embedding.shape[1])]



def decompose_and_embed(data, decomposition_method='cdd', decomposition_func=None,
                        norm_func=None, umap_n_neighbors=30, umap_min_dist=0.0,
                        n_component=2, threshold=None, use_hilbert_amplitude=False,
                        train_fraction=None, train_mask=None, verbose=True,
                        decomposition_max_n=None, msm_filter_sizes='auto'):
    """
    Performs decomposition and UMAP embedding on input data in a single step.

    This high-level wrapper serves as the primary entry point for training a new
    Decomposition-UMAP model. It simplifies the workflow by handling the
    creation of a decomposition function based on the selected method,
    instantiating the main `DecompositionUMAP` class, and returning the
    key results.

    Args:
        data (numpy.ndarray): The input raw data to be decomposed and embedded.
        decomposition_method (str, optional): The name of a built-in decomposition
            method to use. Supported options include: 'cdd', 'emd', 'amd', 'msm'.
            This is ignored if a `decomposition_func` is provided.
            Defaults to 'cdd'.
        decomposition_func (callable, optional): A custom function that takes `data`
            as input and returns its decomposition. If provided, it overrides the
            `decomposition_method` argument. Defaults to None.
        norm_func (callable, optional): A function to normalize each feature vector
            (a single point's set of decomposition components) before UMAP
            processing. If `None`, no normalization is performed.
            Defaults to None.
        umap_n_neighbors (int, optional): The `n_neighbors` parameter for the UMAP
            algorithm, controlling the balance between local and global structure
            in the final embedding. Defaults to 30.
        umap_min_dist (float, optional): The `min_dist` parameter for the UMAP
            algorithm, controlling how tightly packed points are in the embedding.
            Defaults to 0.0.
        n_component (int, optional): The number of dimensions for the output UMAP
            embedding (e.g., 2 for a 2D plot). Defaults to 2.
        threshold (float, optional): A value below which points in the original `data`
            are considered low-signal. These points and their corresponding
            decomposition vectors are masked and excluded from UMAP training
            and transformation. Defaults to None.
        use_hilbert_amplitude (bool, optional): If True, the Hilbert transform is
            applied to each decomposition component to compute its instantaneous
            amplitude before passing it to UMAP. This is useful for analyzing
            oscillatory data. Defaults to False.
        train_fraction (float, optional): A fraction (between 0.0 and 1.0) of the
            valid data points to use for training the UMAP model. The entire
            dataset is then transformed using the trained model. This is highly
            recommended for very large datasets to speed up training.
            Defaults to None, which uses all valid data for training.
        train_mask (numpy.ndarray, optional): A boolean array with the same shape as
            `data`, explicitly specifying which points to use for training the
            UMAP model. This provides finer control than `train_fraction` and
            overrides it if both are provided. Defaults to None.
        verbose (bool, optional): If True, prints progress messages to the console
            during decomposition and embedding. Defaults to True.
        decomposition_max_n (int, optional): A generic parameter to control the
            maximum number of components for decomposition methods that support it,
            such as 'cdd', 'amd', and 'emd'. For 'emd', setting this to `None`
            typically results in computing all Intrinsic Mode Functions (IMFs).
            Defaults to None.
        msm_filter_sizes (list or str, optional): Specific parameter for the 'msm'
            decomposition method. Can be a list of integer filter sizes (e.g.,
            [3, 7, 15]) or the string 'auto' to automatically determine them.
            Defaults to 'auto'.

    Returns:
        tuple[list[np.ndarray], np.ndarray, umap.UMAP]:
        - embed_map (list[np.ndarray]): A list of numpy arrays, where each array
          represents one dimension of the UMAP embedding, reshaped to match the
          original `data` shape. Masked points are filled with NaN.
        - decomposition (np.ndarray): The computed decomposition components of the
          input data, with shape (n_components, ...original_shape).
        - umap_model (umap.UMAP): The fully trained UMAP reducer object. This object
          can be saved and used later to project new data.

    Raises:
        ValueError: If an unknown `decomposition_method` string is provided and
            `decomposition_func` is None.
    """
    if decomposition_func is None:
        def func(d):
            if decomposition_method == 'cdd':
                return cdd_decomposition(d, max_n=decomposition_max_n)
            elif decomposition_method == 'emd':
                # Translate the generic parameter for the EMD function's specific API
                max_imf = decomposition_max_n if decomposition_max_n is not None else -1
                return emd_decomposition(d, max_imf=max_imf)
            elif decomposition_method == 'amd':
                return adaptive_multiscale_decomposition(d, max_n=decomposition_max_n)
            elif decomposition_method == 'msm':
                sizes = msm_filter_sizes
                if sizes == 'auto':
                    # This function would need to be imported
                    sizes = _auto_msm_filter_sizes(d.shape)
                return msm_decomposition(d, *sizes)
            else:
                raise ValueError(f"Unknown decomposition_method: {decomposition_method}")
        decomposition_func = func

    # This assumes the DecompositionUMAP class is defined elsewhere and imported
    instance = DecompositionUMAP(
        original_data=data,
        decomposition_func=decomposition_func,
        norm_func=norm_func,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        n_component=n_component,
        threshold=threshold,
        use_hilbert_amplitude=use_hilbert_amplitude,
        train_fraction=train_fraction,
        train_mask=train_mask,
        verbose=verbose
    )
    return instance.embed_map, instance.decomposition, instance.umap_model


def decompose_with_existing_model(model_filename, data=None, decomposition=None,
                                  decomposition_method='cdd', decomposition_func=None,
                                  norm_func=None, decomposition_max_n=None,
                                  msm_filter_sizes='auto'):
    """
    Applies a pre-trained UMAP model to project new data into an existing embedding.

    This high-level wrapper is the primary entry point for inference. It loads a
    previously trained and saved UMAP model, processes new data using the same
    decomposition method, and then transforms the new data into the learned
    embedding space. This ensures that new data is mapped consistently relative
    to the data used for the original training.

    Args:
        model_filename (str): The path to the saved (pickled) UMAP model file
            that was generated by `decompose_and_embed`.
        data (numpy.ndarray, optional): The new raw input data to be decomposed
            and transformed. You must provide either `data` or `decomposition`.
            Defaults to None.
        decomposition (numpy.ndarray, optional): A pre-computed decomposition of
            the new data. If provided, the decomposition step is skipped. You
            must provide either `data` or `decomposition`. Defaults to None.
        decomposition_method (str, optional): The name of the built-in decomposition
            method ('cdd', 'emd', etc.) to use if `data` is provided. This should
            match the method used when the model was trained. Ignored if
            `decomposition_func` is given. Defaults to 'cdd'.
        decomposition_func (callable, optional): A custom function to decompose the
            new `data`. Overrides `decomposition_method`. This should be the same
            function used during model training. Defaults to None.
        norm_func (callable, optional): The function used to normalize feature
            vectors before transformation. It is crucial that this is the same
            function used to train the original model. If `None`, no
            normalization is performed. Defaults to None.
        decomposition_max_n (int, optional): The generic parameter that controls
            the maximum number of components for the decomposition method. This
            value should be consistent with the one used during training.
            Defaults to None.
        msm_filter_sizes (list or str, optional): The specific parameter for the
            'msm' decomposition method, which should be consistent with the
            value used during training. Defaults to 'auto'.

    Returns:
        tuple[list[np.ndarray], np.ndarray]:
        - embed_map (list[np.ndarray]): A list of numpy arrays, where each array
          is a dimension of the new UMAP embedding, reshaped to match the input
          data's shape.
        - final_decomposition (np.ndarray): The decomposition of the new input
          data, which was used for the transformation.

    Raises:
        ValueError: If both or neither of `data` and `decomposition` are provided,
            if the loaded file is not a valid UMAP model, or if a decomposition
            function is not available when `data` is provided.
        FileNotFoundError: If `model_filename` does not point to a valid file.
    """
    # A dummy instance is created to gain access to the class methods
    temp_instance = DecompositionUMAP(decomposition=np.zeros((1, 1)), verbose=False)
    temp_instance.load_umap_model(model_filename)
    
    # Set the normalization function for the inference process
    temp_instance.norm_func = norm_func
    
    # Create the decomposition function for the new data if not provided
    final_decomposition_func = decomposition_func
    if final_decomposition_func is None and data is not None:
        def func(d):
            if decomposition_method == 'cdd':
                return cdd_decomposition(d, max_n=decomposition_max_n)
            elif decomposition_method == 'emd':
                max_imf = decomposition_max_n if decomposition_max_n is not None else -1
                return emd_decomposition(d, max_imf=max_imf)
            # Add other methods as needed for consistency
            else:
                raise ValueError(f"Unknown decomposition_method: {decomposition_method}")
        temp_instance.decomposition_func = func
    else:
        temp_instance.decomposition_func = final_decomposition_func

    # Use the instance's inference method to get the new embedding
    embed_map = temp_instance.compute_new_embeddings(
        new_original_data=data, new_decomposition=decomposition
    )
    
    # The decomposition must be explicitly computed and returned if it wasn't provided
    if decomposition is None:
        if temp_instance.decomposition_func is None:
            raise ValueError("A decomposition function must be provided to process new raw data.")
        decomp_result = temp_instance.decomposition_func(data)
        final_decomposition = np.array(decomp_result[0] if isinstance(decomp_result, tuple) else decomp_result)
    else:
        final_decomposition = decomposition

    return embed_map, final_decomposition