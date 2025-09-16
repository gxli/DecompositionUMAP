"""
Decomposition UMAP Package
==========================

This package provides a streamlined workflow for applying dimensionality
reduction on complex data by first decomposing it into components and then
using UMAP for embedding.

Exposed Functions:
- decompose_and_embed: A high-level function to train a new model from raw data.
- decompose_with_existing_model: A high-level function to apply a pre-trained model.

Exposed Classes:
- DecompositionUMAP: The core class managing the decomposition and UMAP pipeline.
"""
# Updated import to reflect the file rename
from .decomposition_umap import DecompositionUMAP, decompose_and_embed, decompose_with_existing_model
from . import example

__all__ = [
    "DecompositionUMAP",
    "decompose_and_embed",
    "decompose_with_existing_model", "example"
]
