"""Dataset loaders for the three datasets used in the paper.

Datasets are not redistributed. See docs/datasets.md for download instructions.
Each loader returns framed signals with class labels for the cross-validation
pipeline.
"""
from msif_isac.data import fmcw, dronerf, xiangyu

__all__ = ["fmcw", "dronerf", "xiangyu"]
