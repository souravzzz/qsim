"""
Configuration and environment setup for the quantum simulator.
"""

import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import GPU acceleration libraries if available
try:
    import cupy as cp

    HAS_GPU = True
    logger.info("CUDA GPU acceleration is available.")
except ImportError:
    HAS_GPU = False
    logger.warning("GPU acceleration libraries not found. Running in CPU-only mode.")

# Try to import TensorNetwork library if available
try:
    import tensornetwork as tn

    HAS_TENSOR_NETWORK = True
    logger.info("TensorNetwork library is available for advanced tensor contractions.")
except ImportError:
    HAS_TENSOR_NETWORK = False
    logger.warning(
        "TensorNetwork library not found. Tensor Network simulations will fall back to other methods."
    )
