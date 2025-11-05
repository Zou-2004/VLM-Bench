"""Model wrappers for VLMs."""

from .base_vlm import BaseVLM
from .qwen_vl import QwenVL

# Optional models - import only if needed to avoid dependency issues
try:
    from .llava_model import LLaVAModel
    __all__ = ["BaseVLM", "QwenVL", "LLaVAModel"]
except ImportError:
    __all__ = ["BaseVLM", "QwenVL"]

try:
    from .internvl2_model import InternVL2Model
    __all__.append("InternVL2Model")
except ImportError:
    pass

try:
    from .minicpm_v_model import MiniCPMVModel
    __all__.append("MiniCPMVModel")
except ImportError:
    pass
