"""Base abstract class for Vision-Language Models.

This allows easy extension to other VLMs beyond Qwen.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
from PIL import Image


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models."""
    
    def __init__(self, model_id: str, **kwargs):
        """Initialize the VLM.
        
        Args:
            model_id: Model identifier (e.g., Hugging Face model ID)
            **kwargs: Additional model-specific parameters
        """
        self.model_id = model_id
        self.kwargs = kwargs
    
    @abstractmethod
    def ask_images(
        self, 
        images: List[Image.Image], 
        prompt: str, 
        max_new_tokens: int = 64
    ) -> str:
        """Generate a response given image(s) and a text prompt.
        
        Args:
            images: List of PIL Images
            prompt: Text prompt/question
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def ask_video(
        self, 
        video_path: str, 
        prompt: str, 
        fps: float = 2.0,
        max_frames: int = 32,
        max_new_tokens: int = 64
    ) -> str:
        """Generate a response given a video and a text prompt.
        
        Args:
            video_path: Path to video file
            prompt: Text prompt/question
            fps: Target frames per second for sampling
            max_frames: Maximum number of frames to sample
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        pass
    
    def batch_ask_images(
        self,
        batch: List[tuple],  # List of (images, prompt) tuples
        max_new_tokens: int = 64
    ) -> List[str]:
        """Generate responses for a batch of image queries.
        
        Default implementation processes sequentially. Override for parallel processing.
        
        Args:
            batch: List of (images, prompt) tuples
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            List of generated text responses
        """
        return [
            self.ask_images(images, prompt, max_new_tokens) 
            for images, prompt in batch
        ]
    
    @property
    def supports_video(self) -> bool:
        """Whether the model supports video input."""
        return True
    
    @property
    def supports_multi_image(self) -> bool:
        """Whether the model supports multiple images in one query."""
        return True
