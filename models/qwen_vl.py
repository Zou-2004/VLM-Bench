"""Qwen2.5-VL model wrapper.

Supports single/multi-image and video inputs.
Reference: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
"""

import os
from typing import List
import torch
import numpy as np
from PIL import Image

from .base_vlm import BaseVLM

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import decord
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


class QwenVL(BaseVLM):
    """Qwen2.5-VL model wrapper with video and multi-image support."""
    
    def __init__(
        self, 
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        dtype: str = "bfloat16",
        load_in_8bit: bool = False,
        **kwargs
    ):
        """Initialize Qwen2.5-VL model.
        
        Args:
            model_id: Hugging Face model ID
            device: Device to load model on ("cuda", "cpu", or "auto")
            dtype: Data type ("bfloat16", "float16", "float32")
            load_in_8bit: Whether to load model in 8-bit quantization
            **kwargs: Additional arguments passed to from_pretrained
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Required dependencies not installed. Run:\n"
                "pip install transformers accelerate decord"
            )
        
        super().__init__(model_id, **kwargs)
        
        # Set device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Set dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        
        print(f"Loading Qwen2.5-VL model: {model_id}")
        print(f"Device: {device}, Dtype: {dtype}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        
        # Load model
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "cuda:0" if (device == "cuda" or device == "auto") and not load_in_8bit else ("auto" if device != "cpu" else None),
            "trust_remote_code": True,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        
        model_kwargs.update(kwargs)
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs
        )
        
        if device == "cpu" and not load_in_8bit:
            self.model = self.model.to(device)
        
        print("Model loaded successfully!")
    
    def ask_images(
        self, 
        images: List[Image.Image], 
        prompt: str, 
        max_new_tokens: int = 64
    ) -> str:
        """Generate response for image(s) + prompt.
        
        Args:
            images: List of PIL Images
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Build message content with images + text
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # Prepare inputs using processor
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process both images and text together
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for evaluation
            )
        
        # Decode (skip special tokens)
        generated_text = self.processor.batch_decode(
            output_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Extract only the assistant's response
        # The output includes the full conversation, extract after "assistant\n"
        if "assistant\n" in generated_text:
            generated_text = generated_text.split("assistant\n")[-1]
        elif "\nassistant\n" in generated_text:
            generated_text = generated_text.split("\nassistant\n")[-1]
        
        return generated_text.strip()
    
    def ask_video(
        self, 
        video_path: str, 
        prompt: str, 
        fps: float = 2.0,
        max_frames: int = 32,
        max_new_tokens: int = 64
    ) -> str:
        """Generate response for video + prompt.
        
        Args:
            video_path: Path to video file
            prompt: Text prompt
            fps: Frames per second (for context, actual sampling uses max_frames)
            max_frames: Maximum frames to sample from video
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Load video and sample frames
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        
        # Sample frames uniformly
        num_frames = min(max_frames, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames).astype(int)
        
        # Convert frames to PIL Images
        frames = [
            Image.fromarray(vr[idx].asnumpy()) 
            for idx in frame_indices
        ]
        
        # Prepend video context to prompt
        video_prompt = f"(Video with {num_frames} frames sampled at ~{fps} fps.) {prompt}"
        
        # Use ask_images with sampled frames
        return self.ask_images(frames, video_prompt, max_new_tokens)
