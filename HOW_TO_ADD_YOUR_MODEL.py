"""
Example: How to integrate YOUR custom VLM model (like VLM3R) into this framework.

This shows you EXACTLY how to add your own trained model for evaluation.
"""

from typing import List
import torch
from PIL import Image
import numpy as np

# Import the base class
from models.base_vlm import BaseVLM


class YourCustomVLM(BaseVLM):
    """
    Template for YOUR custom VLM model.
    
    Replace this with your actual model loading and inference code.
    The framework only needs these 2 methods implemented:
    1. ask_images() - for single/multi-image input
    2. ask_video() - for video input (optional)
    """
    
    def __init__(
        self,
        model_id: str = "path/to/your/model/checkpoint.pth",
        device: str = "cuda",
        **kwargs
    ):
        """Initialize YOUR model.
        
        Args:
            model_id: Path to your model checkpoint or config
            device: Device to run on
            **kwargs: Any other parameters your model needs
        """
        super().__init__(model_id, **kwargs)
        
        self.device = device
        
        print(f"Loading YOUR custom VLM from: {model_id}")
        
        # ==================================================================
        # REPLACE THIS SECTION WITH YOUR ACTUAL MODEL LOADING CODE
        # ==================================================================
        
        # Example: Load your model
        # self.model = YourModelClass.from_checkpoint(model_id)
        # self.model = self.model.to(device)
        # self.model.eval()
        
        # Example: Load your tokenizer/processor
        # self.tokenizer = YourTokenizer.from_pretrained(model_id)
        # self.image_processor = YourImageProcessor()
        
        # For demonstration:
        self.model = None  # Replace with your actual model
        self.tokenizer = None  # Replace with your tokenizer
        
        # ==================================================================
        
        print("✓ Model loaded successfully!")
    
    def ask_images(
        self,
        images: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 64
    ) -> str:
        """Generate response for image(s) + text prompt.
        
        THIS IS THE ONLY METHOD YOU NEED TO IMPLEMENT!
        
        Args:
            images: List of PIL Images (can be 1 or multiple)
            prompt: Text question/prompt
            max_new_tokens: Max tokens to generate
            
        Returns:
            Generated text response (just the answer)
        """
        
        # ==================================================================
        # REPLACE THIS WITH YOUR ACTUAL INFERENCE CODE
        # ==================================================================
        
        # Example workflow (adapt to your model):
        
        # 1. Preprocess images
        # image_tensors = [self.image_processor(img) for img in images]
        # image_tensors = torch.stack(image_tensors).to(self.device)
        
        # 2. Tokenize text
        # text_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 3. Forward pass through your model
        # with torch.no_grad():
        #     outputs = self.model(
        #         image=image_tensors,
        #         text=text_tokens,
        #         max_length=max_new_tokens
        #     )
        
        # 4. Decode output
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 5. Return clean response
        # return response.strip()
        
        # ==================================================================
        
        # Placeholder for demonstration:
        return f"[Your model's response to: {prompt[:50]}...]"
    
    def ask_video(
        self,
        video_path: str,
        prompt: str,
        fps: float = 2.0,
        max_frames: int = 32,
        max_new_tokens: int = 64
    ) -> str:
        """Generate response for video + text prompt.
        
        OPTIONAL: Only implement if your model handles video.
        
        Default implementation: sample frames and call ask_images()
        """
        
        # If your model doesn't natively support video, use frame sampling:
        import decord
        
        # Load and sample frames
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        num_frames = min(max_frames, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames).astype(int)
        
        # Convert to PIL Images
        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in frame_indices]
        
        # Add context to prompt
        video_prompt = f"Video with {num_frames} frames. {prompt}"
        
        # Use your ask_images method
        return self.ask_images(frames, video_prompt, max_new_tokens)
    
    @property
    def supports_video(self) -> bool:
        """Does your model support video?"""
        return True  # Change to False if video not supported
    
    @property
    def supports_multi_image(self) -> bool:
        """Does your model support multiple images?"""
        return True  # Change to False if only single image


# ==============================================================================
# USAGE: How to run evaluation with YOUR model
# ==============================================================================

if __name__ == "__main__":
    """
    Once you implement the above, you can evaluate like this:
    """
    
    # Option 1: Python API
    print("\n" + "="*70)
    print("Option 1: Use Python API directly")
    print("="*70)
    
    from datasets.cvbench import CVBenchDataset
    from metrics import format_mcq_prompt, Metrics, answer_to_letter
    
    # Load YOUR model
    model = YourCustomVLM(model_id="path/to/your/checkpoint.pth")
    
    # Load dataset
    dataset = CVBenchDataset(split="test")
    
    # Evaluate one sample
    sample = dataset[0]
    prompt = format_mcq_prompt(sample["question"], sample["choices"])
    response = model.ask_images(sample["images"], prompt)
    
    pred_letter = Metrics.extract_choice_letter(response)
    gt_letter = answer_to_letter(sample["answer"], sample["choices"])
    
    print(f"Question: {sample['question'][:50]}...")
    print(f"Prediction: {pred_letter}")
    print(f"Ground Truth: {gt_letter}")
    print(f"Correct: {pred_letter == gt_letter}")
    
    # Option 2: Command Line (after adding to eval_mcq.py)
    print("\n" + "="*70)
    print("Option 2: Command line evaluation")
    print("="*70)
    print("""
# First, add your model to eval_mcq.py:
# 1. Import: from models.your_model import YourCustomVLM
# 2. Add to choices: choices=["qwen", "llava", "your-model"]
# 3. Add loading logic:
#    elif args.model == "your-model":
#        model = YourCustomVLM(model_id=args.model_id)

# Then run:
python eval_mcq.py \\
    --model your-model \\
    --model_id path/to/your/checkpoint.pth \\
    --benchmark cvbench \\
    --output results/your_model_cvbench.json

# Evaluate on all benchmarks:
bash run_all.sh  # (after updating the script)
    """)
    
    print("\n" + "="*70)
    print("That's it! The framework handles everything else:")
    print("="*70)
    print("✓ Dataset loading (all 5 benchmarks)")
    print("✓ Prompt formatting")
    print("✓ Answer extraction")
    print("✓ Metric computation")
    print("✓ Result saving (JSON/CSV)")
    print("✓ Table formatting")
    print("\nYou only need to implement ask_images()!")
