"""Generic evaluation metrics for VLM benchmarks."""

import re
from typing import List, Dict, Any
import string


class Metrics:
    """Collection of evaluation metrics."""
    
    @staticmethod
    def extract_choice_letter(text: str, valid_choices: str = "ABCDEFGH") -> str:
        """Extract choice letter from model output.
        
        Args:
            text: Model's text output
            valid_choices: Valid choice letters
            
        Returns:
            Extracted letter or empty string if not found
        """
        if not valid_choices:
            return ""
        
        text = text.upper().strip()
        
        # Escape special regex characters in valid_choices
        escaped_choices = re.escape(valid_choices)
        
        # Strategy 1: Look for standalone letter
        pattern = r'\b([' + escaped_choices + r'])\b'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        
        # Strategy 2: Look for "Answer: X" or "The answer is X"
        pattern = r'(?:answer|choice|option)(?:\s+is)?[:\s]+([' + escaped_choices + r'])'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Strategy 3: First letter that matches valid choices
        for char in text:
            if char.upper() in valid_choices:
                return char.upper()
        
        return ""
    
    @staticmethod
    def mcq_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
        """Calculate multiple-choice question accuracy.
        
        Args:
            predictions: List of predicted choice letters
            ground_truths: List of ground truth choice letters
            
        Returns:
            Accuracy (0.0 to 1.0)
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        if len(predictions) == 0:
            return 0.0
        
        correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        return correct / len(predictions)
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for matching (lowercase, remove punctuation, extra spaces)."""
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    @staticmethod
    def exact_match(prediction: str, ground_truths: List[str]) -> bool:
        """Check if prediction exactly matches any ground truth (after normalization).
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers
            
        Returns:
            True if prediction matches any ground truth
        """
        pred_norm = Metrics.normalize_text(prediction)
        
        for gt in ground_truths:
            if pred_norm == Metrics.normalize_text(gt):
                return True
        
        return False
    
    @staticmethod
    def contains_match(prediction: str, ground_truths: List[str]) -> bool:
        """Check if prediction contains any ground truth (after normalization).
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers
            
        Returns:
            True if prediction contains any ground truth
        """
        pred_norm = Metrics.normalize_text(prediction)
        
        for gt in ground_truths:
            gt_norm = Metrics.normalize_text(gt)
            if gt_norm in pred_norm:
                return True
        
        return False


def format_mcq_prompt(
    question: str, 
    choices: List[str],
    instruction: str = "Answer with ONLY the letter of the correct choice."
) -> str:
    """Format a multiple-choice question prompt.
    
    Args:
        question: The question text
        choices: List of choice strings
        instruction: Instruction for how to answer
        
    Returns:
        Formatted prompt string
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    prompt = f"{question}\n\n"
    
    for i, choice in enumerate(choices):
        if i < len(letters):
            prompt += f"({letters[i]}) {choice}\n"
    
    prompt += f"\n{instruction}\nAnswer:"
    
    return prompt


def answer_to_letter(answer: Any, choices: List[str]) -> str:
    """Convert answer (various formats) to choice letter.
    
    Args:
        answer: Answer in various formats (letter, index, text)
        choices: List of choices
        
    Returns:
        Choice letter (A, B, C, ...)
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Case 0: Answer with parentheses like "(C)" - strip them
    if isinstance(answer, str):
        answer_stripped = answer.strip().strip("()")
        if len(answer_stripped) == 1 and answer_stripped.upper() in letters:
            return answer_stripped.upper()
    
    # Case 1: Already a letter
    if isinstance(answer, str) and len(answer) == 1 and answer.upper() in letters:
        return answer.upper()
    
    # Case 2: Integer index
    if isinstance(answer, int):
        if 0 <= answer < len(choices):
            return letters[answer]
    
    # Case 3: String index
    if isinstance(answer, str) and answer.isdigit():
        idx = int(answer)
        if 0 <= idx < len(choices):
            return letters[idx]
    
    # Case 4: Answer text matches a choice
    if isinstance(answer, str):
        answer_norm = Metrics.normalize_text(answer)
        for i, choice in enumerate(choices):
            if answer_norm == Metrics.normalize_text(choice):
                return letters[i]
    
    # Default: return as-is or empty
    return str(answer).upper() if answer else ""
