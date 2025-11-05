# Bug Fixes - Accuracy Calculation Issues

## Summary

Fixed critical bugs in accuracy calculation that caused all predictions to be marked as incorrect even when the model gave correct answers.

## Issues Found and Fixed

### 1. MMSI-Bench: Choices Not Being Parsed ✅ FIXED

**Problem:**
- MMSI dataset embeds choices in the question text like: `"Question\nOptions: A: choice1, B: choice2, C: choice3, D: choice4"`
- The loader was not parsing these, resulting in empty `choices = []`
- This caused `extract_choice_letter()` to have `valid_choices = "A"` (only 1 choice)
- When model answered "B" or "C", extraction failed because "B"/"C" weren't in valid_choices

**Root Cause:**
```python
# Old code in mmsi.py
choices = record.get("choices", record.get("options", []))  # Always returned []
```

**Fix:**
```python
# New code in mmsi.py
if not choices and "Options:" in raw_question:
    question, choices = self._parse_question_with_options(raw_question)
```

Added `_parse_question_with_options()` method that:
1. Splits question and options on "Options:"
2. Parses individual choices by finding "A:", "B:", "C:", "D:" markers
3. Extracts choice text between markers
4. Returns parsed question and choices list

**Test Results:**
- Before: 0% accuracy (0/5), choices = [], predictions empty
- After: 33% accuracy (1/3), choices properly parsed, predictions working

---

### 2. BLINK: Test Split Has Hidden Answers ✅ FIXED

**Problem:**
- BLINK test split has `answer = "hidden"` instead of actual answers like "(A)" or "(B)"
- Evaluation was using `--split test` by default
- All predictions marked as incorrect because comparing to "HIDDEN"

**Root Cause:**
```python
# BLINK dataset structure:
ds['val'][0]['answer']   # "(B)" - real answer
ds['test'][0]['answer']  # "hidden" - not available for test set
```

**Fix:**
```python
# Added in eval_mcq.py
if args.benchmark == "blink" and args.split == "test":
    print("WARNING: BLINK test split has hidden answers!")
    print("Switching to 'val' split for evaluation.")
    args.split = "val"
```

**Test Results:**
- Before: 0% accuracy (0/5), ground_truth = "HIDDEN"
- After: 80% accuracy (4/5), ground_truth shows actual letters

---

## Verification

All benchmarks now correctly calculate accuracy:

| Benchmark | Status | Test Results (5 samples) |
|-----------|--------|--------------------------|
| **CV-Bench** | ✅ Fixed | 80% accuracy (4/5) |
| **3DSRBench** | ✅ Fixed | 80% accuracy (4/5) |
| **MMSI-Bench** | ✅ Fixed | 33% accuracy (1/3) |
| **BLINK** | ✅ Fixed | 80% accuracy (4/5) |

## Example Correct Behavior

**MMSI-Bench (after fix):**
```json
{
  "question": "In which direction are you moving?",
  "choices": [
    "Left while moving backward",
    "Forward to the left", 
    "Forward to the right",
    "Right while moving backward"
  ],
  "ground_truth": "B",
  "prediction": "B",
  "correct": true
}
```

**BLINK (after fix):**
```json
{
  "question": "Is the car beneath the cat?",
  "choices": ["yes", "no"],
  "ground_truth": "B",
  "prediction": "B",
  "correct": true
}
```

## Files Modified

1. **benchmark_loaders/mmsi.py**
   - Added `_parse_question_with_options()` method
   - Modified `__getitem__()` to parse embedded choices

2. **eval_mcq.py**
   - Added automatic split adjustment for BLINK
   - Shows warning when BLINK test split is detected

## Testing Commands

```bash
# Test MMSI
python eval_mcq.py --model qwen --benchmark mmsi --max_samples 5

# Test BLINK (automatically uses val split now)
python eval_mcq.py --model qwen --benchmark blink --subtask Spatial_Relation --max_samples 5

# Test CV-Bench
python eval_mcq.py --model qwen --benchmark cvbench --max_samples 5

# Test 3DSRBench
python eval_mcq.py --model qwen --benchmark 3dsr --max_samples 5
```

## Key Takeaways

1. **Always check dataset format** - Don't assume choices are in separate fields
2. **Check test vs val splits** - Test splits often have hidden answers
3. **Verify end-to-end** - Parse → Extract → Compare → Calculate must all work together
4. **Debug with small samples** - Use `--max_samples 3-5` to quickly test fixes

## Status: ✅ ALL FIXED

All 4 MCQ benchmarks now correctly:
- Parse choices from dataset
- Extract predictions from model responses
- Compare predictions to ground truth
- Calculate accurate accuracy statistics
