# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Cog-based predictor for DeepSeek-OCR, designed for deployment on Replicate. Provides OCR capabilities with 30+ optimized prompt modes for document processing, table extraction, chart analysis, handwriting recognition, and more.

## Development Commands

```bash
# Install Cog
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog

# Run prediction locally (requires GPU)
cog predict -i image=@test.png -i mode=document_markdown

# Deploy to Replicate
cog login
cog push r8.im/your-username/deepseek-ocr

# List available OCR modes
python predict.py
```

## Architecture

The codebase consists of two files:

**[predict.py](predict.py)** - Main predictor containing:
- `Predictor` class (extends `cog.BasePredictor`): Handles model loading in `setup()` and inference in `predict()`
- `PROMPT_LIBRARY` (line 39): Dict of 30+ OCR modes, each with optimized prompt, grounding flag, and recommended resolution
- `RESOLUTION_MODES` (line 260): Size configurations (tiny/small/base/large/gundam)
- `_parse_grounding()` (line 493): Extracts bounding boxes from model output using `<|ref|>...<|/ref|><|det|>...<|/det|>` format
- `_crop_regions()` (line 569): Crops detected regions and returns as base64

**[cog.yaml](cog.yaml)** - Cog configuration:
- GPU-enabled with CUDA 11.8, Python 3.12
- Key dependencies: torch, transformers, flash-attn
- Entry point: `predict.py:Predictor`

## Key Concepts

- **Grounding**: When enabled (via `<|grounding|>` token in prompt), the model returns bounding box coordinates for detected elements
- **Resolution modes**: Control quality/speed tradeoff; "gundam" mode uses dynamic tiling for best quality
- **Custom prompts**: Must start with `<image>\n`, optionally include `<|grounding|>` for bounding boxes

## Model

Uses `deepseek-ai/DeepSeek-OCR` from HuggingFace with flash attention. Model is loaded to GPU in bfloat16 precision.