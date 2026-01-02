# Replicate Model Settings

## Name
deepseek-ocr

## Description
High-quality OCR with 30+ optimized modes for documents, tables, handwriting, receipts, exams, and more.

## GitHub URL
https://github.com/abdussamadbello/deepseek-ocr-cog

## Weights URL
https://huggingface.co/deepseek-ai/DeepSeek-OCR

## Paper URL
(leave blank or add if available)

## License URL
https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/LICENSE

## Hardware
GPU (A40 recommended for production)

## Visibility
Public

---

## Readme (copy below for the Replicate description field)

DeepSeek-OCR is a powerful vision-language model optimized for optical character recognition tasks.

### Features
- **30+ optimized modes** for different OCR tasks
- **Grounding support** - get bounding boxes for detected elements
- **Image cropping** - extract detected figures/charts as separate images
- **Multiple output formats** - JSON, markdown, text, or boxes only

### Quick Start
```python
import replicate

output = replicate.run(
    "abdussamadbello/deepseek-ocr",
    input={
        "image": open("document.png", "rb"),
        "mode": "document_markdown"
    }
)
```

### Popular Modes
| Mode | Use Case |
|------|----------|
| `document_markdown` | General documents with layout |
| `table_markdown` | Table extraction |
| `handwriting` | Handwritten notes |
| `receipt` | Receipt/invoice data |
| `exam_question` | Exam digitization |
| `code` | Source code from screenshots |
| `math_formula` | LaTeX extraction |

### Model
Based on [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) from DeepSeek AI.
