# DeepSeek-OCR on Replicate

High-quality OCR with 30+ optimized prompts and full customization support.

## Quick Start

```python
import replicate

# Basic document OCR
output = replicate.run(
    "your-username/deepseek-ocr",
    input={"image": open("document.png", "rb"), "mode": "document_markdown"}
)

# Custom prompt
output = replicate.run(
    "your-username/deepseek-ocr",
    input={
        "image": open("exam.png", "rb"),
        "mode": "custom",
        "custom_prompt": "<image>\n<|grounding|>Extract all questions and describe any figures in detail."
    }
)
```

---

## Available Modes

### Document Processing

| Mode | Description | Grounding | Best For |
|------|-------------|-----------|----------|
| `document_markdown` | Full OCR with layout preservation | ✅ | General documents |
| `document_text` | Plain text extraction, fastest | ❌ | Speed priority |
| `document_structured` | Hierarchical structure preservation | ✅ | Complex docs |

### Table Extraction

| Mode | Description | Grounding | Best For |
|------|-------------|-----------|----------|
| `table_markdown` | Tables as markdown | ✅ | Data extraction |
| `table_csv` | Tables as CSV format | ❌ | Spreadsheet import |

### Figures & Charts

| Mode | Description | Grounding | Best For |
|------|-------------|-----------|----------|
| `figure_parse` | Parse charts/diagrams | ❌ | Data extraction |
| `chart_data` | Detailed chart analysis | ✅ | Analytics |
| `diagram_describe` | Technical diagram analysis | ✅ | Documentation |

### Specialized OCR

| Mode | Description | Grounding | Best For |
|------|-------------|-----------|----------|
| `handwriting` | Handwritten text | ✅ | Notes, letters |
| `receipt` | Receipt data extraction | ✅ | Expense tracking |
| `invoice` | Invoice field extraction | ✅ | Accounting |
| `business_card` | Contact info extraction | ✅ | CRM |
| `form` | Form field/value pairs | ✅ | Data entry |
| `id_document` | ID/passport/license | ✅ | Verification |

### Academic/Technical

| Mode | Description | Grounding | Best For |
|------|-------------|-----------|----------|
| `math_formula` | LaTeX formula extraction | ✅ | STEM documents |
| `code` | Source code extraction | ✅ | Screenshots |
| `chemical` | Chemical notation | ✅ | Chemistry |
| `scientific_paper` | Academic paper parsing | ✅ | Research |

### Educational

| Mode | Description | Grounding | Best For |
|------|-------------|-----------|----------|
| `exam_question` | Questions + figure descriptions | ✅ | Test digitization |
| `exam_with_figures` | Detailed figure analysis | ✅ | Visual questions |
| `textbook_page` | Textbook content | ✅ | Study materials |
| `worksheet` | Worksheet/assignment | ✅ | Homework |

### Search/Locate

| Mode | Description | Grounding | Best For |
|------|-------------|-----------|----------|
| `find_text` | Find specific text (needs `search_term`) | ✅ | Text location |
| `find_all_images` | Locate all figures with descriptions | ✅ | Image extraction |
| `find_tables` | Locate and extract all tables | ✅ | Table extraction |

### Multilingual

| Mode | Description | Grounding | Best For |
|------|-------------|-----------|----------|
| `multilingual` | Auto language detection | ✅ | Mixed languages |
| `chinese` | Chinese text | ✅ | 中文文档 |
| `arabic` | Arabic text (RTL) | ✅ | النصوص العربية |

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | File | required | Input image (PNG, JPG, WEBP) |
| `mode` | string | `document_markdown` | Preset mode or `custom` |
| `custom_prompt` | string | `""` | Your prompt when mode=`custom` |
| `search_term` | string | `""` | Text to find (for `find_text` mode) |
| `resolution` | string | `auto` | `tiny`/`small`/`base`/`large`/`gundam`/`auto` |
| `enable_grounding` | string | `auto` | Force grounding `on`/`off`/`auto` |
| `crop_detected_images` | bool | `false` | Return cropped figures as base64 |
| `crop_filter` | string | `""` | Only crop matching labels |
| `output_format` | string | `json` | `json`/`text_only`/`markdown`/`boxes_only` |

---

## Resolution Modes

| Mode | Size | Tokens | Speed | Quality | Use Case |
|------|------|--------|-------|---------|----------|
| `tiny` | 512×512 | 64 | ⚡⚡⚡ | ★ | Quick previews |
| `small` | 640×640 | 100 | ⚡⚡ | ★★ | Simple documents |
| `base` | 1024×1024 | 256 | ⚡ | ★★★ | Standard quality |
| `large` | 1280×1280 | 400 | 🐢 | ★★★★ | High detail |
| `gundam` | Dynamic | Varies | ⚡ | ★★★★★ | **Recommended** |

---

## Output Format

### JSON (default)

```json
{
  "success": true,
  "text": "# Document Title\n\nParsed content...",
  "boxes": [
    {
      "label": "figure",
      "box": [120, 340, 580, 720],
      "normalized_box": [120, 340, 580, 720]
    }
  ],
  "image_dims": {"width": 1920, "height": 1080},
  "metadata": {
    "mode": "document_markdown",
    "resolution": "gundam",
    "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
    "grounding_enabled": true
  },
  "cropped_images": [
    {
      "label": "figure",
      "box": [120, 340, 580, 720],
      "width": 460,
      "height": 380,
      "image_base64": "iVBORw0KGgo..."
    }
  ]
}
```

---

## Custom Prompts

### Format

```
<image>\n[<|grounding|>]Your instruction here.
```

- Always start with `<image>\n`
- Add `<|grounding|>` to get bounding boxes
- Be specific and detailed for best results

### Examples

**Extract specific data:**
```python
custom_prompt = "<image>\n<|grounding|>Extract: 1) All names, 2) All dates, 3) All monetary amounts. Format as structured JSON."
```

**Exam with figures:**
```python
custom_prompt = "<image>\n<|grounding|>For this exam page: Extract each question, list answer choices, and for any figures provide a detailed description explaining what the figure shows and how it relates to the question."
```

**Compare documents:**
```python
custom_prompt = "<image>\nIdentify the document type and extract key fields. If this is a contract, extract parties, dates, and key terms. If invoice, extract line items and totals."
```

**Custom language:**
```python
custom_prompt = "<image>\n<|grounding|>Extrae todo el texto en español. Mantén el formato original."
```

---

## Extracting Images from Documents

### Method 1: Use find_all_images mode

```python
output = replicate.run(
    "your-username/deepseek-ocr",
    input={
        "image": open("document.png", "rb"),
        "mode": "find_all_images",
        "crop_detected_images": True
    }
)

# Access cropped images
for img_data in output["cropped_images"]:
    print(f"Found: {img_data['label']}")
    # img_data['image_base64'] contains the cropped image
```

### Method 2: Filter specific elements

```python
output = replicate.run(
    "your-username/deepseek-ocr",
    input={
        "image": open("exam.png", "rb"),
        "mode": "exam_with_figures",
        "crop_detected_images": True,
        "crop_filter": "figure"  # Only crop items labeled "figure"
    }
)
```

---

## Deployment

```bash
# Install Cog
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog

# Login to Replicate
cog login

# Push model
cog push r8.im/your-username/deepseek-ocr

# Test locally first
cog predict -i image=@test.png -i mode=document_markdown
```

---

## Cost Estimation

| Resolution | ~Time/Page | ~Cost/Page |
|------------|------------|------------|
| tiny | 0.5s | $0.0003 |
| small | 0.8s | $0.0005 |
| base | 1.2s | $0.0007 |
| large | 2.0s | $0.0012 |
| gundam | 1.5s | $0.0009 |

*Based on Replicate A40 pricing (~$0.000575/sec)*

---

## License

MIT License - Model weights subject to DeepSeek-OCR license terms.
# deepseek-ocr-cog
