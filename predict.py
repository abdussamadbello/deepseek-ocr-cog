"""
DeepSeek-OCR Replicate Predictor
Comprehensive OCR with optimized prompts and full customization
"""

from cog import BasePredictor, Input, Path, BaseModel
from typing import Optional, List, Any
from PIL import Image
import torch
import base64
import json
import re
import io


class BoundingBox(BaseModel):
    """Detected element with bounding box"""
    label: str
    box: List[int]  # [x1, y1, x2, y2] in pixels
    normalized_box: List[int]  # [x1, y1, x2, y2] in 0-999 range


class OCROutput(BaseModel):
    """Structured OCR response"""
    text: str
    markdown: Optional[str] = None
    boxes: List[BoundingBox] = []
    cropped_images: List[str] = []  # base64 encoded
    image_dims: dict  # {"width": w, "height": h}
    mode_used: str
    prompt_used: str
    tokens_used: Optional[int] = None


# =============================================================================
# PROMPT LIBRARY - Optimized prompts for different use cases
# =============================================================================

PROMPT_LIBRARY = {
    # ---------------------------------------------------------------------
    # DOCUMENT PROCESSING
    # ---------------------------------------------------------------------
    "document_markdown": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "description": "Full document OCR with layout preservation, returns markdown with bounding boxes",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    "document_text": {
        "prompt": "<image>\nFree OCR.",
        "description": "Plain text extraction without layout info, fastest mode",
        "grounding": False,
        "recommended_mode": "base"
    },
    "document_structured": {
        "prompt": "<image>\n<|grounding|>Convert to markdown. Identify all sections, headings, paragraphs, lists, and tables. Preserve document hierarchy.",
        "description": "Structured document with hierarchy preservation",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    
    # ---------------------------------------------------------------------
    # TABLE EXTRACTION
    # ---------------------------------------------------------------------
    "table_markdown": {
        "prompt": "<image>\n<|grounding|>Extract all tables as markdown tables. Preserve column alignment and cell contents exactly.",
        "description": "Table extraction to markdown format",
        "grounding": True,
        "recommended_mode": "large"
    },
    "table_csv": {
        "prompt": "<image>\nExtract the table data as CSV format. Use commas as delimiters. Include headers.",
        "description": "Table extraction to CSV format",
        "grounding": False,
        "recommended_mode": "base"
    },
    
    # ---------------------------------------------------------------------
    # FIGURES & CHARTS
    # ---------------------------------------------------------------------
    "figure_parse": {
        "prompt": "<image>\nParse the figure.",
        "description": "Parse charts, graphs, diagrams - extracts data and structure",
        "grounding": False,
        "recommended_mode": "large"
    },
    "chart_data": {
        "prompt": "<image>\n<|grounding|>Analyze this chart. Extract: 1) Chart type, 2) Title and labels, 3) All data points/values, 4) Legend items. Format as structured markdown.",
        "description": "Detailed chart analysis with data extraction",
        "grounding": True,
        "recommended_mode": "large"
    },
    "diagram_describe": {
        "prompt": "<image>\n<|grounding|>Describe this diagram in detail. Identify all components, labels, arrows, and relationships between elements.",
        "description": "Technical diagram analysis",
        "grounding": True,
        "recommended_mode": "large"
    },
    
    # ---------------------------------------------------------------------
    # IMAGE DESCRIPTION
    # ---------------------------------------------------------------------
    "describe": {
        "prompt": "<image>\nDescribe this image in detail.",
        "description": "General image description",
        "grounding": False,
        "recommended_mode": "base"
    },
    "describe_grounded": {
        "prompt": "<image>\n<|grounding|>Describe this image in detail. Identify and locate all significant elements.",
        "description": "Image description with element locations",
        "grounding": True,
        "recommended_mode": "large"
    },
    
    # ---------------------------------------------------------------------
    # SPECIALIZED OCR
    # ---------------------------------------------------------------------
    "handwriting": {
        "prompt": "<image>\n<|grounding|>Transcribe the handwritten text exactly as written. Preserve line breaks and formatting.",
        "description": "Handwriting recognition",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    "receipt": {
        "prompt": "<image>\n<|grounding|>Extract all information from this receipt: store name, date, items with prices, subtotal, tax, and total. Format as structured markdown.",
        "description": "Receipt/invoice data extraction",
        "grounding": True,
        "recommended_mode": "large"
    },
    "invoice": {
        "prompt": "<image>\n<|grounding|>Extract invoice details: invoice number, date, vendor info, billing address, line items (description, quantity, unit price, total), subtotal, taxes, and grand total. Format as structured markdown.",
        "description": "Invoice data extraction",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    "business_card": {
        "prompt": "<image>\n<|grounding|>Extract all contact information: name, title, company, phone, email, address, website. Format as structured data.",
        "description": "Business card information extraction",
        "grounding": True,
        "recommended_mode": "base"
    },
    "form": {
        "prompt": "<image>\n<|grounding|>Extract all form fields and their values. Format as field_name: value pairs.",
        "description": "Form field extraction",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    "id_document": {
        "prompt": "<image>\n<|grounding|>Extract all visible information from this ID document: name, date of birth, ID number, issue date, expiry date, address. Note: Do not include sensitive biometric data.",
        "description": "ID card/passport/license extraction",
        "grounding": True,
        "recommended_mode": "large"
    },
    
    # ---------------------------------------------------------------------
    # ACADEMIC / TECHNICAL
    # ---------------------------------------------------------------------
    "math_formula": {
        "prompt": "<image>\n<|grounding|>Extract all mathematical formulas and equations. Use LaTeX notation. Preserve equation numbers if present.",
        "description": "Mathematical formula extraction (LaTeX)",
        "grounding": True,
        "recommended_mode": "large"
    },
    "code": {
        "prompt": "<image>\n<|grounding|>Extract the code exactly as shown. Preserve indentation, syntax, and comments. Identify the programming language.",
        "description": "Source code extraction",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    "chemical": {
        "prompt": "<image>\n<|grounding|>Extract chemical formulas, structures, and equations. Use standard chemical notation.",
        "description": "Chemical formula/structure recognition",
        "grounding": True,
        "recommended_mode": "large"
    },
    "scientific_paper": {
        "prompt": "<image>\n<|grounding|>Extract the content preserving: title, authors, abstract, section headings, body text, equations (in LaTeX), figure/table captions, and references. Format as academic markdown.",
        "description": "Scientific paper/article extraction",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    
    # ---------------------------------------------------------------------
    # EXAM / EDUCATIONAL
    # ---------------------------------------------------------------------
    "exam_question": {
        "prompt": "<image>\n<|grounding|>Extract the exam question(s) and all answer options. For any figures or diagrams, describe what they show in detail. Format as markdown with clear question numbering.",
        "description": "Exam/quiz question extraction with figure descriptions",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    "exam_with_figures": {
        "prompt": "<image>\n<|grounding|>Extract this exam content completely. For each question: 1) Extract question text, 2) List all answer options (A, B, C, D), 3) For any figures/diagrams/images, provide detailed description of what they show and how they relate to the question. Locate all visual elements.",
        "description": "Exam questions with detailed figure analysis",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    "textbook_page": {
        "prompt": "<image>\n<|grounding|>Convert this textbook page to markdown. Preserve: headings, paragraphs, numbered lists, bullet points, definitions, examples, and figure captions. For figures, provide a detailed description.",
        "description": "Textbook page with educational content",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    "worksheet": {
        "prompt": "<image>\n<|grounding|>Extract this worksheet/assignment. Include: instructions, questions, blank spaces for answers, and any reference materials or hints provided.",
        "description": "Educational worksheet extraction",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    
    # ---------------------------------------------------------------------
    # LOCATING SPECIFIC CONTENT
    # ---------------------------------------------------------------------
    "find_text": {
        "prompt": "<image>\nLocate <|ref|>{search_term}<|/ref|> in the image.",
        "description": "Find specific text and return its bounding box",
        "grounding": True,
        "recommended_mode": "base",
        "requires_param": "search_term"
    },
    "find_all_images": {
        "prompt": "<image>\n<|grounding|>Identify and locate all images, figures, photos, charts, and diagrams in this document. For each one, provide: 1) Its location (bounding box), 2) Type (photo, chart, diagram, etc.), 3) Detailed description of contents.",
        "description": "Find all visual elements with descriptions",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    "find_tables": {
        "prompt": "<image>\n<|grounding|>Locate all tables in this document. For each table, extract its contents as markdown.",
        "description": "Find and extract all tables",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    
    # ---------------------------------------------------------------------
    # MULTILINGUAL
    # ---------------------------------------------------------------------
    "multilingual": {
        "prompt": "<image>\n<|grounding|>OCR this image. Detect the language(s) and extract all text, preserving the original script and layout.",
        "description": "Multi-language OCR with language detection",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    "chinese": {
        "prompt": "<image>\n<|grounding|>提取图片中的所有中文文本，保持原始布局和格式。",
        "description": "Chinese text extraction",
        "grounding": True,
        "recommended_mode": "gundam"
    },
    "arabic": {
        "prompt": "<image>\n<|grounding|>استخراج جميع النصوص العربية من الصورة مع الحفاظ على التخطيط.",
        "description": "Arabic text extraction (RTL)",
        "grounding": True,
        "recommended_mode": "gundam"
    },
}


# Resolution mode configurations
RESOLUTION_MODES = {
    "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False, "tokens": 64},
    "small": {"base_size": 640, "image_size": 640, "crop_mode": False, "tokens": 100},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False, "tokens": 256},
    "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False, "tokens": 400},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True, "tokens": "dynamic"},
}


class Predictor(BasePredictor):
    
    def setup(self):
        """Load the model into memory"""
        from transformers import AutoModel, AutoTokenizer
        
        model_name = "deepseek-ai/DeepSeek-OCR"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True
        )
        self.model = self.model.eval().cuda().to(torch.bfloat16)
    
    def predict(
        self,
        image: Path = Input(description="Input image (PNG, JPG, WEBP)"),
        
        # Mode selection
        mode: str = Input(
            description="Preset mode with optimized prompt. Use 'custom' for your own prompt.",
            default="document_markdown",
            choices=[
                # Document
                "document_markdown", "document_text", "document_structured",
                # Tables
                "table_markdown", "table_csv",
                # Figures
                "figure_parse", "chart_data", "diagram_describe",
                # Description
                "describe", "describe_grounded",
                # Specialized
                "handwriting", "receipt", "invoice", "business_card", "form", "id_document",
                # Academic
                "math_formula", "code", "chemical", "scientific_paper",
                # Educational
                "exam_question", "exam_with_figures", "textbook_page", "worksheet",
                # Search
                "find_text", "find_all_images", "find_tables",
                # Multilingual
                "multilingual", "chinese", "arabic",
                # Custom
                "custom"
            ]
        ),
        
        # Custom prompt (used when mode="custom")
        custom_prompt: str = Input(
            description="Your custom prompt. Start with '<image>\\n' and add '<|grounding|>' for bounding boxes. Only used when mode='custom'.",
            default=""
        ),
        
        # For find_text mode
        search_term: str = Input(
            description="Text to search for (only for 'find_text' mode)",
            default=""
        ),
        
        # Resolution override
        resolution: str = Input(
            description="Resolution mode. 'auto' uses the recommended setting for the selected mode.",
            default="auto",
            choices=["auto", "tiny", "small", "base", "large", "gundam"]
        ),
        
        # Grounding override
        enable_grounding: str = Input(
            description="Force grounding on/off. 'auto' uses mode's default.",
            default="auto",
            choices=["auto", "on", "off"]
        ),
        
        # Image cropping from detected boxes
        crop_detected_images: bool = Input(
            description="Crop and return detected figures/diagrams as base64 images",
            default=False
        ),
        
        # Filter crops by label
        crop_filter: str = Input(
            description="Only crop boxes matching this label (e.g., 'figure', 'chart'). Empty = crop all.",
            default=""
        ),
        
        # Output format
        output_format: str = Input(
            description="Output format",
            default="json",
            choices=["json", "text_only", "markdown", "boxes_only"]
        ),
        
    ) -> str:
        """Run OCR on an image with optimized prompts"""
        
        import tempfile
        import os
        
        # Load and get image dimensions
        img = Image.open(str(image)).convert("RGB")
        img_width, img_height = img.size
        
        # Determine prompt
        if mode == "custom":
            if not custom_prompt:
                raise ValueError("custom_prompt is required when mode='custom'")
            prompt = custom_prompt
            if not prompt.startswith("<image>"):
                prompt = f"<image>\n{prompt}"
            has_grounding = "<|grounding|>" in prompt
        else:
            mode_config = PROMPT_LIBRARY[mode]
            prompt = mode_config["prompt"]
            has_grounding = mode_config["grounding"]
            
            # Handle search_term for find_text mode
            if mode == "find_text":
                if not search_term:
                    raise ValueError("search_term is required for 'find_text' mode")
                prompt = prompt.replace("{search_term}", search_term)
        
        # Apply grounding override
        if enable_grounding == "on" and "<|grounding|>" not in prompt:
            prompt = prompt.replace("<image>\n", "<image>\n<|grounding|>")
            has_grounding = True
        elif enable_grounding == "off" and "<|grounding|>" in prompt:
            prompt = prompt.replace("<|grounding|>", "")
            has_grounding = False
        
        # Determine resolution
        if resolution == "auto":
            if mode != "custom":
                resolution = PROMPT_LIBRARY[mode].get("recommended_mode", "base")
            else:
                resolution = "gundam"  # Default for custom
        
        res_config = RESOLUTION_MODES[resolution]
        
        # Save image to temp file for model
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name, "JPEG", quality=95)
            temp_image_path = tmp.name
        
        try:
            # Create temp output directory
            with tempfile.TemporaryDirectory() as output_dir:
                # Run inference
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=temp_image_path,
                    output_path=output_dir,
                    base_size=res_config["base_size"],
                    image_size=res_config["image_size"],
                    crop_mode=res_config["crop_mode"],
                    save_results=False,
                    test_compress=False
                )
                
                # Parse the result
                raw_text = result if isinstance(result, str) else str(result)
                
                # Extract bounding boxes if grounding was enabled
                boxes = []
                clean_text = raw_text
                
                if has_grounding:
                    boxes, clean_text = self._parse_grounding(raw_text, img_width, img_height)
                
                # Crop detected images if requested
                cropped_images = []
                if crop_detected_images and boxes:
                    cropped_images = self._crop_regions(img, boxes, crop_filter)
                
                # Build response based on output format
                if output_format == "text_only":
                    return clean_text
                
                elif output_format == "markdown":
                    return clean_text
                
                elif output_format == "boxes_only":
                    return json.dumps({
                        "boxes": [self._box_to_dict(b) for b in boxes],
                        "image_dims": {"width": img_width, "height": img_height}
                    }, indent=2)
                
                else:  # json (default)
                    response = {
                        "success": True,
                        "text": clean_text,
                        "boxes": [self._box_to_dict(b) for b in boxes],
                        "image_dims": {"width": img_width, "height": img_height},
                        "metadata": {
                            "mode": mode,
                            "resolution": resolution,
                            "prompt": prompt,
                            "grounding_enabled": has_grounding
                        }
                    }
                    
                    if cropped_images:
                        response["cropped_images"] = cropped_images
                    
                    return json.dumps(response, indent=2)
                    
        finally:
            # Clean up temp file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    
    def _box_to_dict(self, box: BoundingBox) -> dict:
        """Convert BoundingBox to dict"""
        return {
            "label": box.label,
            "box": box.box,
            "normalized_box": box.normalized_box
        }
    
    def _parse_grounding(self, text: str, img_width: int, img_height: int) -> tuple:
        """
        Parse grounding tags from model output.
        Returns (list of BoundingBox, clean_text)
        
        Format: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
        Coordinates are normalized 0-999
        """
        boxes = []
        clean_text = text
        
        # Pattern to match grounding annotations
        pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
        
        matches = re.findall(pattern, text)
        
        for label, coords_str in matches:
            try:
                # Parse coordinates - can be [[x1,y1,x2,y2]] or [[x1,y1,x2,y2],[x1,y1,x2,y2],...]
                coords_str = coords_str.strip()
                
                # Handle nested array format
                if coords_str.startswith("[["):
                    import ast
                    coords_list = ast.literal_eval(coords_str)
                    if isinstance(coords_list[0], list):
                        # Multiple boxes
                        for coords in coords_list:
                            box = self._create_box(label, coords, img_width, img_height)
                            if box:
                                boxes.append(box)
                    else:
                        # Single box in nested format
                        box = self._create_box(label, coords_list, img_width, img_height)
                        if box:
                            boxes.append(box)
                elif coords_str.startswith("["):
                    # Single array [x1,y1,x2,y2]
                    import ast
                    coords = ast.literal_eval(coords_str)
                    box = self._create_box(label, coords, img_width, img_height)
                    if box:
                        boxes.append(box)
                        
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse coordinates '{coords_str}': {e}")
                continue
        
        # Remove grounding tags from text for clean output
        clean_text = re.sub(pattern, r'\1', text)
        
        # Also remove any remaining special tokens
        clean_text = re.sub(r'<\|[^|]+\|>', '', clean_text)
        
        return boxes, clean_text.strip()
    
    def _create_box(self, label: str, coords: list, img_width: int, img_height: int) -> Optional[BoundingBox]:
        """Create a BoundingBox from normalized coordinates"""
        if len(coords) != 4:
            return None
            
        # Normalized coordinates (0-999)
        nx1, ny1, nx2, ny2 = coords
        
        # Scale to actual pixels
        x1 = int((nx1 / 999) * img_width)
        y1 = int((ny1 / 999) * img_height)
        x2 = int((nx2 / 999) * img_width)
        y2 = int((ny2 / 999) * img_height)
        
        return BoundingBox(
            label=label.strip(),
            box=[x1, y1, x2, y2],
            normalized_box=[int(nx1), int(ny1), int(nx2), int(ny2)]
        )
    
    def _crop_regions(self, img: Image.Image, boxes: List[BoundingBox], label_filter: str = "") -> List[dict]:
        """
        Crop regions from image and return as base64 strings with metadata
        
        Args:
            img: PIL Image
            boxes: List of BoundingBox objects
            label_filter: Only crop boxes with labels containing this string (empty = all)
        
        Returns:
            List of dicts with 'label', 'box', 'image_base64'
        """
        cropped = []
        
        for box in boxes:
            # Apply filter if specified
            if label_filter and label_filter.lower() not in box.label.lower():
                continue
                
            try:
                x1, y1, x2, y2 = box.box
                
                # Ensure valid coordinates
                x1, x2 = max(0, min(x1, x2)), min(img.width, max(x1, x2))
                y1, y2 = max(0, min(y1, y2)), min(img.height, max(y1, y2))
                
                if x2 > x1 and y2 > y1:
                    cropped_img = img.crop((x1, y1, x2, y2))
                    
                    # Convert to base64
                    buffer = io.BytesIO()
                    cropped_img.save(buffer, format="PNG")
                    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    
                    cropped.append({
                        "label": box.label,
                        "box": box.box,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "image_base64": b64_str
                    })
                    
            except Exception as e:
                print(f"Warning: Could not crop region {box.box}: {e}")
                continue
        
        return cropped


# =============================================================================
# Helper function to list all available modes (for documentation)
# =============================================================================

def list_modes():
    """Print all available modes and their descriptions"""
    print("\n" + "="*80)
    print("DeepSeek-OCR Available Modes")
    print("="*80)
    
    categories = {
        "Document Processing": ["document_markdown", "document_text", "document_structured"],
        "Table Extraction": ["table_markdown", "table_csv"],
        "Figures & Charts": ["figure_parse", "chart_data", "diagram_describe"],
        "Image Description": ["describe", "describe_grounded"],
        "Specialized OCR": ["handwriting", "receipt", "invoice", "business_card", "form", "id_document"],
        "Academic/Technical": ["math_formula", "code", "chemical", "scientific_paper"],
        "Educational": ["exam_question", "exam_with_figures", "textbook_page", "worksheet"],
        "Search/Locate": ["find_text", "find_all_images", "find_tables"],
        "Multilingual": ["multilingual", "chinese", "arabic"],
    }
    
    for category, modes in categories.items():
        print(f"\n## {category}")
        print("-" * 40)
        for mode in modes:
            config = PROMPT_LIBRARY[mode]
            print(f"\n  {mode}:")
            print(f"    Description: {config['description']}")
            print(f"    Grounding: {config['grounding']}")
            print(f"    Recommended: {config.get('recommended_mode', 'base')}")


if __name__ == "__main__":
    list_modes()
