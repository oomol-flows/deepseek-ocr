#region generated meta
import typing
class Inputs(typing.TypedDict):
    image: str
    resolution: typing.Literal["tiny", "small", "base", "large", "gundam"]
    prompt: str
    use_gpu: bool
class Outputs(typing.TypedDict):
    text: typing.NotRequired[str]
#endregion

from oocana import Context
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
import os

# Global model cache
_model = None
_tokenizer = None
_current_device = None

def load_model(use_gpu: bool):
    """
    Load DeepSeek-OCR model with caching

    Args:
        use_gpu: Whether to use GPU acceleration

    Returns:
        Tuple of (model, tokenizer, device)
    """
    global _model, _tokenizer, _current_device

    model_name = 'deepseek-ai/DeepSeek-OCR'

    # Determine device
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Load model if not cached or device changed
    if _model is None or _current_device != device:
        print(f"Loading DeepSeek-OCR model on {device}...")

        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model with appropriate settings
        model_kwargs = {
            'trust_remote_code': True,
            'use_safetensors': True
        }

        # Check if flash attention is available for GPU
        if device == 'cuda':
            try:
                import flash_attn
                model_kwargs['_attn_implementation'] = 'flash_attention_2'
                print("Using Flash Attention 2 for GPU acceleration")
            except ImportError:
                print("Flash attention not installed, using default attention (slower but functional)")
                model_kwargs['_attn_implementation'] = 'eager'

        _model = AutoModel.from_pretrained(model_name, **model_kwargs)
        _model = _model.eval()

        # Move to device and set dtype
        if device == 'cuda':
            _model = _model.cuda().to(torch.bfloat16)
        else:
            _model = _model.to(device)

        _current_device = device
        print("Model loaded successfully")

    return _model, _tokenizer, _current_device

def get_resolution_config(resolution: str) -> dict:
    """
    Get resolution configuration based on mode

    Args:
        resolution: Resolution mode (tiny/small/base/large/gundam)

    Returns:
        Dictionary with resolution parameters
    """
    configs = {
        'tiny': {'size': (512, 512), 'tokens': 64},
        'small': {'size': (640, 640), 'tokens': 100},
        'base': {'size': (1024, 1024), 'tokens': 256},
        'large': {'size': (1280, 1280), 'tokens': 400},
        'gundam': {'dynamic': True}  # Dynamic resolution
    }

    return configs.get(resolution, configs['base'])

def main(params: Inputs, context: Context) -> Outputs:
    """
    Main OCR function using DeepSeek-OCR

    Args:
        params: Input parameters (image, resolution, prompt, use_gpu)
        context: OOMOL context object

    Returns:
        Outputs dictionary with extracted text
    """
    # Validate image path
    image_path = params['image']
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load model
    model, tokenizer, device = load_model(params['use_gpu'])

    # Get resolution config
    resolution = params['resolution']
    res_config = get_resolution_config(resolution)
    print(f"Processing image: {os.path.basename(image_path)}")
    print(f"Using resolution mode: {resolution}")

    # Prepare prompt
    prompt = params['prompt']
    full_prompt = f"<image>\n<|grounding|>{prompt}"

    # Create temporary output directory
    import tempfile
    output_dir = tempfile.mkdtemp()

    # Map resolution to base_size
    resolution_to_size = {
        'tiny': 512,
        'small': 640,
        'base': 1024,
        'large': 1280,
        'gundam': 1024  # gundam uses crop_mode
    }

    base_size = resolution_to_size.get(resolution, 1024)
    image_size = 640 if resolution == 'gundam' else base_size
    crop_mode = (resolution == 'gundam')

    # Run OCR inference using model.infer()
    print("Running OCR inference...")
    result = model.infer(
        tokenizer,
        prompt=full_prompt,
        image_file=image_path,
        output_path=output_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=False,
        test_compress=False
    )

    # Extract text from result
    text = result if isinstance(result, str) else str(result)

    # Clean up temporary directory
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)

    print(f"OCR completed. Extracted {len(text)} characters")

    # Preview result as markdown
    preview_content = f"# OCR Result\n\n**Image:** {os.path.basename(image_path)}\n\n**Resolution:** {resolution}\n\n**Extracted Text:**\n\n{text}"
    context.preview({
        "type": "markdown",
        "data": preview_content
    })

    return {
        "text": text
    }
