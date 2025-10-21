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

import os
# Set environment variables to suppress verbose output before importing other libraries
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from oocana import Context
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
import sys
from io import StringIO
from contextlib import contextmanager
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set logging levels
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr output at file descriptor level"""
    import builtins

    # Save original print and file descriptors
    _original_print = builtins.print
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # Duplicate original file descriptors
    stdout_dup = os.dup(stdout_fd)
    stderr_dup = os.dup(stderr_fd)

    # Open /dev/null
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        # Override print function
        builtins.print = lambda *args, **kwargs: None

        # Redirect file descriptors to /dev/null
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)

        # Flush to ensure all buffered data is written
        sys.stdout.flush()
        sys.stderr.flush()

        yield
    finally:
        # Flush again before restoring
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore file descriptors
        os.dup2(stdout_dup, stdout_fd)
        os.dup2(stderr_dup, stderr_fd)

        # Close duplicates and devnull
        os.close(stdout_dup)
        os.close(stderr_dup)
        os.close(devnull)

        # Restore print function
        builtins.print = _original_print

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
        with suppress_output():
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
                except ImportError:
                    model_kwargs['_attn_implementation'] = 'eager'

            _model = AutoModel.from_pretrained(model_name, **model_kwargs)
            _model = _model.eval()

            # Move to device and set dtype
            if device == 'cuda':
                _model = _model.cuda().to(torch.bfloat16)
            else:
                _model = _model.to(device)

            _current_device = device

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
    # Note: model.infer() returns text when save_results=True and reads from output file
    with suppress_output():
        model.infer(
            tokenizer,
            prompt=full_prompt,
            image_file=image_path,
            output_path=output_dir,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=True,  # Enable saving to get the result
            test_compress=False
        )

    # Read the result from the saved markdown file
    result_file = os.path.join(output_dir, 'result.mmd')
    text = ""

    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        text = ""

    # Clean up temporary directory
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)

    # Preview result as markdown
    preview_content = f"# OCR Result\n\n**Image:** {os.path.basename(image_path)}\n\n**Resolution:** {resolution}\n\n**Extracted Text:**\n\n{text}"
    context.preview({
        "type": "markdown",
        "data": preview_content
    })

    return {
        "text": text
    }
