#region generated meta
import typing
from oocana import LLMModelOptions
class Inputs(typing.TypedDict):
    image_path: str
    llm: LLMModelOptions
    prompt: str | None
class Outputs(typing.TypedDict):
    text: typing.NotRequired[str]
#endregion

import base64
from pathlib import Path
from oocana import Context
from openai import OpenAI


async def main(params: Inputs, context: Context) -> Outputs:
    # Validate input
    image_path = params.get("image_path")
    if not image_path:
        raise ValueError("image_path parameter is required")

    # Check if file exists
    image_file = Path(image_path)
    if not image_file.exists():
        raise ValueError(f"Image file not found: {image_path}")

    # Read image and convert to base64
    try:
        with open(image_file, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Failed to read image file: {str(e)}")

    # Determine image MIME type
    suffix = image_file.suffix.lower()
    mime_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp"
    }
    mime_type = mime_type_map.get(suffix, "image/jpeg")

    # Get LLM configuration
    llm = params.get("llm", {})
    model = llm.get("model", "deepseek-ocr")
    temperature = llm.get("temperature", 0)
    max_tokens = llm.get("max_tokens", 8000)

    # Get custom prompt or use default
    custom_prompt = params.get("prompt")
    if custom_prompt and custom_prompt.strip():
        prompt_text = custom_prompt
    else:
        prompt_text = "Convert the document to markdown."

    # Initialize OpenAI client with OOMOL token
    client = OpenAI(
        base_url=context.oomol_llm_env.get("base_url_v1"),
        api_key=await context.oomol_token(),
    )

    # Call LLM for OCR
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )

        # Extract text from response
        if not response.choices:
            raise ValueError("LLM returned no choices in response")

        extracted_text = response.choices[0].message.content
        if extracted_text is None:
            raise ValueError(f"LLM returned None content. Response: {response}")

        # Return the extracted text (even if empty string)
        return {"text": extracted_text.strip() if extracted_text else ""}

    except Exception as e:
        # Re-raise if already a ValueError or RuntimeError
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        # Wrap other exceptions
        raise RuntimeError(f"OCR failed: {str(e)}")
