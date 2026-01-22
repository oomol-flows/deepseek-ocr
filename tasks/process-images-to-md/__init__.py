#region generated meta
import typing
from oocana import LLMModelOptions
class Inputs(typing.TypedDict):
    image_dir: str
    llm: LLMModelOptions
    prompt: str | None
    output_file: str | None
class Outputs(typing.TypedDict):
    markdown_file: typing.NotRequired[str]
    text: typing.NotRequired[str]
#endregion

import base64
from pathlib import Path
from oocana import Context
from openai import OpenAI


async def main(params: Inputs, context: Context) -> Outputs:
    # Validate input
    image_dir = params.get("image_dir")
    if not image_dir:
        raise ValueError("image_dir parameter is required")

    # Check if directory exists
    dir_path = Path(image_dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {image_dir}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {image_dir}")

    # Get all image files and sort them numerically
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    image_files = [
        f for f in dir_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    # Sort files numerically (assumes names like page_001.png)
    image_files.sort()

    if not image_files:
        raise ValueError(f"No image files found in directory: {image_dir}")

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
        prompt_text = """Convert this image to standard Markdown format. Follow these rules:

1. Use proper Markdown syntax for all elements (headings, lists, tables, links, etc.)
2. Preserve document structure and hierarchy with appropriate heading levels (# ## ###)
3. Format tables using Markdown table syntax with proper alignment
4. Convert lists to proper Markdown lists (- or 1. 2. 3.)
5. Preserve code blocks with triple backticks and language tags if applicable
6. Keep image references and links in Markdown format
7. Maintain text formatting: **bold**, *italic*, `code`
8. Return only the Markdown content without any explanations or meta-text"""

    # Initialize OpenAI client with OOMOL token
    client = OpenAI(
        base_url=context.oomol_llm_env.get("base_url_v1"),
        api_key=await context.oomol_token(),
    )

    # Process each image
    all_markdown = []
    total = len(image_files)

    for idx, image_file in enumerate(image_files):
        # Report progress
        progress = int((idx + 1) / total * 100)
        context.report_progress(progress)

        # Read image and convert to base64
        try:
            with open(image_file, "rb") as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to read image file {image_file.name}: {str(e)}")

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
                raise ValueError(f"LLM returned no choices for {image_file.name}")

            extracted_text = response.choices[0].message.content
            if extracted_text is None:
                raise ValueError(f"LLM returned None content for {image_file.name}")

            # Add page separator and content
            page_number = idx + 1
            all_markdown.append(f"<!-- Page {page_number} -->\n\n{extracted_text.strip()}\n\n")

        except Exception as e:
            # Re-raise if already a ValueError or RuntimeError
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            # Wrap other exceptions
            raise RuntimeError(f"OCR failed for {image_file.name}: {str(e)}")

    # Combine all markdown
    combined_markdown = "\n".join(all_markdown)

    # Determine output file path
    output_file = params.get("output_file")
    if output_file:
        output_path = Path(output_file)
    else:
        # Default to session directory
        output_path = Path(context.session_dir) / "output.md"

    # Write markdown to file
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(combined_markdown, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to write markdown file: {str(e)}")

    context.report_progress(100)

    return {
        "markdown_file": str(output_path),
        "text": combined_markdown
    }
