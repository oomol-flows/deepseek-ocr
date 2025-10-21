# DeepSeek OCR - Intelligent Image Text Recognition Tool

## Project Overview

DeepSeek OCR is a powerful optical character recognition (OCR) tool that automatically extracts text content from images and saves the results in Markdown format. Whether it's scanned documents, screenshots, or text in photos, it can easily recognize and convert them into editable text.

## Key Features

### 📸 Image Text Recognition

This tool helps you:
- **Extract text from images** - Upload any image containing text and automatically extract the text content
- **Support multiple resolutions** - Choose appropriate recognition accuracy based on image quality
- **Smart document conversion** - Convert scanned documents, screenshots, etc. into Markdown format
- **GPU acceleration** - Support GPU acceleration for faster processing

## Feature Modules

### DeepSeek OCR Recognition Module

This is the core functionality module of the project, providing the following capabilities:

#### Input Parameters

1. **Image File** (`image`)
   - Upload the image to be recognized
   - Supports common image formats (PNG, JPG, etc.)

2. **Recognition Resolution** (`resolution`)
   - **Tiny (Ultra-fast)** - Suitable for simple text, fastest speed
   - **Small (Fast)** - Suitable for general text recognition
   - **Base (Standard)** - Balanced speed and accuracy, recommended
   - **Large (High precision)** - Suitable for complex documents, higher accuracy
   - **Gundam (Ultra-high precision)** - Suitable for documents requiring high quality

3. **Custom Prompt** (`prompt`)
   - Specify particular recognition requirements
   - For example: "Convert to Markdown format" or "Extract table content only"

4. **GPU Acceleration** (`use_gpu`)
   - When enabled, uses GPU acceleration for faster processing
   - When disabled, uses CPU processing (slower but better compatibility)

#### Output Results

- **Markdown File** (`markdown`)
  - The recognized text content will be saved as a Markdown format file
  - Files are stored in the system storage directory and can be directly downloaded

## Use Cases

### Suitable for the following scenarios

✅ **Document digitization** - Convert scanned paper documents into editable electronic documents
✅ **Screenshot text extraction** - Extract text from web screenshots and software interfaces
✅ **Image to text** - Convert images containing text into plain text
✅ **Multi-language recognition** - Recognize text content in different languages
✅ **Batch processing** - Process multiple images in batches through workflows

## Quick Start

### Basic Usage Flow

1. **Upload image** - Select the image file to be recognized
2. **Choose resolution** - Select recognition resolution based on needs (recommend "Standard" mode)
3. **Run recognition** - Click run and wait for processing to complete
4. **Get results** - Download the generated Markdown file

### Usage Tips

- 📌 **Image clarity** - The clearer the image, the higher the recognition accuracy
- 📌 **Choose appropriate resolution** - "Standard" mode is sufficient for general documents, "High precision" for complex documents
- 📌 **GPU acceleration** - If your system supports GPU, it's recommended to enable it for improved speed
- 📌 **Custom prompts** - You can specify particular conversion requirements through prompts

## Technical Features

- 🚀 **Advanced AI model** - Uses DeepSeek-OCR vision-language model
- ⚡ **High-performance processing** - Supports GPU acceleration and various optimization techniques
- 🎯 **High accuracy** - Multiple resolution modes adapt to images of different quality
- 📝 **Markdown output** - Direct output in Markdown format for easy subsequent editing

## System Requirements

### Hardware Recommendations

- **Recommended configuration** - NVIDIA GPU with CUDA support (optional, for acceleration)
- **Minimum configuration** - Regular CPU can also run (slower processing speed)

### Software Environment

- Python 3.x environment
- Related dependencies will be installed automatically

## Project Information

- **Version**: 0.0.1
- **Project URL**: [https://github.com/oomol-flows/deepseek-ocr](https://github.com/oomol-flows/deepseek-ocr)
- **Development Platform**: OOMOL Workflow Platform

## FAQ

### Q: What should I do if recognition is slow?
A: You can try the following methods:
- Reduce recognition resolution (use Tiny or Small mode)
- Enable GPU acceleration (if your system supports it)
- Reduce image size

### Q: Recognition accuracy is not high?
A: Suggestions:
- Increase recognition resolution (use Large or Gundam mode)
- Ensure the image is clear with high contrast in text areas
- Use custom prompts to describe recognition requirements

### Q: What languages are supported?
A: The DeepSeek-OCR model supports multiple languages, including Chinese, English, and other common languages.

### Q: Where are output files saved?
A: Recognition results are saved in the `/oomol-driver/oomol-storage/deepseek_ocr/` directory with automatically generated filenames.

## Changelog

### v0.0.1 (Initial Release)
- ✨ Implemented image text recognition based on DeepSeek-OCR
- ✨ Support for 5 resolution modes
- ✨ Support for GPU acceleration
- ✨ Automatic saving in Markdown format

---

**Note**: This tool works best on the OOMOL workflow platform and can be combined with other functional modules to achieve more complex automated processing workflows.
