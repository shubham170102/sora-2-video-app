# Sora 2 API Python Client

Python client implementation for OpenAI's Sora 2 video generation API.

## Overview

This repository provides a Python interface for the Sora 2 API, supporting video generation from text prompts with optional reference images. The implementation includes automatic image resizing, batch processing, and comprehensive error handling.

## Features

### Core Functionality
- Video generation using sora-2 and sora-2-pro models
- Reference image support with automatic dimension validation
- Model, resolution, and duration selection
- Organized file management with designated input/output directories
- Video library management (list, monitor, download, delete)
- Concurrent batch processing
- Asset downloads (video, thumbnail, spritesheet)

### Additional Capabilities
- Automatic image resizing to match target video dimensions
- Directory scanning for reference images
- Real-time cost calculation based on selected parameters
- Progress monitoring with status indicators
- Detailed error messages for API failures
- Timestamp-based file naming to prevent overwrites
- Test image generation for validation

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key with Sora access

### Setup

```bash
# Clone repository
git clone <repository-url>
cd sora-2-video-app

# Install dependencies using uv
uv pip install python-dotenv openai print-color pillow

# Alternative: Using standard pip with venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install python-dotenv openai print-color pillow

# Configure API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Project Structure

```
sora2-test/
├── reference_images/      # Input directory for reference images
├── generated_videos/      # Output directory for generated videos
├── examples.py           # Main entry point with menu interface
├── sora_client.py        # Core API client implementation
├── video_generator.py    # Interactive video generation interface
├── video_utils.py        # Utility functions and helpers
├── create_test_image.py  # Test image generator
└── .env                  # API key configuration (not tracked in git)
```

## Usage

### Running the Application

```bash
# Interactive menu
./examples.py

# Direct command execution
./examples.py 3  # Reference image generation
./examples.py g  # Custom video generator
./examples.py s  # Safe reference test
./examples.py t  # Test API connection
```

### Menu Options

```
Interactive Generator:
  G. Custom Video Generator - Full parameter control
  S. Safe Reference Test - Test with generated images

Video Generation:
  1. Interactive Generation - User customizable
  2. High Quality Production - Pro model, 12-second
  3. Reference Image Generation - Full flexibility
  4. Remix Workflow - Modify existing videos
  5. Download All Assets - Video, thumbnail, sprites
  6. Batch Generation - Multiple videos concurrently
  7. Async Generation - Asynchronous processing

Management:
  8. Library Management - List, monitor, download
  9. Advanced Prompting - Cinematic techniques

Utilities:
  T. Test Connection - Verify API setup
```

## Reference Image Feature

The reference image feature (Option 3) provides comprehensive control over video generation with image inputs.

### Capabilities
- Automatic directory scanning in `reference_images/` folder
- Support for .jpg, .jpeg, .png, .webp, .bmp formats
- Automatic image resizing to match target video dimensions
- Model selection between sora-2 and sora-2-pro
- Resolution options based on selected model
- Duration selection (4, 8, or 12 seconds)
- Real-time cost calculation before generation

### Resolution Options

**sora-2 model:**
- 720x1280 (Portrait)
- 1280x720 (Landscape)

**sora-2-pro model:**
- 720x1280 (Portrait)
- 1280x720 (Landscape)
- 1024x1792 (HD Portrait)
- 1792x1024 (HD Landscape)

### Pricing Structure

| Model | Resolution | Cost per Second |
|-------|------------|-----------------|
| sora-2 | 720x1280, 1280x720 | $0.10 |
| sora-2-pro | 720x1280, 1280x720 | $0.30 |
| sora-2-pro | 1024x1792, 1792x1024 | $0.50 |

### Workflow

1. Place reference images in `reference_images/` directory
2. Execute `./examples.py 3`
3. Select image from displayed list
4. Choose model (sora-2 or sora-2-pro)
5. Select target resolution
6. Choose duration (4, 8, or 12 seconds)
7. Review cost estimate
8. Enter or select prompt
9. Confirm generation

## Content Guidelines

### API Restrictions

The API enforces content policies that will reject requests containing:

- Copyrighted characters (e.g., Spider-Man, Batman, Disney properties)
- Real people or celebrity likenesses
- Trademarked logos or brand imagery
- Adult or inappropriate content
- Human faces in reference images (strict enforcement)
- Dangerous or violent actions

### Reference Image Guidelines

**Acceptable Images:**
- Landscape and nature scenes
- Abstract patterns and designs
- Objects without human presence
- Architecture and buildings
- Computer-generated test images

**Problematic Images:**
- Photos containing human faces
- Identifiable individuals
- Copyrighted character imagery
- Brand logos or trademarks

### Prompt Recommendations

**Recommended Prompts (Technical/Camera-based):**
- "The camera slowly zooms in by 10 percent"
- "Smooth pan from left to right across the scene"
- "Gentle fade transition to warmer color temperature"
- "Gradual tilt upward revealing more of the scene"
- "Time-lapse transition from day to night"

**Avoid These Prompts:**
- "Person waves goodbye"
- "Character jumps off cliff"
- "Spider-Man swinging through city"
- "Celebrity walking forward"

### Test Image Generation

Generate safe test images without human faces:

```bash
python create_test_image.py
```

This creates:
- `test_landscape.jpeg` - Gradient landscape
- `test_abstract.jpeg` - Abstract pattern
- `test_landscape_wrong_size.jpeg` - For resize testing

## Code Examples

### Basic Video Generation

```python
from sora_client import SoraClient

client = SoraClient()

video = client.create_and_poll(
    prompt="Sunset over mountains",
    model="sora-2",
    size="1280x720",
    seconds="8"
)

if video.status == "completed":
    client.download_video(video.id, "generated_videos/sunset.mp4")
```

### Reference Image with Auto-Resize

```python
# Image automatically resized if dimensions don't match
video = client.create_and_poll(
    prompt="Camera slowly zooms in",
    input_reference="reference_images/landscape.jpg",
    size="720x1280",  # Image resized to match
    seconds="4"
)
```

### Custom Generator Interface

```python
from video_generator import VideoGenerator

generator = VideoGenerator()
generator.generate_custom_video()  # Interactive prompts for all parameters
```

## API Parameters

### Models
- `sora-2` - Standard quality, faster processing
- `sora-2-pro` - Enhanced quality, professional output

### Resolutions
- `1280x720` - HD Landscape (16:9)
- `720x1280` - HD Portrait (9:16)
- `1792x1024` - Wide Landscape (7:4)
- `1024x1792` - Wide Portrait (4:7)

### Durations
- `"4"` - Preview length
- `"8"` - Standard length
- `"12"` - Extended length

Note: Duration parameter must be a string type.

## File Organization

### Input Structure
- Reference images: `reference_images/`
- Supported formats: .jpg, .jpeg, .png, .webp, .bmp

### Output Structure
- Generated videos: `generated_videos/`
- Naming convention: `{type}_{timestamp}_{prompt}_{id}.mp4`
- Example: `ref_20240108_143022_camera_zoom_abc12345.mp4`

### Directory Creation
The following directories are created automatically:
- `reference_images/` - For input images
- `generated_videos/` - For output videos

## Troubleshooting

### Common Issues

**Moderation Block**
- Likely cause: Human face in reference image
- Solution: Use landscape or object images
- Alternative: Generate test images without faces

**Dimension Mismatch**
- Automatic resolution: System auto-resizes images
- Output location: Resized images saved in `reference_images/`

**Reference Image Not Found**
- Check: Images placed in `reference_images/` directory
- Verify: File extension is supported

**API Key Error**
- Solution: Create `.env` file with `OPENAI_API_KEY=your_key`

### Diagnostic Commands

```bash
# Test API connection
./examples.py t

# Safe mode testing with generated images
./examples.py s
```

## Environment Configuration

Create `.env` file in project root:

```bash
OPENAI_API_KEY=sk-...your_key_here...
```

Obtain API key from: https://platform.openai.com/api-keys

## Best Practices

1. Start with generated test images for validation
2. Use technical prompts focusing on camera movements
3. Allow automatic resizing to handle dimension mismatches
4. Review cost estimates before confirming generation
5. Maintain organized file structure using designated directories
6. Test with safe mode before using custom images

## Performance Considerations

- Generation time: 1-5 minutes typical
- sora-2: Faster processing, standard quality
- sora-2-pro: Slower processing, enhanced quality
- Batch processing: Available for multiple concurrent videos
- URL expiration: Download links valid for 24 hours

## License

MIT License