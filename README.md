# Sora 2 API Python Implementation

Comprehensive Python implementation for OpenAI's Sora 2 video generation API based on official documentation.

## Features

### Core Functionality
- **Video Generation**: Create videos from text prompts using sora-2 or sora-2-pro models
- **Status Monitoring**: Poll and track video generation progress with live updates
- **Asset Downloads**: Download videos, thumbnails, and spritesheets
- **Reference Images**: Use images as the first frame for video generation
- **Video Remixing**: Create variations of existing videos
- **Library Management**: List, filter, monitor, and delete videos

### Advanced Features
- **Batch Processing**: Generate multiple videos concurrently with queue management
- **Async Support**: Asynchronous video generation for better performance
- **Webhook Handling**: Process webhook notifications for completed/failed videos
- **Caching System**: Avoid duplicate generations with smart caching
- **Progress Monitoring**: Real-time progress bars with colored output
- **Retry Logic**: Automatic retry on failures with exponential backoff
- **Platform Optimization**: Auto-configure parameters for Instagram, YouTube, Twitter

## Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd sora2-test

# Install dependencies using uv (recommended)
uv pip install python-dotenv openai print-color

# Or using pip with venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install python-dotenv openai print-color

# Set up API key
cp .env.example .env
# Edit .env and add your API key
```

### Run Everything from One File

```bash
# Interactive mode
./examples.py  # or .venv/bin/python examples.py

# Direct commands
./examples.py 1  # Basic video generation
./examples.py 2  # High quality production
./examples.py 3  # With reference image
./examples.py 4  # Remix workflow
./examples.py 5  # Download all assets
./examples.py 6  # Batch generation
./examples.py 7  # Async generation
./examples.py 8  # VIDEO LIBRARY MANAGEMENT (monitor, download, delete)
./examples.py 9  # Advanced prompting
./examples.py t  # Test API connection
```

**Library Management Features (Option 8):**
- List all videos with status indicators
- Monitor in-progress videos with live progress bar
- Download completed videos with optional thumbnails
- Delete unwanted videos
- Check detailed video information

## Project Structure

```
sora2-test/
├── examples.py       # Main entry point - all features in one place
│                     # Video generation examples (1-7)
│                     # Video library management (8)
│                     # Connection testing (T)
│                     # Monitor, download, delete videos
├── sora_client.py    # Core Sora API client implementation
├── video_utils.py    # Advanced utilities (caching, queues, webhooks)
├── .env             # Your API key (not committed to git)
├── .env.example     # Template for environment variables
├── .gitignore       # Git ignore configuration
├── pyproject.toml   # Project dependencies
└── README.md        # This documentation
```

**Only 3 Python files needed.** Everything accessible through examples.py.

## API Documentation

### Models

- **sora-2**: Fast generation for exploration and iteration
- **sora-2-pro**: Higher quality, production-ready output

### Video Parameters

**Supported Sizes:**
- `1280x720` - HD Landscape
- `720x1280` - HD Portrait (vertical)
- `1792x1024` - Wide Landscape
- `1024x1792` - Wide Portrait (vertical)

**Duration Options:**
- `"4"` seconds - Quick preview
- `"8"` seconds - Standard length
- `"12"` seconds - Extended content

### Content Restrictions

- Content must be suitable for audiences under 18
- No copyrighted characters or music
- No real people or public figures
- No faces in input reference images

## Usage Examples

### Basic Video Generation

```python
from sora_client import SoraClient

client = SoraClient()

# Generate video
video = client.create_and_poll(
    prompt="A serene sunset over mountain peaks",
    model="sora-2",
    size="1280x720",
    seconds="8"
)

# Download if successful
if video.status == "completed":
    client.download_video(video.id, "sunset.mp4")
```

### With Reference Image

```python
video = client.create_and_poll(
    prompt="Character turns and walks forward",
    input_reference="character.jpg",
    seconds="4"
)
```

### Video Remixing

```python
# Create remix of existing video
remix = client.remix_video(
    original_video_id="video_abc123",
    remix_prompt="Change color palette to warm tones"
)
```

### Batch Processing

```python
from video_utils import VideoQueue

queue = VideoQueue(client, max_concurrent=3)

# Add tasks to queue
queue.add_task("Sunset over ocean", seconds="4")
queue.add_task("City skyline timelapse", seconds="8")
queue.add_task("Forest in autumn", seconds="4")

# Process queue
queue.process_queue()
```

### Platform-Optimized Generation

```python
from video_utils import VideoOptimizer

# Get optimized parameters for Instagram Reels
params = VideoOptimizer.optimize_parameters(
    prompt="Dancing in the rain",
    target_platform="instagram_reel"
)

video = client.create_and_poll(**params)
```

## API Endpoints

The client implements all Sora API endpoints:

- `POST /videos` - Create new video
- `GET /videos/{id}` - Get video status
- `GET /videos/{id}/content` - Download video/assets
- `GET /videos` - List all videos
- `DELETE /videos/{id}` - Delete video
- `POST /videos/{id}/remix` - Create remix

## Effective Prompting

Best results with structured prompts including:
- **Shot type**: Wide, close-up, medium, aerial
- **Subject**: Clear description of main focus
- **Action**: What's happening in the scene
- **Setting**: Environment and location
- **Lighting**: Time of day, mood, atmosphere
- **Camera movement**: Static, pan, dolly, tracking

Example:
```python
prompt = "Wide tracking shot of a vintage red car cruising along coastal highway, golden hour sunlight, smooth dolly movement"
```

## Environment Variables

Create a `.env` file with:

```bash
OPENAI_API_KEY=your_api_key_here
```

Get your API key from: https://platform.openai.com/api-keys

## Performance Tips

- Video generation typically takes 1-5 minutes depending on model and parameters
- Use `sora-2` for rapid prototyping and iteration
- Use `sora-2-pro` for final production output
- Batch multiple requests for efficiency
- Implement caching to avoid duplicate generations

## Important Notes

- Download URLs expire after 24 hours
- The API is currently in beta and subject to changes
- Rate limits apply based on your account tier
- Videos are stored temporarily on OpenAI servers

## Troubleshooting

If you encounter issues:
1. Check your internet connection
2. Verify your API key is valid and has credits
3. Ensure you're using correct video parameters (sizes and durations)
4. Check if the Sora API is temporarily unavailable
5. Review error messages for specific issues (content policy, rate limits)

## Contributing

Feel free to submit issues and enhancement requests.

## License

MIT License - use freely in your projects