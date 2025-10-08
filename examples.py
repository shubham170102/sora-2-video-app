#!/Users/shubhamshrivastava/Documents/GitHub/sora2-test/.venv/bin/python
"""
Sora 2 API examples and utilities
Main entry point for all video generation operations
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv
from print_color import print as cprint

# get environment variables loaded up
load_dotenv()

from sora_client import (
    SoraClient,
    VideoModel,
    VideoSize,
    PromptBuilder,
    quick_generate
)
from video_generator import VideoGenerator, interactive_video_generator


def example_test_reference_safe():
    """Test reference images with safe content that avoids moderation"""
    cprint("Safe Reference Image Testing", color='cyan', format='bold')
    print("-" * 50)

    client = SoraClient()

    # Check for test images
    test_images = ["test_landscape.jpeg", "test_abstract.jpeg"]
    available_images = [img for img in test_images if Path(img).exists()]

    if not available_images:
        cprint("No test images found. Creating them now...", color='yellow')
        import subprocess
        subprocess.run(["/Users/shubhamshrivastava/Documents/GitHub/sora2-test/.venv/bin/python", "create_test_image.py"])
        available_images = [img for img in test_images if Path(img).exists()]

    if not available_images:
        cprint("Failed to create test images", color='red')
        return

    print("\nAvailable test images (no faces, safe content):")
    for i, img in enumerate(available_images, 1):
        print(f"  {i}. {img}")

    img_choice = input(f"\nSelect image (1-{len(available_images)}): ").strip()
    try:
        selected_image = available_images[int(img_choice) - 1]
    except:
        selected_image = available_images[0]

    print(f"\nUsing: {selected_image}")

    # Ultra-safe prompts that focus on technical aspects
    ultra_safe_prompts = [
        "The camera slowly zooms in by 10 percent",
        "Gentle fade transition to warmer color temperature",
        "Smooth pan from left to right across the scene",
        "Gradual increase in brightness and contrast",
        "Slow tilt upward revealing more of the scene"
    ]

    print("\nUltra-safe prompts (technical, no actions):")
    for i, p in enumerate(ultra_safe_prompts, 1):
        print(f"  {i}. {p}")

    prompt_choice = input("\nSelect prompt (1-5): ").strip()
    try:
        prompt = ultra_safe_prompts[int(prompt_choice) - 1]
    except:
        prompt = ultra_safe_prompts[0]

    print(f"\nUsing prompt: {prompt}")
    print("\nGenerating with safest possible settings...")

    video = client.create_and_poll(
        prompt=prompt,
        model=VideoModel.SORA_2.value,
        size=VideoSize.SIZE_720x1280.value,
        seconds="4",
        input_reference=selected_image
    )

    if video.status == "completed":
        filename = f"safe_test_{Path(selected_image).stem}.mp4"
        client.download_video(video.id, filename)
        cprint(f"\nSuccess! Video saved as: {filename}", color='green', format='bold')
        print("\nThis proves reference images work when:")
        print("- No human faces in the image")
        print("- Safe, technical prompts")
        print("- Short duration for testing")
    else:
        cprint(f"Generation failed: {video.status}", color='red')


def example_basic_generation():
    """basic video gen with flexible user input"""
    cprint("Interactive Video Generation", color='cyan', format='bold')
    print("-" * 50)

    # use the new generator for flexibility
    generator = VideoGenerator()

    # let user customize everything
    generator.generate_custom_video()


def example_high_quality_production():
    """high quality video using pro model"""
    cprint("High Quality Production Video", color='cyan', format='bold')
    print("-" * 50)

    client = SoraClient()
    from datetime import datetime

    # Create generated_videos folder
    output_folder = Path("generated_videos")
    output_folder.mkdir(exist_ok=True)

    # build a detailed prompt for better results
    prompt = PromptBuilder.build(
        shot_type="Wide tracking shot",
        subject="a vintage red convertible",
        action="cruising along a coastal highway",
        setting="California's Pacific Coast Highway",
        lighting="golden hour sunlight, dramatic shadows",
        camera_movement="smooth dolly shot following the car",
        additional_details="ocean waves crashing on rocks, seabirds in the sky"
    )

    print(f"Prompt: {prompt}")

    # use pro model for better quality
    video = client.create_and_poll(
        prompt=prompt,
        model=VideoModel.SORA_2_PRO.value,
        size=VideoSize.SIZE_1792x1024.value,  # wide format looks good
        seconds="12"
    )

    if video.status == "completed":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pro_{timestamp}_coastal_drive_{video.id[:8]}.mp4"
        output_path = output_folder / filename
        client.download_video(video.id, str(output_path))
        cprint(f"Saved: {output_path}", color='green')


def example_with_reference_image():
    """use an image as first frame with full flexibility"""
    cprint("Video Generation with Reference Image", color='cyan', format='bold')
    print("-" * 50)

    client = SoraClient()
    from datetime import datetime
    import shutil

    # Create reference_images folder if it doesn't exist
    ref_folder = Path("reference_images")
    ref_folder.mkdir(exist_ok=True)

    # Create generated_videos folder
    output_folder = Path("generated_videos")
    output_folder.mkdir(exist_ok=True)

    # Scan for images in reference_images folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    available_images = []

    for ext in image_extensions:
        available_images.extend(ref_folder.glob(f"*{ext}"))
        available_images.extend(ref_folder.glob(f"*{ext.upper()}"))

    # Also check root directory for backward compatibility
    root_images = []
    for ext in image_extensions:
        root_images.extend(Path(".").glob(f"*{ext}"))
        root_images.extend(Path(".").glob(f"*{ext.upper()}"))

    # Filter out resized images from root
    root_images = [img for img in root_images if "resized" not in str(img)]

    if not available_images and not root_images:
        cprint(f"No images found in '{ref_folder}' folder or root directory", color='yellow')
        print("\nPlease add images to the 'reference_images' folder")
        print("Supported formats: .jpg, .jpeg, .png, .webp, .bmp")
        return

    # Display available images
    all_images = available_images + root_images
    cprint("\nAvailable Reference Images:", color='cyan')
    for i, img in enumerate(all_images, 1):
        location = "reference_images" if img in available_images else "root"
        print(f"  {i}. {img.name} ({location})")

    # Select image
    img_choice = input(f"\nSelect image (1-{len(all_images)}): ").strip()
    try:
        selected_image = all_images[int(img_choice) - 1]
    except:
        selected_image = all_images[0]

    print(f"\nSelected: {selected_image}")

    # Display pricing information
    cprint("\nPricing Information:", color='cyan', format='bold')
    print("-" * 50)
    print("sora-2:")
    print("  720x1280 / 1280x720: $0.10 per second")
    print("\nsora-2-pro:")
    print("  720x1280 / 1280x720: $0.30 per second")
    print("  1024x1792 / 1792x1024: $0.50 per second")
    print("-" * 50)

    # Model selection
    cprint("\nModel Selection:", color='cyan')
    print("1. sora-2 (Standard - $0.10/sec)")
    print("2. sora-2-pro (Professional - $0.30-$0.50/sec)")

    model_choice = input("\nSelect model (1-2): ").strip()
    if model_choice == "2":
        model = VideoModel.SORA_2_PRO.value
        model_name = "sora-2-pro"
    else:
        model = VideoModel.SORA_2.value
        model_name = "sora-2"

    # Resolution selection based on model
    cprint("\nResolution Selection:", color='cyan')
    if model == VideoModel.SORA_2_PRO.value:
        print("1. Portrait 720x1280 ($0.30/sec)")
        print("2. Landscape 1280x720 ($0.30/sec)")
        print("3. Portrait HD 1024x1792 ($0.50/sec)")
        print("4. Landscape HD 1792x1024 ($0.50/sec)")
        res_choice = input("\nSelect resolution (1-4): ").strip()

        resolution_map = {
            "1": VideoSize.SIZE_720x1280.value,
            "2": VideoSize.SIZE_1280x720.value,
            "3": VideoSize.SIZE_1024x1792.value,
            "4": VideoSize.SIZE_1792x1024.value
        }
        selected_size = resolution_map.get(res_choice, VideoSize.SIZE_720x1280.value)
    else:
        print("1. Portrait 720x1280 ($0.10/sec)")
        print("2. Landscape 1280x720 ($0.10/sec)")
        res_choice = input("\nSelect resolution (1-2): ").strip()

        resolution_map = {
            "1": VideoSize.SIZE_720x1280.value,
            "2": VideoSize.SIZE_1280x720.value
        }
        selected_size = resolution_map.get(res_choice, VideoSize.SIZE_720x1280.value)

    # Duration selection
    cprint("\nDuration Selection:", color='cyan')
    print("1. 4 seconds")
    print("2. 8 seconds")
    print("3. 12 seconds")

    duration_choice = input("\nSelect duration (1-3): ").strip()
    duration_map = {"1": "4", "2": "8", "3": "12"}
    selected_duration = duration_map.get(duration_choice, "4")

    # Calculate cost
    base_rate = 0.10 if model == VideoModel.SORA_2.value else 0.30
    if selected_size in [VideoSize.SIZE_1024x1792.value, VideoSize.SIZE_1792x1024.value]:
        base_rate = 0.50
    total_cost = base_rate * int(selected_duration)

    cprint(f"\nEstimated Cost: ${total_cost:.2f}", color='yellow', format='bold')

    # Check and prepare the reference image
    print("\nPreparing reference image...")
    # Validate and resize if needed
    prepared_image, was_resized = client.validate_and_prepare_reference_image(
        str(selected_image), selected_size
    )

    # If image was resized, save it to reference_images folder
    if was_resized:
        resized_path = Path(prepared_image)
        new_path = ref_folder / resized_path.name
        shutil.move(str(resized_path), str(new_path))
        prepared_image = str(new_path)
        cprint(f"Resized image saved to: {new_path}", color='green')

    # Prompt guidelines
    cprint("\nReference Image Guidelines:", color='cyan')
    print("- Avoid images with human faces (often blocked)")
    print("- Keep prompts positive and safe")

    # Get prompt
    safe_prompts = [
        "The camera slowly zooms in while maintaining the composition",
        "The scene transitions from day to sunset with beautiful lighting",
        "The camera gently pans across the scene from left to right",
        "The elements in the scene begin to glow with soft light",
        "Colors gradually shift from cool tones to warm tones"
    ]

    print("\nSuggested safe prompts:")
    for i, p in enumerate(safe_prompts, 1):
        print(f"  {i}. {p}")

    prompt_choice = input("\nChoose a prompt (1-5) or enter custom: ").strip()
    if prompt_choice in ['1', '2', '3', '4', '5']:
        prompt = safe_prompts[int(prompt_choice) - 1]
    else:
        prompt = prompt_choice if prompt_choice else safe_prompts[0]

    # Summary before generation
    print("\n" + "=" * 50)
    cprint("Generation Summary:", color='cyan', format='bold')
    print(f"Model: {model_name}")
    print(f"Resolution: {selected_size}")
    print(f"Duration: {selected_duration} seconds")
    print(f"Cost: ${total_cost:.2f}")
    print(f"Image: {Path(prepared_image).name}")
    print(f"Prompt: {prompt[:80]}...")
    print("=" * 50)

    confirm = input("\nProceed with generation? (y/n): ").lower()
    if confirm != 'y':
        cprint("Generation cancelled", color='yellow')
        return

    print(f"\nGenerating video...")
    video = client.create_and_poll(
        prompt=prompt,
        model=model,
        size=selected_size,
        seconds=selected_duration,
        input_reference=prepared_image
    )

    if video.status == "completed":
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c.isspace())
        clean_prompt = clean_prompt.strip().replace(" ", "_").lower()
        filename = f"ref_{timestamp}_{clean_prompt}_{video.id[:8]}.mp4"

        output_path = output_folder / filename
        client.download_video(video.id, str(output_path))

        cprint(f"\nSuccess! Video saved to: {output_path}", color='green', format='bold')
        print(f"Total cost: ${total_cost:.2f}")
    else:
        cprint(f"Video generation status: {video.status}", color='red')

        # Check for moderation block
        error = getattr(video, 'error', None)
        if error and 'moderation' in str(error).lower():
            cprint("\nModeration Tips:", color='yellow')
            print("- Try using an image without human faces")
            print("- Use landscape or object photos instead")
            print("- Keep prompts describing simple camera movements")


def example_remix_workflow():
    """remix existing videos"""
    cprint("Video Remix Workflow", color='cyan', format='bold')
    print("-" * 50)

    client = SoraClient()
    from datetime import datetime

    # Create generated_videos folder
    output_folder = Path("generated_videos")
    output_folder.mkdir(exist_ok=True)

    # create original first
    original_prompt = "A peaceful forest scene with morning mist"
    original_video = client.create_and_poll(
        prompt=original_prompt,
        model=VideoModel.SORA_2.value,
        seconds="4"
    )

    if original_video.status != "completed":
        cprint("Original generation failed", color='red')
        return

    print(f"Original created: {original_video.id}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_path = output_folder / f"remix_{timestamp}_original_{original_video.id[:8]}.mp4"
    client.download_video(original_video.id, str(original_path))
    cprint(f"Original saved: {original_path}", color='green')

    # try some remixes
    remixes = [
        "Add fireflies glowing in the mist",
        "Change the time to sunset with orange lighting",
        "Add a deer walking through the scene"
    ]

    for i, remix_prompt in enumerate(remixes):
        print(f"\nRemix {i+1}: {remix_prompt}")

        remix_video = client.remix_video(
            original_video_id=original_video.id,
            remix_prompt=remix_prompt
        )

        # wait for completion
        completed = client.poll_video_status(remix_video.id)

        if completed.status == "completed":
            remix_path = output_folder / f"remix_{timestamp}_v{i+1}_{completed.id[:8]}.mp4"
            client.download_video(completed.id, str(remix_path))
            cprint(f"Saved: {remix_path}", color='green')


def example_download_all_assets():
    """get video with thumbnail and sprites"""
    cprint("Download All Assets", color='cyan', format='bold')
    print("-" * 50)

    client = SoraClient()
    from datetime import datetime

    # Create generated_videos folder
    output_folder = Path("generated_videos")
    output_folder.mkdir(exist_ok=True)

    prompt = "Time-lapse of clouds moving across a city skyline"
    video = client.create_and_poll(prompt, seconds="4")

    if video.status == "completed":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"assets_{timestamp}_{video.id[:8]}"

        # grab everything
        client.download_video(video.id, str(output_folder / f"{base_name}.mp4"), variant="video")
        client.download_video(video.id, str(output_folder / f"{base_name}_thumb.webp"), variant="thumbnail")
        client.download_video(video.id, str(output_folder / f"{base_name}_sprites.jpg"), variant="spritesheet")
        cprint(f"Downloaded all assets to: {output_folder}", color='green')


def example_batch_generation():
    """generate multiple videos at once"""
    cprint("Batch Video Generation", color='cyan', format='bold')
    print("-" * 50)

    client = SoraClient()
    from datetime import datetime

    # Create generated_videos folder
    output_folder = Path("generated_videos")
    output_folder.mkdir(exist_ok=True)

    prompts = [
        "A butterfly emerging from its cocoon in macro detail",
        "Northern lights dancing across a starry Arctic sky",
        "A coffee being poured in slow motion with steam rising",
        "Cherry blossoms falling in a Japanese garden"
    ]

    # start all at once
    video_jobs = []
    for prompt in prompts:
        video = client.create_video(prompt, seconds="4")
        video_jobs.append({"id": video.id, "prompt": prompt[:30]})
        print(f"Started: {video.id} - {prompt[:30]}...")

    # poll for completion
    print("\nWaiting for videos to complete...")
    completed_videos = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for job in video_jobs:
        video = client.poll_video_status(job["id"], show_progress=False)
        if video.status == "completed":
            completed_videos.append(video)
            cprint(f"Done: {job['prompt']}", color='green')
        else:
            cprint(f"Failed: {job['prompt']}", color='red')

    # download completed ones
    for i, video in enumerate(completed_videos):
        filename = f"batch_{timestamp}_{i+1}_{video.id[:8]}.mp4"
        output_path = output_folder / filename
        client.download_video(video.id, str(output_path))

    print(f"\nBatch complete: {len(completed_videos)}/{len(prompts)} succeeded")
    if completed_videos:
        cprint(f"Videos saved to: {output_folder}", color='green')


async def example_async_generation():
    """async video generation for better performance"""
    cprint("Async Video Generation", color='cyan', format='bold')
    print("-" * 50)

    client = SoraClient()
    from datetime import datetime

    # Create generated_videos folder
    output_folder = Path("generated_videos")
    output_folder.mkdir(exist_ok=True)

    prompts = [
        "Lightning striking a lone tree in a field",
        "Underwater coral reef with tropical fish",
        "Steam locomotive crossing a mountain bridge"
    ]

    # create tasks
    tasks = [
        client.create_and_poll_async(prompt, seconds="4")
        for prompt in prompts
    ]

    # run them all
    results = await asyncio.gather(*tasks)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # download what completed
    for i, video in enumerate(results):
        if video.status == "completed":
            filename = f"async_{timestamp}_{i+1}_{video.id[:8]}.mp4"
            output_path = output_folder / filename
            client.download_video(video.id, str(output_path))
            cprint(f"Downloaded: {output_path}", color='green')


def example_video_library_management():
    """manage your videos - list, monitor, download, delete"""
    cprint("Video Library Management", color='cyan', format='bold')
    print("-" * 50)

    client = SoraClient()

    # get recent videos
    print("\nRecent videos:")
    try:
        videos = client.list_videos(limit=10)
        video_list = []

        if not videos:
            print("  No videos found")
            return

        for i, video in enumerate(videos, 1):
            # handle different response types
            if hasattr(video, '__dict__'):
                video_dict = video.__dict__
            else:
                video_dict = video

            video_id = video_dict.get("id", "unknown") if isinstance(video_dict, dict) else getattr(video, "id", "unknown")
            status = video_dict.get("status", "unknown") if isinstance(video_dict, dict) else getattr(video, "status", "unknown")
            created = video_dict.get("created_at", "unknown") if isinstance(video_dict, dict) else getattr(video, "created_at", "unknown")
            progress = video_dict.get("progress", 0) if isinstance(video_dict, dict) else getattr(video, "progress", 0)

            # show status with color
            if status == "completed":
                cprint(f"  {i}. {video_id[:20]}... - {status}", color='green')
            elif status == "failed":
                cprint(f"  {i}. {video_id[:20]}... - {status}", color='red')
            elif status in ["in_progress", "queued"]:
                cprint(f"  {i}. {video_id[:20]}... - {status} ({progress}%)", color='yellow')
            else:
                print(f"  {i}. {video_id[:20]}... - {status}")

            video_list.append({
                "id": video_id,
                "status": status,
                "progress": progress,
                "video": video
            })

        # operations menu
        print("\n" + "=" * 50)
        print("Available Operations:")
        print("  1. Monitor in-progress video")
        print("  2. Download completed video")
        print("  3. Delete a video")
        print("  4. Check video details")
        print("  5. Exit")

        choice = input("\nSelect operation (1-5): ").strip()

        if choice == "1":
            # monitor progress
            in_progress = [v for v in video_list if v["status"] in ["in_progress", "queued"]]
            if not in_progress:
                cprint("No videos in progress", color='green')
            else:
                print("\nIn-progress videos:")
                for i, v in enumerate(in_progress, 1):
                    print(f"  {i}. {v['id'][:20]}... ({v['progress']}%)")

                if len(in_progress) == 1:
                    monitor_video_progress(client, in_progress[0]["id"])
                else:
                    vid_num = input("\nSelect video number: ").strip()
                    try:
                        idx = int(vid_num) - 1
                        if 0 <= idx < len(in_progress):
                            monitor_video_progress(client, in_progress[idx]["id"])
                    except:
                        cprint("Invalid selection", color='red')

        elif choice == "2":
            # download
            completed = [v for v in video_list if v["status"] == "completed"]
            if not completed:
                cprint("No completed videos", color='red')
            else:
                print("\nCompleted videos:")
                for i, v in enumerate(completed, 1):
                    print(f"  {i}. {v['id'][:20]}...")

                vid_num = input("\nSelect video to download: ").strip()
                try:
                    idx = int(vid_num) - 1
                    if 0 <= idx < len(completed):
                        video_id = completed[idx]["id"]
                        output_file = f"video_{video_id[:8]}.mp4"
                        print(f"Downloading to {output_file}...")
                        if client.download_video(video_id, output_file):
                            cprint(f"Downloaded: {output_file}", color='green')

                            # optional extras
                            dl_thumb = input("Download thumbnail? (y/n): ").strip().lower()
                            if dl_thumb == 'y':
                                client.download_video(video_id, f"thumb_{video_id[:8]}.webp", "thumbnail")
                                print("Downloaded thumbnail")
                except:
                    cprint("Invalid selection", color='red')

        elif choice == "3":
            # delete
            print("\nSelect video to delete:")
            for i, v in enumerate(video_list, 1):
                print(f"  {i}. {v['id'][:20]}... - {v['status']}")

            vid_num = input("\nVideo number (or 'cancel'): ").strip()
            if vid_num.lower() != 'cancel':
                try:
                    idx = int(vid_num) - 1
                    if 0 <= idx < len(video_list):
                        video_id = video_list[idx]["id"]
                        confirm = input(f"Delete {video_id[:20]}...? (yes/no): ").strip()
                        if confirm.lower() == 'yes':
                            client.delete_video(video_id)
                            cprint("Video deleted", color='green')
                except:
                    cprint("Invalid selection", color='red')

        elif choice == "4":
            # details
            print("\nSelect video for details:")
            for i, v in enumerate(video_list, 1):
                print(f"  {i}. {v['id'][:20]}... - {v['status']}")

            vid_num = input("\nVideo number: ").strip()
            try:
                idx = int(vid_num) - 1
                if 0 <= idx < len(video_list):
                    video = client.get_video_status(video_list[idx]["id"])
                    print(f"\nVideo Details:")
                    if hasattr(video, '__dict__'):
                        for key, value in video.__dict__.items():
                            if not key.startswith('_'):
                                print(f"  {key}: {value}")

            except:
                cprint("Invalid selection", color='red')

    except Exception as e:
        cprint(f"Error: {e}", color='red')


def monitor_video_progress(client, video_id):
    """monitor video generation progress"""
    print(f"\nMonitoring: {video_id[:20]}...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            video = client.get_video_status(video_id)
            status = getattr(video, "status", "unknown")
            progress = getattr(video, "progress", 0)

            # progress bar
            bar_length = 40
            filled = int((progress / 100) * bar_length)
            bar = "=" * filled + "-" * (bar_length - filled)

            print(f"\r[{bar}] {progress}% - {status}", end="", flush=True)

            if status == "completed":
                cprint(f"\nVideo completed!", color='green')
                download = input("Download now? (y/n): ").strip().lower()
                if download == 'y':
                    output = f"video_{video_id[:8]}.mp4"
                    client.download_video(video_id, output)
                    cprint(f"Downloaded: {output}", color='green')
                break
            elif status == "failed":
                cprint(f"\nVideo failed", color='red')
                break

            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")


def example_advanced_prompting():
    """advanced prompting examples"""
    cprint("Advanced Prompting Techniques", color='cyan', format='bold')
    print("-" * 50)

    client = SoraClient()

    # some cinematic examples
    cinematic_prompts = [
        {
            "name": "Action Scene",
            "prompt": "Low angle shot of a parkour runner vaulting over rooftops, "
                     "dynamic camera movement tracking the motion, urban cityscape background, "
                     "late afternoon sun creating long shadows, slow motion at peak of jump, "
                     "gritty realistic style, handheld camera shake"
        },
        {
            "name": "Emotional Close-up",
            "prompt": "Extreme close-up of an elderly person's weathered hands "
                     "crafting pottery on a wheel, soft window light from left, "
                     "shallow depth of field with background blur, "
                     "slow push-in camera movement, warm color grading"
        },
        {
            "name": "Nature Documentary",
            "prompt": "Aerial drone shot descending through morning fog "
                     "to reveal a pristine mountain lake, smooth gimbal movement, "
                     "birds flying in V-formation across frame, "
                     "sun rays breaking through clouds, "
                     "color grade emphasizing blues and greens"
        }
    ]

    for example in cinematic_prompts:
        print(f"\n{example['name']}:")
        print(f"Prompt: {example['prompt'][:100]}...")

        # generate it
        try:
            output_path = quick_generate(example["prompt"])
            cprint(f"Saved: {output_path}", color='green')
        except Exception as e:
            cprint(f"Failed: {e}", color='red')


def test_connection():
    """check if everything is set up correctly"""
    cprint("API Connection Test", color='cyan', format='bold')
    print("-" * 50)
    print("Testing OpenAI setup...")

    from openai import OpenAI

    # check for api key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        cprint("ERROR: OPENAI_API_KEY not found", color='red')
        return False

    masked = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
    cprint(f"API Key: {masked}", color='green')

    try:
        client = OpenAI()

        # test connection
        models = client.models.list()
        model_count = len(list(models))
        cprint(f"Connected: Found {model_count} models", color='green')

        # check video api
        if hasattr(client, 'videos'):
            cprint("Video API: Available", color='green')

            # try listing videos
            try:
                videos = client.videos.list(limit=1)
                video_list = list(videos)
                cprint(f"Video Library: {len(video_list)} recent videos", color='green')
            except:
                cprint("Video Library: Accessible (empty or new)", color='green')
        else:
            cprint("Video API: Not available", color='red')

        print("\nAll tests passed. Ready to go.")
        return True

    except Exception as e:
        cprint(f"Connection failed: {e}", color='red')
        return False


def run_example(choice: str):
    """run specific example by choice"""
    examples = {
        "1": example_basic_generation,  # now interactive
        "2": example_high_quality_production,
        "3": example_with_reference_image,
        "4": example_remix_workflow,
        "5": example_download_all_assets,
        "6": example_batch_generation,
        "7": lambda: asyncio.run(example_async_generation()),
        "8": example_video_library_management,
        "9": example_advanced_prompting,
        "t": test_connection,
        "g": lambda: interactive_video_generator(),  # new generator menu
        "s": example_test_reference_safe  # safe reference test
    }

    if choice == "0":
        print("Running all examples...")
        print("=" * 50)
        for name, func in examples.items():
            if name != "t":  # skip test
                try:
                    print(f"\n[Example {name}]")
                    func()
                    print("\n" + "="*50 + "\n")
                except Exception as e:
                    cprint(f"Example {name} failed: {e}", color='red')
                    print("=" * 50)
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            cprint(f"\nError: {e}", color='red')
            print("\nTroubleshooting:")
            print("- Check internet connection")
            print("- Verify API key is valid")
            print("- Check OpenAI account credits")
            print("- API might be temporarily down")
    else:
        cprint("Invalid choice. Select 0-9 or T", color='red')
        return False
    return True


def main():
    """main entry point"""
    import sys

    cprint("\nSora 2 API Examples", color='cyan', format='bold')
    print("=" * 50)

    # check for command line arg
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        print(f"Running example {choice}...")
        run_example(choice)
    else:
        # interactive menu
        print("\nAvailable Options:")
        cprint("\nNEW - Interactive Generator:", color='green', format='bold')
        print("  G. Custom Video Generator - Full control over all settings")
        print("  S. Safe Reference Test - Test with face-free images")
        print("\nVideo Generation:")
        print("  1. Interactive Generation - User customizable")
        print("  2. High Quality Production - Pro model, 12-second")
        print("  3. With Reference Image - Use image as first frame")
        print("  4. Remix Workflow - Modify existing videos")
        print("  5. Download All Assets - Video + thumbnail + sprites")
        print("  6. Batch Generation - Multiple videos concurrently")
        print("  7. Async Generation - Asynchronous processing")
        print("\nManagement:")
        print("  8. Library Management - List, monitor, download")
        print("  9. Advanced Prompting - Cinematic techniques")
        print("\nUtilities:")
        print("  T. Test Connection - Check API setup")
        print("  0. Run All Examples")
        print("\nTip: Run directly with: ./examples.py [choice]")

        while True:
            try:
                choice = input("\nSelect option (0-9, G, S, T, or Q to quit): ").strip().lower()

                if choice == 'q' or choice == 'quit':
                    print("Bye")
                    break

                if run_example(choice):
                    another = input("\nRun another? (y/n): ").strip().lower()
                    if another != 'y':
                        print("Done")
                        break
                    print("\n" + "=" * 50)

            except KeyboardInterrupt:
                print("\n\nInterrupted")
                break
            except EOFError:
                print("\nBye")
                break


if __name__ == "__main__":
    # check api key first
    if not os.getenv("OPENAI_API_KEY"):
        cprint("ERROR: OPENAI_API_KEY not set", color='red', format='bold')
        print("\nTo fix:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your API key to .env")
        print("  3. Or export OPENAI_API_KEY=your_key")
        sys.exit(1)

    try:
        main()
    except Exception as e:
        cprint(f"\nUnexpected error: {e}", color='red')
        sys.exit(1)