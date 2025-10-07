#!/usr/bin/env python3
"""
Flexible video generation utilities
Handles user input, aspect ratios, durations, and unique naming
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple
from print_color import print as cprint

from sora_client import SoraClient, VideoModel, VideoSize


class VideoGenerator:
    """handles video generation with user customization"""

    # aspect ratio mappings with descriptions
    ASPECT_RATIOS = {
        "1": {
            "name": "Landscape HD",
            "size": VideoSize.SIZE_1280x720.value,
            "description": "16:9 - Standard widescreen (YouTube, TV)",
            "best_for": "General content, presentations, tutorials"
        },
        "2": {
            "name": "Portrait HD",
            "size": VideoSize.SIZE_720x1280.value,
            "description": "9:16 - Vertical format",
            "best_for": "Instagram Reels, TikTok, YouTube Shorts"
        },
        "3": {
            "name": "Wide Landscape",
            "size": VideoSize.SIZE_1792x1024.value,
            "description": "7:4 - Cinematic wide",
            "best_for": "Cinematic shots, panoramic views"
        },
        "4": {
            "name": "Wide Portrait",
            "size": VideoSize.SIZE_1024x1792.value,
            "description": "4:7 - Tall vertical",
            "best_for": "Mobile-first content, stories"
        }
    }

    # duration options with use cases
    DURATIONS = {
        "1": {
            "seconds": "4",
            "name": "Quick Preview",
            "description": "4 seconds - Fast iteration and testing",
            "credits": "Lower cost"
        },
        "2": {
            "seconds": "8",
            "name": "Standard",
            "description": "8 seconds - Balanced length for most content",
            "credits": "Medium cost"
        },
        "3": {
            "seconds": "12",
            "name": "Extended",
            "description": "12 seconds - Longer scenes and narratives",
            "credits": "Higher cost"
        }
    }

    # model selection
    MODELS = {
        "1": {
            "model": VideoModel.SORA_2.value,
            "name": "Sora 2 Standard",
            "description": "Faster generation, good quality",
            "best_for": "Prototyping, drafts, social media"
        },
        "2": {
            "model": VideoModel.SORA_2_PRO.value,
            "name": "Sora 2 Pro",
            "description": "Highest quality, slower generation",
            "best_for": "Final production, marketing, professional use"
        }
    }

    def __init__(self):
        self.client = SoraClient()

    def display_aspect_ratios(self):
        """show available aspect ratios with guidelines"""
        cprint("\nAspect Ratio Options:", color='cyan', format='bold')
        print("-" * 60)

        for key, ratio in self.ASPECT_RATIOS.items():
            print(f"\n  {key}. {ratio['name']} ({ratio['size']})")
            print(f"     {ratio['description']}")
            cprint(f"     Best for: {ratio['best_for']}", color='gray')

    def display_durations(self):
        """show duration options with use cases"""
        cprint("\nDuration Options:", color='cyan', format='bold')
        print("-" * 60)

        for key, duration in self.DURATIONS.items():
            print(f"\n  {key}. {duration['name']}")
            print(f"     {duration['description']}")
            cprint(f"     Credits: {duration['credits']}", color='gray')

    def display_models(self):
        """show model options"""
        cprint("\nModel Options:", color='cyan', format='bold')
        print("-" * 60)

        for key, model in self.MODELS.items():
            print(f"\n  {key}. {model['name']}")
            print(f"     {model['description']}")
            cprint(f"     Best for: {model['best_for']}", color='gray')

    def display_prompt_guidelines(self):
        """show prompting best practices"""
        cprint("\nPrompt Guidelines:", color='cyan', format='bold')
        print("-" * 60)

        # Content restrictions first - important!
        cprint("\nIMPORTANT - Content Restrictions:", color='red', format='bold')
        print("  - NO copyrighted characters (Spider-Man, Batman, Mickey Mouse, etc.)")
        print("  - NO real people or celebrities")
        print("  - NO trademarked logos or brands")
        print("  - NO inappropriate or adult content")
        print("  - Must be suitable for all audiences")

        print("\nInclude these elements for best results:")
        print("  - Shot type: wide, close-up, aerial, tracking")
        print("  - Subject: clear description of main focus")
        print("  - Action: what's happening in the scene")
        print("  - Setting: environment and location")
        print("  - Lighting: time of day, mood")
        print("  - Style: realistic, animated, cinematic")

        print("\nExample prompts:")
        cprint("  'Wide shot of a surfer riding a wave at sunset, golden hour lighting'", color='gray')
        cprint("  'Close-up of coffee being poured, steam rising, shallow depth of field'", color='gray')
        cprint("  'Aerial view of autumn forest, drone slowly moving forward, misty morning'", color='gray')

    def get_user_prompt(self) -> str:
        """get prompt from user with guidelines"""
        self.display_prompt_guidelines()

        print("\n" + "-" * 60)
        prompt = input("\nEnter your prompt: ").strip()

        # validate prompt
        if not prompt:
            cprint("Error: Prompt cannot be empty", color='red')
            return self.get_user_prompt()

        if len(prompt) < 10:
            cprint("Warning: Very short prompt may produce unpredictable results", color='yellow')
            confirm = input("Continue anyway? (y/n): ").lower()
            if confirm != 'y':
                return self.get_user_prompt()

        return prompt

    def select_aspect_ratio(self) -> str:
        """let user select aspect ratio"""
        self.display_aspect_ratios()

        choice = input("\nSelect aspect ratio (1-4): ").strip()

        if choice not in self.ASPECT_RATIOS:
            cprint("Invalid choice. Please select 1-4", color='red')
            return self.select_aspect_ratio()

        selected = self.ASPECT_RATIOS[choice]
        cprint(f"Selected: {selected['name']}", color='green')
        return selected['size']

    def select_duration(self) -> str:
        """let user select duration"""
        self.display_durations()

        choice = input("\nSelect duration (1-3): ").strip()

        if choice not in self.DURATIONS:
            cprint("Invalid choice. Please select 1-3", color='red')
            return self.select_duration()

        selected = self.DURATIONS[choice]
        cprint(f"Selected: {selected['name']}", color='green')
        return selected['seconds']

    def select_model(self) -> str:
        """let user select model"""
        self.display_models()

        choice = input("\nSelect model (1-2): ").strip()

        if choice not in self.MODELS:
            cprint("Invalid choice. Please select 1-2", color='red')
            return self.select_model()

        selected = self.MODELS[choice]
        cprint(f"Selected: {selected['name']}", color='green')
        return selected['model']

    def generate_unique_filename(self, video_id: str, prompt: str) -> str:
        """create unique filename for video"""
        # get timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # clean prompt for filename (first 30 chars, alphanumeric only)
        clean_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c.isspace())
        clean_prompt = clean_prompt.strip().replace(" ", "_").lower()

        # create filename: timestamp_prompt_shortid.mp4
        short_id = video_id[:8] if video_id else "unknown"
        filename = f"{timestamp}_{clean_prompt}_{short_id}.mp4"

        return filename

    def generate_custom_video(self):
        """full custom video generation flow"""
        cprint("\nCustom Video Generation", color='cyan', format='bold', tag='NEW')
        print("=" * 60)

        # get all inputs
        prompt = self.get_user_prompt()
        print("\n" + "-" * 60)

        model = self.select_model()
        print("\n" + "-" * 60)

        size = self.select_aspect_ratio()
        print("\n" + "-" * 60)

        seconds = self.select_duration()

        # show summary
        print("\n" + "=" * 60)
        cprint("Generation Summary:", color='cyan', format='bold')
        print(f"Prompt: {prompt[:100]}...")
        print(f"Model: {model}")
        print(f"Size: {size}")
        print(f"Duration: {seconds} seconds")
        print("=" * 60)

        confirm = input("\nProceed with generation? (y/n): ").lower()
        if confirm != 'y':
            cprint("Generation cancelled", color='yellow')
            return None

        # generate video
        cprint("\nStarting generation...", color='cyan')
        try:
            video = self.client.create_and_poll(
                prompt=prompt,
                model=model,
                size=size,
                seconds=seconds
            )

            if video.status == "completed":
                # generate unique filename
                filename = self.generate_unique_filename(video.id, prompt)

                # create videos directory if needed
                videos_dir = Path("generated_videos")
                videos_dir.mkdir(exist_ok=True)

                filepath = videos_dir / filename

                # download
                cprint(f"\nDownloading video...", color='cyan')
                self.client.download_video(video.id, str(filepath))

                cprint(f"Success! Video saved as: {filepath}", color='green', format='bold')

                # ask about additional assets
                dl_thumb = input("\nDownload thumbnail? (y/n): ").lower()
                if dl_thumb == 'y':
                    thumb_path = filepath.with_suffix('.webp')
                    self.client.download_video(video.id, str(thumb_path), "thumbnail")
                    cprint(f"Thumbnail saved: {thumb_path}", color='green')

                return str(filepath)
            else:
                cprint(f"Generation failed: {video.status}", color='red')
                return None

        except Exception as e:
            cprint(f"Error: {e}", color='red')
            return None

    def batch_generate_custom(self):
        """generate multiple videos with custom settings"""
        cprint("\nBatch Custom Generation", color='cyan', format='bold', tag='BATCH')
        print("=" * 60)

        num_videos = input("How many videos to generate? ").strip()
        try:
            num_videos = int(num_videos)
            if num_videos < 1 or num_videos > 10:
                raise ValueError
        except:
            cprint("Invalid number. Please enter 1-10", color='red')
            return

        videos_config = []

        for i in range(num_videos):
            print(f"\n--- Video {i+1} of {num_videos} ---")
            prompt = self.get_user_prompt()

            # option to use same settings as previous
            if i > 0:
                use_same = input("\nUse same model/size/duration as previous? (y/n): ").lower()
                if use_same == 'y':
                    config = videos_config[-1].copy()
                    config['prompt'] = prompt
                else:
                    model = self.select_model()
                    size = self.select_aspect_ratio()
                    seconds = self.select_duration()
                    config = {
                        'prompt': prompt,
                        'model': model,
                        'size': size,
                        'seconds': seconds
                    }
            else:
                model = self.select_model()
                size = self.select_aspect_ratio()
                seconds = self.select_duration()
                config = {
                    'prompt': prompt,
                    'model': model,
                    'size': size,
                    'seconds': seconds
                }

            videos_config.append(config)

        # show summary
        print("\n" + "=" * 60)
        cprint("Batch Summary:", color='cyan', format='bold')
        for i, config in enumerate(videos_config, 1):
            print(f"\n{i}. {config['prompt'][:50]}...")
            print(f"   Model: {config['model']}, Size: {config['size']}, Duration: {config['seconds']}s")

        confirm = input("\nProceed with batch generation? (y/n): ").lower()
        if confirm != 'y':
            cprint("Batch cancelled", color='yellow')
            return

        # generate all videos
        videos_dir = Path("generated_videos")
        videos_dir.mkdir(exist_ok=True)

        results = []
        for i, config in enumerate(videos_config, 1):
            print(f"\n[{i}/{num_videos}] Generating: {config['prompt'][:50]}...")

            try:
                video = self.client.create_and_poll(**config)

                if video.status == "completed":
                    filename = self.generate_unique_filename(video.id, config['prompt'])
                    filepath = videos_dir / filename
                    self.client.download_video(video.id, str(filepath))
                    cprint(f"Success: {filename}", color='green')
                    results.append(str(filepath))
                else:
                    cprint(f"Failed: {video.status}", color='red')
                    results.append(None)
            except Exception as e:
                cprint(f"Error: {e}", color='red')
                results.append(None)

        # summary
        print("\n" + "=" * 60)
        successful = sum(1 for r in results if r)
        cprint(f"Batch complete: {successful}/{num_videos} successful",
               color='green' if successful == num_videos else 'yellow')

        if successful > 0:
            print("\nGenerated files:")
            for filepath in results:
                if filepath:
                    print(f"  - {filepath}")


def interactive_video_generator():
    """main interactive generator interface"""
    generator = VideoGenerator()

    while True:
        print("\n" + "=" * 60)
        cprint("Video Generator Menu", color='cyan', format='bold')
        print("-" * 60)
        print("1. Generate single video (custom settings)")
        print("2. Batch generate videos")
        print("3. Quick generate (with prompts only)")
        print("4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            generator.generate_custom_video()
        elif choice == "2":
            generator.batch_generate_custom()
        elif choice == "3":
            # quick mode - just prompt, use defaults
            prompt = input("\nEnter prompt (using default settings): ").strip()
            if prompt:
                try:
                    video = generator.client.create_and_poll(
                        prompt=prompt,
                        model=VideoModel.SORA_2.value,
                        size=VideoSize.SIZE_1280x720.value,
                        seconds="8"
                    )
                    if video.status == "completed":
                        filename = generator.generate_unique_filename(video.id, prompt)
                        videos_dir = Path("generated_videos")
                        videos_dir.mkdir(exist_ok=True)
                        filepath = videos_dir / filename
                        generator.client.download_video(video.id, str(filepath))
                        cprint(f"Success: {filepath}", color='green')
                except Exception as e:
                    cprint(f"Error: {e}", color='red')
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            cprint("Invalid choice", color='red')


if __name__ == "__main__":
    # test the generator
    interactive_video_generator()