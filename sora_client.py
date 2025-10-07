"""
Comprehensive Sora 2 API Client
Implements all features from the OpenAI Sora 2 video generation API
"""

import os
import sys
import time
import asyncio
from typing import Optional, Dict, Any, List, Literal
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from openai import OpenAI, AsyncOpenAI


class VideoModel(Enum):
    """Available Sora video generation models"""
    SORA_2 = "sora-2"  # Speed and flexibility, good for exploration
    SORA_2_PRO = "sora-2-pro"  # Higher quality, production-ready output


class VideoStatus(Enum):
    """Video generation job statuses"""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoSize(Enum):
    """Supported video resolutions - from Sora API"""
    SIZE_1280x720 = "1280x720"      # HD Landscape
    SIZE_720x1280 = "720x1280"      # HD Portrait (vertical)
    SIZE_1792x1024 = "1792x1024"    # Wide Landscape
    SIZE_1024x1792 = "1024x1792"    # Wide Portrait (vertical)


@dataclass
class VideoConfig:
    """Configuration for video generation"""
    model: str = VideoModel.SORA_2.value
    size: str = VideoSize.SIZE_1280x720.value
    seconds: int = 5
    prompt: str = ""
    input_reference: Optional[str] = None
    remix_video_id: Optional[str] = None


class SoraClient:
    """
    Comprehensive client for OpenAI Sora 2 API
    Handles video generation, monitoring, downloading, and management
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Sora client with API key

        Args:
            api_key: OpenAI API key (uses env var OPENAI_API_KEY if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY or pass api_key parameter")

        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    def create_video(
        self,
        prompt: str,
        model: str = VideoModel.SORA_2.value,
        size: str = VideoSize.SIZE_1280x720.value,
        seconds: str = "8",
        input_reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start a video generation job

        Args:
            prompt: Text description of the video to generate
            model: Model to use (sora-2 or sora-2-pro)
            size: Video resolution
            seconds: Video duration - must be "4", "8", or "12"
            input_reference: Path to reference image (first frame)

        Returns:
            Video job object with id and status
        """
        # Validate and convert seconds parameter
        valid_seconds = ["4", "8", "12"]
        seconds_str = str(seconds) if not isinstance(seconds, str) else seconds
        if seconds_str not in valid_seconds:
            # Use closest valid value
            try:
                seconds_int = int(seconds_str)
                if seconds_int <= 4:
                    seconds_str = "4"
                elif seconds_int <= 8:
                    seconds_str = "8"
                else:
                    seconds_str = "12"
                print(f"Adjusted seconds from {seconds} to {seconds_str} (valid values: 4, 8, 12)")
            except ValueError:
                seconds_str = "8"  # Default to 8 if invalid

        params = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "seconds": seconds_str
        }

        # Add reference image if provided
        try:
            if input_reference and Path(input_reference).exists():
                # Try opening the file in binary mode and passing it to the API
                with open(input_reference, "rb") as img_file:
                    params["input_reference"] = img_file
                    video = self.client.videos.create(**params)
            else:
                video = self.client.videos.create(**params)
        except Exception as e:
            # If the API doesn't support reference images yet, proceed without it
            print(f"Note: Reference image upload may not be supported yet: {e}")
            video = self.client.videos.create(
                model=model,
                prompt=prompt,
                size=size,
                seconds=seconds_str
            )

        return video

    def create_and_poll(
        self,
        prompt: str,
        model: str = VideoModel.SORA_2.value,
        size: str = VideoSize.SIZE_1280x720.value,
        seconds: str = "8",
        input_reference: Optional[str] = None,
        poll_interval: int = 10,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Create video and poll until completion

        Args:
            prompt: Text description of the video
            model: Model to use
            size: Video resolution
            seconds: Video duration
            input_reference: Path to reference image
            poll_interval: Seconds between status checks
            show_progress: Display progress bar

        Returns:
            Completed video object or error
        """
        # Start generation
        video = self.create_video(prompt, model, size, seconds, input_reference)
        print(f"Video generation started: {video.id}")

        # Poll for completion
        return self.poll_video_status(video.id, poll_interval, show_progress)

    def poll_video_status(
        self,
        video_id: str,
        poll_interval: int = 10,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Poll video status until completion

        Args:
            video_id: ID of the video job
            poll_interval: Seconds between checks
            show_progress: Display progress bar

        Returns:
            Final video object
        """
        from print_color import print as pc
        bar_length = 40

        while True:
            video = self.client.videos.retrieve(video_id)
            progress = getattr(video, "progress", 0)

            if show_progress:
                filled = int((progress / 100) * bar_length)
                bar = "=" * filled + "-" * (bar_length - filled)
                status = "Queued" if video.status == "queued" else "Processing"
                sys.stdout.write(f"\r{status}: [{bar}] {progress:.1f}%")
                sys.stdout.flush()

            if video.status in ["completed", "failed"]:
                if show_progress:
                    sys.stdout.write("\n")

                # Handle failure with detailed error message
                if video.status == "failed":
                    error_msg = getattr(video, 'error', None)
                    if error_msg:
                        pc(f"\nVideo generation failed: {error_msg}", color='red', tag='ERROR', tag_color='red')

                        # Check for common content policy violations
                        if "content policy" in str(error_msg).lower():
                            pc("\nContent Policy Violation Detected:", color='yellow', tag='INFO', tag_color='yellow')
                            pc("- No copyrighted characters (Spider-Man, Batman, etc.)", color='white')
                            pc("- No real people or public figures", color='white')
                            pc("- No inappropriate or adult content", color='white')
                            pc("- No trademarked logos or brands", color='white')
                            pc("- Content must be suitable for all audiences", color='white')
                    else:
                        pc(f"\nVideo generation failed with status: {video.status}", color='red', tag='ERROR', tag_color='red')
                break

            time.sleep(poll_interval)

        return video

    async def create_and_poll_async(
        self,
        prompt: str,
        model: str = VideoModel.SORA_2.value,
        size: str = VideoSize.SIZE_1280x720.value,
        seconds: str = "8",
        input_reference: Optional[str] = None,
        poll_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Asynchronously create and poll video

        Args:
            prompt: Text description
            model: Model to use
            size: Resolution
            seconds: Duration - must be "4", "8", or "12"
            input_reference: Reference image path
            poll_interval: Poll interval

        Returns:
            Completed video object
        """
        # Validate seconds parameter
        valid_seconds = ["4", "8", "12"]
        seconds_str = str(seconds) if not isinstance(seconds, str) else seconds
        if seconds_str not in valid_seconds:
            try:
                seconds_int = int(seconds_str)
                if seconds_int <= 4:
                    seconds_str = "4"
                elif seconds_int <= 8:
                    seconds_str = "8"
                else:
                    seconds_str = "12"
            except ValueError:
                seconds_str = "8"

        params = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "seconds": seconds_str
        }

        # Start generation
        video = await self.async_client.videos.create(**params)
        print(f"Async video generation started: {video.id}")

        # Poll asynchronously
        while video.status in ["queued", "in_progress"]:
            await asyncio.sleep(poll_interval)
            video = await self.async_client.videos.retrieve(video.id)
            progress = getattr(video, "progress", 0)
            print(f"Progress: {progress}%")

        return video

    def get_video_status(self, video_id: str) -> Dict[str, Any]:
        """
        Get current status of a video job

        Args:
            video_id: ID of the video job

        Returns:
            Video object with current status
        """
        return self.client.videos.retrieve(video_id)

    def download_video(
        self,
        video_id: str,
        output_path: str = "output.mp4",
        variant: Literal["video", "thumbnail", "spritesheet"] = "video"
    ) -> bool:
        """
        Download completed video or assets

        Args:
            video_id: ID of completed video
            output_path: Where to save the file
            variant: Type of asset to download

        Returns:
            True if successful
        """
        try:
            # Check if video is completed
            video = self.get_video_status(video_id)
            if video.status != "completed":
                print(f"Video not ready. Status: {video.status}")
                return False

            # Download content
            content = self.client.videos.download_content(video_id, variant=variant)

            # Save to file
            content.write_to_file(output_path)
            print(f"Downloaded {variant} to {output_path}")
            return True

        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def remix_video(
        self,
        original_video_id: str,
        remix_prompt: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a remix of an existing video

        Args:
            original_video_id: ID of the video to remix
            remix_prompt: Description of changes to make
            model: Model to use (defaults to original)

        Returns:
            New video job for the remix
        """
        params = {"prompt": remix_prompt}
        if model:
            params["model"] = model

        # Note: This uses a hypothetical remix endpoint based on the docs
        # The actual implementation might differ
        video = self.client.videos.create(
            remix_video_id=original_video_id,
            **params
        )

        return video

    def list_videos(
        self,
        limit: int = 20,
        after: Optional[str] = None,
        order: Literal["asc", "desc"] = "desc"
    ) -> List[Dict[str, Any]]:
        """
        List all videos in your library

        Args:
            limit: Number of videos to return
            after: Pagination cursor
            order: Sort order

        Returns:
            List of video objects
        """
        params = {"limit": limit, "order": order}
        if after:
            params["after"] = after

        response = self.client.videos.list(**params)
        return response.data

    def delete_video(self, video_id: str) -> bool:
        """
        Delete a video from OpenAI storage

        Args:
            video_id: ID of video to delete

        Returns:
            True if successful
        """
        try:
            self.client.videos.delete(video_id)
            print(f"Deleted video: {video_id}")
            return True
        except Exception as e:
            print(f"Delete failed: {e}")
            return False

    def generate_with_best_practices(
        self,
        prompt: str,
        quality: Literal["fast", "high"] = "fast",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using best practices from the documentation

        Args:
            prompt: Well-structured prompt with shot type, subject, action, setting, lighting
            quality: "fast" for sora-2, "high" for sora-2-pro
            **kwargs: Additional parameters

        Returns:
            Completed video object
        """
        # Select model based on quality preference
        model = VideoModel.SORA_2.value if quality == "fast" else VideoModel.SORA_2_PRO.value

        # Apply best practice defaults
        defaults = {
            "size": VideoSize.SIZE_1280x720.value,
            "seconds": "8",
            "poll_interval": 10,
            "show_progress": True
        }
        defaults.update(kwargs)

        # Generate and poll
        return self.create_and_poll(prompt, model, **defaults)


class PromptBuilder:
    """Helper class to build effective prompts for Sora"""

    @staticmethod
    def build(
        shot_type: str,
        subject: str,
        action: str,
        setting: str,
        lighting: Optional[str] = None,
        camera_movement: Optional[str] = None,
        additional_details: Optional[str] = None
    ) -> str:
        """
        Build a well-structured prompt following best practices

        Args:
            shot_type: e.g., "Wide shot", "Close-up", "Medium shot"
            subject: Main subject of the video
            action: What the subject is doing
            setting: Where the scene takes place
            lighting: Lighting conditions
            camera_movement: Camera motion description
            additional_details: Extra details

        Returns:
            Formatted prompt string
        """
        parts = [f"{shot_type} of {subject} {action} in {setting}"]

        if lighting:
            parts.append(lighting)

        if camera_movement:
            parts.append(camera_movement)

        if additional_details:
            parts.append(additional_details)

        return ", ".join(parts)


# Utility functions for common operations
def quick_generate(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Quick function to generate a video with defaults

    Args:
        prompt: Video description
        api_key: Optional API key

    Returns:
        Path to downloaded video
    """
    client = SoraClient(api_key)
    video = client.create_and_poll(prompt)

    if video.status == "completed":
        output_path = f"video_{video.id[:8]}.mp4"
        client.download_video(video.id, output_path)
        return output_path
    else:
        raise Exception(f"Video generation failed: {video.status}")


def batch_generate(prompts: List[str], api_key: Optional[str] = None) -> List[str]:
    """
    Generate multiple videos from a list of prompts

    Args:
        prompts: List of video descriptions
        api_key: Optional API key

    Returns:
        List of video IDs
    """
    client = SoraClient(api_key)
    video_ids = []

    for prompt in prompts:
        video = client.create_video(prompt)
        video_ids.append(video.id)
        print(f"Started: {video.id} - {prompt[:50]}...")

    return video_ids