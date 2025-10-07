"""
Utility functions for Sora 2 video management
Includes webhooks, monitoring, and advanced features
"""

import os
import json
import time
import hashlib
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from pathlib import Path
import threading
from queue import Queue

from openai import OpenAI
from sora_client import SoraClient, VideoModel, VideoSize


class VideoMonitor:
    """Monitor multiple video generation jobs"""

    def __init__(self, client: SoraClient):
        self.client = client
        self.active_jobs = {}
        self.completed_jobs = []
        self.failed_jobs = []
        self.monitoring = False

    def add_job(self, video_id: str, metadata: Optional[Dict] = None):
        """Add a video job to monitor"""
        self.active_jobs[video_id] = {
            "id": video_id,
            "start_time": datetime.now(),
            "metadata": metadata or {},
            "status": "queued",
            "progress": 0
        }

    def start_monitoring(self, interval: int = 10, callback: Optional[Callable] = None):
        """Start monitoring all active jobs"""
        self.monitoring = True

        def monitor_loop():
            while self.monitoring and self.active_jobs:
                for video_id in list(self.active_jobs.keys()):
                    video = self.client.get_video_status(video_id)
                    job_info = self.active_jobs[video_id]

                    # Update job info
                    job_info["status"] = video.status
                    job_info["progress"] = getattr(video, "progress", 0)

                    # Call callback if provided
                    if callback:
                        callback(video_id, video)

                    # Move completed/failed jobs
                    if video.status == "completed":
                        job_info["end_time"] = datetime.now()
                        job_info["duration"] = (job_info["end_time"] - job_info["start_time"]).total_seconds()
                        self.completed_jobs.append(job_info)
                        del self.active_jobs[video_id]
                        print(f"Job {video_id[:8]}... completed in {job_info['duration']:.1f}s")

                    elif video.status == "failed":
                        job_info["end_time"] = datetime.now()
                        job_info["error"] = getattr(video, "error", "Unknown error")
                        self.failed_jobs.append(job_info)
                        del self.active_jobs[video_id]
                        print(f"Job {video_id[:8]}... failed: {job_info['error']}")

                time.sleep(interval)

        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring jobs"""
        self.monitoring = False

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about monitored jobs"""
        total_jobs = len(self.active_jobs) + len(self.completed_jobs) + len(self.failed_jobs)

        if self.completed_jobs:
            avg_duration = sum(j["duration"] for j in self.completed_jobs) / len(self.completed_jobs)
        else:
            avg_duration = 0

        return {
            "total_jobs": total_jobs,
            "active": len(self.active_jobs),
            "completed": len(self.completed_jobs),
            "failed": len(self.failed_jobs),
            "success_rate": len(self.completed_jobs) / total_jobs * 100 if total_jobs > 0 else 0,
            "average_duration": avg_duration
        }


class VideoWebhookHandler:
    """
    Handle webhook notifications for video events
    Note: This is a simplified example. In production, you'd use a web framework
    """

    def __init__(self, client: SoraClient):
        self.client = client
        self.handlers = {
            "video.completed": self.on_video_completed,
            "video.failed": self.on_video_failed
        }

    def process_webhook(self, event_data: Dict[str, Any]):
        """Process incoming webhook event"""
        event_type = event_data.get("type")
        video_id = event_data.get("data", {}).get("id")

        if event_type in self.handlers:
            self.handlers[event_type](video_id, event_data)
        else:
            print(f"Unhandled event type: {event_type}")

    def on_video_completed(self, video_id: str, event_data: Dict):
        """Handle video completion"""
        print(f"Video {video_id} completed via webhook")

        # Auto-download completed video
        output_path = f"webhook_{video_id[:8]}.mp4"
        self.client.download_video(video_id, output_path)
        print(f"Auto-downloaded to {output_path}")

        # Could trigger other actions:
        # - Send notification
        # - Update database
        # - Start next job in queue

    def on_video_failed(self, video_id: str, event_data: Dict):
        """Handle video failure"""
        print(f"Video {video_id} failed via webhook")

        # Could implement retry logic
        video = self.client.get_video_status(video_id)
        error = getattr(video, "error", "Unknown error")
        print(f"Error: {error}")


class VideoQueue:
    """Queue system for batch video processing"""

    def __init__(self, client: SoraClient, max_concurrent: int = 3):
        self.client = client
        self.max_concurrent = max_concurrent
        self.queue = Queue()
        self.active_jobs = []
        self.results = []
        self.processing = False

    def add_task(self, prompt: str, **kwargs):
        """Add a video generation task to the queue"""
        task = {
            "prompt": prompt,
            "params": kwargs,
            "added_at": datetime.now()
        }
        self.queue.put(task)

    def process_queue(self):
        """Process queued tasks with concurrency limit"""
        self.processing = True

        while self.processing:
            # Remove completed jobs from active list
            self.active_jobs = [
                job for job in self.active_jobs
                if self.client.get_video_status(job["id"]).status in ["queued", "in_progress"]
            ]

            # Start new jobs if below limit
            while len(self.active_jobs) < self.max_concurrent and not self.queue.empty():
                task = self.queue.get()
                video = self.client.create_video(task["prompt"], **task["params"])

                job_info = {
                    "id": video.id,
                    "task": task,
                    "start_time": datetime.now()
                }
                self.active_jobs.append(job_info)
                print(f"Started job {video.id[:8]}... ({len(self.active_jobs)}/{self.max_concurrent} active)")

            # Check for completed jobs
            for job in self.active_jobs[:]:
                video = self.client.get_video_status(job["id"])
                if video.status == "completed":
                    job["end_time"] = datetime.now()
                    job["status"] = "completed"
                    self.results.append(job)
                    print(f"Job {job['id'][:8]}... completed")

                    # Auto-download
                    output_path = f"queue_{job['id'][:8]}.mp4"
                    self.client.download_video(job["id"], output_path)

                elif video.status == "failed":
                    job["end_time"] = datetime.now()
                    job["status"] = "failed"
                    self.results.append(job)
                    print(f"Job {job['id'][:8]}... failed")

            if self.queue.empty() and not self.active_jobs:
                break

            time.sleep(5)

        self.processing = False
        print("Queue processing complete")

    def stop_processing(self):
        """Stop queue processing"""
        self.processing = False


class VideoCache:
    """Cache video generation results to avoid duplicates"""

    def __init__(self, cache_dir: str = ".sora_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "video_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def get_prompt_hash(self, prompt: str, params: Dict) -> str:
        """Generate unique hash for prompt and parameters"""
        cache_key = f"{prompt}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(cache_key.encode()).hexdigest()

    def get_cached_video(self, prompt: str, **params) -> Optional[str]:
        """Check if video exists in cache"""
        prompt_hash = self.get_prompt_hash(prompt, params)

        if prompt_hash in self.cache:
            cached = self.cache[prompt_hash]
            video_path = Path(cached["video_path"])

            # Check if file still exists and not expired
            if video_path.exists():
                created_time = datetime.fromisoformat(cached["created_at"])
                if datetime.now() - created_time < timedelta(days=7):  # 7-day cache
                    print(f"Using cached video: {video_path}")
                    return str(video_path)

        return None

    def add_to_cache(self, prompt: str, video_id: str, video_path: str, **params):
        """Add video to cache"""
        prompt_hash = self.get_prompt_hash(prompt, params)

        self.cache[prompt_hash] = {
            "video_id": video_id,
            "video_path": video_path,
            "prompt": prompt,
            "params": params,
            "created_at": datetime.now().isoformat()
        }

        self._save_cache()

    def clear_expired(self):
        """Remove expired entries from cache"""
        current_time = datetime.now()
        expired = []

        for prompt_hash, entry in self.cache.items():
            created_time = datetime.fromisoformat(entry["created_at"])
            if current_time - created_time > timedelta(days=7):
                expired.append(prompt_hash)

                # Delete video file if exists
                video_path = Path(entry["video_path"])
                if video_path.exists():
                    video_path.unlink()

        for prompt_hash in expired:
            del self.cache[prompt_hash]

        self._save_cache()
        print(f"Cleared {len(expired)} expired cache entries")


class VideoOptimizer:
    """Optimize video generation parameters for cost and quality"""

    @staticmethod
    def suggest_model(prompt: str, requirements: Dict) -> str:
        """Suggest best model based on requirements"""

        # Keywords that suggest need for high quality
        quality_keywords = [
            "cinematic", "production", "professional", "high-quality",
            "marketing", "advertisement", "commercial", "film"
        ]

        # Check if high quality is needed
        needs_pro = any(keyword in prompt.lower() for keyword in quality_keywords)

        # Check requirements
        if requirements.get("quality") == "high":
            needs_pro = True
        if requirements.get("budget") == "low":
            needs_pro = False
        if requirements.get("speed") == "fast":
            needs_pro = False

        return VideoModel.SORA_2_PRO.value if needs_pro else VideoModel.SORA_2.value

    @staticmethod
    def optimize_parameters(prompt: str, target_platform: str = "general") -> Dict:
        """Optimize video parameters for target platform"""

        platform_configs = {
            "instagram_reel": {
                "size": VideoSize.SIZE_720x1280.value,  # Portrait for Instagram
                "seconds": "12",
                "model": VideoModel.SORA_2.value
            },
            "youtube_short": {
                "size": VideoSize.SIZE_720x1280.value,  # Portrait for Shorts
                "seconds": "12",  # Max allowed duration
                "model": VideoModel.SORA_2.value
            },
            "youtube": {
                "size": VideoSize.SIZE_1792x1024.value,  # Wide landscape for YouTube
                "seconds": "12",
                "model": VideoModel.SORA_2_PRO.value
            },
            "twitter": {
                "size": VideoSize.SIZE_1280x720.value,  # Standard HD
                "seconds": "4",
                "model": VideoModel.SORA_2.value
            },
            "general": {
                "size": VideoSize.SIZE_1280x720.value,  # Standard HD
                "seconds": "8",
                "model": VideoModel.SORA_2.value
            }
        }

        config = platform_configs.get(target_platform, platform_configs["general"])

        return config


def batch_download_completed_videos(client: SoraClient, output_dir: str = "downloads"):
    """Download all completed videos from your library"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    videos = client.list_videos(limit=100)
    completed_videos = [v for v in videos if v.get("status") == "completed"]

    print(f"Found {len(completed_videos)} completed videos")

    for i, video in enumerate(completed_videos, 1):
        video_id = video["id"]
        filename = output_path / f"video_{video_id[:8]}.mp4"

        if not filename.exists():
            print(f"Downloading {i}/{len(completed_videos)}: {video_id[:12]}...")
            client.download_video(video_id, str(filename))
        else:
            print(f"Skipping {i}/{len(completed_videos)}: {filename.name} (already exists)")

    print(f"Downloaded videos saved to {output_path}")


def estimate_generation_time(model: str, seconds: str, size: str) -> float:
    """Estimate generation time based on parameters"""

    # Convert seconds to int if string
    seconds_int = int(seconds) if isinstance(seconds, str) else seconds

    # Base times (in seconds) - these are rough estimates
    base_times = {
        VideoModel.SORA_2.value: 60,  # 1 minute base
        VideoModel.SORA_2_PRO.value: 180  # 3 minutes base
    }

    # Size multipliers
    size_multipliers = {
        VideoSize.SIZE_1280x720.value: 1.0,      # HD baseline
        VideoSize.SIZE_720x1280.value: 1.0,      # HD portrait
        VideoSize.SIZE_1792x1024.value: 1.3,     # Wide landscape (more pixels)
        VideoSize.SIZE_1024x1792.value: 1.3,     # Wide portrait
    }

    base_time = base_times.get(model, 120)
    size_mult = size_multipliers.get(size, 1.0)
    duration_mult = seconds_int / 8  # Assuming 8 seconds is baseline

    estimated_time = base_time * size_mult * duration_mult

    return estimated_time


def create_video_with_retry(
    client: SoraClient,
    prompt: str,
    max_retries: int = 3,
    **kwargs
) -> Optional[Dict]:
    """Create video with automatic retry on failure"""

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}")
            video = client.create_and_poll(prompt, **kwargs)

            if video.status == "completed":
                return video
            elif video.status == "failed":
                error = getattr(video, "error", "Unknown error")
                print(f"Generation failed: {error}")

                # Check if error is retryable
                if "rate limit" in str(error).lower():
                    wait_time = 60 * (attempt + 1)  # Progressive backoff
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif "content policy" in str(error).lower():
                    print("Content policy violation. Cannot retry.")
                    return None

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")

        if attempt < max_retries - 1:
            time.sleep(10)

    print(f"Failed after {max_retries} attempts")
    return None


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = SoraClient()

    # Example: Monitor multiple jobs
    monitor = VideoMonitor(client)

    # Start some jobs
    prompts = [
        "A lighthouse beam sweeping across foggy ocean",
        "Time-lapse of flowers blooming in a garden",
        "Northern lights over snowy mountains"
    ]

    for prompt in prompts:
        video = client.create_video(prompt, seconds="4")
        monitor.add_job(video.id, {"prompt": prompt})

    # Start monitoring
    monitor.start_monitoring(interval=10)

    # Wait for completion
    while monitor.active_jobs:
        stats = monitor.get_statistics()
        print(f"Stats: {stats}")
        time.sleep(30)

    monitor.stop_monitoring()
    print("All jobs completed!")