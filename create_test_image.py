#!/usr/bin/env python3
"""
Create a simple test image without faces for reference image testing
This avoids moderation blocks from human faces
"""

from PIL import Image, ImageDraw
import random
from pathlib import Path


def create_landscape_gradient(width=720, height=1280):
    """Create a simple gradient landscape image"""
    # Create new image
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    # Create sky gradient (top half)
    for y in range(height // 2):
        # Gradient from light blue to darker blue
        ratio = y / (height // 2)
        r = int(135 + (50 * ratio))
        g = int(206 + (-50 * ratio))
        b = int(235 + (-35 * ratio))
        draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b))

    # Create ground gradient (bottom half)
    for y in range(height // 2, height):
        # Gradient from light green to darker green
        ratio = (y - height // 2) / (height // 2)
        r = int(124 + (-40 * ratio))
        g = int(195 + (-60 * ratio))
        b = int(115 + (-40 * ratio))
        draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b))

    # Add some simple shapes (mountains)
    mountain_color = (100, 100, 120)
    points1 = [(0, height // 2), (width // 3, height // 3), (width // 2, height // 2)]
    draw.polygon(points1, fill=mountain_color)

    points2 = [(width // 3, height // 2), (2 * width // 3, height // 4), (width, height // 2)]
    draw.polygon(points2, fill=(120, 120, 140))

    # Add sun
    sun_x = width - 100
    sun_y = 100
    sun_radius = 40
    draw.ellipse(
        [(sun_x - sun_radius, sun_y - sun_radius),
         (sun_x + sun_radius, sun_y + sun_radius)],
        fill=(255, 220, 100)
    )

    return img


def create_abstract_pattern(width=720, height=1280):
    """Create an abstract pattern image"""
    img = Image.new('RGB', (width, height), color=(40, 40, 60))
    draw = ImageDraw.Draw(img)

    # Create random circles
    random.seed(42)  # Fixed seed for consistency
    for _ in range(30):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(20, 100)
        color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255)
        )
        # Semi-transparent effect by drawing with lower opacity
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=color,
            outline=None
        )

    return img


def main():
    """Generate test images for reference"""
    print("Creating test reference images...")

    # Create landscape image
    landscape = create_landscape_gradient()
    landscape.save("test_landscape.jpeg", quality=95)
    print("Created: test_landscape.jpeg (720x1280) - Simple landscape")

    # Create abstract pattern
    abstract = create_abstract_pattern()
    abstract.save("test_abstract.jpeg", quality=95)
    print("Created: test_abstract.jpeg (720x1280) - Abstract pattern")

    # Create different sizes for testing resize functionality
    landscape_wrong_size = create_landscape_gradient(1920, 1080)
    landscape_wrong_size.save("test_landscape_wrong_size.jpeg", quality=95)
    print("Created: test_landscape_wrong_size.jpeg (1920x1080) - For testing auto-resize")

    print("\nTest images created successfully!")
    print("\nThese images are designed to avoid moderation blocks:")
    print("- No human faces or identifiable people")
    print("- Simple, abstract compositions")
    print("- Safe, non-controversial content")
    print("\nUse these with safe prompts like:")
    print("- 'The camera slowly pans across the scene'")
    print("- 'Colors gradually shift to warmer tones'")
    print("- 'Elements begin to glow with soft light'")


if __name__ == "__main__":
    main()