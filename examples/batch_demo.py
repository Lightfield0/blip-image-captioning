#!/usr/bin/env python3
"""
Batch Image Captioning Demo
==========================

This script demonstrates how to process multiple images at once
using the BLIP image captioning model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blip_captioner import BLIPImageCaptioner

def main():
    print("🚀 BLIP Batch Image Captioning Demo")
    print("=" * 50)
    
    # Initialize the captioner
    captioner = BLIPImageCaptioner()
    
    # List of sample images (mix of URLs and potential local paths)
    sample_images = [
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Bp47tyvcJN9LUK7fUNHNpQ/149-22-JPG-jpg-rf-4899cbb6f4aad9588fa3811bb886c34d.jpg",
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500",
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=500",
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500&q=80"
    ]
    
    print(f"📦 Processing {len(sample_images)} images...")
    print("-" * 50)
    
    # Generate captions for all images
    captions = captioner.generate_captions_batch(sample_images)
    
    # Display results
    print("\n" + "=" * 50)
    print("📋 RESULTS")
    print("=" * 50)
    
    for i, (image_url, caption) in enumerate(zip(sample_images, captions), 1):
        print(f"\n🖼️  Image {i}:")
        print(f"   URL: {image_url[:60]}{'...' if len(image_url) > 60 else ''}")
        print(f"   📝 Caption: {caption}")
    
    print("\n" + "=" * 50)
    print("✅ Batch processing completed!")

if __name__ == "__main__":
    main()
