#!/usr/bin/env python3
"""
BLIP Image Captioning Tool
=========================

A Python script that generates automatic captions for images using 
BLIP (Bootstrapping Language-Image Pre-training) model.

Usage:
    python blip_captioner.py                    # Test with sample images
    python blip_captioner.py image.jpg          # Analyze local file
    python blip_captioner.py --interactive      # Interactive mode
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
import argparse
import os

def load_model():
    """Loads BLIP model and processor."""
    print("📥 Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on: {device}")
    
    return processor, model, device

def load_image_from_url(url):
    """Loads image from URL."""
    try:
        print(f"🌐 Loading image from URL: {url}")
        raw_image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        print(f"✅ Image loaded! Size: {raw_image.size}")
        return raw_image
    except Exception as e:
        print(f"❌ URL error: {e}")
        return None

def load_image_from_file(file_path):
    """Loads image from local file."""
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
            
        print(f"📁 Loading image from file: {file_path}")
        raw_image = Image.open(file_path).convert('RGB')
        print(f"✅ Image loaded! Size: {raw_image.size}")
        return raw_image
    except Exception as e:
        print(f"❌ File error: {e}")
        return None

def generate_caption(image, processor, model, device):
    """Generates caption for the image."""
    print("🧠 Generating caption...")
    
    # Prepare inputs for the model
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Generate caption
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, num_beams=5)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

def test_with_sample_images():
    """Tests the model with sample images."""
    processor, model, device = load_model()
    
    # Check for images in the images directory
    images_dir = "images"
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if image_files:
            print(f"\n📁 Found {len(image_files)} sample images!")
            for i, img_file in enumerate(image_files[:3], 1):  # First 3 images
                print(f"\n--- Image {i}: {img_file} ---")
                image_path = os.path.join(images_dir, img_file)
                image = load_image_from_file(image_path)
                if image:
                    caption = generate_caption(image, processor, model, device)
                    print(f"🎯 Caption: {caption}")
            return
    
    # If no local images, test with URL
    print("📁 No local sample images found, testing with URL...")
    img_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Bp47tyvcJN9LUK7fUNHNpQ/149-22-JPG-jpg-rf-4899cbb6f4aad9588fa3811bb886c34d.jpg"
    image = load_image_from_url(img_url)
    
    if image:
        caption = generate_caption(image, processor, model, device)
        print(f"\n🎯 Result: {caption}")
    else:
        print("❌ Test failed!")

def analyze_local_file(file_path):
    """Analyzes a local file."""
    processor, model, device = load_model()
    image = load_image_from_file(file_path)
    
    if image:
        caption = generate_caption(image, processor, model, device)
        print(f"\n🎯 Result: {caption}")
        return True
    return False

def interactive_mode():
    """Interactive mode - user can analyze multiple images."""
    print("🔄 Interactive mode started!")
    print("💡 Type 'q' to quit")
    
    processor, model, device = load_model()
    
    while True:
        print("\n" + "="*50)
        user_input = input("📁 Enter image file path or URL: ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("👋 Goodbye!")
            break
            
        if not user_input:
            print("⚠️ Please enter a valid path or URL")
            continue
        
        # Check if it's URL or file
        if user_input.startswith(('http://', 'https://')):
            image = load_image_from_url(user_input)
        else:
            image = load_image_from_file(user_input)
        
        if image:
            caption = generate_caption(image, processor, model, device)
            print(f"\n🎯 Result: {caption}")
        else:
            print("❌ Failed to load image, please try again")

def main():
    print("🎯 BLIP Image Captioning Tool")
    print("=" * 40)
    
    parser = argparse.ArgumentParser(description='BLIP image captioning')
    parser.add_argument('image_path', nargs='?', help='Image file path')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    try:
        if args.interactive:
            interactive_mode()
        elif args.image_path:
            if not analyze_local_file(args.image_path):
                print("❌ File analysis failed!")
        else:
            # Default: test with sample images
            test_with_sample_images()
            
    except KeyboardInterrupt:
        print("\n👋 Program interrupted!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
