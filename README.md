# 🎯 BLIP Image Captioning Project

This project uses BLIP (Bootstrapping Language-Image Pre-training) model to generate automatic captions for images.

## 🚀 Quick Demo

```bash
# Clone and run in 3 commands!
git clone https://github.com/username/blip-image-captioning.git
cd blip-image-captioning && pip install -r requirements.txt
python blip_captioner.py  # Test with sample images immediately!
```

**Example output:**
```
🎯 Caption: a small dog with a stick in its mouth
🎯 Caption: two people holding cups of coffee in their hands  
🎯 Caption: a city street filled with traffic and tall buildings
```

## ✨ Features

- 🌐 **URL image loading**: Analyze images from the internet
- 📁 **Local file support**: Analyze images from your computer  
- 🖼️ **Sample images included**: 10 ready-to-test images in different categories
- 💬 **Interactive mode**: Analyze multiple images sequentially
- 🚀 **GPU support**: Automatic GPU usage when CUDA is available

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/username/blip-image-captioning.git
cd blip-image-captioning
```

### 2. Create virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

### 3. Install packages
```bash
pip install -r requirements.txt
```

## 📖 Usage

### 🎮 Quick Start

#### 1. Quick test (with sample images)
```bash
python blip_captioner.py
```
This command tests the model with included sample images (cat, dog, car, etc.).

#### 2. Analyze your own image
```bash
python blip_captioner.py image.jpg
```

#### 3. Interactive mode (for multiple images)
```bash
python blip_captioner.py --interactive
```

### 💻 Use as Python Code

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

# 1. Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2. Load image (from URL)
img_url = "https://example.com/image.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# 3. Prepare inputs for model
inputs = processor(images=raw_image, return_tensors="pt")

# 4. Generate caption
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print("Caption:", caption)
```

## 📂 Project Structure

```
blip-image-captioning/
├── blip_captioner.py      # Main script (single file)
├── images/                # 10 sample images (~667KB total)
├── requirements.txt       # Python packages
├── README.md             # This file
└── .gitignore           # Git ignore
```

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Pillow (PIL)
- requests

See `requirements.txt` for the complete list.

## 🎯 Example Use Cases

- **Social media**: Automatic caption generation for photos
- **E-commerce**: Product image descriptions  
- **Accessibility**: Alternative text for visually impaired users
- **Content management**: Categorizing large image archives

## 🤝 Contributing

1. Fork this repository
2. Make your changes
3. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**What this means:**
- ✅ **Commercial use** - Use it in your business
- ✅ **Modification** - Change and improve the code
- ✅ **Distribution** - Share it with others
- ✅ **Private use** - Use it for personal projects
- ✅ **No liability** - Use at your own risk

## 🙏 Acknowledgments

- **Salesforce Research** - for creating the amazing BLIP model
- **Hugging Face** - for the excellent Transformers library
- **Unsplash contributors** - for the beautiful sample images

---

⭐ **If you like this project, please give it a star!**