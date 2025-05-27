import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel, AutoModelForCausalLM

# -----------------------------
# Directories
# -----------------------------
IMAGE_DIR = "Image/"
OCR_DIR = "ocr/"
TEXT_DIR = "text/"

# -----------------------------
# Load Models
# -----------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

text_encoder = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")
text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

generator_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
generator_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# -----------------------------
# MM-CAD Classifier Model
# -----------------------------
class MM_CAD(nn.Module):
    def __init__(self, clip_dim=512, text_dim=4096, fusion_dim=1024):
        super(MM_CAD, self).__init__()
        self.proj_image = nn.Linear(clip_dim, fusion_dim)
        self.proj_text = nn.Linear(text_dim, fusion_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * fusion_dim, 1)
        )

    def forward(self, image_feat, text_feat):
        v = self.proj_image(image_feat)
        t = self.proj_text(text_feat)
        z = torch.cat([v, t], dim=-1)
        logits = self.classifier(z)
        return logits, z

# -----------------------------
# Feature Processing Functions
# -----------------------------
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        img_features = clip_model.get_image_features(**inputs)
    return img_features

def encode_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = text_encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def generate_explanation(prompt_text, max_new_tokens=50):
    inputs = generator_tokenizer(prompt_text, return_tensors="pt").to(generator_model.device)
    with torch.no_grad():
        output = generator_model.generate(**inputs, max_new_tokens=max_new_tokens)
    return generator_tokenizer.decode(output[0], skip_special_tokens=True)

# -----------------------------
# Full Pipeline: Classification + Explanation
# -----------------------------
def run_pipeline(filename, mmcad_model):
    image_path = os.path.join(IMAGE_DIR, filename)
    ocr_path = os.path.join(OCR_DIR, filename.replace(".jpg", ".txt"))
    text_path = os.path.join(TEXT_DIR, filename.replace(".jpg", ".txt"))

    # Read OCR + caption text and concatenate
    with open(ocr_path, 'r', encoding='utf-8') as f:
        ocr_text = f.read().strip()
    with open(text_path, 'r', encoding='utf-8') as f:
        caption_text = f.read().strip()
    combined_text = ocr_text + " " + caption_text

    # Extract features
    img_feat = process_image(image_path)
    txt_feat = encode_text(combined_text)

    # Predict
    mmcad_model.eval()
    with torch.no_grad():
        logits, fused = mmcad_model(img_feat, txt_feat)
        prob = torch.sigmoid(logits).item()
        label = "Abusive" if prob > 0.5 else "Non-Abusive"
        print(f"[Prediction]: {label} (confidence: {prob:.2f})")

        # Explanation only for abusive cases
        if label == "Abusive":
            prompt = f"Why is this meme abusive? It contains the following text: '{combined_text}' and image features."
            rationale = generate_explanation(prompt)
            print("[Generated Explanation]:", rationale)
