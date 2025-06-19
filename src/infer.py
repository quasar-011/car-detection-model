import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from src.config import *
from src.utils import visualize_prediction

# Load the model
def load_model(model_path, num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Inference on one image
def run_inference(image_path, model, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(DEVICE)
    with torch.no_grad():
        prediction = model([image_tensor])[0]

    boxes = prediction['boxes'].cpu()
    scores = prediction['scores'].cpu()
    labels = prediction['labels'].cpu()

    selected = scores > threshold
    boxes = boxes[selected]
    scores = scores[selected]
    labels = labels[selected]

    label_names = ["background", "car"]
    labels_named = [label_names[l] for l in labels]

    visualize_prediction(image_tensor.cpu(), boxes, scores, labels_named)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="../saved_models/fasterrcnn.pth", help="Path to model weights")
    args = parser.parse_args()

    model = load_model(args.weights, num_classes=2)
    run_inference(args.image, model)
