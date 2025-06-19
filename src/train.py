# train.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision
from torch.utils.data import DataLoader
from src.utils import load_voc_dataset, collate_fn
from src.config import *
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train():
    dataset = load_voc_dataset(DATA_DIR, image_set="train")
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for images, targets in data_loader:
            try:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                del images, targets, loss_dict, losses
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print("⚠️ Skipping batch due to OOM")
                torch.cuda.empty_cache()
                continue

        lr_scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "../saved_models/fasterrcnn.pth")

if __name__ == "__main__":
    train()
