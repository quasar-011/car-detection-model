import torch
import torchvision
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ComposeTransform:
    def __init__(self):
        self.transforms = T.ToTensor()

    def __call__(self, image, target):
        image = self.transforms(image)
        boxes = []
        labels = []

        objs = target['annotation'].get('object', [])
        if not isinstance(objs, list):
            objs = [objs]  # single object case

        for obj in objs:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # You can refine label mapping here later

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
        return image, target

def load_voc_dataset(data_dir, image_set="train"):
    return VOCDetection(
        root=data_dir,
        year="2012",
        image_set=image_set,
        download=False,
        transforms=ComposeTransform()
    )

def collate_fn(batch):
    return tuple(zip(*batch))

def visualize_prediction(img, boxes, labels):
    fig, ax = plt.subplots(1)
    ax.imshow(img.permute(1, 2, 0))
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, label, color='white', fontsize=8,
                bbox=dict(facecolor='red', edgecolor='none', boxstyle='round,pad=0.2'))
    plt.show()


