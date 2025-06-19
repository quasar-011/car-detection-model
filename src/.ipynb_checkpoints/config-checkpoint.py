DATA_DIR = "../data/VOC"
MODEL_NAME = "fasterrcnn_resnet50_fpn"
NUM_CLASSES = 21  # 20 classes + background (adjust for car-only if needed)
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 0.005
DEVICE = "cuda"  # or "cpu"
