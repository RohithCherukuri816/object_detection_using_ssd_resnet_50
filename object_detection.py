import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from sklearn.metrics import precision_score, recall_score

# Define paths to your dataset
voc_root_dir = 'C:\\Users\\rohit\\OneDrive\\Desktop\\VOC\\VOCdevkit\\VOC2007'  # Update this to your VOC dataset path
train_images_dir = os.path.join(voc_root_dir, 'JPEGImages')
train_annotations_dir = os.path.join(voc_root_dir, 'Annotations')

# Classes in Pascal VOC dataset
classes = ["__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
num_classes = len(classes)

class VOCDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.dataset = self.load_voc_annotations()

    def load_voc_annotations(self):
        dataset = []
        for annotation_file in os.listdir(self.annotations_dir):
            annotation_path = os.path.join(self.annotations_dir, annotation_file)
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            img_filename = root.find('filename').text
            img_path = os.path.join(self.images_dir, img_filename)

            objects = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_idx = class_to_idx[class_name]
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                objects.append((class_idx, xmin, ymin, xmax, ymax))

            dataset.append((img_path, objects))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, objects = self.dataset[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = []
        labels = []
        for obj in objects:
            class_idx, xmin, ymin, xmax, ymax = obj
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_idx)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

def get_transform():
    def transform(image, target):
        image = F.to_tensor(image)
        return image, target

    return transform

dataset = VOCDataset(train_images_dir, train_annotations_dir, transform=get_transform())

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Define the SSD model
model = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=num_classes)

# Load the pretrained model weights
#model.load_state_dict(torch.load("C:\\Users\\rohit\\OneDrive\\Desktop\\SSD\\ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"))
model.eval()

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training function
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {losses.item()}")

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            for i, output in enumerate(outputs):
                all_predictions.append(output['boxes'].cpu())
                all_targets.append(targets[i]['boxes'].cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_predictions, all_targets

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    lr_scheduler.step()

    predictions, targets = evaluate(model, val_loader, device)
    precision = precision_score(targets.numpy(), predictions.numpy(), average='weighted')
    recall = recall_score(targets.numpy(), predictions.numpy(), average='weighted')

    print(f"Epoch {epoch} - Precision: {precision}, Recall: {recall}")

# Save the trained model
torch.save(model.state_dict(), "ssd_mobilenet_v3_large.pth")

# Load the model for evaluation
model.load_state_dict(torch.load("ssd_mobilenet_v3_large.pth"))
model.eval()

# Real-time deployment with OpenCV
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = F.to_tensor(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(input_tensor)[0]

    pred_boxes = predictions['boxes'].cpu().numpy()
    pred_scores = predictions['scores'].cpu().numpy()
    pred_labels = predictions['labels'].cpu().numpy()

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score > 0.5:
            xmin, ymin, xmax, ymax = box.astype(int)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            class_name = classes[label]
            cv2.putText(frame, f"{class_name}: {score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('SSD Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
