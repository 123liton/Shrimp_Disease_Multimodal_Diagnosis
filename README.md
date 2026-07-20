1.	Python Code for Yolov5
# clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
!git reset --hard 064365d8683fd002e9ad789c1e91fa3d021b44f0
# install dependencies as necessary
!pip install -qr requirements.txt  # install dependencies (ignore errors)
import torch
from IPython.display import Image, clear_output  # to display images
from utils.downloads import attempt_download  # to download models/datasets
# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="ATKPWFvTMs1RnT1DoIL")
project = rf.workspace("liton-library").project("diseases-r837z")
dataset = project.version(4).download("yolov5")
# this is the YAML file Roboflow wrote for us that we're loading into this notebook with our data
%cat {dataset.location}/data.yaml
# define number of classes based on YAML
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])
#this is the model configuration we will use for our tutorial
%cat /content/yolov5/models/yolov5s.yaml
#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic
@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
%%writetemplate /content/yolov5/models/custom_yolov5s.yaml
# parameters
nc: {num_classes}  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]
# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)
   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5) ]
# train yolov5s on custom data for 100 epochs
# time its performance
%%time
%cd /content/yolov5/
!python train.py --img 416 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache
# Evaluate Custom YOLOv5 Detector Performance
from utils.plots import plot_results  # plot results.txt as results.png
Image(filename='/content/yolov5/runs/train/yolov5s_results/results.png', width=1000)  # view results.png
Image(filename='/content/yolov5/runs/train/yolov5s_results/val_batch0_labels.jpg', width=900)
# print out an augmented training example
print("GROUND TRUTH AUGMENTED TRAINING DATA:")
Image(filename='/content/yolov5/runs/train/yolov5s_results/train_batch0.jpg', width=900)
# Run Inference With Trained Weights
Next, we can run inference with a pretrained checkpoint on all images in the `test/images` folder to understand how our model performs on our test set.
# trained weights are saved by default in our weights folder
%ls runs/
%ls runs/train/yolov5s_results/weights
In the snippet below, replace `Cash-Counter-10` with the name of the folder in which your dataset is stored.
%cd /content/yolov5/
!python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source Cash-Counter-10/test/images/
import glob
from IPython.display import Image, display
for imageName in glob.glob('/content/yolov5/runs/detect/exp3/*.jpg')[:10]: #assuming JPG
    display(Image(filename=imageName))
project.version(dataset.version).deploy(model_type="yolov5", model_path=f"/content/yolov5/runs/train/yolov5s_results/")
#Run inference on your model on a persistant, auto-scaling, cloud API
#load model
model = project.version(dataset.version).model
#choose random test set image
import os, random
test_set_loc = dataset.location + "/test/images/"
random_test_image = random.choice(os.listdir(test_set_loc))
print("running inference on " + random_test_image)
pred = model.predict(test_set_loc + random_test_image, confidence=40, overlap=30).json()
pred

2.	Python based code for BERT, NLP
!pip install tensorflow-text
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import pandas as pd

import chardet

with open("/content/drive/MyDrive/sample_labeled_wssv.csv", 'rb') as f:
    result = chardet.detect(f.read())

df = pd.read_csv("/content/drive/MyDrive/sample_labeled_wssv.csv", encoding=result['encoding'])
df.head(100)
import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/sample_labeled_wssv.csv", encoding='latin-1')
df.head(5)
df.groupby('category').describe()
df['category'].value_counts()
df_yes = df[df['category']=='yes']
df_yes.shape

df_no = df[df['category']=='no']
df_no.shape

df_no_downsampled = df_no.sample(df_yes.shape[0], replace=True)
df_yes_downsampled = df_yes.sample(df_no.shape[0])
df_yes_downsampled.shape

df_balanced = pd.concat([df_yes_downsampled, df_no])
df_balanced.shape
df_balanced['category'].value_counts()

df_balanced['yes']=df_balanced['category'].apply(lambda x: 1 if x=='yes' else 0)
df_balanced.sample(5)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_balanced['message'], df_balanced['yes'], stratify=df_balanced['yes'])

X_train.head(4)

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

def get_sentence_embeding(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']

get_sentence_embeding([
    "500$ discount. hurry up",
    "Bhavin, are you up for a volleybal game tomorrow?"]
)

e = get_sentence_embeding([
    "shrimp",
    "water quality",
    "temperature",
    "soil",
    "dog",
    "bill gates"
]
)

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity([e[0]],[e[1]])

cosine_similarity([e[0]],[e[3]])

cosine_similarity([e[0]],[e[2]])

# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])

model.summary()

len(X_train)

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)

model.fit(X_train, y_train, epochs=10)

model.evaluate(X_test, y_test)

y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()

import numpy as np

y_predicted = np.where(y_predicted > 0.5, 1, 0)
y_predicted

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_predicted)

from matplotlib import pyplot as plt
import seaborn as sn
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
print(classification_report(y_test, y_predicted))
reviews = [
    'Erratic swimming patterns and disorientation can be signs of abnormal behavior in shrimps affected by certain diseases',
    'Decreased resistance to infections or increased susceptibility to opportunistic pathogens.',
    'Shrimps hierarchical structures foster stability and efficient resource utilization, ensuring the well-being of the entire group',
    'Through intricate chemical signals and captivating visual displays, shrimps engage in sophisticated forms of communication',
    "Excessive lethargy or reduced swimming activity."
]
model.predict(reviews)

4. Faster R-CNN Image detection code
Enable GPU:
Runtime → Change runtime type → GPU

STEP 2 — Install Required Libraries

!pip install roboflow pycocotools torchmetrics seaborn -q

STEP 3 — Download Dataset from Roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="ATKPWFvTMs1RnT1DoIL")

project = rf.workspace("liton-library").project("diseases-r837z")

version = project.version(1)

dataset = version.download("coco")

STEP 4 — Import Libraries

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO

STEP 5 — Create Custom COCO Dataset Loader

class CocoDataset(Dataset):
    def __init__(self, root, annotation):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        img_id = self.ids[index]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        img = F.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.ids)

STEP 6 — Load Dataset

train_dataset = CocoDataset(
    root=f"{dataset.location}/train",
    annotation=f"{dataset.location}/train/_annotations.coco.json"
)

valid_dataset = CocoDataset(
    root=f"{dataset.location}/valid",
    annotation=f"{dataset.location}/valid/_annotations.coco.json"
)

STEP 7 — Create DataLoader

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

STEP 8 — Load Faster R-CNN Model

model = fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2  # background + disease

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    num_classes
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)

STEP 9 — Define Optimizer

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(params, lr=0.0001)

STEP 10 — Train Faster R-CNN Model

num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    epoch_loss = 0

    for images, targets in train_loader:

        images = list(img.to(device) for img in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        epoch_loss += losses.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

STEP 11 — Save Model

torch.save(model.state_dict(), "faster_rcnn_disease.pth")

STEP 12 — Load Trained Model

model.load_state_dict(torch.load("faster_rcnn_disease.pth"))
model.eval()

STEP 13 — Prediction on Test Image

from google.colab import files

uploaded = files.upload()

image_path = list(uploaded.keys())[0]

img = Image.open(image_path).convert("RGB")

img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

model.eval()

with torch.no_grad():
    prediction = model(img_tensor)

STEP 14 — Visualize Predictions

import matplotlib.patches as patches

fig, ax = plt.subplots(1, figsize=(12,8))

ax.imshow(img)

for box in prediction[0]['boxes'].cpu():

    x1, y1, x2, y2 = box

    rect = patches.Rectangle(
        (x1, y1),
        x2-x1,
        y2-y1,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )

    ax.add_patch(rect)

plt.show()

STEP 15 — Evaluation Metrics

!pip install scikit-learn -q

STEP 16 — Calculate Precision, Recall, F1

import torch
import torchvision.ops as ops

def calculate_precision_recall_f1_object_detection(model, data_loader, device, iou_threshold=0.5, score_threshold=0.0):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(outputs)):
                gt_boxes = targets[i]['boxes'].to(device) # Ground truth boxes
                pred_boxes = outputs[i]['boxes'].to(device) # Predicted boxes
                pred_scores = outputs[i]['scores'].to(device) # Predicted scores

                # Filter predictions by score threshold
                high_score_indices = pred_scores >= score_threshold
                pred_boxes = pred_boxes[high_score_indices]
                pred_scores = pred_scores[high_score_indices]

                num_gt = len(gt_boxes)
                num_pred = len(pred_boxes)

                if num_gt == 0 and num_pred == 0:
                    continue
                if num_gt == 0 and num_pred > 0:
                    total_fp += num_pred
                    continue
                if num_gt > 0 and num_pred == 0:
                    total_fn += num_gt
                    continue

                # Calculate IoU matrix between predicted and ground truth boxes
                # iou_matrix will have shape (num_pred, num_gt)
                iou_matrix = ops.box_iou(pred_boxes, gt_boxes)

                # Keep track of matched ground truth boxes to avoid double counting
                matched_gt = torch.zeros(num_gt, dtype=torch.bool)

                # Iterate over predictions, sorted by confidence score (descending)
                # This ensures higher confidence predictions get priority in matching
                sorted_pred_indices = torch.argsort(pred_scores, descending=True)

                tp_current_image = 0
                fp_current_image = 0
                fn_current_image = 0

                for pred_idx in sorted_pred_indices:
                    best_iou_for_pred = 0
                    best_gt_idx = -1

                    # Find the best ground truth match for the current prediction
                    for gt_idx in range(num_gt):
                        if not matched_gt[gt_idx]: # Only consider unmatched ground truths
                            current_iou = iou_matrix[pred_idx, gt_idx]
                            if current_iou > best_iou_for_pred:
                                best_iou_for_pred = current_iou
                                best_gt_idx = gt_idx

                    if best_gt_idx != -1 and best_iou_for_pred >= iou_threshold:
                        # This is a True Positive (TP)
                        tp_current_image += 1
                        matched_gt[best_gt_idx] = True # Mark ground truth as matched
                    else:
                        # This is a False Positive (FP) - either no match or IoU too low
                        fp_current_image += 1

                # Any unmatched ground truth boxes are False Negatives (FN)
                fn_current_image = num_gt - torch.sum(matched_gt).item()

                total_tp += tp_current_image
                total_fp += fp_current_image
                total_fn += fn_current_image

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

# Calculate object detection specific precision, recall, and F1
precision, recall, f1 = calculate_precision_recall_f1_object_detection(model, valid_loader, device)

print("Object Detection Precision:", precision)
print("Object Detection Recall:", recall)
print("Object Detection F1 Score:", f1)

from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision.ops import box_iou
import numpy as np

y_true = []
y_pred = []

iou_threshold = 0.5

model.eval()

with torch.no_grad():

    for images, targets in valid_loader:

        images = [img.to(device) for img in images]

        outputs = model(images)

        for output, target in zip(outputs, targets):

            pred_boxes = output['boxes'].cpu()
            pred_labels = output['labels'].cpu()

            true_boxes = target['boxes'].cpu()
            true_labels = target['labels'].cpu()

            # If no predictions
            if len(pred_boxes) == 0:
                y_true.extend(true_labels.tolist())
                y_pred.extend([0] * len(true_labels))
                continue

            # Compute IoU matrix
            ious = box_iou(pred_boxes, true_boxes)

            matched_true = set()

            for pred_idx in range(len(pred_boxes)):

                max_iou, true_idx = torch.max(ious[pred_idx], dim=0)

                if max_iou >= iou_threshold and true_idx.item() not in matched_true:

                    y_pred.append(pred_labels[pred_idx].item())
                    y_true.append(true_labels[true_idx].item())

                    matched_true.add(true_idx.item())

                else:
                    # False positive
                    y_pred.append(pred_labels[pred_idx].item())
                    y_true.append(0)

            # False negatives
            for idx in range(len(true_boxes)):
                if idx not in matched_true:
                    y_true.append(true_labels[idx].item())
                    y_pred.append(0)

precision = precision_score(
    y_true,
    y_pred,
    average='weighted',
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average='weighted',
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average='weighted',
    zero_division=0
)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

STEP 17 — Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

STEP 18 — Accuracy

accuracy = np.mean(np.array(y_true) == np.array(y_pred))

print("Accuracy:", accuracy)

STEP 19 — mAP Score (Advanced)

!pip install torchmetrics -q

from torchmetrics.detection.mean_ap import MeanAveragePrecision

metric = MeanAveragePrecision()

model.eval()

with torch.no_grad():

    for images, targets in valid_loader:

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] # Move targets to the same device
        preds = model(images)
        metric.update(preds, targets)
result = metric.compute()
print(result)

STEP 20 — Download Trained Model
from google.colab import files
files.download("faster_rcnn_disease.pth")
Improve Accuracy Further
num_epochs = 30
batch_size = 8
learning_rate = 0.00005

5. Shrimp disease nlp classification using Naive Bayes
# Import libraries
import pandas as pd
import numpy as np
# NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer
# Model
from sklearn.naive_bayes import MultinomialNB
# Train-test split
from sklearn.model_selection import train_test_split
# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# LOAD DATASET
df = pd.read_csv("shrimp_disease_dataset.csv")

print(df.head())
# INPUT AND OUTPUT

X = df["message"]
y = df["class"]
# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#  TEXT VECTORIZATION
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# TRAIN NAIVE BAYES MODEL
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# PREDICTION
y_pred = model.predict(X_test_tfidf)

# EVALUATION
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# TEST WITH NEW INPUT

new_text = [
    "Shrimp shows white spots and reduced feeding"
]
new_text_tfidf = vectorizer.transform(new_text)
prediction = model.predict(new_text_tfidf)
if prediction[0] == 0:
    print("\nPrediction: Disease Detected")
else:
    print("\nPrediction: Normal Shrimp Behavior")






 
6. Python based code for YOLO26s
Install Packages
!pip install -q ultralytics roboflow
Import Libraries
import os
from ultralytics import YOLO
from roboflow import Roboflow
from IPython.display import Image, display
Download Dataset
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="ATKPWFvTMs1RnT1DoIL")
project = rf.workspace("liton-library").project("diseases-r837z")
version = project.version(2)
dataset = version.download("yolo26")
Load YOLO26s
model = YOLO("yolo26s.pt")
Train Model
results = model.train(

    data=f"{dataset.location}/data.yaml",

    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=2,
    pretrained=True,
    optimizer="auto",
    project="Shrimp_Disease",
    name="YOLO26s",
    plots=True,
    save=True,
    verbose=True
)
Validate Model
metrics = model.val(
    data=f"{dataset.location}/data.yaml"
)

print(metrics)
Predict Test Images
model.predict(

    source=f"{dataset.location}/test/images",

    conf=0.25,

    save=True
)
Display Results Folder
!ls runs/detect/YOLO26s
Training Curve
display(Image("runs/detect/YOLO26s/results.png"))
Confusion Matrix
display(Image("runs/detect/YOLO26s/confusion_matrix.png"))
Precision-Recall Curve
display(Image("runs/detect/YOLO26s/PR_curve.png"))
Precision Curve
display(Image("runs/detect/YOLO26s/P_curve.png"))
Recall Curve
display(Image("runs/detect/YOLO26s/R_curve.png"))
F1 Curve
display(Image("runs/detect/YOLO26s/F1_curve.png")
Sample Predictions
display(Image("runs/detect/YOLO26s/val_batch0_pred.jpg"))
Download Best Model
from google.colab import files

files.download("runs/detect/YOLO26s/weights/best.pt")
Read Final Metrics
import pandas as pd

results = pd.read_csv("runs/detect/YOLO26s/results.csv")

results.tail()


