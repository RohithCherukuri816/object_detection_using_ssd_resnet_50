# SSD MobileNetV3 Object Detection using Pascal VOC Dataset

This project implements an object detection pipeline using the SSD MobileNetV3 model with the Pascal VOC dataset. The model is trained to detect various object classes and is deployed for real-time object detection using OpenCV.

## Project Structure

- `VOC2007/`: Directory containing the Pascal VOC 2007 dataset.
  - `JPEGImages/`: Contains all the images.
  - `Annotations/`: Contains XML annotation files for the images.
- `ssd_mobilenet_v3_large.pth`: Saved model weights after training.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- Scikit-learn
- Torchvision

## Pretrained Model

Download a [SSD Resnet-50](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) model from a collection of pretrained models [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and move it to the `object_detection` folder.
