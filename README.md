# OpenCV--Background-Modelling-Detection-and-Tracking-of-Pedestrians

# Video Processing for Object Classification and Pedestrian Tracking

This repository contains Python scripts for video processing tasks, including background modeling and object classification, as well as pedestrian detection and tracking. The project utilizes OpenCV for computer vision tasks and a pre-trained MobileNet SSD model for object detection.

## Table of Contents

- [Introduction](#introduction)
- [Tasks](#tasks)
  - [Task 1: Background Modeling and Object Classification](#task-1-background-modeling-and-object-classification)
  - [Task 2: Pedestrian Detection and Tracking](#task-2-pedestrian-detection-and-tracking)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [License](#license)

## Introduction

This project aims to demonstrate how to process video streams for object detection and classification, as well as pedestrian detection and tracking. It involves background subtraction, object classification based on aspect ratio, and pedestrian tracking using a pre-trained MobileNet SSD model.

## Tasks

### Task 1: Background Modeling and Object Classification

This task uses a background subtractor to detect moving objects in a video stream and classify them based on their aspect ratio into three categories: persons, cars, and others. The processed video displays four views: the original frame, background model, binary mask, and color-highlighted objects.

#### Features

- Background subtraction using MOG2.
- Object classification based on aspect ratio.
- Visualization of original frame, background model, binary mask, and color-highlighted objects.

### Task 2: Pedestrian Detection and Tracking

This task uses a pre-trained MobileNet SSD model to detect pedestrians in a video stream and tracks them using KCF trackers. It calculates the distance of each pedestrian from the camera and highlights the three closest pedestrians.

#### Features

- Pedestrian detection using MobileNet SSD.
- Tracking of detected pedestrians using KCF trackers.
- Calculation of pedestrian distances from the camera.
- Visualization of detected and tracked pedestrians.

## Installation

### Prerequisites

- Python 3.x
- OpenCV
- NumPy
