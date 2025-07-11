## Obstacle Detection for the Visually Impaired Using Deep Learning and Audio Cues
A real-time multi-modal perception system that combines computer vision and spatial audio to provide environmental awareness for visually impaired individuals navigating Indonesian roads.

## Brief Overview
This project implements an intelligent assistive navigation system that transforms visual information into intuitive spatial audio feedback. By mounting a smartphone and wearing headphones, users receive real-time audio cues about obstacles, people, and vehicles in their environment.

## System Architecture
-Primary Detection: YOLOv11-Nano for moving objects (people, vehicles)
-Secondary Detection: OpenCV contour analysis for environmental structures
-Additional Detection: Custom YOLO model for specific road obstacles
-Depth Analysis: Depth Anything V2 for distance estimation
-Audio Processing: Stereo panning with volume-based proximity feedback

## Running the System

## Changing the Depth Analysis Model
To change the model used for the depth analysis model, simply replace the .pth inside the checkpoints folder to one of these 3 models
| Base Model | Params | Indoor (Hypersim) | Outdoor (Virtual KITTI 2) |
|:-|-:|:-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true) | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth?download=true) | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true) | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true) |

## Demonstration Video
[![Demonstration Video](https://img.youtube.com/vi/njkRDEua8Kk/hqdefault.jpg)](https://youtu.be/njkRDEua8Kk)

## Poster
![Image description](https://raw.githubusercontent.com/Rayes36/object-detection-with-distance/audio-branch/obstacle-detection-ai-research-poster/1.png)
