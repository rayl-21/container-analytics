# YOLOv12 Configuration

## Project Configuration
The Container Analytics project is set up to use YOLOv12 for object detection, specifically for detecting containers and trucks in port gate camera images.

## Key Points
- YOLOv12 is the attention-centric object detection model specified in the project requirements
- The model weights should be stored in `data/models/` directory 
- Model file: `yolov12x.pt` or similar YOLOv12 variant
- Currently falling back to YOLOv8 when YOLOv12 models are not available

## Implementation Notes
- The detection module is located at `modules/detection/yolo_detector.py`
- Uses ultralytics library for YOLO implementation
- Configuration parameters:
  - confidence_threshold: 0.2
  - iou_threshold: 0.7
  - batch_size: 8 for batch processing

