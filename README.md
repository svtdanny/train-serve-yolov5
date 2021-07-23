# train-serve-yolov5

In this project I:
1. Converted MS COCO annotation to YOLO annotation 
2. Built object detection model 
3. Accelerated the model, 
4. Suggested metrics for tracking model degradation and data consistency in production
5. Converted model to ONNX format
6. Deployed model on Nvidia Triton with TensorRT backend engine (for maximum inference speed)
7. Wrote grpc client and tested inference FPS

Main Goal: fastest inference

Test machine config:
1. No GPU
2. Core i7-8th
3. 16 GB RAM

Results:


Object_Detection.ipynb [MS COCO -> YOLO labels, transfer learning, accelerating, converting]
Online_validation.ipynb [online metrics]

