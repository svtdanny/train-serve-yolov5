# train-serve-yolov5

In this project I:
1. Converted MS COCO annotations to YOLO annotations
2. Built object detection model 
3. Accelerated the model (quantization, smaller inference size, optimization), 
4. Suggested metrics for tracking model degradation and data consistency in production
5. Converted model to ONNX format
6. Deployed model on Nvidia Triton with TensorRT backend engine (for maximum inference speed)
7. Wrote grpc client and tested inference FPS

Main Goal: fastest inference

Test machine config:
1. No GPU
2. Core i7-8th
3. 16 GB RAM

Learning artefacts + metric plots:  
https://wandb.ai/svtdanny/yolov5/overview

Results (imahes sended sequentially, batch_size = 1):

![results_table](https://user-images.githubusercontent.com/66482706/126733557-fa2dd7fb-0c31-4812-baf9-ff8a1f602b06.png)
  
Object_Detection.ipynb [MS COCO -> YOLO labels, transfer learning, accelerating, converting]  
Online_validation.ipynb [online metrics]  

