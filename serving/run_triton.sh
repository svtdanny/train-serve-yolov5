sudo docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v /home/danil/Desktop/yolo_deploy/deploy_models/:/models \
    nvcr.io/nvidia/tritonserver:21.06.1-py3 tritonserver --model-repository=/models --strict-model-config=false --log-verbose=1 #--model-control-mode=EXPLICIT
