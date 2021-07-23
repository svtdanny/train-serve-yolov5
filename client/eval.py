from client import RemoteClient

import cv2
from glob import glob

from datetime import datetime


client = RemoteClient(
    url = '0.0.0.0:8001',
    model_name = 'yolo5s_sm_480_pruned',
    img_size = (480, 480),
)

if __name__=='__main__':
    paths = glob('test/*jpg')

    imgs = [cv2.imread(p) for p in paths]

    start=datetime.now()

    for i, img in enumerate(imgs):
        print(i)
        _ = client.infere_labels(img)

    end = datetime.now()
    delta = end - start
    
    print('FPS')
    print((i+1)/(delta.seconds + delta.microseconds / 1000000.0))