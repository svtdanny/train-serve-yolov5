from typing import Tuple, Optional

import numpy as np
import cv2

import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype


class RemoteClient:
    def __init__(self, 
                url: str,
                model_name: str,
                img_size: Tuple[int, int],
                inf_type: Optional[str] = 'FP32',
                ) -> None:
        
        self.url = url
        self.client = grpcclient.InferenceServerClient(url=self.url, verbose=0)
        self.model_name = model_name

        self.img_size = img_size
        self.inf_type = inf_type

    def _preprocess_img(self, img: np.array) -> np.array:
        assert len(img.shape) == 3, 'Err shape: should be (h, w, n_channels)'
        assert img.shape[2] == 3, 'Err n_channels: should be 3'

        if (img.shape[0] != self.img_size[0]) or (img.shape[1] != self.img_size[1]):
            img = cv2.resize(img, self.img_size)
        
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        #img = np.expand_dims(img, axis=0)
        img = np.moveaxis(img, -1, 0)

        return img

    def _send_req_grpc(self, batch_imgs: np.array, url: str) -> np.array:
        inputs = []
        #print(batch_imgs.dtype)
        # Set the input data
        inputs.append(grpcclient.InferInput('images', batch_imgs.shape, self.inf_type))
        inputs[0].set_data_from_numpy(batch_imgs)

        results = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
     
            compression_algorithm=None
        )
        #print(results)
        res = results.as_numpy('output')    
        #print(res)
        return res

    def infere_labels(self, imgs: np.array) -> np.array:
        imgs = imgs if len(imgs.shape) == 4 else np.expand_dims(imgs, axis=0)

        imgs_prep = np.zeros((imgs.shape[0], 3, self.img_size[0], self.img_size[1]))
        for i in range(len(imgs)):
            imgs_prep[i] = self._preprocess_img(imgs[i])
        
        npdtype = triton_to_np_dtype(self.inf_type)
        imgs_prep = imgs_prep.astype(npdtype)

        res = self._send_req_grpc(imgs_prep, self.url)

        return res