import imp
from typing import Dict

import SimpleITK
import tqdm
import json
from pathlib import Path
import tifffile
import numpy as np
import imgaug.augmenters as iaa
import torch
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.models import Model

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from evalutils.io import ImageLoader


class DummyLoader(ImageLoader):
    @staticmethod
    def load_image(fname):
        return str(fname)


    @staticmethod
    def hash_image(image):
        return hash(image)


class airogs_algorithm(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        self._file_loaders = dict(input_image=DummyLoader())

        self.output_keys = ["multiple-referable-glaucoma-likelihoods", 
                            "multiple-referable-glaucoma-binary",
                            "multiple-ungradability-scores",
                            "multiple-ungradability-binary"]
        
        self.aug = iaa.GammaContrast(2)
        self.INPUT_SHAPE = (224,224,3)
        base_model = keras.applications.ResNet50(input_shape=self.INPUT_SHAPE,
                                         include_top=False,
                                         weights=None)
        x = Flatten()(base_model.layers[-1].output)
        x = Dense(1024, activation='relu', name='fc1024')(x)
        x = Dropout(rate=0.2, name='dropout1')(x, training=True)
        x = Dense(256, activation='relu', name='fc256')(x)
        x = Dropout(rate=0.2, name='dropout2')(x, training=True)
        x = Dense(64, activation='relu', name='fc64')(x)
        x = Dropout(rate=0.2, name='dropout3')(x, training=True)
        x = Dense(1, activation='sigmoid', name='fc1')(x)
        self.model = Model(inputs=base_model.input,outputs=x)

        self.model.load_weights('weights.h5')
        self.yolo_model = torch.load('model')
        #self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')

    
    def load(self):
        for key, file_loader in self._file_loaders.items():
            fltr = (
                self._file_filters[key] if key in self._file_filters else None
            )
            self._cases[key] = self._load_cases(
                folder=Path("/input/images/color-fundus/"),
                file_loader=file_loader,
                file_filter=fltr,
            )

        pass
    
    def combine_dicts(self, dicts):
        out = {}
        for d in dicts:
            for k, v in d.items():
                if k not in out:
                    out[k] = []
                out[k].append(v)
        return out
    
    def process_case(self, *, idx, case):
        # Load and test the image(s) for this case
        if case.path.suffix == '.tiff':
            results = []
            with tifffile.TiffFile(case.path) as stack:
                for page in tqdm.tqdm(stack.pages):
                    input_image_array = page.asarray()
                    results.append(self.predict(input_image_array=input_image_array))
        else:
            input_image = SimpleITK.ReadImage(str(case.path))
            input_image_array = SimpleITK.GetArrayFromImage(input_image)
            results = [self.predict(input_image_array=input_image_array)]
        
        results = self.combine_dicts(results)

        # Test classification output
        if not isinstance(results, dict):
            raise ValueError("Expected a dictionary as output")

        return results

    def predict(self, *, input_image_array: np.ndarray) -> Dict:
        # From here, use the input_image to predict the output
        # We are using a not-so-smart algorithm to predict the output, you'll want to do your model inference here

        # Replace starting here
        
        roi_cropped = cv2.resize(input_image_array,(1200,1200))
        roi_cropped = self.yolo_model(roi_cropped)
        roi_cropped = roi_cropped.crop()
        confidence = 0.0001
        if roi_cropped is not None and len(roi_cropped)>0:
            roi_cropped = roi_cropped[0]
            if roi_cropped is not None and 'im' in roi_cropped:
                confidence = roi_cropped['conf']
                roi_cropped = roi_cropped['im']
                roi_cropped = cv2.resize(roi_cropped, (self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]))
            else:
                roi_cropped = cv2.resize(input_image_array, (self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]))
        else:
            roi_cropped = cv2.resize(input_image_array, (self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]))


        image_arr = self.aug.augment_image(roi_cropped)
        image_arr[:,:,0] = image_arr[:,:,1]
        image_arr[:,:,2] = image_arr[:,:,1]
        if np.max(image_arr)>0:
            image_arr=image_arr/(np.max(image_arr))
        else:
            image_arr = image_arr
        
        preds = []
        for i in range(5):
            preds.append(self.model.predict(np.expand_dims(image_arr, axis=0))[0][0])
            
            
        preds = np.array(preds)
        rg_likelihood = np.mean(preds, axis=0)
        aleatoric = np.mean(preds*(1-preds), axis=0)
    
        rg_binary = bool(rg_likelihood > .5)
        ungradability_score = max(rg_likelihood, 1-rg_likelihood)
        ungradability_binary = bool(aleatoric > .2)
       

        out = {
            "multiple-referable-glaucoma-likelihoods": float(rg_likelihood),
            "multiple-referable-glaucoma-binary": rg_binary,
            "multiple-ungradability-scores": 10*float(confidence),
            "multiple-ungradability-binary": ungradability_binary
        }

        return out

    def save(self):
        for key in self.output_keys:
            with open(f"/output/{key}.json", "w") as f:
                out = []
                for case_result in self._case_results:
                    out += case_result[key]
                json.dump(out, f)


if __name__ == "__main__":
    airogs_algorithm().process()
