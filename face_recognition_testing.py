import os
import cv2
import glob
import random
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from Utils.model_utils import classify_image, triplet_loss

np.random.seed(42)
Join = os.path.join
TEST_DATASET = './datasets'
INPUT_SHAPE = [160, 160, 3]
FaceNet_classifier_path = './model_data/facenet_mobilenet_classifier.h5'

# helper function
accuracy_score = lambda y_true, y_pred, decimal=3: np.round((np.sum(np.equal(y_true, y_pred)) / len(y_true)), decimals=decimal)

def relu6(x):
    return K.relu(x, max_value=6)

def main(model_path:str=FaceNet_classifier_path,
         input_shape:list=INPUT_SHAPE,
         data_path:str=TEST_DATASET,
         rounds:int=10,
         target_score:float=0.9):
    """
    Test the face_recognition class.
    
    Args:
        model_path: What type of algorithm to be loaded.
        input_shape: Input shape for model.
        data_path: Path for dataset expected to be tested.
        rounds: Rounds for testing number. The default is 10.
        target_score: The score to be beat. The default is 0.9.
    Returns:
        Pass or Failure
    """
    # load model
    FaceNet_model = load_model(model_path, custom_objects={'_triplet_loss': triplet_loss, 'relu6': relu6})
    
    # read dataset
    classes = os.listdir(data_path)
    
    # testing
    y_true = []
    y_pred = []
    for _ in range(rounds):
        real_idx = random.randint(0, len(classes)-1)
        real_class = classes[real_idx]
        faces = glob.glob(Join(TEST_DATASET, real_class, '*'))
        image_path = random.choice(faces)
        
        # face classification
        image = cv2.imread(image_path) # read image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_idx = classify_image(image_rgb, model=FaceNet_model, input_shape=input_shape)
        pred_class = classes[pred_idx]
        print('real label: %s | prediction:  %s' % (real_class, pred_class))
        y_true.append(real_idx)
        y_pred.append(pred_idx)

    acc_score = accuracy_score(y_true, y_pred)
    result = 'Pass' if acc_score >= target_score else 'Fail'
    print('Accuracy score is %s, result: %s' % (acc_score, result))

if __name__ == '__main__':
    main(rounds=50)