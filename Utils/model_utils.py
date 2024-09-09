import math
import random

import cv2
import numpy as np
from PIL import Image
from functools import partial

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from Utils.FaceNet_model import FaceNet

def preprocess_input(image):
    """
    Rescale image into [0, 1] range with float data type.

    Args:
        image: numpy array of image.

    Returns:
        new preprocessed image
    """
    image /= 255.0 
    return image

def resize_image(image, size, mode='train'):
    """
    Resize image into specific size

    Args:
        image: numpy array of image.
        size: specific size for resizing

    Returns:
        new preprocessed image
    """
    if mode == 'train':
        new_image = image.resize(size, Image.BICUBIC)
    elif mode == 'test':
        new_image = cv2.resize(image, size, interpolation = cv2.INTER_CUBIC)
    
    return new_image


def detect_image(image_1, image_2, model, input_shape, verbose=0):
    """
    Detect two images and output L2 norm after model prediction.

    Args:
        image_1: numpy array of image
        image_2: numpy array of image
        model: tensorflow trained model
        input_shape: input shape of images
        show_result: whether to show predicted results
        verbose: "auto", 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line. 
    
    Returns:
        L2 norm values
    """
    # resize image
    image_1 = resize_image(image_1, [input_shape[1], input_shape[0]], mode='test')
    image_2 = resize_image(image_2, [input_shape[1], input_shape[0]], mode='test')
    
    # rescale image
    photo_1 = np.expand_dims(preprocess_input(np.array(image_1, np.float32)), 0)
    photo_2 = np.expand_dims(preprocess_input(np.array(image_2, np.float32)), 0)

    #---------------------------------------------------#
    # model predictions
    #---------------------------------------------------#
    output1 = model.predict(photo_1, verbose=verbose)
    output2 = model.predict(photo_2, verbose=verbose)
    
    #---------------------------------------------------#
    #  calculate distance between them
    #---------------------------------------------------#
    l2 = np.linalg.norm(output1-output2, axis=1)
    # l2 = np.sum(np.square(output1 - output2), axis=-1)
    
    return l2, (photo_1, photo_2)

def classify_image(image, model, input_shape, verbose=0):
    """
    Detect two images and output L2 norm after model prediction.

    Args:
        image: numpy array of image
        model: tensorflow trained model
        input_shape: input shape of images
        verbose: "auto", 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line. 
    Returns:
        Classification results
    """
    # resize image
    image = resize_image(image, [input_shape[1], input_shape[0]], mode='test')
    
    # rescale image
    photo = np.expand_dims(preprocess_input(np.array(image, np.float32)), 0)

    # model predictions
    prediction, _ = model.predict(photo, verbose=verbose)
    class_result = np.argmax(prediction)
    
    return class_result



def cvtColor(image):
    """
    Convert image into RGB channels
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


def triplet_loss(alpha = 0.2):
    """
    Triplet loss for model training.
    """
    def _triplet_loss(y_true, y_pred):
        batch_size = tf.shape(y_pred)[0] // 3
        anchor, positive, negative = y_pred[:batch_size], y_pred[batch_size:2 * batch_size], y_pred[-batch_size:]

        pos_dist    = K.sqrt(K.sum(K.square(anchor - positive), axis=-1))
        neg_dist    = K.sqrt(K.sum(K.square(anchor - negative), axis=-1))

        basic_loss  = pos_dist - neg_dist + alpha
        
        idxs        = tf.where(basic_loss > 0)
        select_loss = tf.gather_nd(basic_loss, idxs)

        loss        = K.sum(K.maximum(basic_loss, 0)) / tf.cast(tf.maximum(1, tf.shape(select_loss)[0]), tf.float32)
        return loss
    return _triplet_loss


# get number of classes
def get_num_classes(annotation_path):
    """
    Get number of classes.
    """
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes


def show_config(**kwargs):
    """
    Show all the configuration about training.
    """
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


class FaceNetDataset(keras.utils.Sequence):
    def __init__(self, input_shape, lines, batch_size, num_classes, ngpus_per_node, random, cluster=False):
        self.input_shape    = input_shape
        self.lines          = lines
        self.length         = len(lines)
        self.batch_size     = batch_size
        self.num_classes    = num_classes
        self.ngpus_per_node = ngpus_per_node
        self.random         = random
        self.cluster        = cluster
        
        #  Paths and labels
        self.paths  = []
        self.labels = []

        self.load_dataset()
        
    def __len__(self):
        return math.ceil(self.length / float(self.batch_size))

    def __getitem__(self, index):
        multi_gpus_images = []
        multi_gpus_labels = []

        for gpu in range(self.ngpus_per_node):

            # calculate number of gpu
            bs_per_gpu = self.batch_size // self.ngpus_per_node // 3
            
            #  create zero matrices
            images = np.zeros((bs_per_gpu, 3, self.input_shape[0], self.input_shape[1], 3))
            labels = np.zeros((bs_per_gpu, 3))
            
            for i in range(bs_per_gpu):
                
                # Choose 2 face images from one person: Anchor and Positive
                c               = random.randint(0, self.num_classes - 1)
                selected_path   = self.paths[self.labels[:] == c]
                while len(selected_path) < 2:
                    c               = random.randint(0, self.num_classes - 1)
                    selected_path   = self.paths[self.labels[:] == c]

                # choose 2 images randomly
                image_indexes = np.random.choice(range(0, len(selected_path)), 2)
                
                # open images and put into matrix: Anchor
                image = cvtColor(Image.open(selected_path[image_indexes[0]]))

                # data augmentation: flip images randomly
                if self.rand()<.5 and self.random: 
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image = resize_image(image, [self.input_shape[1], self.input_shape[0]])
                image = preprocess_input(np.array(image, dtype='float32'))

                images[i, 0, :, :, :] = image
                labels[i, 0] = c
                
                # open images and put into matrix: Positive
                image = cvtColor(Image.open(selected_path[image_indexes[1]]))
             
                # data augmentation: flip images randomly
                if self.rand()<.5 and self.random: 
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image = resize_image(image, [self.input_shape[1], self.input_shape[0]])
                image = preprocess_input(np.array(image, dtype='float32'))
                
                images[i, 1, :, :, :] = image
                labels[i, 1] = c

                # choose another face: Negative
                different_c         = list(range(self.num_classes))
                different_c.pop(c) # [0, num_classes-1]
                
                different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
                current_c           = different_c[different_c_index[0]]
                selected_path       = self.paths[self.labels == current_c]
                while len(selected_path) < 1:
                    different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
                    current_c           = different_c[different_c_index[0]]
                    selected_path       = self.paths[self.labels == current_c]

                # choose one image randomly
                image_indexes       = np.random.choice(range(0, len(selected_path)), 1)
                image               = cvtColor(Image.open(selected_path[image_indexes[0]]))

                # data augmentation: flip images randomly
                if self.rand()<.5 and self.random: 
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image = resize_image(image, [self.input_shape[1], self.input_shape[0]])
                image = preprocess_input(np.array(image, dtype='float32'))
                
                images[i, 2, :, :, :] = image
                labels[i, 2] = current_c

            #--------------------------------------------------------------#
            # Assume batch size is 32 (0, 32, 64) belongs to same group
            # 0 and 32 is the same person, 0 and 64 not
            #--------------------------------------------------------------#
            images1 = np.array(images)[:, 0, :, :, :]
            images2 = np.array(images)[:, 1, :, :, :]
            images3 = np.array(images)[:, 2, :, :, :]
            images = np.concatenate([images1, images2, images3], 0)
            
            labels1 = np.array(labels)[:, 0]
            labels2 = np.array(labels)[:, 1]
            labels3 = np.array(labels)[:, 2]
            labels = np.concatenate([labels1, labels2, labels3], 0)

            labels = np_utils.to_categorical(np.array(labels), num_classes = self.num_classes)  

            multi_gpus_images.append(images)
            multi_gpus_labels.append(labels)

        multi_gpus_images = np.concatenate(multi_gpus_images, 0)
        multi_gpus_labels = np.concatenate(multi_gpus_labels, 0)

        if self.cluster:
            return multi_gpus_images, {'cluster_Embedding' : np.zeros_like(multi_gpus_labels), 'cluster_Softmax' : multi_gpus_labels}
        else:
            return multi_gpus_images, {'Embedding' : np.zeros_like(multi_gpus_labels), 'Softmax' : multi_gpus_labels}
   
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def load_dataset(self):
        for path in self.lines:
            path_split = path.split(";")
            self.paths.append(path_split[1].split()[0])
            self.labels.append(int(path_split[0]))
        self.paths  = np.array(self.paths, dtype=object) # np.object
        self.labels = np.array(self.labels)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


class FaceNetExperiment:
    def __init__(self, 
                 annotation_path:str="cls_train.txt", 
                 input_shape:list=[160, 160, 3], 
                 model_path:str="model_data/facenet_mobilenet.h5", 
                 pre_train_model=None, 
                 init_epoch:int=0, 
                 epoch:int=30, 
                 batch_size:int=24, 
                 val_split:float=0.1, 
                 optimizer_type:str="adam", 
                 lr_decay_type:str="cos", 
                 init_lr:float=1e-2, 
                 cluster=False):
        """
        Initialization of FaceNet model experiment.

        Args:
            annotation_path: Path for face classes and image paths
            input_shape: Input shape for model acceptable format
            model_path: If a training model already exists, pass the model path in the parameter.
            pre_train_model: If a pre-trained model has been instantiated, pass directly into the parameter.
            init_epoch: Initial epoch of the training
            epoch: Total epochs of the training
            batch_size: Batch size for feeding to the model each time
            val_split: Splitting ratio of the validation dataset
            optimizer_type: Algorithms for optimization during mdoel training
            lr_decay_type: Methods using learning rate decay, 'step' or 'cos'
            init_lr: Initial learning rate during model training
            cluster:: Whether to apply the cluster model and dataset, this determines the loss names.
        """
        self.annotation_path = annotation_path
        self.input_shape = input_shape
        self.model_path = model_path
        self.pre_train_model = pre_train_model
        self.Init_Epoch = init_epoch
        self.Epoch = epoch

        # check the resonability of batch size
        self.batch_size = self.batch_size_validation(batch_size)
        
        # Set hyperparameters
        self.optimizer_type = optimizer_type
        self.lr_decay_type = lr_decay_type
        self.cluster = cluster
        
        self.prepared = False
        self.ngpus_per_node = 1 # number of GPU
        self.val_split = val_split # validation split

        # ----------------------------------
        # 1. Create number of face classes 
        # 2. Generate num_train and num_val
        # 3. Check resonability of epochs
        # ----------------------------------
        self.lines, self.num_classes = self.get_lines() 
        print(f"Number of classes: {self.num_classes}")
                
        # load model
        self.model = self.pre_train_model if self.pre_train_model else self.load_model()

        # -----------------------------------------------
        # Generate learning rate parameters
        # 1. Set limitation of learning rate
        # 2. Generate initial and minimum learning rate
        # -----------------------------------------------
        self.Init_lr = init_lr
        self.set_lr_fit()
        self.optimizers = {
            'adam'  : Adam(learning_rate = self.Init_lr_fit, beta_1 = self.momentum),
            'sgd'   : SGD(learning_rate = self.Init_lr_fit, momentum = self.momentum, nesterov=True)
        }

        # load datasets
        self.train_dataset, self.val_dataset = self.crate_dataset()
        
        self.prepared = True
        self.ready_to_run = False
        if self.prepared:
            show_config(
                num_classes = self.num_classes, model_path = self.model_path, \
                input_shape = self.input_shape, optimizer = self.optimizer_type, \
                learning_rate_fit = self.Init_lr_fit, learning_rate_decay = self.lr_decay_type, \
                Init_Epoch = self.Init_Epoch, Epoch = self.Epoch, batch_size = self.batch_size, \
                num_train = self.num_train, num_val = self.num_val, cluster = self.cluster
            )

            self.ready_to_run = True
            print(f"Ready to train !")

    def batch_size_validation(self, batch_size):
        """
        Batch size must be the multiple of 3.
        """
        if batch_size % 3 != 0:
            raise ValueError("Batch_size must be the multiple of 3.")
        else: 
            return batch_size
        
    def set_lr_fit(self):
        """
        Set initial and minimum learning rate for model fitting.
        """
        self.Min_lr         = self.Init_lr * 0.01
        self.momentum       = 0.9
        self.nbs            = 16
        self.lr_limit_max   = 1e-3 if self.optimizer_type == 'adam' else 1e-1
        self.lr_limit_min   = 3e-4 if self.optimizer_type == 'adam' else 5e-4
        self.Init_lr_fit    = min(max(self.batch_size / self.nbs * self.Init_lr, self.lr_limit_min), 
                                  self.lr_limit_max)
        self.Min_lr_fit     = min(max(self.batch_size / self.nbs * self.Min_lr, self.lr_limit_min * 1e-2), 
                                  self.lr_limit_max * 1e-2)

    def crate_dataset(self) -> tuple:
        """
        Create train dataset for model training and validation dataset for model validation.
        """
        train_dataset   = FaceNetDataset(self.input_shape, self.lines[:self.num_train], self.batch_size, 
                                         self.num_classes, self.ngpus_per_node, random = True, cluster=self.cluster)
        val_dataset     = FaceNetDataset(self.input_shape, self.lines[self.num_train:], self.batch_size, 
                                         self.num_classes, self.ngpus_per_node, random = False, cluster=self.cluster)
        return (train_dataset, val_dataset)
        
    def load_model(self):
        """
        Load FaceNet model from model_path if pre_train_model is None.
        """
        model = FaceNet(input_shape=self.input_shape, num_classes=self.num_classes, mode="train")
        if self.model_path != '':
            model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
            return model
        else: return

    def check_epoch_steps(self):
        """
        Check whether epoch step and epoch step val are reasonable.
        """
        self.epoch_step = self.num_train // self.batch_size
        self.epoch_step_val = self.num_val // self.batch_size
            
        if self.epoch_step == 0 or self.epoch_step_val == 0:
            raise ValueError('Dataset is too small to train, please collect more data.')
            
    def get_lines(self):
        """
        Get all the lines from annotation_path including labels and face paths.
        """
        with open(self.annotation_path,"r") as f:
            lines = f.readlines()
        
        # prpare labels
        labels = []
        for line in lines:
            path_split = line.split(";")
            labels.append(int(path_split[0]))
        num_classes = np.max(labels) + 1

        # prepare dataset fro training
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        self.num_val = int(len(lines) * self.val_split)
        self.num_train = len(lines) - self.num_val
        
        self.check_epoch_steps() # check the reasonability of epoch steps.
        
        return lines, num_classes

    def train(self, loss_names=['Embedding', 'Softmax']):
        """
        Train the FaceNet model load from path, there are two loss functions during training, \
        you have to pass the loss names to the argument.

        Args:
            loss_names: Loss names for triplet loss and categorical_crossentropy loss funcitons.
        Returns:
            Training history
        """
        if self.ready_to_run:
            self.lr_scheduler_func = get_lr_scheduler(lr_decay_type = self.lr_decay_type, 
                                                      lr = self.Init_lr_fit, 
                                                      min_lr = self.Min_lr_fit, 
                                                      total_iters = self.Epoch)
            self.optimizer = self.optimizers[self.optimizer_type]
            triplet_loss_name, softmax_loss_name = loss_names
            self.model.compile(
                loss={triplet_loss_name : triplet_loss(), softmax_loss_name : 'categorical_crossentropy'},
                optimizer = self.optimizer, metrics = {softmax_loss_name : 'categorical_accuracy'}
            )
            self.early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 5, verbose = 1)
            self.lr_scheduler    = LearningRateScheduler(self.lr_scheduler_func, verbose = 1)
            self.callbacks = [self.early_stopping, self.lr_scheduler]
        
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(self.num_train, 
                                                                                       self.num_val, 
                                                                                       self.batch_size))
            self.facenet_model_hitory = self.model.fit(x                   = self.train_dataset,
                                                       steps_per_epoch     = self.epoch_step,
                                                       validation_data     = self.val_dataset,
                                                       validation_steps    = self.epoch_step_val,
                                                       epochs              = self.Epoch,
                                                       initial_epoch       = self.Init_Epoch,
                                                       callbacks           = self.callbacks)
            return self.facenet_model_hitory
        else:
            raise ValueError('Something is not prepared, please check again !')

    def save_model(self, 
                   weight_path:str='./model_data/facenet_mobilenet_fine_tune_weights.h5', 
                   model_path:str='./model_data/facenet_mobilenet_classifier.h5'):
        self.model.save_weights(weight_path)
        self.model.save(model_path)
        print('Model has been saved as {} and {}'.format(weight_path, model_path))



def plot_facenet_training_history(history):
    """
    Plot training history of facenet model.
    """ 
    epochs = range(1, len(history['loss']) + 1)
    
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    
    # Loss
    axs[0, 0].plot(epochs, history['loss'], label='Training Loss')
    axs[0, 0].plot(epochs, history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    # Softmax Loss
    axs[0, 1].plot(epochs, history['Softmax_loss'], label='Training Softmax Loss')
    axs[0, 1].plot(epochs, history['val_Softmax_loss'], label='Validation Softmax Loss')
    axs[0, 1].set_title('Softmax Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Softmax Loss')
    axs[0, 1].legend()
    
    # Embedding Loss
    axs[1, 0].plot(epochs, history['Embedding_loss'], label='Training Embedding Loss')
    axs[1, 0].plot(epochs, history['val_Embedding_loss'], label='Validation Embedding Loss')
    axs[1, 0].set_title('Embedding Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Embedding Loss')
    axs[1, 0].legend()
    
    # Softmax Categorical Accuracy
    axs[1, 1].plot(epochs, history['Softmax_categorical_accuracy'], label='Training Softmax Categorical Accuracy')
    axs[1, 1].set_title('Softmax Categorical Accuracy')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].legend()
    
    # Learning Rate
    axs[2, 0].plot(epochs, history['lr'], label='Learning Rate', color='orange')
    axs[2, 0].set_title('Learning Rate')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Learning Rate')
    axs[2, 0].legend()
    
    # white figure
    axs[2, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_cluster_facenet_training_history(history):
    """
    Plot training history of cluster facenet model.
    """ 
    epochs = range(1, len(history['loss']) + 1)
    
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    
    # Loss
    axs[0, 0].plot(epochs, history['loss'], label='Training Loss')
    axs[0, 0].plot(epochs, history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    # Softmax Loss
    axs[0, 1].plot(epochs, history['cluster_Softmax_loss'], label='Training Softmax Loss')
    axs[0, 1].plot(epochs, history['val_cluster_Softmax_loss'], label='Validation Softmax Loss')
    axs[0, 1].set_title('Softmax Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Softmax Loss')
    axs[0, 1].legend()
    
    # Embedding Loss
    axs[1, 0].plot(epochs, history['cluster_Embedding_loss'], label='Training Embedding Loss')
    axs[1, 0].plot(epochs, history['val_cluster_Embedding_loss'], label='Validation Embedding Loss')
    axs[1, 0].set_title('Embedding Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Embedding Loss')
    axs[1, 0].legend()
    
    # Softmax Categorical Accuracy
    axs[1, 1].plot(epochs, history['cluster_Softmax_categorical_accuracy'], label='Training Softmax Categorical Accuracy')
    axs[1, 1].plot(epochs, history['val_cluster_Softmax_categorical_accuracy'], label='Validation Softmax Categorical Accuracy')
    axs[1, 1].set_title('Softmax Categorical Accuracy')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].legend()
    
    # Learning Rate
    axs[2, 0].plot(epochs, history['lr'], label='Learning Rate', color='orange')
    axs[2, 0].set_title('Learning Rate')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Learning Rate')
    axs[2, 0].legend()
    
    # white figure
    axs[2, 1].axis('off')
    
    plt.tight_layout()
    plt.show()