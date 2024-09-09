import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
from PIL import Image, ImageOps 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

class FaceClassifier:
    def __init__(self, 
                 root_dir:str, 
                 face_datasets:str='datasets',
                 threshold:int=50,
                 img_size=(224, 224)):
        """
        Run face classification with given model.

        Args:
            root_dir: Working directory for experiments.
            face_datasets: directory of face images.
            threshold: Number of images filtered.
            img_size: Image size of each face image in dataset.
        """
        self.root_dir = root_dir
        self.face_datasets = face_datasets
        self.threshold = threshold
        self.img_size = img_size

        # Rescaling
        self.scaler = StandardScaler()

        # Create training and testing dataset
        self.X_train, self.X_test, self.y_train, self.y_test = self.create_tain_and_test_data()

    def read_images(self) -> tuple:
        """
        Read images and labels from face datasets.
    
        Returns:
            tuple of face features, face labels, face paths.
        """
        face_features = []
        face_labels = []
        label_names = {}

        path_join = os.path.join
        face_cls = 0
        for data in os.listdir(self.face_datasets):
            faces = os.listdir(path_join(self.face_datasets, data))
            if len(faces) >= self.threshold:
                label_names[face_cls] = data
                for face in faces:
                    # path
                    face_path = path_join(self.face_datasets, data, face)

                    # image
                    image = ImageOps.grayscale(Image.open(face_path))
                    image = np.array(image)
                    face_features.append(image)

                    # label
                    face_labels.append(face_cls)
                face_cls += 1

        return (np.array(face_features), np.array(face_labels), label_names)

    def create_tain_and_test_data(self) -> tuple:
        """
        Create training and testing dataset.

        Returns:
            Tuple of X_train, X_test, y_train, y_test
        """
        # Read images
        self.face_features, self.face_labels, self.label_names = self.read_images()

        # introspect the images arrays to find the shapes (for plotting)
        n_samples, height, width = self.face_features.shape

        # create X and y
        X = self.face_features.reshape((n_samples, height*width))
        
        # the label to predict is the id of the person
        y = self.face_labels
        n_classes = np.unique(self.face_labels).shape[0]

        print("Total dataset size: %d" % (n_classes))

        # split into a training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Size of training set: ", X_train.shape)
        print("Size of testing set: ", X_test.shape)
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return (X_train, X_test, y_train, y_test)
    
    def train(self, classifier, pca_components:int) -> tuple:
        """
        Train a classifier for face recognition from pca -> model fitting process.

        Args:
            classifier: Sklearn model for training and predicting.
            pca_components: Number of components to keep during PCA fitting.
        Returns:
            Tuple of training results: pca_components, acc_score, fitting_time, face_selected
        """
        print("Extracting the top %d eigenfaces from %d faces" % (pca_components, self.X_train.shape[0]))
        pca = PCA(n_components=pca_components, 
                  svd_solver='randomized', 
                  whiten=True).fit(self.X_train) # PCA

        # Transform the training and testing data
        X_train_pca = pca.transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)

        # Train ML model
        start_time = time()
        classifier = classifier.fit(X_train_pca, self.y_train)
        fitting_time = time() - start_time
        print("Algorithm fitting done in %0.3fs" % (fitting_time))

        y_pred = classifier.predict(X_test_pca)
        acc_score = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy Score: {acc_score}")

        # Eigen faces for saving
        height, width = self.img_size
        eigenfaces = pca.components_.reshape((pca_components, height, width))
        face_selected = random.choices(eigenfaces, k=10)

        return (pca_components, acc_score, fitting_time, face_selected)
    
    def run(self, classifier, start:int=10, end:int=101, steps:int=5) -> dict:
        """
        Run experiments by given classifier on training and testing dataset for {[(end-1) - start] / steps} times.

        Args:
            classifier: Sklearn model for training and predicting.
            start: Start number for range object.
            end: End number for range object.
            steps: Step number for range object.
        Returns:
            Dictionary of training results.
        """
        results = dict(pca_components=list(), acc_score=list(), fitting_time=list(), face_selected=dict())
        iteration = 1
        for n_components in range(start, end, steps):
            print('='*10, f' {str(iteration)} Iteration', '='*10)
            pca_components, acc_score, fitting_time, face_selected = self.train(classifier, n_components)
            results['pca_components'].append(pca_components)
            results['acc_score'].append(acc_score)
            results['fitting_time'].append(fitting_time)
            results['face_selected'][n_components] = face_selected
            iteration += 1
        return results
    
    def plot_results(self, results, clf_name, fig_type='accuracy'):
        if fig_type == 'accuracy':
            y_values = results['acc_score']
            y_label = 'Accuracy Scores'
        elif fig_type == 'fitting_time':
            y_values = results['fitting_time']
            y_label = 'Fitting Time'
        else:
            raise ValueError('Wrong fid_type: accuracy or fitting_time')
        x_values = results['pca_components']
        plt.figure(figsize=(8, 5))
        plt.bar(x_values, y_values, width=1.0)
        plt.xlabel('N Components')
        plt.ylabel(y_label)
        plt.title('%s: components-%s bar graph'%(clf_name, fig_type))
        plt.show()
    
    def fit_one_model(self, classifier, pca_components:int, pca_file:str='', model_file:str=''):
        """
        Train a classifier for face recognition from pca -> model fitting process.

        Args:
            classifier: Sklearn model for training and predicting.
            pca_components: Number of components to keep during PCA fitting.
        """
        print("Extracting the top %d eigenfaces from %d faces" % (pca_components, self.X_train.shape[0]))
        start_time = time()
        pca = PCA(n_components=pca_components, 
                  svd_solver='randomized', 
                  whiten=True).fit(self.X_train) # PCA
        fitting_time = time() - start_time
        print("PCA fitting done in %0.3fs" % (fitting_time))

        # Transform the training and testing data
        X_train_pca = pca.transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)

        # Train ML model
        classifier = classifier.fit(X_train_pca, self.y_train)

        y_pred = classifier.predict(X_test_pca)
        acc_score = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy Score: {acc_score}")

        # save PCA algorithm
        if pca_file:
            with open(pca_file, 'wb') as file:
                pickle.dump(pca, file)
            print(f"Save PCA algorithm at {pca_file}")
        
        # save ML algorithm
        if model_file:
            with open(model_file, 'wb') as file:
                pickle.dump(classifier, file)
            print(f"Save ML algorithm at {model_file}")

        return self.model_evaluation(self.y_test, y_pred)

    def model_evaluation(self, y_true, y_pred):
        acc_score = accuracy_score(y_true, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        return {'Accuracy': acc_score, 'Precision': precision, 'Recall': recall, 'f1_score': f1_score}


def extract_best_results(**kwargs):
    """
    Extract the best results from experimental results generated before, and return the collections.
    """
    collections = dict()
    for classifier_name in kwargs:
        results = kwargs[classifier_name]
        best_index = np.argmax(np.array(results['acc_score']))
        best_acc = np.round(results['acc_score'][best_index], 3)
        best_fitting_time = np.round(results['fitting_time'][best_index], 3)
        best_components = 10+(best_index*5)
        collections[classifier_name] = {'best_accuracy': best_acc, 
                                        'fitting_time': best_fitting_time, 
                                        'best_components': best_components}
        eigen_faces = results['face_selected'][best_components]

        nrows = len(eigen_faces)
        plt.figure(figsize=(nrows*1.5, 3))
        plt.suptitle('%s: Acc: %s, Components: %s, Fitting_Time: %s'%\
                     (classifier_name, str(best_acc), str(best_components), str(best_fitting_time)), 
                     y=0.8)
        for i, eigenface in enumerate(eigen_faces):
            plt.subplot(1, nrows, i+1)
            plt.imshow(eigenface)
            plt.title("eigenface %s" % (i+1))
            plt.axis("off")
        plt.show()
    
    return collections