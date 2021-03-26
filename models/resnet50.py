from .basemodel import BaseModel

import os
# import pickle
import re
import pathlib
import json
import h5py

import cv2
import matplotlib.pyplot as plt                                                                                                                                                                             
# import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

import keras
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.callbacks import LearningRateScheduler

# from sklearn.model_selection import train_test_split 

import tensorflow_addons as tfa
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ResNet50(BaseModel):
    def __init__(self, cat, config, name_tensorboard):
        super().__init__(config)
        self.base_model = tf.keras.applications.ResNet50(
                            include_top=False,
                            weights="imagenet",
                            input_tensor=None,
                            input_shape=(224,224,3),
                            pooling=None,
                            classes=1000)
        self.name_tensorboard=name_tensorboard
        self.model= None
        self.dataset = None                  
        self.info=None
        self.validation_steps = 0
        self.train_length = 0
        self.steps_per_epoch = 0
        self.cat = cat

        #train - batch size -  epochs - dropout - learning rate
        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epochs
        self.dropout = self.config.train.dropout
        self.lr = self.config.train.opt_lr
        self.fine_tune_at=  self.config.train.fine_tune_at

        # data and model
        self.num_classes =  self.config.data.num_classes
        self.image_size = self.config.data.image_size
        self.path_to_data = pathlib.Path(self.config.data.path_to_data) /self.cat #images folder separate by classes 
        self.path_to_model = pathlib.Path(self.config.data.path_to_model) / self.cat / self.base_model.name  #save model history    
    
    @staticmethod
    def data_augment():
        inputs = tf.keras.Input(shape=(224, 224, 3))
        data_augmentation = keras.Sequential(
            [
#                   tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
#                 tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
                  tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
                  tf.keras.layers.experimental.preprocessing.RandomContrast(0.5),
#                   tf.keras.layers.experimental.preprocessing.RandomZoom(0.4),
                tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2),
                RandomCutout((224,20)),
        #           tf.keras.layers.experimental.preprocessing.RandomCrop(224,224,),
        #         tf.keras.layers.experimental.preprocessing.Rescaling(),
            ]
        )
        return data_augmentation(inputs)
    
    @staticmethod
    def plot_history(history,
                     save=False ,
                     save_path='./train_validation.png'):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,5.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        if save:
            plt.savefig(save_path, dpi=300)
            print('plot_saved')
            print('save_path\n',save_path)
        plt.show()
    
    
        return plt.gcf()
            
    @staticmethod
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)


    def load_data_from_dir(self):
        
        train_data_dir = self.path_to_data / 'train-val'
        test_data_dir = self.path_to_data / 'test'

        img_height , img_width = (self.image_size,
                                  self.image_size)
        batch_size = self.batch_size

        # TODO what the heck is going on
        list_clasess = sorted({x for x in
                                 os.listdir(train_data_dir) if
                                 re.search('[0-9]+', x)})

        num_samples = len(list(pathlib.Path(train_data_dir).rglob('*.jpg')))

        test_datagen = ImageDataGenerator(
#             rescale=1./255,
        )

        validation_datagen = ImageDataGenerator(
#             rescale=1./255,
              validation_split=0.2)

        train_datagen = ImageDataGenerator(
#             rescale=1./255,
            shear_range=0.2,
            zoom_range=0.4,
            fill_mode="nearest",
            zca_whitening=False,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
#             preprocessing_function=ResNet50.preprocessing_fun,
            validation_split=0.2) # set validation split

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            shuffle=True,
            target_size=(img_height, img_width),
            seed=1,
            classes=list_clasess,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training') # set as training data

        validation_generator = validation_datagen.flow_from_directory(
            train_data_dir, # same directory as training data
            shuffle=True,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            seed=1,
            classes=list_clasess,
            class_mode='categorical',
            subset='validation') # set as validation data

        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            shuffle=False,
            target_size=(img_height, img_width),
            seed=1,
            batch_size=1,
            classes=list_clasess,
            class_mode='categorical',
                )
        
        self.dataset = dict([('train', train_generator),
                    ('validation', validation_generator),
                    ('test',test_generator),
                    ('num_classes',train_generator.num_classes),
                    ('mapping', train_generator.class_indices),
                    ('total_train_val_samples', num_samples )
                            ])

        
    def build(self):
        inputs = tf.keras.Input(shape=(224, 224, 3))
        
        num_classes = self.dataset.get('num_classes') if self.dataset.get('num_classes') else self.num_classes

#         x = self.data_augment(inputs)
        x = preprocess_input(inputs)
    
        x = self.base_model(x, training=False) # As previously mentioned, use training=False as our model contains a BatchNormalization layer.
        
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        x = global_average_layer(x)
        
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(num_classes//2, activation='relu')(x)
        
        prediction_layer = tf.keras.layers.Dense(num_classes,activation='softmax')
        
        outputs = prediction_layer(x)
        
        
        self.model = tf.keras.Model(inputs, outputs)
        
        return 
        
    def compile(self, fine_tune_at=None, lr = None):
        self.base_model.trainable = True
        learning_rate = lr if lr else self.lr
        ft = fine_tune_at if fine_tune_at else self.fine_tune_at
        for layer in self.base_model.layers[:ft]:
            layer.trainable =  False
            
        print('len(self.model.trainable_variables) ', len(self.model.trainable_variables))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy',
#                       tf.keras.metrics.AUC(),
                      tf.keras.metrics.Precision(), 
                      tf.keras.metrics.Recall(),
                    #   tf.keras.metrics.SparseCategoricalCrossentropy() #error
                      ],)
        print(self.model.optimizer.get_config())
    
    
    def train(self,
              save=False,
              epochs = None,
              save_name='no_name',
              initial_epoch=0,
              verbose=1
             ):
        
        epochs = epochs if epochs else self.epochs 
        history = self.model.fit(self.dataset['train'],
                                 batch_size=self.batch_size,
                                 epochs=epochs,
                                 initial_epoch = initial_epoch,
                                 validation_data=self.dataset['validation'],
                                 callbacks= self.callbacks(),
                                 verbose=verbose,
                                 steps_per_epoch=self.dataset['total_train_val_samples'] // self.batch_size,
                                 workers=4,
                           )
        
        if save:
            if not os.path.exists(self.path_to_model / save_name):
                os.makedirs(self.path_to_model / save_name)
            self.model.save(self.path_to_model / save_name) 
            plot = self.plot_history(history, save=True,
                                     save_path= self.path_to_model / save_name/ 'cross_val.png')
            #TODO function history to pandas dataftame
            mapping = self.dataset['mapping']
            mapping = pd.DataFrame.from_dict(mapping, orient='index')
            mapping.columns = ['code']
            mapping.index.name = 'vms'
            mapping.to_csv(self.path_to_model / save_name/ 'vms_code_mapping.csv')
            print('saved')
        else:
            plot = self.plot_history(history,save=False)
            
        return history
        

    def evaluate(self, test_data):
        self.model.evaluate(
                        test_data,
                        batch_size=None,
                        verbose=1,
                        sample_weight=None,
                        steps=None,
                        callbacks=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        return_dict=False,
                        )
        
    def predict(self, image):
        return self.model.predict(image)
    
    def callbacks(self,):
        lr_callback = LearningRateScheduler(self.lr_scheduler)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                log_dir=self.path_to_model / 'logs' / self.name_tensorboard,
                                histogram_freq=0,
                                write_graph=True,
                                write_images=False,
                                update_freq="epoch",
                                profile_batch=2,
                                embeddings_freq=0,
                                embeddings_metadata=None,)
        save_checkpoint_callback= tf.keras.callbacks.ModelCheckpoint(
                self.path_to_model /  'checkpoints' / f'{self.name_tensorboard}' / 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                    monitor='val_loss', verbose=0, save_best_only=True,
                    save_weights_only=False, mode='auto', save_freq='epoch',
                    options=None)
        return [lr_callback,
                tensorboard_callback,
                save_checkpoint_callback]