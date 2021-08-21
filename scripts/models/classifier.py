from scripts.models.basemodel import BaseModel
from scripts.utils import image_pp, data_loader
from scripts.utils import image_pp, data_loader
from scripts.utils import callbacks as callb

import os
import re
import pathlib
import json
import h5py

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

import keras

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

from tensorflow.keras.callbacks import (LearningRateScheduler,
                                        ReduceLROnPlateau,
                                        EarlyStopping,
                                        TensorBoard)

from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnet_v2
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_res_v2

from keras.applications.resnet50 import ResNet50





class Classifier(BaseModel):
    def __init__(self, cat, config, name_tensorboard, base_model):
        super().__init__(config,
                         base_model=base_model)
        self.image_size = self.config.data.image_size
        self.name_tensorboard = name_tensorboard
        self.model = None
        self.info = None
        self.validation_steps = 0
        self.train_length = 0
        self.steps_per_epoch = 0
        self.cat = cat
        
        
        self.optimizer = self.config.train.optimizer.type
        
        self.learning_rate_callback_type = self.config.train.optimizer.lr_callback.type
        if self.learning_rate_callback_type == 'cosine_learning_rate_decay':
            self.learning_rate_callback_params = {'warmup_epoch' : self.config.train.optimizer.lr_callback.params.warmup_epoch,
                                                  'hold_base_rate_steps' : self.config.train.optimizer.lr_callback.params.hold_base_rate_steps}
        
        
        # self.data_augmentation_options = self.config.train.data_augmentation
        
        # train - batch size -  epochs - dropout - learning rate
        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epochs
        self.dropout = self.config.train.dropout
        self.lr = self.config.train.opt_lr
        self.fine_tune_at = self.config.train.fine_tune_at

        # data and model
        # self.num_classes = self.config.data.num_classes
        self.image_size = self.config.data.image_size

        # path 
        self.path_to_data = pathlib.Path(self.config.data.path_to_data)  # images folder separate by classes
        self.path_to_save_model = (pathlib.Path(self.config.data.path_to_model) /
                              self.cat /
                              self.base_model.name)  # save model history
        self.path_to_tensroboard_log = (pathlib.Path(self.config.data.path_to_model) /
                              self.cat /
                              'tensorboard_logs')

        print(f'model_path ::\n{self.path_to_save_model}')
        print(f'tensorboard_logs_path_ ::\n{self.path_to_tensroboard_log}')
        print(f'data_path_ ::\n{self.path_to_data}')
        print('init_done')

    # @staticmethod
    # def data_augment(inputs):
    #     data_augmentation = keras.Sequential(
    #         [
    #             #                   tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    #             #                 tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
    #             # tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    #             # tf.keras.layers.experimental.preprocessing.RandomContrast(0.5),
    #             #                   tf.keras.layers.experimental.preprocessing.RandomZoom(0.4),
    #             # tf.keras.layers.experimental.preprocessing.RandomTranslation(
    #                 # height_factor=0.2, width_factor=0.2),
    #             image_pp.RandomCutout((220, 40)),
    #             #           tf.keras.layers.experimental.preprocessing.RandomCrop(224,224,),
    #             #         tf.keras.layers.experimental.preprocessing.Rescaling(),
    #         ]
    #     )
    #     return data_augmentation(inputs)
   

    def load_data(self):
        self.dataset = data_loader.load_data_flow_from_dataframe(
            self.path_to_data,
            target_image_size=self.image_size,
            batch_size=self.batch_size,
            # load_with_mixup_generator=bool(self.data_augmentation_options.mixup)
            )

        try:
            assert set(['train', 'test', 'samples', 'class_indices']).issubset(self.dataset.keys())
        except AssertionError:
            print('dictionary loaded from util/dataloader is different')
    # build %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def build(self):
        
        num_classes = self.dataset.get('num_classes')
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        
        if self.base_model.name == 'resnet152v2':
            print('resnet152v2 preprocess')
            x = preprocess_input_resnet_v2(inputs)
            
        elif self.base_model.name  == 'resnet50':
            x = preprocess_input_resnet(inputs)
            print('resnet50 preprocess')

        elif self.base_model.name  == 'densenet121':
            x = preprocess_input_densnet(inputs)
            print('densenet121 preprocess')
            
        elif self.base_model.name  == 'densenet201':
            x = preprocess_input_densnet(inputs)
            print('DenseNet201 preprocess')
            
        elif self.base_model.name  == 'nasnetLarge':
            x = preprocess_input_nasnet(inputs)
            print('nasnetLarge preprocess')
            
        elif self.base_model.name == 'vgg16':
            print('resnetvgg16 preprocess')
            x = preprocess_input_vgg16(inputs)
            
        elif self.base_model.name == 'inception_resnet_v2':
            print('inception resnet preprocess')
            x = preprocess_input_inception_res_v2(inputs)
            
        elif self.base_model.name == 'resnet50v2':
            print('resnet50v2 preprocess')
            x = preprocess_input_resnet_v2(inputs)
        #Note: each Keras Application expects a specific kind of input preprocessing.
        # For EfficientNet, input preprocessing is included as part of the model
        # (as a Rescaling layer), and thus tf.keras.applications.efficientnet.
        # preprocess_input is actually a pass-through function.
        elif self.base_model.name == 'efficientnetb7':
            print('efficientnetb7 preprocess')
            x = preprocess_input_efficientnet(inputs)
        else:
            # x = preprocess_input_resnet(inputs)
            raise ValueError(f'hmmm {self.base_model.name} not in this list ? [resnet152v2 , resnet50 , efficientnetb7, ...]')
            # print(f'hmmm {self.base_model.name} not in this list ? [resnet152v2 , resnet50, resnet50v2, ...]')


        # use training=False as our model contains a BatchNormalization layer.
        x = self.base_model(x, training=False)

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        x = global_average_layer(x)

        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(1024,
                                activation='relu',
                                # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                # bias_regularizer=regularizers.l2(1e-4),
                                # activity_regularizer=regularizers.l2(1e-5)
                                )(x)

        prediction_layer = tf.keras.layers.Dense(
            num_classes, activation='softmax')

        outputs = prediction_layer(x)

        self.model = tf.keras.Model(inputs, outputs)

        return
    # compile %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def compile(self, fine_tune_at : int, lr):
        self.base_model.trainable = True

        num_classes = self.dataset.get('num_classes')

        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False

        print('len(self.model.trainable_variables) ',
              len(self.base_model.trainable_variables))

        print(self.optimizer.lower())
        if self.optimizer.lower() == 'adam':
            optimizer=tf.keras.optimizers.Adam(lr=lr,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=1e-01,
                                        # epsilon=1e-01,
                                        # epsilon=1e-07, #vanilla
                                        amsgrad=False,
                                        name="Adam",)
        elif self.optimizer.lower() == 'sgd':
            optimizer=tf.keras.optimizers.SGD(lr=lr,
                                        decay=lr / self.epochs,
                                        momentum=0.9,
                                         name="SGD")
        elif self.optimizer.lower() == 'rmsprop':
            optimizer=tf.keras.optimizers.RMSprop(lr=lr,
                                        rho=0.9,
                                        momentum=0.9,
                                        epsilon=1e-07,
                                        centered=False,
                                        name="RMSprop")
        elif self.optimizer.lower() == 'adadelta':
            optimizer=tf.keras.optimizers.Adadelta(
                                        learning_rate=lr,
                                        rho=0.95,
                                        epsilon=1e-07,
                                        name="Adadelta",
                                         )
        else:
            raise ValueError('optimizer s :: |.sgd.|.rmsprop.|.adam.|.adadelta.|]')

 
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                        # loss=tf.nn.softmax_cross_entropy_with_logits,
                           metrics=['accuracy',
                                    tf.keras.metrics.TopKCategoricalAccuracy(
                                            k=3, name="top_3_acc", dtype=None),
                                    tf.keras.metrics.Precision(thresholds=0.5,
                                                               name = 'PRCN-tr-0.5'),
                                    tf.keras.metrics.Precision(thresholds=0.9,
                                                               name = 'PRCN-tr-0.9'),
                                    tf.keras.metrics.Recall(thresholds=0.5,
                                                            name='recall-tr--0.5'),
                                    tf.keras.metrics.Recall(thresholds=0.9,
                                                            name='recall-tr--0.9')
                                    ],)
        print(['*']*100)
        print(self.model.optimizer.get_config())
        print(['*']*100)
    # train %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def train(self,
              save=False,
              epochs=None,
              save_name='no_name',
              initial_epoch=0,
              verbose=1,
            #   train_info=None
              ):

        epochs = epochs if epochs else self.epochs
        cbs = self.callbacks()
        
        print(f'callbacks ::{cbs}')
        history = self.model.fit(self.dataset['train'],
                                 batch_size=self.batch_size,
                                 epochs=epochs,
                                 initial_epoch=initial_epoch,
                                 validation_data=self.dataset['validation'],
                                 callbacks=cbs,
                                 verbose=verbose,
                                 steps_per_epoch=self.dataset['samples'] // self.batch_size,
                                 validation_steps = self.dataset['validation'].samples // self.batch_size,
                                 workers=4,
                                #  use_multiprocessing=True
                                 )

        if save:
            DIR = self.path_to_save_model / save_name
            if not os.path.exists(DIR):
                os.makedirs(DIR)
            self.model.save(DIR)
            
            mapping = self.dataset['class_indices']
            mapping = pd.DataFrame.from_dict(mapping, orient='index')
            mapping.columns = ['CODE']
            mapping.index.name = 'CLASS'
            mapping.to_csv(DIR /'label_mapping.csv')
            print('saved')
            
            train_labels = self.dataset['train'].labels
            counts = np.unique(train_labels, return_counts=True)
            train_info = pd.DataFrame(counts).T
            train_info.to_csv(DIR /'train_info.csv')
            
            # if train_info is not None:
            #     train_info.to_csv(DIR /'train_info.csv')
                
    #             def get_traindata_info(model):
    # train_labels = model.dataset['train'].labels
    # counts = np.unique(train_labels, return_counts=True)
    # return pd.DataFrame(counts).T



        return history
    # evaluate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def evaluate(self, test_data):
        return self.model.evaluate(
            test_data,
            batch_size=None,
            verbose=1,
            sample_weight=None,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            return_dict=True,
        )
    # predict %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def predict(self, image):
        return self.model.predict(image)
    # callbacks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    def callbacks(self,):
        
        callbacks_list = []
        
        # either ""cosine_learning_rate_decay"" or ""ReduceLROnPlateau""
        if self.learning_rate_callback_type == 'cosine_learning_rate_decay':
            # self.learning_rate_callback_params = {'warmup_epoch' : self.config.train.optimizer.lr_callback.params.warmup_epoch} 
            sample_count = self.dataset['train'].samples
            total_steps = int(self.epochs * sample_count / self.batch_size)
            warmup_epoch = self.learning_rate_callback_params['warmup_epoch']
            hold_base_rate_steps = self.learning_rate_callback_params['hold_base_rate_steps']
            warmup_steps = int(warmup_epoch * sample_count / self.batch_size)
            
            print('*******************************************************   cosine_learning_rate_decay   ****************************************************************************************************************************************************')
            print(f'total_steps in train:: {total_steps}\nsample count :: {sample_count} in {self.epochs} epochs *********************************************************************')
            print(f'warmup_steps {warmup_steps}')
            print('*******************************************************************************************************************************************************************************************************************************************')
            learning_rate_callback = callb.WarmUpCosineDecayScheduler(learning_rate_base=self.lr,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0.0,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=hold_base_rate_steps)
            
            callbacks_list.append(learning_rate_callback)
        else:
            callbacks_list.append(callb.Learning_date_log_callback())
            
            callbacks_list.append(ReduceLROnPlateau(
                        monitor='loss',
                        factor=0.1,
                        patience=1,
                        verbose=1,
                        mode='auto',
                        epsilon=0.0001,
                        cooldown=0,
                        min_lr=0))
        
        
        callbacks_list.append(callb.LRTensorBoard(
                        log_dir=str(self.path_to_tensroboard_log/ self.name_tensorboard),
                        histogram_freq=2,
                        write_graph=False,
                        write_images=True,
                        update_freq="epoch",
                        profile_batch=2,
                        embeddings_freq=0,
                        embeddings_metadata=None,))
    
        
        callbacks_list.append(EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=3,
                        verbose=1,
                        mode='auto',
                        baseline=None,
                        restore_best_weights=True
                    ))
        
    
        return callbacks_list
