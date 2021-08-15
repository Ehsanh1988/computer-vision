from scripts.models.basemodel import BaseModel
from scripts.utils import image_pp, data_loader
from scripts.utils import image_pp, data_loader
from scripts.utils import callbacks as callb# import WarmUpCosineDecayScheduler, learning_date_log_callback



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
from keras.models import Model



from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnet_v2
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet

from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from tensorflow.keras import regularizers

# from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard


from keras.applications.resnet50 import ResNet50
# from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
        
        
        self.data_augmentation_options = self.config.train.data_augmentation
        
        # train - batch size -  epochs - dropout - learning rate
        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epochs
        self.dropout = self.config.train.dropout
        self.lr = self.config.train.opt_lr
        self.fine_tune_at = self.config.train.fine_tune_at

        # data and model
        self.num_classes = self.config.data.num_classes
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

    @staticmethod
    def data_augment(inputs):
        data_augmentation = keras.Sequential(
            [
                #                   tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                #                 tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
                # tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
                # tf.keras.layers.experimental.preprocessing.RandomContrast(0.5),
                #                   tf.keras.layers.experimental.preprocessing.RandomZoom(0.4),
                # tf.keras.layers.experimental.preprocessing.RandomTranslation(
                    # height_factor=0.2, width_factor=0.2),
                image_pp.RandomCutout((220, 40)),
                #           tf.keras.layers.experimental.preprocessing.RandomCrop(224,224,),
                #         tf.keras.layers.experimental.preprocessing.Rescaling(),
            ]
        )
        return data_augmentation(inputs)
    # plot_history %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    @staticmethod
    def plot_history(history,
                     save=False,
                     save_path='./train_validation.png'):
        print(history.history.keys)
        # acc = history.history['accuracy']
        # val_acc = history.history['val_accuracy']

        # loss = history.history['loss']
        # val_loss = history.history['val_loss']

        # plt.figure(figsize=(8, 8))
        # plt.subplot(2, 1, 1)
        # plt.plot(acc, label='Training Accuracy')
        # plt.plot(val_acc, label='Validation Accuracy')
        # plt.legend(loc='lower right')
        # plt.ylabel('Accuracy')
        # plt.ylim([min(plt.ylim()), 1])
        # plt.title('Training and Validation Accuracy')

        # plt.subplot(2, 1, 2)
        # plt.plot(loss, label='Training Loss')
        # plt.plot(val_loss, label='Validation Loss')
        # plt.legend(loc='upper right')
        # plt.ylabel('Cross Entropy')
        # plt.ylim([0, 5.0])
        # plt.title('Training and Validation Loss')
        # plt.xlabel('epoch')
        # if save:
        #     plt.savefig(save_path, dpi=300)
        #     print('plot_saved')
        #     print('save_path\n', save_path)
        # plt.show()

        return plt.gcf()

    def load_data(self):
        self.dataset = data_loader.load_data_flow_from_dataframe(
            self.path_to_data,
            # self.path_to_data / 'test',
            target_image_size=self.image_size,
            batch_size=self.batch_size,
            # csv_path_classes=csv_path,
            load_with_mixup_generator=bool(self.data_augmentation_options.mixup)
            )
        
        # self.dataset = data_loader.load_data_flow_from_dataframe(
        #     self.path_to_data / 'train-val',
        #     self.path_to_data / 'test',
        #     target_image_size=self.image_size,
        #     batch_size=self.batch_size,
        #     csv_path_classes=csv_path)
        
        # TODO remove it man! seriouslyy . I mean it. NOW!. . . .
        try:
            assert set(['train', 'test', 'samples', 'class_indices']).issubset(self.dataset.keys())
        except AssertionError:
            print('dictionary loaded from util/dataloader is different')
    # build %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def build(self):

        # num_classes = self.dataset.get('num_classes') if self.dataset.get(
        #     'num_classes') else self.num_classes
        num_classes = self.dataset.get('num_classes')
        # if self.dataset.get('num_classes') != self.num_classes:
        #     print('________________________________________________WARNING!___________________________________________')
        #     print('something is WRONGGGG!! num  classes in CONFIG is DIFF from what I get from Train-validation folder')
        #     print('num classes is calculated based on  folder structure in xxx/train-val')
        #     print(f'num_classes : {num_classes}')
        #     print('________________________________________________WARNING!___________________________________________')

        inputs = tf.keras.Input(shape=(224, 224, 3))
        # x = self.data_augment(inputs)
        
        if self.base_model.name == 'resnet152v2':
            print('resnet152v2 preprocess')
            x = preprocess_input_resnet(inputs)
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
            print('resnet50 preprocess')
            
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
            raise ValueError(f'hmmm {self.base_model.name} not in this list ? [resnet152v2 , resnet50 , efficientnetb7]')
            print(f'hmmm {self.base_model.name} not in this list ? [resnet152v2 , resnet50, resnet50v2]')
        # x = preprocess_input_resnet(inputs)

        # As previously mentioned, use training=False as our model contains a BatchNormalization layer.
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
        # x = tf.keras.layers.Dense(num_classes//2, activation='relu')(x)

        prediction_layer = tf.keras.layers.Dense(
            num_classes, activation='softmax')

        outputs = prediction_layer(x)

        self.model = tf.keras.Model(inputs, outputs)

        return
    # compile %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def compile(self, fine_tune_at : int, lr):
        self.base_model.trainable = True

        # learning_rate = lr if lr else self.lr
        # ft = fine_tune_at if fine_tune_at else self.fine_tune_at
        num_classes = self.dataset.get('num_classes') if self.dataset.get(
            'num_classes') else self.num_classes

        # print(f'overwrite learning_rate {learning_rate},finetune {ft}')
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

        # if self.ob
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                        # loss=tf.nn.softmax_cross_entropy_with_logits,
                           metrics=['accuracy',
                                    #                       tf.keras.metrics.AUC(),
                                    tf.keras.metrics.Precision(thresholds=0.5,
                                                               name = 'PRCN-tr-0.5'),
                                    tf.keras.metrics.Precision(thresholds=0.9,
                                                               name = 'PRCN-tr-0.9'),
                                    # tf.keras.metrics.TruePositives(name='tp'),
                                    # tf.keras.metrics.FalsePositives(name='fp'),
                                    # tf.keras.metrics.TrueNegatives(name='tn'),
                                    # tf.keras.metrics.FalseNegatives(name='fn'),
                                    tf.keras.metrics.Recall(thresholds=0.5,
                                                            name='recall-tr--0.5'),
                                    tf.keras.metrics.Recall(thresholds=0.9,
                                                            name='recall-tr--0.9')
                                    # tfa.metrics.F1Score(num_classes=num_classes),
                                    # tf.keras.metrics.SparseCategoricalCrossentropy() #error
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
              train_info=None
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
                                 use_multiprocessing=True
                                 )

        if save:
            DIR = self.path_to_save_model / save_name
            if not os.path.exists(DIR):
                os.makedirs(DIR)
            self.model.save(DIR)
            plot = self.plot_history(history, save=True,
                                     save_path=DIR / 'cross_val.png')
            # TODO function history to pandas dataftame
            mapping = self.dataset['class_indices']
            mapping = pd.DataFrame.from_dict(mapping, orient='index')
            mapping.columns = ['code']
            mapping.index.name = 'vms'
            mapping.to_csv(DIR /'vms_code_mapping.csv')
            print('saved')
            if train_info is not None:
                train_info.to_csv(DIR /'train_info.csv')
        else:
            plot = self.plot_history(history, save=False)

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
            learning_date_log_callback = callb.Learning_date_log_callback()
            callbacks_list.append(learning_date_log_callback)
            callbacks_list.append(ReduceLROnPlateau(
                        monitor='loss',
                        factor=0.1,
                        patience=1,
                        verbose=1,
                        mode='auto',
                        epsilon=0.0001,
                        cooldown=0,
                        min_lr=0))
        
        
        # learning_date_log_callback = callb.Learning_date_log_callback()
        callbacks_list.append(callb.LRTensorBoard(
                        log_dir=str(self.path_to_tensroboard_log/ self.name_tensorboard),
                        histogram_freq=2,
                        write_graph=False,
                        write_images=True,
                        update_freq="epoch",
                        profile_batch=2,
                        embeddings_freq=0,
                        embeddings_metadata=None,))
    
        
        # callbacks_list.append(TensorBoard(
        #                 log_dir=str(self.path_to_tensroboard_log/ self.name_tensorboard),
        #                 histogram_freq=2,
        #                 write_graph=False,
        #                 write_images=True,
        #                 update_freq="epoch",
        #                 profile_batch=2,
        #                 embeddings_freq=0,
        #                 embeddings_metadata=None,))
        
        
        callbacks_list.append(EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=3,
                        verbose=1,
                        mode='auto',
                        baseline=None,
                        restore_best_weights=True
                    ))
        
        
        
        # callbacks_list.append(tf.keras.callbacks.ModelCheckpoint(
        #                 str(self.path_to_save_model /
        #                         'checkpoints' /
        #                         f'{self.name_tensorboard}' /
        #                         'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
        #                 monitor='val_loss',
        #                 verbose=1,
        #                 save_best_only=True,
        #                 save_weights_only=False,
        #                 mode='auto',
        #                 save_freq='epoch',
        #                 options=None))
        return callbacks_list
