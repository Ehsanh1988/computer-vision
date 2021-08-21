# -*- coding: utf-8 -*-
""" 
main.py

# TODO  add efficientnetb6

python main.py --backbone 'resnet152v2' --category "by_category" --save False --desc 'WHY RES 152 not working'

downloaded models :: 
'resnet50'
'resnet50v2'
'resnet152v2'
'efficientnetb7'
'vgg16'
'densenet201'
'densenet121'
'nasnetLarge'
'inception_resnet'
"""

import argparse

from configs.config import *

from scripts.models.classifier import Classifier
import os
import tensorflow as tf
import datetime
import numpy as np
import pandas as pd  
import pathlib
import json 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--backbone',     help='Backbone model used', default='resnet50', type=str)
    parser.add_argument('--category',     help='which category milk , yogurt .. or by_category', default='milk', type=str)
    parser.add_argument('--save',         help='save.', default=False, type=bool)
    parser.add_argument('--desc',         help='description to save model and tensorboad logs .', default='untitled', type=str)

    return parser.parse_args(args)


# def get_traindata_info(model):
#     train_labels = model.dataset['train'].labels
#     counts = np.unique(train_labels, return_counts=True)
#     return pd.DataFrame(counts).T



def train(model, cat , config, name_tag, save):
    """Builds model, loads data, trains and evaluates"""
    model = Classifier(cat = cat,
    # config =CFG_RESNET,
                     config =config,
                     base_model = model,
                     name_tensorboard=name_tag,
                      )

    print('len(base_model.layers) :\n',len(model.base_model.layers))
    print(f'model.count_params() _ {model.base_model.count_params()}')

    
    model.load_data()
    # train_info = get_traindata_info(model)
    model.build()
    model.compile(fine_tune_at=model.fine_tune_at,
                            lr=model.lr)
    

    history = model.train(save=save,
                          save_name=name_tag,
                        #   train_info=train_info
                          )
    

    number_of_epochs_it_ran = len(history.history['loss'])
    #TODO save result 
    print('________________________________________________________________________')
    print('____________________eval on test set____________________________________')
    test_r = model.evaluate(model.dataset['test'])
    print('________________________________________________________________________')
    
    # if save:
    #     path_to_save_model = pathlib.Path((config['data']['path_to_model']))/ cat/ model.base_model.name /name_tag   # save model history

    # d = datetime.datetime.today().strftime('%H%M%S')
    p = f'/workspace/detect-me/product_classifier/saved_models/{cat}/test_result/'
    if not os.path.exists(p):
        os.mkdir(p)
    with open(os.path.join(p ,
                           f'{model.base_model.name}_{name_tag}-(number_of_epochs_it_ran-{number_of_epochs_it_ran})-(count_params-{model.base_model.count_params()}).json'),
              'w') as writer:
        json.dump(test_r,
                  writer,
                  indent=4)

def finetune():
    model = Classifier(cat = 'milk',
                    config =CFG_RESNET,
                    name_tensorboard=model_nametag,
                    base_model = base_model)
    model.load_data()

    
    model.build()
    ##
    model.compile(fine_tune_at=1000, lr=0.008)
    history = model.train(epochs = 6, save=False,)
    ##
    model.compile(fine_tune_at=120, lr=0.0001)
    history1 = model.train(save=False,
                    save_name=model_nametag,
                    initial_epoch=history.epoch[-1],
                    epochs=12)
    ##
    model.compile(fine_tune_at=50, lr=0.0008)
    history2 = model.train(save=True,
                    save_name=model_nametag,
                    initial_epoch=history1.epoch[-1],
                    epochs=22)

    print(history2.history.keys()) 

def test(cat):
    base_model = tf.keras.applications.ResNet50(
                             include_top=False,
                             weights="imagenet",
                             input_tensor=None,
                             # input_shape=(*self.image_size, 3),
                             pooling=None,
                             classes=1000)
    model_nametag ='TT'
    
    model = Classifier(cat = cat,
                    config =CFG_RESNET50,
                    name_tensorboard=model_nametag,
                    base_model = base_model)
    model.load_data()
    print(model.base_model.name)
    print(dir(model))

    model = Classifier(cat = cat,
                    config =train_config,
                    name_tensorboard='TEST_freeze_up_to_zero_lr00005',
                    base_model=base_model)
    model.load_data()
    print(model.base_model.name)
    print(dir(model))

def main(args=None):
    import sys
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    backbone =      args.backbone
    should_I_save = args.save
    desc =          args.desc
    cat  =          args.category                    

    print(backbone)
    if backbone=='resnet50':
        train_config = CFG_RESNET50
        base_model = tf.keras.applications.ResNet50(
                             include_top=False,
                             weights="imagenet",
                             input_tensor=None,
                             # input_shape=(*self.image_size, 3),
                             pooling=None,
                             classes=1000)
    elif backbone=='resnet50v2':
        train_config = CFG_RESNET50V2
        base_model = tf.keras.applications.ResNet50V2(
                            include_top=False, weights='imagenet', input_tensor=None,
                            input_shape=None, pooling=None, classes=1000,
                            classifier_activation='softmax')
    elif backbone=='resnet152v2':
        train_config=CFG_RESNET152
        base_model = tf.keras.applications.ResNet152V2(
                        include_top=False,
                        weights="imagenet",
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        classes=1000,
                        classifier_activation="softmax",)
    elif backbone=='efficientnetb7':
        train_config = CFG_EfficientNet
        base_model = tf.keras.applications.EfficientNetB7(
                    include_top=False,
                    weights="imagenet",
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=1000,
                    classifier_activation="softmax",)
    elif backbone=='vgg16':
        train_config = CFG_VGG16
        base_model = tf.keras.applications.VGG16(include_top=False,
                                  weights='imagenet',
                                  classes=1000)
    elif backbone=='densenet201':
        train_config = CFG_DENSE201
        base_model = tf.keras.applications.DenseNet201(
                            include_top=False,
                            weights="imagenet",
                            input_tensor=None,
                            input_shape=None,
                            pooling=None,
                            classes=1000,
                        )
    elif backbone=='densenet121':
        train_config = CFG_DENSE121
        base_model = tf.keras.applications.DenseNet121(
                            include_top=False,
                            weights="imagenet",
                            input_tensor=None,
                            input_shape=None,
                            pooling=None,
                            classes=1000,
                        )
    elif backbone=='nasnetLarge':
        train_config = CFG_NASNET
        base_model = tf.keras.applications.NASNetLarge(
                            input_shape=None,
                            include_top=False,
                            weights="imagenet",
                            input_tensor=None,
                            pooling=None,
                            classes=1000,
                            )
    elif backbone=='inception_resnet':
        train_config = CFG_INCEPTION_RES
        base_model = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",)

    elif backbone=='vgg16':
        train_config = CFG_VGG16
        base_model = tf.keras.applications.VGG16(include_top=False,
                                  weights='imagenet',
                                  classes=1000)    
    else:
        raise ValueError('saved models in ./keras folder :: efficientnetb7-vgg16-resnet152v2-resnet50-resnet50v2 ')
        
    freeze_up_to = train_config['train']['fine_tune_at']
    lr = train_config['train']['opt_lr']
    batchs = train_config['train']['batch_size']
    optimizer = train_config['train']['optimizer']['type']
    optimizer__callback = train_config['train']['optimizer']['lr_callback']['type']
    
    # optimizer__callback_params = train_config['train']['optimizer']['lr_callback']['params']
    # optimizer__callback_params = '___'.join([f'{k}-{v}' for k,v in optimizer__callback_params.items()])
    
    epochs = train_config['train']['epochs']
    dropout = train_config['train']['dropout']
    base_model_name = train_config['model']['name']
    # data_augmentation = '___'.join([f'{k}-{v}' for k,v in train_config['train']['data_augmentation'].items()])

    
    time = datetime.datetime.utcnow() + datetime.timedelta(hours=4, minutes=30)
    timest = time.strftime("%Y-%b-%d----%H%M%S")

    desc = f"{desc}__{timest}"
    
    model_nametag = f'{base_model_name}---(Freeze-{freeze_up_to})-(lr-{lr}-{optimizer})-(opt_cb-{optimizer__callback})-(epoch-{epochs})-(batch-{batchs})-(dropout-{dropout})-{desc}'
    
    train(base_model,
          cat,
          train_config,
          model_nametag ,
          should_I_save )

if __name__ == '__main__':
    main()
    # train()
    # test()
    # finetune()
    # get_traindata_info()