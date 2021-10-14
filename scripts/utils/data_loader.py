# 'all_categories' , 'milk' , ....

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple
import os
import re
import pathlib
from pathlib import Path
import pandas as pd
import numpy as np

            
def load_data_tree(directory) -> pd.DataFrame:
    print('loading data tree .. .. ')
    list_images = list(Path(directory).rglob('*.jpg'))
    # next line, pffff . .. it might go wonrg and it will make a messssss                
    df = pd.DataFrame( {'filepath' : list_images,
                        'partial_filepath' : [pathlib.Path(str(x).split('/data/')[1]) for x in list_images]})
    
    df_parts = pd.DataFrame(
        df.partial_filepath.apply(lambda x : x.parts).to_list())
    
    df = pd.concat([df,
                    df_parts],
                   axis=1)
    
    # remove checkpoints
    df = df[~df.filepath.astype(str).str.contains('ipynb_checkpoints')]
    
    # make sure your data folder structure is  data/ LEVEL1 / ('train-val'| 'test') / LEVEL2 / x.jpg
    df = df.rename({0:'first_level', 
                    1 :'subset',
                    2:'second_level',
                    3:'filename'},
                axis=1)
    print(f'data tree return ed {df.shape}')
    assert(set(df.subset) - set(['test', 'train-val']))==set(),\
           'something is wrong here .. it is expected hah aha ha hah!!!'
    return df 


def down_sample(dataframe:pd.DataFrame,
                ref_column:str,
                n_sample:int,
                random_state:int=1
                ) -> pd.DataFrame:
    
    print(f'before downsample {dataframe.shape}')
    count = dataframe.groupby(ref_column,
                                as_index=False).count()

    skus_with_lots_of_labels = count[count['filepath'] > n_sample][ref_column]
    
    if len(skus_with_lots_of_labels) > 0:
        oversamples = dataframe[dataframe[ref_column]\
                            .isin(skus_with_lots_of_labels)]
        oversamples = oversamples.groupby(ref_column)\
                            .sample(n_sample,
                                    random_state = random_state)

        normalsamples = dataframe[~dataframe[ref_column]\
                            .isin(skus_with_lots_of_labels)]

        df = pd.concat([normalsamples,
                        oversamples])
    else:
        df = dataframe.copy()

    print(f'after downsample {df.shape}')
    
    return df

# this function does a lot and do it messy ,, i know :|  someone need to QC trainset , testset  
def prepare_dataframe(dataframe:pd.DataFrame,
                        category:str ,
                        subset:str,
                        downsample_n:int,
                        **kwargs) -> pd.DataFrame:
    print('preparing dataframe ')
#     assert(dataframe.columns)
    df = dataframe[dataframe['subset'] ==subset]
    df = df.reset_index(drop=True).reset_index()

    #TODO
    if category != 'all_categories':
#         print('<_____________________')
#         print(df.shape)
#         df = df[~df.filename.str.contains('DATABASE_')]
#         print('after removing images from database (DB_)')
#         print(df.shape)
#         print('_____________________>')
    
                    
                    
        # ALL BILBO DATA is in finglish 
#         if category=='milk':
#             print('adding images from biblo team "shyr"')
#             print("first_level=='shyr'")
#             print(df[df.first_level=='shyr'].shape)
#             print("first_level=='milk'")
#             print(df[df.first_level=='milk'].shape)
#             df = df.assign(first_level = np.where(df.first_level == 'shyr', 'milk',
#                                                 df.first_level))
            
        df = df[df['first_level']==category]
        print(f'df.first_level == {category}')
        print(df.shape)
        df = down_sample(df,
                'second_level',
                downsample_n,
                random_state = 1)
         
        df = df[['filepath' , 'second_level']]
        df.columns = ['filepath', 'class']
    else:
        df = down_sample(df,
                'second_level',
                downsample_n,
                random_state = 1)
        df = df[['filepath' , 'first_level']]
        # TODO change here
#         df = df.assign(first_level = np.where(df.first_level == 'milk', 'shyr',
#                                                 df.first_level))
#         df = df.assign(first_level = np.where(df.first_level == 'yogurt', 'mast',
#                                                 df.first_level))

        df.columns = ['filepath', 'class']
        
    df = df.astype({'filepath' :str})
    
    
    if df.empty:
        raise Exception(
            '''NO NO returned dataframe is empty,
            maybe folder struct is not right.
            \nData/ Second category / vms ''')
    

    return df
    
                

def load_data_flow_from_dataframe(
                        data_dir:str,
                        target_image_size,
                        batch_size: int,
                        category='all_categories'
                        ):
    
     # SPLIT :: Train - Validation
    val_percent = 0.8

    # downsample , if n or more labels from one vms
    test_downsample=10
    train_downsample=150
                    
    dataframe = load_data_tree(data_dir)                    
    train_validation_df = prepare_dataframe(dataframe ,
                                            category,
                                            subset='train-val',
                                            downsample_n=train_downsample)
    
    # train_validation_df = return_dataframe(data_dir, 'train-val')
    print(train_validation_df.head())
    print('train_validation_df class nunique')
    print(train_validation_df['class'].nunique())

    train_df = train_validation_df.groupby('class', group_keys=False).\
                                        apply(lambda x: x.sample(max(int(val_percent*len(x)), 1)))
    validation_df = train_validation_df[~train_validation_df.filepath.isin(train_df.filepath)]  
     # TESTset
     
    test_df = prepare_dataframe(dataframe,
                                category,
                                subset='test',
                                downsample_n=test_downsample
                                 )
    # test_df = return_dataframe(data_dir, 'test')
    
    
    classes_set = sorted(list(set(train_validation_df['class'])))
    num_classes = len(classes_set)
    print(f'len classes_set {len(classes_set)}')

    # img_height , img_width = target_image_size

    test_datagen = ImageDataGenerator(
        #rescale=1./255,
    )


    validation_datagen = ImageDataGenerator(
        #rescale=1./255,
            )

    train_datagen = ImageDataGenerator(
            # rescale=1./255,
        shear_range=0.2,
        zoom_range= 0.3,
        fill_mode="constant",
        rotation_range=9,
        cval = 0,
        zca_whitening=False,
        brightness_range=[0,2],
        # channel_shift_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        # validation_split=0.2
        )

       
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=None,
        x_col="filepath",
        y_col="class",
        weight_col=None,
        target_size=target_image_size,
        color_mode="rgb",
        classes=classes_set,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=1,
        # save_to_dir=None,
        # save_prefix="",
        # save_format="png",
        # subset='training',
        interpolation="nearest",
        validate_filenames=True,
        )
        

    validation_generator = validation_datagen.flow_from_dataframe(
        validation_df,
        directory=None,
        x_col="filepath",
        y_col="class",
        weight_col=None,
        target_size=target_image_size,
        color_mode="rgb",
        classes=classes_set,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=1,
        # save_to_dir=None,
        # save_prefix="",
        # save_format="png",
        # subset='validation',
        interpolation="nearest",
        validate_filenames=True,
        )

    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        directory=None,
        x_col="filepath",
        y_col="class",
        weight_col=None,
        target_size=target_image_size,
        color_mode="rgb",
        classes=classes_set,
        class_mode="categorical",
        batch_size=1,
        shuffle=False,
        seed=1,
        # save_to_dir=None,
        # save_prefix="",
        # save_format="png",
        subset=None,
        interpolation="nearest",
        validate_filenames=True,
            )
    
    return dict([('train', train_generator),
                ('validation', validation_generator),
                ('test',test_generator),
                ('num_classes', num_classes),
                ('class_indices', train_generator.class_indices),
                ('samples', train_generator.n )
                        ])


def main():
    data_dir='/workspace/detect-me/product_classifier/data/'
    dataframe = load_data_tree(data_dir) 
    df = prepare_dataframe(dataframe,
                                'all_categories',
                                subset='train-val',
                                downsample_n=100
                                 )
    c = df.groupby('class').filepath.count().sort_values(ascending=False)
    print('train shape')
    print(df.shape)
    print('train head 10 ---------------------------------------------------')
    print(c.head(10))
    print('train tail 10  ---------------------------------------------------')
    print(c.tail(10))
    print('train shape ---------------------------------------------------')
    print(df.shape)
    print('train shape ---------------------------------------------------')
    print('-------------------------')
    df = prepare_dataframe(dataframe,
                                category='all_categories',
                                subset='test',
                                downsample_n=30
                                 )
    c = df.groupby('class').filepath.count().sort_values(ascending=False)
    print('test head 10 ---------------------------------------------------')
    print(c.head(10))
    print('test tail 10 ---------------------------------------------------')
    print(c.tail(10))
    print('test shape ---------------------------------------------------')
    print(df.shape)
    print('test shape ---------------------------------------------------')

    
if __name__ == '__main__':
    # main() 
    data_dir='/workspace/detect-me/product_classifier/data/'
    # dataframe = load_data_tree(data_dir)    
    # df = prepare_dataframe(dataframe,
    #                             category='yogurt',
    #                             subset='train-val',
    #                             downsample_n=100
    #                              )
    df = load_data_flow_from_dataframe(data_dir,(224,224), 2, category='milk')
