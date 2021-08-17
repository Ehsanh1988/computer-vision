from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple
import os
import re
import pathlib
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import shutil

    
def return_dataframe(dirr, subset : str):
    list_images = list(Path(dirr).rglob('*.jpg'))
    df = pd.DataFrame( {'filepath' : list_images,
                        'partial_filepath' : [pathlib.Path(str(x).split('/data/')[1]) for x in list_images]})
    
    df_parts = pd.DataFrame(
        df.partial_filepath.apply(lambda x : x.parts).to_list())
    
    df = pd.concat([df, df_parts],axis=1)

    df = df.rename({0:'seccat', 
                    1 :'subset',
                    2:'vms',
                    3:'filename'},
                   axis=1)

    df = df[df['subset'] == subset]
    df = df[~df.filepath.astype(str).str.contains('ipynb_checkpoints')]
    
    
    # remove over sample data from training 
    if subset == 'train-val':
        df = df.reset_index(drop=True)
        df = df.reset_index()

        counts = df.groupby('seccat').count()['filepath']

        oversample_data = counts[(np.abs(stats.zscore(counts)) > 3)]
        print('oversample_data',oversample_data.index.values)

        oversample_data_df = df[df.seccat.isin(oversample_data.index)]

        counts_oversample_data =  oversample_data_df.groupby(['seccat', 'vms']).count()['filepath']

        counts_oversample_data_ix = counts_oversample_data[np.abs(stats.zscore(counts_oversample_data)) > 2]

        D = df[df.vms.isin(counts_oversample_data_ix.reset_index()['vms'])]

        include = D.groupby('vms')['index'].sample(int(counts_oversample_data.mean()))
        exclude = D[~D.index.isin(include.index)]
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(f'excluding {exclude.shape} from trainig')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        df = df.assign(exclude_flag = np.where(df.index.isin(exclude.index), True, False))
        df = df[df.exclude_flag==False]
    

    if df.empty:
        raise Exception(
            '''NO NO returned dataframe is empty,
            maybe folder struct is not right.
            \nData/ Second category / vms ''')
    
    df_selected = df[['filepath','seccat']]

    df_selected = df_selected.assign(seccat = np.where(df_selected.seccat == 'milk', 'shyr', df_selected.seccat))
    df_selected = df_selected.assign(seccat = np.where(df_selected.seccat == 'yogurt', 'mast', df_selected.seccat))

    df_selected.columns = ['filename', 'class']

    df_selected = df_selected.astype({'filename' :str})

    return df_selected

def load_data_flow_from_dataframe(
                        data_dir:str,
                        target_image_size: (int, int),
                        batch_size: int,
                        ):
    
     # SPLIT :: Train - Validation
    val_percent = 0.8
                    
    train_validation_df = return_dataframe(data_dir, 'train-val')
    print(train_validation_df.head())
    
    print(train_validation_df['class'].nunique())

    train_df = train_validation_df.groupby('class', group_keys=False).\
                                        apply(lambda x: x.sample(max(int(val_percent*len(x)), 1)))
    validation_df = train_validation_df[~train_validation_df.filename.isin(train_df.filename)]  
     # TESTset
    test_df = return_dataframe(data_dir, 'test')
    
    
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
        brightness_range=[0.6,1.4],
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
        x_col="filename",
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
        x_col="filename",
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
        x_col="filename",
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
    data_directory='/workspace/detect-me/product_classifier/data/'
    df = return_dataframe(data_directory, 'train')
    print('train shape')
    print(df.shape)
    print(df.head())
    print('-------------------------')
    df = return_dataframe(data_directory, 'test')
    print('test shape')
    print(df.shape)
    print(df.head())
if __name__ == '__main__':
    main()                    

                    