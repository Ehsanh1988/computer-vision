from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple
import os
import re
import pathlib
from pathlib import Path
import pandas as pd
import numpy as np

# import fnmatch
import os
import shutil


# def clear_directory(directory):
#     # DELETE .ipynb_checkpoints .. 
#     folder_to_delete = list(pathlib.Path(directory).rglob('.ipynb_checkpoints'))
#     for file in folder_to_delete:
#         try:
#             shutil.rmtree(file) 
#             print(f'folder removed {file}')
#             print('folder is clear')
#         except FileNotFoundError:
#             print('no ipynb_checkpoints shit s here )')
        
        
# def return_dataframes(folder, classes):
#     list_files = list(pathlib.Path(folder).\
#                                   rglob('*.jpg'))

#     df = pd.DataFrame({
#             'filename':[str(x) for x in list_files],
#             'class':[x.parent.name for x in list_files]},
#                  )
#     df = df.astype({
#             'filename':str,
#             'class' :str}
#     )
    
#     selected_df = df[df['class'].isin(list(classes))].\
#                         sort_values('class').\
#                             reset_index(drop=True)
#     return selected_df


# def load_classes_from_csv(csv):
#     if isinstance(csv, pd.DataFrame):
#         vms = csv.iloc[:,csv.columns.str.contains('vms')].iloc[:,0]
#         return sorted(vms.astype(str), key=float)
#     else:
#         csv = pd.read_csv(csv)
#         vms = csv.iloc[:,csv.columns.str.contains('vms')].iloc[:,0]
#         print(f'loading classes form csv {csv}')
#         return sorted(vms.astype(str), key=float)
    
def return_dataframe(dirr, subset):
    list_images = list(Path(dirr).rglob('*.jpg'))
    df = pd.DataFrame(
    {'filepath' : list_images,
    'partial_filepath' : [pathlib.Path(str(x).split('/data/')[1]) for x in list_images]})
    df_parts = pd.DataFrame(
        df.partial_filepath.apply(lambda x : x.parts).to_list())
    df_j = pd.concat([df, df_parts],
                     axis=1)

    df_j = df_j.rename({0:'seccat', 
                1 :'subset',
                2:'vms',
                3:'filename'},axis=1)

    df_j = df_j[df_j['subset'] == subset]
    df_j = df_j[~df_j.filepath.astype(str).str.contains('ipynb_checkpoints')]
    
    if df_j.empty:
        raise Exception('NO NO returned dataframe is empty, maybe folder struct is not right..\nData/ Second category / vms ')
    
    df_selected = df_j[['filepath','seccat']]

    df_selected = df_selected.assign(seccat = np.where(df_selected.seccat == 'milk', 'shyr', df_selected.seccat))

    df_selected = df_selected.assign(seccat = np.where(df_selected.seccat == 'yogurt', 'mast', df_selected.seccat))

    df_selected.columns = ['filename', 'class']

    df_selected = df_selected.astype({'filename' :str})
    
    print('____________________')
    print(df_selected['class'].nunique())
    return df_selected

def load_data_flow_from_dataframe(
                        data_dir:str,
#                         test_dir : str,
                        target_image_size: (int, int),
                        batch_size,
#                         csv_path_classes=None,
                        load_with_mixup_generator= False,
                        mixup_alpha = 0.2
                        ):
#     _=[clear_directory(x) for x in [train_validation_dir,test_dir]]
#     if csv_path_classes is not None:
#         classes = load_classes_from_csv(csv_path_classes) 
#     else:
#         classes  = list(pathlib.Path(train_validation_dir).glob('*'))
#         classes = sorted([str(x.name) for x in classes])
    
    
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

    if load_with_mixup_generator:
        print('load_with_mixup_generator')
        train_generator = MixupImageDataGenerator(generator=train_datagen,
                                            directory=None,
                                            dataframe=train_df,
                                            batch_size=batch_size,
                                            target_image_size=target_image_size,
                                            classes_set = classes_set,
                                            alpha=mixup_alpha
                                            #   img_height=target_image_size[0],
                                            #   img_width=target_image_size[1],
                                            #   subset='training'
                                            )
    else:
        print('load_with OUT_mixup_generator')
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



def load_data_flow_from_directory(
                        train_validation_dir:str,
                        test_dir : str,
                        target_image_size: (int, int),
                        batch_size,
                        csv_path_classes=None,
                        load_with_mixup_generator= True):
    _=[clear_directory(x) for x in [train_validation_dir,test_dir]]
    
    
    all_classes_in_train_val_dir  = list(pathlib.Path(train_validation_dir).glob('*'))
    all_classes_in_train_val_dir = [str(x.name) for x in all_classes_in_train_val_dir]
    
    
    if csv_path_classes:
        list_classes = load_classes_from_csv(csv_path_classes)
        list_classes = set(list_classes) & set(all_classes_in_train_val_dir)
        print(f'a subset of classes in training: read by {csv_path_classes}')
    else:
        list_classes = all_classes_in_train_val_dir
    
    img_height , img_width = target_image_size

    test_datagen = ImageDataGenerator(
                #rescale=1./255,
    )


    validation_datagen = ImageDataGenerator(
            #rescale=1./255,   
            validation_split=0.2)

    train_datagen = ImageDataGenerator(
            # rescale=1./255,
        shear_range=0.2,
        zoom_range= 0.3,
        fill_mode="constant",
        rotation_range=9,
        cval = 0,
        zca_whitening=False,
        # channel_shift_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        #preprocessing_function=ResNet50.preprocessing_fun,
        validation_split=0.2) # set validation split

    # list_classes_generator = 
    
    train_generator = train_datagen.flow_from_directory(
        train_validation_dir,
        shuffle=True,
        target_size=(img_height, img_width),
        seed=1,
        classes=list_classes,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training') # set as training data

    validation_generator = validation_datagen.flow_from_directory(
        train_validation_dir, # same directory as training data
        shuffle=True,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        seed=1,
        classes=list_classes,
        # classes=list(train_generator.class_indices.keys()),
        class_mode='categorical',
        subset='validation') # set as validation data

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        shuffle=False,
        target_size=(img_height, img_width),
        seed=1,
        batch_size=1,
        classes=list_classes,
        # classes=list(train_generator.class_indices.keys()),
        class_mode='categorical',
            )
    
    return dict([('train', train_generator),
                ('validation', validation_generator),
                ('test',test_generator),
                ('num_classes',train_generator.num_classes),
                ('class_indices', train_generator.class_indices),
                ('samples', train_generator.n )
                        ])
    



import numpy as np


# def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
#     def eraser(input_img):
#         img_h, img_w, _ = input_img.shape
#         p_1 = np.random.rand()

#         if p_1 > p:
#             return input_img

#         while True:
#             s = np.random.uniform(s_l, s_h) * img_h * img_w
#             r = np.random.uniform(r_1, r_2)
#             w = int(np.sqrt(s / r))
#             h = int(np.sqrt(s * r))
#             left = np.random.randint(0, img_w)
#             top = np.random.randint(0, img_h)

#             if left + w <= img_w and top + h <= img_h:
#                 break

#         c = np.random.uniform(v_l, v_h)
#         input_img[top:top + h, left:left + w, :] = c

#         return input_img

#     return eraser

# class MixupGenerator():
#     def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
#         self.X_train = X_train
#         self.y_train = y_train
#         self.batch_size = batch_size
#         self.alpha = alpha
#         self.shuffle = shuffle
#         self.sample_num = len(X_train)
#         self.datagen = datagen

#     def __call__(self):
#         while True:
#             indexes = self.__get_exploration_order()
#             itr_num = int(len(indexes) // (self.batch_size * 2))

#             for i in range(itr_num):
#                 batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
#                 X, y = self.__data_generation(batch_ids)

#                 yield X, y

#     def __get_exploration_order(self):
#         indexes = np.arange(self.sample_num)

#         if self.shuffle:
#             np.random.shuffle(indexes)

#         return indexes

#     def __data_generation(self, batch_ids):
#         _, h, w, c = self.X_train.shape
#         l = np.random.beta(self.alpha, self.alpha, self.batch_size)
#         X_l = l.reshape(self.batch_size, 1, 1, 1)
#         y_l = l.reshape(self.batch_size, 1)

#         X1 = self.X_train[batch_ids[:self.batch_size]]
#         X2 = self.X_train[batch_ids[self.batch_size:]]
#         X = X1 * X_l + X2 * (1 - X_l)

#         if self.datagen:
#             for i in range(self.batch_size):
#                 X[i] = self.datagen.random_transform(X[i])
#                 X[i] = self.datagen.standardize(X[i])

#         if isinstance(self.y_train, list):
#             y = []

#             for y_train_ in self.y_train:
#                 y1 = y_train_[batch_ids[:self.batch_size]]
#                 y2 = y_train_[batch_ids[self.batch_size:]]
#                 y.append(y1 * y_l + y2 * (1 - y_l))
#         else:
#             y1 = self.y_train[batch_ids[:self.batch_size]]
#             y2 = self.y_train[batch_ids[self.batch_size:]]
#             y = y1 * y_l + y2 * (1 - y_l)

#         return X, y 







class MixupImageDataGenerator():
    def __init__(self, generator, directory, dataframe, batch_size, target_image_size,classes_set, alpha=0.2, subset=None):
        """Constructor for mixup image data generator.

        Arguments:
            generator {object} -- An instance of Keras ImageDataGenerator.
            directory {str} -- Image directory.
            batch_size {int} -- Batch size.
            img_height {int} -- Image height in pixels.
            img_width {int} -- Image width in pixels.

        Keyword Arguments:
            alpha {float} -- Mixup beta distribution alpha parameter. (default: {0.2})
            subset {str} -- 'training' or 'validation' if validation_split is specified in
            `generator` (ImageDataGenerator).(default: {None})
        """

        self.batch_index = 0
        self.batch_size = batch_size
        self.alpha = alpha

        # First iterator yielding tuples of (x, y)
        # self.generator1 = generator.flow_from_directory(directory,
        #                                                 target_size=(
        #                                                     img_height, img_width),
        #                                                 class_mode="categorical",
        #                                                 batch_size=batch_size,
        #                                                 shuffle=True,
        #                                                 subset=subset)
        
        self.generator1 = generator.flow_from_dataframe(
                        dataframe, 
                        directory=directory,
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


        # Second iterator yielding tuples of (x, y)
        # self.generator2 = generator.flow_from_directory(directory,
        #                                                 target_size=(
        #                                                     img_height, img_width),
        #                                                 class_mode="categorical",
        #                                                 batch_size=batch_size,
        #                                                 shuffle=True,
        #                                                 subset=subset)
        
        self.generator2 = generator.flow_from_dataframe(
                        dataframe,
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
                        seed=2,
                        # save_to_dir=None,
                        # save_prefix="",
                        # save_format="png",
                        # subset='training',
                        interpolation="nearest",
                        validate_filenames=True,
                        )

        # Number of images across all classes in image directory.
        self.n = self.generator1.samples
        self.class_indices = self.generator1.class_indices
        self.labels = self.generator1.labels
        self.samples = self.generator1.samples
        self.classes = self.generator1.classes

    def reset_index(self):
        """Reset the generator indexes array.
        """

        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def on_epoch_end(self):
        self.reset_index()

    def reset(self):
        self.batch_index = 0

    def __len__(self):
        # round up
        return (self.n + self.batch_size - 1) // self.batch_size

    def get_steps_per_epoch(self):
        """Get number of steps per epoch based on batch size and
        number of images.

        Returns:
            int -- steps per epoch.
        """

        return self.n // self.batch_size

    def __next__(self):
        """Get next batch input/output pair.

        Returns:
            tuple -- batch of input/output pair, (inputs, outputs).
        """

        if self.batch_index == 0:
            self.reset_index()

        current_index = (self.batch_index * self.batch_size) % self.n
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        # random sample the lambda value from beta distribution.


        # Get a pair of inputs and outputs from two iterators.
        X1, y1 = self.generator1.next()
        X2, y2 = self.generator2.next()
        
        l = np.random.beta(self.alpha, self.alpha, X1.shape[0])

        X_l = l.reshape(X1.shape[0], 1, 1, 1)
        y_l = l.reshape(X1.shape[0], 1)

        # Perform the mixup.
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return X, y

    def __iter__(self):
        while True:
            yield next(self)


# if __name__ == '__main__':
#     return_dataframe                    
    
                    