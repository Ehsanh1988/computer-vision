import pandas as pd
import json
import numpy as np
import re
import traceback
from PIL import ExifTags
from tensorflow.keras.models import load_model

def get_best_classifier(path_to_test_result:str,
                        metric:str):
    """
    metrics : 
     are defined in classifier model
        PRCN-tr-0.5
        PRCN-tr-0.9
        recall-tr--0.5
        recall-tr--0.9
        top_3_acc
        ________________________________________________
        date ::in case you want to get the most recent model
        f1_score  ::is defined in this function 
        length_classes :: in case you want to load a model trained on more classes
    """
    test_results = (path_to_test_result / 'test_result').glob('*.json')
    data_dict = dict() 
    for filep in test_results:
        with open(filep) as file:
            data = json.load(file)
        data_dict[filep.stem] = data

    dataframe_testResult = pd.DataFrame(data_dict).T.reset_index()
    dataframe_testResult = \
            dataframe_testResult.rename({'index' : 'tag'},
                                axis=1)
    patterns = [('freeze', '(?<=Freeze-)\d+'),
               ('lr' , '(?<=lr-)[\d|.]+'),
               ('dropout' , '(?<=dropout-)[\d|.]+'),
               ('epochs_it_ran' , '(?<=epochs_it_ran-)\d+'),
               ('tag_HMS' , '(?<=----)\d+'),
               ('modelname' , '(\w+)-{3}'),
               ('date' , '(?<=__).*?(?=----)')
               ]

    for (n , p) in patterns:
        if n == 'modelname':
            group = 1
        else:
            group=0
        dataframe_testResult.loc[:,n] =\
            dataframe_testResult['tag'].\
                    apply(lambda x : re.search(p,x).group(group) if re.search(p,x) else np.nan)


    prec_at90 = dataframe_testResult['PRCN-tr-0.9']
    recall_at90 = dataframe_testResult['recall-tr--0.9']
    # f1_score is calculated here ___________________________
    dataframe_testResult = \
        dataframe_testResult.assign(
            f1_score = ((2*recall_at90*prec_at90)/
                               (recall_at90+prec_at90)))

    dataframe_testResult = dataframe_testResult.sort_values([metric], ascending=False)
    
    best_model_baseon_testset = dataframe_testResult.head(1)
    
    tag= best_model_baseon_testset['tag_HMS'].item()
    print('best model')
    print(best_model_baseon_testset)
    root_tree = (path_to_test_result).rglob('*')

    df = pd.DataFrame(root_tree)

    df.columns = ['fullpath']

    df = pd.concat([pd.DataFrame((df['fullpath'].apply(lambda x: x.parts)).to_list()),
              df],axis=1)
    print(tag)
    model_path = df[(df[8] == 'saved_model.pb') & df[7].str.contains(tag)]['fullpath']

    model_path = model_path.item().parent
    return model_path


def exiff(image):
    '''
    rotate image base on exiff info'''
    try :
        for orientation in ExifTags.TAGS.keys() : 
            if ExifTags.TAGS[orientation]=='Orientation' : break 
        exif=dict(image._getexif().items())
        print(f'orientation :: {orientation}')
        if   exif[orientation] == 3 : 
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6 : 
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8 : 
            image=image.rotate(90, expand=True)
    except:
        traceback.print_exc()
        return image
    return image



def load_classifier(model_path):    
    model = load_model(model_path)
    csv = pd.read_csv(
        model_path /'label_mapping.csv')

    train_count_labels = \
        pd.read_csv(model_path /'train_info.csv',
                    index_col=0)
    train_count_labels.columns = ['code', 'counts']

    train_info = csv.set_index('CODE').\
            join(train_count_labels.set_index('code'))

    train_info.columns = ['label', 'count_in_trainset']
    return model, train_info



def predict_on_batch(model,list_crops):
    images_array = np.empty((len(list_crops), 224,224,3), np.uint8)

    for i, (img, _,_ ) in enumerate(list_crops):
        images_array[i,:,:,:] = img
    
    ps = model.predict(images_array)

    return ps