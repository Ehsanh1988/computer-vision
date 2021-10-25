import tensorflow as tf
import os;os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
import re
import datetime
import PIL
import os
import shutil
import datetime
import pathlib
import json

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.get_logger().setLevel('ERROR')

import warnings
warnings.filterwarnings("ignore")

def regex_timestamp(s):
    grs = re.search("\d{4}(-|_)+\d{2}(-|_)+\d{2}(-|_)+\d{2}(-|_|:)+\d{2}(-|_|:)+\d{2}", s)
    if grs:
        return grs.group()
    else:
        return '0000-00-00-00-00-00'

# def load_classifier(basepath,
#                     basemodel,
#                     modelname):    
#     model = load_model(basepath / basemodel /modelname)
#     csv = pd.read_csv(
#         basepath/ basemodel / modelname /'label_mapping.csv')

#     train_count_labels = \
#         pd.read_csv(basepath/ basemodel / modelname /'train_info.csv',
#                     index_col=0)
#     train_count_labels.columns = ['code', 'counts']

#     train_info = csv.set_index('CODE').\
#             join(train_count_labels.set_index('code'))

#     train_info.columns = ['vms', 'count_in_trainset']  
#     return model, train_info

# load best model
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

    train_info.columns = ['vms', 'count_in_trainset']
    return model, train_info


def predict_on_batch(model,list_crops, train_info):
    images_array = np.empty((len(list_crops), 224,224,3), np.uint8)
    images_path = []

    for i, filep in enumerate(list_crops):
        images_array[i,:,:,:] = np.array(PIL.Image.open(filep))
        images_path.append(filep)
    
    
    ps = model.predict(images_array)

    
    return images_path,ps

def post_predict(images_p,ps, train_info):
    df = pd.DataFrame({'np.max' : np.max(ps,axis=1),
               'np.argmax' : np.argmax(ps,axis=1),
               'images_path' : images_p})

    df = df.assign(vms=df['np.argmax'].map(train_info['vms']))

    # FIRST FIVE GUESS
    dd = train_info['vms'].to_dict()
    f5 = pd.DataFrame(ps.argsort(axis=1)[:,-5:][:,::-1],
                                columns=['p1','p2','p3', 'p4', 'p5'])
    f5 = f5.stack().map(dd).unstack()

    #add shape
    df_shapes = df.images_path.apply(lambda x : x.name).str.extract('(\d+)_(\d+).jpg')
    df_shapes.columns=['height', 'width']


    df = pd.concat([df,
                    df_shapes,
                    f5,
                    pd.DataFrame(ps)],
                   axis=1)

    df = df.assign(f5 = df[['p1','p2','p3', 'p4', 'p5']].apply(lambda x: '-'.join(x.astype(str)), axis=1))

    df = df.assign(datetime = df.images_path.apply(lambda x : regex_timestamp(Path(x).stem)))

    df = df.assign(camera = df.images_path.apply(lambda x :x.name).str.extract('([a-zA-Z]{3,8}-camera\d+)'))   
    return df


def move_to_vms_folder(df,temp_dir):
    import uuid
    for row in df.itertuples():
        dest = Path(temp_dir) / 'vms_preds' / str(row.vms)
        if not dest.exists():
            os.makedirs(dest)
            print(f'making dir {dest}')
        rndm_filename = str(uuid.uuid4())
        shutil.move(row.images_path,
                    dest/ f'{row._1:.3}-{row.f5}_{rndm_filename}.jpg')
    print(f'moved  {df.shape[0]} files')
    
def get_best_classifier(path_to_test_result:str,
                        metric=None):
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
               ]

    for (n , p) in patterns:
        if n == 'modelname':
            group = 1
        else:
            group=0
        dataframe_testResult.loc[:,n] =\
            dataframe_testResult['tag'].\
                    apply(lambda x : re.search(p,x).group(group) if re.search(p,x) else np.nan)


    dataframe_testResult.head(1)

    prec_at90 = dataframe_testResult['PRCN-tr-0.9']
    recall_at90 = dataframe_testResult['recall-tr--0.9']
    dataframe_testResult = \
        dataframe_testResult.assign(
            f1score = (2*(recall_at90*prec_at90)/
                               (recall_at90+prec_at90)
                      ))
    dataframe_testResult.f1score = dataframe_testResult.f1score.fillna(0)

#     import pdb;pdb.set_trace()
    dataframe_testResult = dataframe_testResult.sort_values(['f1score'], ascending=False)
    
    best_model_baseon_testset = dataframe_testResult.head(1)
    
    tag= best_model_baseon_testset['tag_HMS'].item()
    print('best model')
    print(best_model_baseon_testset)
    root_tree = (path_to_test_result).rglob('*')

    df = pd.DataFrame(root_tree)

    df.columns = ['fullpath']

    df = pd.concat([pd.DataFrame((df['fullpath'].apply(lambda x: x.parts)).to_list()),
              df],axis=1)
#     print(tag)
    model_path = df[(df[8] == 'saved_model.pb') & df[7].str.contains(tag)]['fullpath']

    model_path = model_path.item().parent
    return model_path

def main():
    root_classifier = Path(F'/workspace/detect-me/product_classifier/saved_models/')

    # BasePATH = Path('/workspace/detect-me/product_classifier/saved_models/milk')

    temp_dir = '/workspace/detect-me/cronjobs/cronjob_result/images_from_store/temp'    
    # classifier_model_name='EfficientNetB7_Freeze_50_lr-0.0023-adam_epoch-50-batch-16_dropout-0.0-BEST__2021-Jun-29----130811'
    # classifier_model_name='RESNET152V2---(Freeze-200)-(lr-0.0004-adam)-(opt_cb-cosine_learning_rate_decay)-(epoch-25)-(batch-16)-(dropout-0.1)-N__2021-Aug-23----230559'
    # base_model = 'resnet152v2'

    # classifier_model_name_2 = 'RESNET50V2_Freeze_50_lr-0.001-adam_epoch-50-batch-24_dropout-0.1-TEST_scheduler__2021-Jun-29----000837'
    # base_model_2 = 'resnet50v2'
    # classifier_model_name_2='RESNET50V2_Freeze_50_lr-0.001-adam_epoch-50-batch-24_dropout-0.1-TEST_scheduler__2021-Jun-29----000837'

    # !mkdir '/workspace/keras-retinanet/images_from_store/saved_result/'

    list_crops = (Path(temp_dir) / 'crops').glob('*.jp*')
    list_crops = list(list_crops)

    list_crops_df = pd.DataFrame(list_crops)
    list_crops_df.columns= ['filepath']

    list_crops_df = list_crops_df.assign(
            stem = list_crops_df.filepath.apply(lambda x:pathlib.Path(x).stem ))

    list_crops_df = list_crops_df.assign(
        category = list_crops_df.stem.apply(lambda x : x.split('-')[0]))

    if not list_crops_df.empty:
        result_all = pd.DataFrame()
        ALL_PRODUCTS = pd.read_csv('/workspace/detect-me/all_products_small.csv',
                                    index_col=0,
                                    )
        ALL_PRODUCTS = ALL_PRODUCTS.set_index('vms_food_id')
        ALL_PRODUCTS.index = ALL_PRODUCTS.index.astype(str)
        ALL_PRODUCTS = ALL_PRODUCTS[['title', 'SecCat']]
    #     kmeans = KMeans(n_clusters=4)

        for cat in list_crops_df.category.unique():
            print(cat)
            crops_fp = list_crops_df[list_crops_df.category==cat]['filepath']

            model_path = get_best_classifier(root_classifier/cat)
            model, train_info = load_classifier(model_path)

            images_p, ps = predict_on_batch(model,crops_fp,train_info)

            print('done predicting')
            df = post_predict(images_p, ps, train_info)
            move_to_vms_folder(df,temp_dir)

            # d8te = list(set(['-'.join(re.findall('(\d+)-(\d+)-(\d+)_(\d+):(\d+):(\d+)',str(x))[0]) for x in df.images_path]))
            # d8te = str(d8te[0])

            results = df[['np.max',
                        'vms',
                        'height',
                        'width',
                        'datetime',
                        'camera']]

            results.columns= ['score',
                             'vms',
                             'height',
                             'width',
                             'datetime',
                             'camera']


            results = results[results.score > 0.95]

            # preds_df.columns = [*['H','W','C'],  *preds_df.columns[3:]]
    #         kmeans_result = kmeans.fit_predict(results[['height','width']])
    #         results = pd.concat([results.reset_index(drop=True),
    #                             pd.Series(kmeans_result)], axis=1)

    #         results = results.rename({0:'cluster'},axis=1)

            RESULT = results[['vms', 'score', 'height' ,'width', 'datetime', 'camera']]

            # RESULT.columns = ['vms', 'score', 'h', 'w', 'cluster']

            RESULT = RESULT.astype({'vms' : str})
            RESULT = RESULT.set_index('vms').join(ALL_PRODUCTS)
            RESULT.index.name = 'vms'
            RESULT = RESULT.reset_index()

            RESULT = RESULT.astype({'vms' :int,
                          'score' : float,
                          'height':int,
                          'width' :int,
    #                       'cluster' : int,
                          'title' : str})

            RESULT.sort_values('height', inplace=True)
            result_all = pd.concat([result_all,RESULT])
            print(result_all.shape)

    #     gr = df.groupby('camera').datetime.unique().reset_index()
    #     num_cams = len(gr)
    #     # d8te = '_'.join(gr.astype(str).apply('-'.join, axis=1))
    #     # d8te = datetime.datetime.today().strftime('%m-%d-%H%M')
        d8te = (datetime.datetime.utcnow() +
               datetime.timedelta(hours=4, minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
        

    #     n = f"/workspace/sftp-files-from-BI-SRV/predictions_{base_model}_{num_cams}_camera_{d8te}.csv"
        num_cams = len(result_all.camera.unique())
        n = f"/workspace/sftp-files-from-BI-SRV/predictions_{num_cams}_camera_{d8te}.csv"
        result_all.to_csv(n)
    #     print(f'saved {n}')
    #     print(RESULT.shape)
    #     print(RESULT.head())   
        print(result_all.shape)
    else:
        print('nothing to predict (classify)')
        
#     _ = [os.remove(x) for x in list_crops]
#     print(len(list_crops))
if __name__=='__main__':
    main()
    