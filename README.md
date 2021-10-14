## train
**--category** *milk yogurt,... or all_categories* 
**--backbone**
- 'vgg16'
- 'resnet50'
- 'resnet50v2'
- 'resnet152v2'
- 'inception_resnet'
- 'efficientnetb7'
- 'densenet121'
- 'densenet201'

`python main.py --backbone 'densenet201' --category "milk" --save True --desc 'ALL_minus_DB' --weights 'imagenet'`

