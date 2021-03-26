# -*- coding: utf-8 -*-
""" main.py """

from configs.config import CFG
from models.resnet50 import ResNet50


def run():
    """Builds model, loads data, trains and evaluates"""
    model = ResNet50(cat = 'milk',
                     config =CFG,
                     name_tensorboard='freeze_up_to_120_lr0005T')

    print('len(base_model.layers) :\n',len(model.base_model.layers))

    model.load_data_from_dir()
    model.build(model.dataset['num_classes'],)
    model.compile(fine_tune_at=None, lr=None)
    history = model.train(save=False)
    model.evaluate(model.dataset['test'])

if __name__ == '__main__':
    run()