# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2024) B-
# ytedance Inc..  
# *************************************************************************

import os
import imp
import time

import numpy as np
import torch
from configs import cfg
from .dataset_args import DatasetArgs


def _query_dataset(data_type):
    module = cfg[data_type].dataset_module
    module_path = module.replace(".", "/") + ".py"
    dataset = imp.load_source(module, module_path).Dataset
    return dataset


def create_dataset(data_type='train', subject=None):
    dataset_name = cfg[data_type].dataset

    args = DatasetArgs.get(dataset_name)

    # customize dataset arguments according to dataset type
    args['bgcolor'] = None if data_type == 'train' else cfg.bgcolor
    args['data_type'] = data_type
    if data_type == 'progress':
        if cfg.progress.get('skip', -1) > 0:
            args['skip'] = cfg.progress.skip
            args['maxframes'] = -1
        else:
            total_train_imgs = 20000 
            args['skip'] = 16
            args['maxframes'] = 16
        
    args['subject'] = subject
    
    # if data_type in ['freeview', 'tpose', 'freepose', 'infer']:
    #     args['skip'] = cfg[data_type].get('skip', 1)
    # if data_type in ['infer']:
    #     args['subject'] = subject

    dataset = _query_dataset(data_type)
    dataset = dataset(**args)
    return dataset


def _worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def create_dataloader(data_type='train', **kwargs):
    cfg_node = cfg[data_type]

    batch_size = cfg_node.batch_size
    shuffle = cfg_node.shuffle
    drop_last = cfg_node.drop_last
    num_workers = cfg_node.num_workers

    dataset = create_dataset(data_type=data_type, **kwargs)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=num_workers,
                                              worker_init_fn=_worker_init_fn)

    return data_loader
