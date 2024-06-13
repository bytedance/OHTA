# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2024) B-
# ytedance Inc..  
# *************************************************************************

import imp

from configs import cfg


def create_lr_updater():
    module = cfg.lr_updater_module
    lr_updater_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, lr_updater_path).update_lr
