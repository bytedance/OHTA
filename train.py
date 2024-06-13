# *************************************************************************
# Copyright 2024 ByteDance and/or its affiliates
#
# Copyright 2024 OHTA Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# *************************************************************************

# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2024) B-
# ytedance Inc..  
# *************************************************************************

from configs import cfg
from core.utils.log_util import Logger, Board
from core.data import create_dataloader
from core.nets import create_network
from core.train import create_trainer, create_optimizer
import os
import torch


def main():
    log = Logger()
    # log.print_config()
    
    model = create_network()
    phase = cfg.get('phase', 'train')

    if phase == 'val':
        trainer = create_trainer(model, None, board=None)
        trainer.progress()
    else:
        board = Board()
        optimizer = create_optimizer(model)
        trainer = create_trainer(model, optimizer, board=board)
        train_loader = create_dataloader('train')
        # estimate start epoch
        epoch = trainer.iter // len(train_loader) + 1
        while True:
            if trainer.iter > cfg.train.maxiter: #cfg.train.maxepoch * len(train_loader):
                break
            
            trainer.train(epoch=epoch,
                        train_dataloader=train_loader)
            epoch += 1

        trainer.finalize()

if __name__ == '__main__':
    main()
