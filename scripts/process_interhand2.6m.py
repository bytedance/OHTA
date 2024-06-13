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


import os 
import cv2 
import pickle
import numpy as np 
from configs import cfg
from core.data.interhand.train import Dataset
from core.data.dataset_args import DatasetArgs
from core.utils.augm_util import augmentation, trans_point2d


class DataProcessor(Dataset):
    def __init__(self, dataset_path, keyfilter=None, maxframes=-1, bgcolor=None, ray_shoot_mode='image', skip=1, subject=None, **kwargs):
        super().__init__(dataset_path, keyfilter, maxframes, bgcolor, ray_shoot_mode, skip, subject, **kwargs)

    def process_frame(self, frame_name, output_folder):
        '''
        '''
        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(frame_name, bgcolor, use_mask=(True, False)[self.phase=='infer'])
        bbox = self.bbox[frame_name]
        img, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, alpha = augmentation(img, self.bbox[frame_name], 'eval',
                                                                                     exclude_flip=True,
                                                                                     input_img_shape=(256, 256), mask=alpha,
                                                                                     base_scale=1.3,
                                                                                     scale_factor=0.2,
                                                                                     rot_factor=0,
                                                                                     shift_wh=[bbox[2], bbox[3]],
                                                                                     gaussian_std=3,
                                                                                     bordervalue=bgcolor.tolist())

        img = (img / 255.).astype('float32')
        alpha = alpha.astype('float32')

        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2]*aug_param[1])
        self.cameras[frame_name]['intrinsics'][:3, :3] = K

        for i in range(len(self.mesh_infos[frame_name]['joint_img'])):
            self.mesh_infos[frame_name]['joint_img'][i] = trans_point2d(self.mesh_infos[frame_name]['joint_img'][i], img2bb_trans)

        img_name = frame_name.replace('/', '_')
        img_folder = f'{output_folder}/img'
        anno_folder = f'{output_folder}/anno'
        mask_folder = f'{output_folder}/mask'
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(anno_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)

        img_path = os.path.join(img_folder, img_name)
        anno_path = os.path.join(anno_folder, img_name.replace('jpg', 'pkl'))
        mask_path = os.path.join(mask_folder, img_name.replace('jpg', 'png'))

        fi = open(anno_path, 'wb')
        pickle.dump([self.cameras[frame_name], self.mesh_infos[frame_name], self.bbox[frame_name], 'rgb'], fi)
        cv2.imwrite(mask_path, alpha[:, :, ::-1] * 255)
        cv2.imwrite(img_path, img[:, :, ::-1] * 255)

if __name__ == '__main__':
    process_subject = 'test/Capture0/ROM03_RT_No_Occlusion'
    process_img_list = ['test/Capture0/ROM03_RT_No_Occlusion/cam400272/image15012.jpg']
    output_folder = 'example_data/interhand2.6m'
    
    args = DatasetArgs.get('interhand_train')
    args['subject'] = process_subject
    args['data_type'] = 'process'
    args['bgcolor'] = cfg.bgcolor
    dataset = DataProcessor(**args)

    for img in process_img_list:
        dataset.process_frame(img, output_folder)
        print(f'[INFO] Processed {img}.')
