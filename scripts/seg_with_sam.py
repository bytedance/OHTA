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
import torch
import numpy as np
from configs import cfg
from core.data import create_dataloader
from metaseg import SegManualMaskPredictor
from core.utils.vis_util import draw_2d_skeleton


@torch.no_grad()
def run(subject):
    cfg.perturb = 0.
    predictor = SegManualMaskPredictor()
    for phase in ['train', 'progress']:
        test_loader = create_dataloader(phase, subject=subject)
        output_folder = './vis_sam'
        os.makedirs(output_folder, exist_ok=True)
        for i, batch in enumerate(test_loader):
            print(i, len(test_loader))

            for k, v in batch.items():
                batch[k] = v[0]

            MANO2SIMPLE = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]

            img_path = batch['img_path']
            joints_uv = batch['joints_uv'][MANO2SIMPLE]

            min_joints = joints_uv.min(0)[0]
            max_joints = joints_uv.max(0)[0]

            bbox = [min_joints[0], min_joints[1], max_joints[0], max_joints[1]]
            center = (np.array(bbox[0:2]) + np.array(bbox[2:4])) / 2
            scale = 1.5
            x_length = (bbox[2] - bbox[0]) * scale
            y_length = (bbox[3] - bbox[1]) * scale
            bbox = [center[0] - x_length // 2, center[1] - y_length // 2, center[0] + x_length // 2, center[1] + y_length // 2]
            bbox = [int(i) for i in bbox]
            
            save_mask_path = img_path.replace('images', 'masks_SAM').replace('.jpg', '.png')
            assert 'masks_SAM' in save_mask_path
            os.makedirs(os.path.dirname(save_mask_path), exist_ok=True)


            input_point = joints_uv.numpy().astype(int)
            input_label = np.ones(input_point.shape[0], dtype=int)
            save_path = os.path.join(output_folder, '{:>04d}.png'.format(i))


            if i < 10:
                save = True
            else:
                save = False 

            sam_results = predictor.image_predict(
                source=img_path,
                model_type="vit_h", 
                input_point=input_point,
                input_label=input_label,
                input_box=bbox,  # XYXY
                multimask_output=False,
                random_color=False,
                show=False,
                output_path=save_path,
                save=save,
            )

            new_mask = (sam_results[0] * 255)[:, :, None].repeat(3, 2)
            cv2.imwrite(save_mask_path, new_mask)
            print(f'[INFO] Saving mask: {save_mask_path}')

            if i < 10:
                save_image = cv2.imread(save_path)
                save_image = draw_2d_skeleton(save_image, input_point)
                cv2.imwrite(save_path, save_image)
                print(f'[INFO] Saving debug image for SAM: {save_image}')
            

if __name__ == '__main__':
    subjects = ['train/prior_learning_data']
    run(subjects)


