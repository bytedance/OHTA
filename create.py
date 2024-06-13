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
import time
import torch
import torch.nn as nn
import numpy as np 
from tqdm import tqdm
from core.nets import create_network
from core.train.trainers.ohta.trainer import _unpack_imgs
from core.utils.network_util import set_requires_grad
from core.utils.train_util import cpu_data_to_gpu
from core.utils.create_util import load_ckpt, get_data
from lpips import LPIPS
from configs import cfg
import shutil


class Creator:
    def __init__(self, cfg):
        # get parameters
        self.lr_inversion = cfg.one_shot.lr_inversion
        self.lr_finetune = cfg.one_shot.lr_finetune
        self.inversion_img_scale = cfg.one_shot.inversion_img_scale
        self.total_view = cfg.one_shot.total_view 
        self.input_view = cfg.one_shot.input_view
        self.other_view = [i for i in range(0, self.total_view) if i not in self.input_view]
        self.iter_inversion = cfg.one_shot.iter_inversion
        self.iter_finetune = cfg.one_shot.iter_finetune
        self.texture_editing = cfg.one_shot.texture_editing
        if self.texture_editing:
            self.iter_inversion = 0
            self.iter_finetune = cfg.one_shot.iter_finetune_edit
        self.iter_total = self.iter_inversion + self.iter_finetune
        self.channel = {
            'rgb': 3, 
            'alpha': 1,
            'albedo': 3, 
            'shadow': 1,
            'part': 1
        }

        # get paths
        self.ckpt_path = cfg.checkpoint
        self.image_path = cfg.input
        self.image_name = self.image_path.split('/')[-1].split('.')[0]
        self.root = os.path.dirname(os.path.dirname(self.image_path))
        self.folder_list = [self.root]
        self.image_list = [self.image_name]
        self.folder_indicator = self.folder_list[0].split('/')[-1]
        self.save_folder = f'./output/ohta/one_shot/{self.folder_indicator}_{self.image_name}_{self.input_view[0]}_{self.total_view}'
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(cfg.logdir, exist_ok=True)
        self.save_checkpoint_path = f'{self.save_folder}/one_shot.tar'
        self.save_checkpoint_path_for_eval = f'{cfg.logdir}/one_shot.tar'


        # set input data
        assert len(self.folder_list) == len(self.image_list)
        self.img_path = {self.input_view[i]: f'{self.folder_list[i]}/img/{self.image_list[i]}.jpg' for i in range(len(self.folder_list))}
        self.mask_path = {self.input_view[i]: f'{self.folder_list[i]}/mask/{self.image_list[i]}.png' for i in range(len(self.folder_list))}
        self.anno_path = {self.input_view[i]: f'{self.folder_list[i]}/anno/{self.image_list[i]}.pkl' for i in range(len(self.folder_list))}

        # copy inputs to output folder
        for key, img in self.img_path.items():
            shutil.copy(img, os.path.join(self.save_folder, 'input.jpg'))
            print('[INFO] Copy input images to output folder.')

        # set model
        self.set_network(cfg.checkpoint)
        

    def set_network(self, ckpt_path):
        # set pre-trained model
        self.model = create_network()
        self.model.train().cuda()
        self.model = load_ckpt(self.model, ckpt_path)
        set_requires_grad(self.model, requires_grad=False)

        # set lpips
        lpips = LPIPS(net='vgg')
        set_requires_grad(lpips, requires_grad=False)
        self.lpips = nn.DataParallel(lpips).cuda()


    def set_inversion_optimizer(self):
        params = [{'params': [self.model.color_shift, self.model.color_scale, self.model.id_feature]}]
        self.optimizer = torch.optim.Adam(params, lr=self.lr_inversion)
        self.model.color_shift.requires_grad = True
        self.model.color_scale.requires_grad = True
        self.model.id_feature.requires_grad = True
        if self.texture_editing:
            self.model.color_shift.requires_grad = False
            self.model.color_scale.requires_grad = False
            self.model.id_feature.requires_grad = False
    

    def set_finetune_optimizer(self):
        set_requires_grad(self.model.id_net_list, requires_grad=True)
        set_requires_grad(self.model.dict_fusion, requires_grad=True)
        params = []
        for key, value in self.model.named_parameters():
            params += [{"params": [value], 
                        "lr": self.lr_finetune}]
        self.optimizer = torch.optim.Adam(params, lr=self.lr_finetune)
        

    def scale_for_lpips(self, image_tensor): 
        return image_tensor * 2. - 1.


    def inversion_loss_func(self, pred, gt):
        return ((torch.abs(pred - gt) + 1e-5) ** 0.3).mean()


    def finetune_loss_func(self, pred, gt):
        l1 = torch.abs(pred - gt).mean() * 10
        lpips_loss = self.lpips(self.scale_for_lpips(pred.permute(0, 3, 1, 2)), self.scale_for_lpips(gt.permute(0, 3, 1, 2))).mean() * 1
        return l1 + lpips_loss


    def save_img_from_tensor(self, img_tensor, path):
        img_tensor = (255 * img_tensor.detach().cpu().numpy()).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(path, img_tensor)


    def save_reference_img(self, all_output):
        for sample_idx, rendering in all_output.items():
            for img_type in ['rgb', 'alpha']:
                self.save_img_from_tensor(rendering[img_type][0], f'./{self.save_folder}/ref_{sample_idx}_{img_type}.png')


    def save_final_img(self, all_output):
        for sample_idx, rendering in all_output.items():
            for img_type in ['rgb']:
                self.save_img_from_tensor(rendering[img_type][0], f'./{self.save_folder}/final_{sample_idx}_{img_type}.png')


    def get_all_data(self, img_size, scale, patch, all_view):
        all_input_numpy = []
        all_input_tensor = []
        for get_idx in range(self.total_view):
            if get_idx in self.input_view:
                all_input_numpy.append(get_data('right', self.anno_path[get_idx], self.input_view, self.save_folder, img_size, self.total_view, scale, get_idx, patch=patch))
            elif all_view:
                all_input_numpy.append(get_data('right', self.anno_path[self.input_view[0]], self.input_view, self.save_folder, img_size, self.total_view, scale, get_idx, patch=patch))
        results_tensor_list = [{k: torch.FloatTensor(v) for k, v in all_input_numpy[idx].items()} for idx in range(len(all_input_numpy))]
        for s_idx in range(len(all_input_numpy)):
            data = cpu_data_to_gpu(results_tensor_list[s_idx], exclude_keys=['target_rgbs'])
            all_input_tensor.append(data)
        return all_input_tensor


    @torch.no_grad()
    def get_all_view_rendering(self):
        all_output = {}
        img_size = (256, 256)
        all_input = self.get_all_data(img_size=img_size, scale=1, patch=False, all_view=True)
        for sample_idx, data in enumerate(all_input):
            rendering = {}
            net_output = self.model(**data, offset=False)
            for key, channel in self.channel.items():
                rendered = torch.zeros((img_size[0] * img_size[1], channel)).cuda().squeeze(-1)
                try:
                    rendered[data['ray_mask'].bool()] = net_output[key]
                except:
                    rendered = rendered.unsqueeze(-1)
                    rendered[data['ray_mask'].bool()] = net_output[key]
                rendered = rendered.reshape(1, img_size[0], img_size[1], channel)
                rendering[key] = rendered
            rendering['alpha'] = rendering['alpha'].repeat(1, 1, 1, 3)
            all_output[sample_idx] = rendering
        return all_output


    def inversion(self):
        self.set_inversion_optimizer()
        scale = self.inversion_img_scale
        img_size = (int(256 * scale), int(256 * scale))
        all_input = self.get_all_data(img_size=img_size, scale=scale, patch=False, all_view=False)
        for i in tqdm(range(self.iter_inversion)):
            for sample_idx, data in enumerate(all_input):
                net_output = self.model(**data, offset=False)
                rendering = {}
                for key, channel in self.channel.items():
                    rendered = torch.zeros((img_size[0] * img_size[1], channel)).cuda().squeeze(-1)
                    try:
                        rendered[data['ray_mask'].bool()] = net_output[key]
                    except:
                        rendered = rendered.unsqueeze(-1)
                        rendered[data['ray_mask'].bool()] = net_output[key]
                    rendered = rendered.reshape(1, img_size[0], img_size[1], channel)
                    rendering[key] = rendered

                rendering['alpha'] = rendering['alpha'].repeat(1, 1, 1, 3)
                rendering['part'] = rendering['part'].repeat(1, 1, 1, 3)

                # get mask without fingertips
                mask_tensor = data['target_alpha_img'][None, ..., None].repeat(1, 1, 1, 3)
                mask_tensor_bitwise = torch.where(mask_tensor == 1, 1, 0)
                rendered_part_bitwise = torch.where(rendering['part'] >= 0.5, 1, 0)
                mask_without_fingertip = torch.bitwise_xor(mask_tensor_bitwise.bool()[..., 0], rendered_part_bitwise[..., 0].bool())[..., None]                             
                
                masked_img_tensor = data['target_img'][None] * mask_tensor
                rendering['rgb'] = rendering['rgb'] * mask_tensor 

                loss = self.inversion_loss_func(masked_img_tensor * mask_without_fingertip, rendering['rgb'] * mask_without_fingertip)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    self.save_img_from_tensor(rendering['rgb'][0], f'{self.save_folder}/inversion_{i}_{sample_idx}.png')
                    self.save_img_from_tensor(masked_img_tensor[0], f'{self.save_folder}/inversion_{i}_{sample_idx}_gt.png')


    def finetune(self):
        self.set_finetune_optimizer()
        for i in tqdm(range(self.iter_inversion, self.iter_total)):
            all_input = self.get_all_data(img_size=(256, 256), scale=1, patch=True, all_view=True)
            for sample_idx, data in enumerate(all_input):
                rendering = {}
                net_output = self.model(**data, offset=False)
                rendering['rgb'], rendering['alpha'] = _unpack_imgs(net_output['rgb'], net_output['alpha'], data['patch_masks'].bool(), data['bgcolor'] / 255.,
                                            data['target_patches'], data['patch_div_indices'].long())
                rendering['part'], _ = _unpack_imgs(net_output['part'].repeat(1, 3), net_output['alpha'], data['patch_masks'].bool(), data['bgcolor'] / 255., data['target_patches'], data['patch_div_indices'].long())
                mask_tensor = data['target_alpha_patches'][..., None].repeat(1, 1, 1, 3)
                masked_img_tensor = data['target_patches'] * mask_tensor
                rendering['rgb'] = rendering['rgb'] * mask_tensor 

                input_weight = 4
                ref_weight = 1
                if self.texture_editing:
                    input_weight = 15
                    if sample_idx == self.total_view // 2:
                        ref_weight = 4
                if sample_idx in self.input_view:
                    loss = self.finetune_loss_func(masked_img_tensor, rendering['rgb']) * input_weight
                    tip_loss = torch.abs(masked_img_tensor * rendering['part'] - rendering['rgb'] * rendering['part']).mean() * 30 
                    loss += tip_loss
                else:
                    loss = self.finetune_loss_func(masked_img_tensor, rendering['rgb']) * ref_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    self.save_img_from_tensor(rendering['rgb'][0], f'{self.save_folder}/finetune_{i}_{sample_idx}.png')
                    self.save_img_from_tensor(masked_img_tensor[0], f'{self.save_folder}/finetune_{i}_{sample_idx}_gt.png')


    def save_checkpoint(self, path):
        torch.save({
            'iter': -1,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)


    def run(self):
        start_time = time.time()
        self.inversion()
        print(f'[INFO] Finished inversion: {time.time() - start_time}')
        self.save_reference_img(self.get_all_view_rendering())
        print(f'[INFO] Saved reference image: {time.time() - start_time}')
        self.finetune()
        print(f'[INFO] Finished tinetune: {time.time() - start_time}')
        self.save_final_img(self.get_all_view_rendering())
        print(f'[INFO] Saved final image: {time.time() - start_time}')
        self.save_checkpoint(self.save_checkpoint_path)
        self.save_checkpoint(self.save_checkpoint_path_for_eval)
        print(f'[INFO] Saved checkpoint: {time.time() - start_time}')

if __name__ == '__main__':
    creator = Creator(cfg)
    creator.run()




    



    