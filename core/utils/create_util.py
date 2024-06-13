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
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
import pickle
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation
from third_parties import smplx
from core.utils.image_util import load_image
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox, \
    cam2pixel
from core.utils.augm_util import process_bbox
from configs import cfg
from core.utils.image_util import load_image


def select_rays(select_inds, rays_o, rays_d, ray_img, ray_alpha, near, far):
    rays_o = rays_o[select_inds]
    rays_d = rays_d[select_inds]
    ray_img = ray_img[select_inds]
    ray_alpha = ray_alpha[select_inds]
    near = near[select_inds]
    far = far[select_inds]
    return rays_o, rays_d, ray_img, ray_alpha, near, far

def get_patch_ray_indices(
        N_patch, 
        ray_mask, 
        subject_mask, 
        bbox_mask,
        patch_size, 
        H, W):

    assert subject_mask.dtype == np.bool_
    assert bbox_mask.dtype == np.bool_

    bbox_exclude_subject_mask = np.bitwise_and(
        bbox_mask,
        np.bitwise_not(subject_mask)
    )

    list_ray_indices = []
    list_mask = []
    list_xy_min = []
    list_xy_max = []

    total_rays = 0
    patch_div_indices = [total_rays]
    for _ in range(N_patch):
        # let p = cfg.patch.sample_subject_ratio
        # prob p: we sample on subject area
        # prob (1-p): we sample on non-subject area but still in bbox
        if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
            candidate_mask = subject_mask
        else:
            candidate_mask = bbox_exclude_subject_mask

        ray_indices, mask, xy_min, xy_max = \
            _get_patch_ray_indices(ray_mask, candidate_mask, 
                                        patch_size, H, W)

        assert len(ray_indices.shape) == 1
        total_rays += len(ray_indices)

        list_ray_indices.append(ray_indices)
        list_mask.append(mask)
        list_xy_min.append(xy_min)
        list_xy_max.append(xy_max)
        
        patch_div_indices.append(total_rays)

    select_inds = np.concatenate(list_ray_indices, axis=0)
    patch_info = {
        'mask': np.stack(list_mask, axis=0),
        'xy_min': np.stack(list_xy_min, axis=0),
        'xy_max': np.stack(list_xy_max, axis=0)
    }
    patch_div_indices = np.array(patch_div_indices)

    return select_inds, patch_info, patch_div_indices


def _get_patch_ray_indices(
        ray_mask, 
        candidate_mask, 
        patch_size, 
        H, W):

    assert len(ray_mask.shape) == 1
    assert ray_mask.dtype == np.bool_
    assert candidate_mask.dtype == np.bool_

    valid_ys, valid_xs = np.where(candidate_mask)

    # determine patch center
    select_idx = np.random.choice(valid_ys.shape[0], 
                                    size=[1], replace=False)[0]
    center_x = valid_xs[select_idx]
    center_y = valid_ys[select_idx]

    # determine patch boundary
    half_patch_size = patch_size // 2
    x_min = np.clip(a=center_x-half_patch_size, 
                    a_min=0, 
                    a_max=W-patch_size)
    x_max = x_min + patch_size
    y_min = np.clip(a=center_y-half_patch_size,
                    a_min=0,
                    a_max=H-patch_size)
    y_max = y_min + patch_size

    sel_ray_mask = np.zeros_like(candidate_mask)
    sel_ray_mask[y_min:y_max, x_min:x_max] = True

    #####################################################
    ## Below we determine the selected ray indices
    ## and patch valid mask

    sel_ray_mask = sel_ray_mask.reshape(-1)
    inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
    select_masked_inds = np.where(inter_mask)

    masked_indices = np.cumsum(ray_mask) - 1
    select_inds = masked_indices[select_masked_inds]
    
    inter_mask = inter_mask.reshape(H, W)

    return select_inds, \
            inter_mask[y_min:y_max, x_min:x_max], \
            np.array([x_min, y_min]), np.array([x_max, y_max])



def sample_patch_rays(img, alpha, H, W,
                        subject_mask, bbox_mask, ray_mask,
                        rays_o, rays_d, ray_img, ray_alpha, near, far):

    select_inds, patch_info, patch_div_indices = \
        get_patch_ray_indices(
            N_patch=cfg.patch.N_patches, 
            ray_mask=ray_mask, 
            subject_mask=subject_mask, 
            bbox_mask=bbox_mask,
            patch_size=cfg.patch.size, 
            H=H, W=W)

    rays_o, rays_d, ray_img, ray_alpha, near, far = select_rays(
        select_inds, rays_o, rays_d, ray_img, ray_alpha, near, far)
    
    targets = []
    targets_alpha = []
    for i in range(cfg.patch.N_patches):
        x_min, y_min = patch_info['xy_min'][i] 
        x_max, y_max = patch_info['xy_max'][i]
        targets.append(img[y_min:y_max, x_min:x_max])
        targets_alpha.append(alpha[y_min:y_max, x_min:x_max])
    target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)
    target_alpha_patches = np.stack(targets_alpha, axis=0)

    patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

    return rays_o, rays_d, ray_img, ray_alpha, near, far, \
            target_patches, target_alpha_patches, patch_masks, patch_div_indices


def load_masked_image(imagepath, maskpath, bg_color, use_mask=True):
    orig_img = np.array(load_image(imagepath))

    if use_mask:
        alpha_mask = np.array(load_image(maskpath))
    else:
        alpha_mask = np.ones_like(orig_img) * 255
    
    img = alpha_mask / 255. * orig_img + (1.0 - alpha_mask / 255.) * bg_color[None, None, :]
    if cfg.resize_img_scale != 1.:
        img = cv2.resize(img, None, 
                            fx=cfg.resize_img_scale,
                            fy=cfg.resize_img_scale,
                            interpolation=cv2.INTER_LANCZOS4)
        alpha_mask = cv2.resize(alpha_mask, None,
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LINEAR)
                            
    return img, alpha_mask


def query_dst_skeleton(mesh_infos):
    return {
        'poses': mesh_infos['poses'].astype('float32'),
        'shape': mesh_infos['shape'].astype('float32'),
        'dst_tpose_joints': \
            mesh_infos['tpose_joints'].astype('float32'),
        'bbox': mesh_infos['bbox'].copy(),
        'Rh': mesh_infos['Rh'].astype('float32'),
        'Th': mesh_infos['Th'].astype('float32'),
        'joint_img': mesh_infos['joint_img'].astype('int32'),
        'joint_cam': mesh_infos['joint_cam'].astype('float32'),
        'joint_valid': mesh_infos['joint_valid'].astype('float32')
    }


def load_ckpt(model, ckpt_path):
    '''
    '''
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    new_sd = model.state_dict()  # take the "default" state_dict
    pre_trained_sd = ckpt['network']  # load the old version pre-trained weights
    # merge information from pre_trained_sd into new_sd
    for k, v in new_sd.items():
        if k in pre_trained_sd:
            if v.shape == pre_trained_sd[k].shape:
                new_sd[k] = pre_trained_sd[k]
            else:
                print(f'[Warning] <Loading Model> Size mismatch: {k}, {v.shape}, {pre_trained_sd[k].shape}')
        else:
            print(f'[Warning] <Loading Model> {k} not in pretrained model.')
    weight = new_sd
    model.load_state_dict(weight, strict=False)
    return model 



def skeleton_to_bbox(skeleton):
    min_xyz = np.min(skeleton, axis=0) - 0.5
    max_xyz = np.max(skeleton, axis=0) + 0.5
    return {
        'min_xyz': min_xyz,
        'max_xyz': max_xyz
    }


def get_data(hand_type, anno_path, INPUT_IDX, SAVE_FOLDER, IMG_SIZE, TOTAL_IMAGE, SCALE, idx=-1, patch=False):
    results = {}
    bgcolor = np.array((0, 0, 0), dtype='float32')
    if idx in INPUT_IDX:
        imagepath = anno_path.replace('anno', 'img').replace('pkl', 'jpg')
        maskpath = imagepath.replace('img', 'mask').replace('.jpg', '.png')
        img, alpha = load_masked_image(imagepath, maskpath, bgcolor, True)  
        if IMG_SIZE[0] != 256:
            img = cv2.resize(img, (int(IMG_SIZE[0]), int(IMG_SIZE[1])))
            alpha = cv2.resize(alpha, (int(IMG_SIZE[0]), int(IMG_SIZE[1])))
    else:
        if patch:
            imagepath = f'./{SAVE_FOLDER}/ref_{idx}_rgb.png'
            maskpath = f'./{SAVE_FOLDER}/ref_{idx}_alpha.png'
            img, alpha = load_masked_image(imagepath, maskpath, bgcolor, True)
            assert img.shape[0] == 256
        else:
            img = np.zeros(shape=(int(IMG_SIZE[0]), int(IMG_SIZE[1]), 3))
            alpha = np.ones_like(img) * 255

    alpha = (alpha > 150).astype(np.uint8)
    img = (img / 255.).astype('float32')
    alpha = alpha.astype('float32')
    results['target_alpha_img'] = alpha[..., 0]

    H, W = img.shape[0:2]

    with open(anno_path, 'rb') as fi:
        cameras, mesh_infos, bbox, img_type = pickle.load(fi)

    dst_skel_info = query_dst_skeleton(mesh_infos)
    dst_bbox = dst_skel_info['bbox']
    dst_poses = dst_skel_info['poses']
    dst_shape = dst_skel_info['shape'].reshape(1, 10)
    dst_tpose_joints = dst_skel_info['dst_tpose_joints']
    dst_cam_joints = dst_skel_info['joint_cam']
    dst_valid_joints = dst_skel_info['joint_valid']

    def generate_regularize_data_annot(idx):

        mano_cfg = cfg.smpl_cfg.copy()
        mano_cfg['is_rhand'] = True if hand_type == 'right' else False
        mano = smplx.create(**mano_cfg)
        if hand_type == 'left':
            mano.shapedirs[:,0,:] *= -1

        
        ori_posed_res = mano(
            torch.from_numpy(dst_shape).float(), 
            torch.zeros(1,3).float(), 
            torch.from_numpy(dst_poses)[None, 3:].float(),
            return_verts=True)

        # MCP idx=4, wrist idx=0
        rot_axis = (ori_posed_res.joints[0][4] - ori_posed_res.joints[0][0]).detach().numpy()
        rot_axis_norm = rot_axis / np.linalg.norm(rot_axis)
        # rot_axis_norm = np.array([1,0,0], dtype=np.float32)
        delta_r = Rotation.from_rotvec(np.deg2rad(360 * idx / TOTAL_IMAGE) * rot_axis_norm).as_matrix()
        ori_rot = cv2.Rodrigues(dst_poses[:3])[0]
        new_rot_wrt_cam = ori_rot @ delta_r
        new_dst_Rh = cv2.Rodrigues(new_rot_wrt_cam)[0]
        dst_poses[:3] = new_dst_Rh.reshape(3)

        posed_res = mano(
            torch.from_numpy(dst_shape).float(), 
            torch.from_numpy(dst_poses)[None, :3].float(), 
            torch.from_numpy(dst_poses)[None, 3:].float(),
            return_verts=True)
        joints = posed_res.joints[0].detach().numpy() # / cfg.smpl_cfg.scale #世界坐标系
        verts = posed_res.vertices[0].detach().numpy() # / cfg.smpl_cfg.scale
        dst_bbox = skeleton_to_bbox(verts)


        # use mean pose for reference view
        dst_poses[3:] *= 0

        return dst_poses, dst_bbox
    
    if idx not in [-1] + INPUT_IDX:
        dst_poses, dst_bbox = generate_regularize_data_annot(idx)
    

    K = cameras['intrinsics'][:3, :3].copy()
    K[:2] *= SCALE

    try:
        E = cameras['extrinsics']
    except:
        E = np.eye(4)

    

    E = apply_global_tfm_to_camera(
            E=E, 
            Rh=dst_skel_info['Rh'],
            Th=dst_skel_info['Th'])
    R = E[:3, :3]
    T = E[:3, 3]
    # print(T, 3)
    distort = np.zeros(1)  # cameras['distort']
    xi = np.zeros(1)  # cameras['xi']
    results.update({
        'cam_K': K,
        'cam_R': R,
        'cam_T': T,
        'cam_distort': distort,
        'cam_xi': xi,
    })

    if img_type == 'rgb':
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
    else:
        assert False, 'Invalid image type!'

    ray_img = img.reshape(-1, 3) 
    rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
    rays_d = rays_d.reshape(-1, 3)
    ray_alpha = alpha.reshape(-1, 3) 


    # (selected N_samples, ), (selected N_samples, ), (N_samples, )
    near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)

    rays_o = rays_o[ray_mask]
    rays_d = rays_d[ray_mask]
    ray_img = ray_img[ray_mask]
    ray_alpha = ray_alpha[ray_mask]

    near = near[:, None].astype('float32')
    far = far[:, None].astype('float32')

    if patch:
        rays_o, rays_d, ray_img, ray_alpha, near, far, \
        target_patches, target_alpha_patches, patch_masks, patch_div_indices = \
            sample_patch_rays(img=img, alpha=alpha, H=H, W=W,
                                    subject_mask=alpha[:, :, 0] > 0.,
                                    bbox_mask=ray_mask.reshape(H, W),
                                    ray_mask=ray_mask,
                                    rays_o=rays_o, 
                                    rays_d=rays_d, 
                                    ray_img=ray_img, 
                                    ray_alpha=ray_alpha,
                                    near=near, 
                                    far=far)
        results.update({
            'patch_div_indices': patch_div_indices,
            'patch_masks': patch_masks,
            'target_patches': target_patches,
            'target_alpha_patches': target_alpha_patches[..., 0],
            })

    batch_rays = np.stack([rays_o, rays_d], axis=0) 

    results.update({
        'img_width': W,
        'img_height': H,
        'ray_mask': ray_mask,
        'rays': batch_rays,
        'near': near,
        'far': far,
        'bgcolor': bgcolor})

    results['target_rgbs'] = ray_img
    results['target_alpha'] = ray_alpha[..., 0]
    results['target_img'] = img

    # Set ID
    results['id'] = np.array([0])

    dst_posevec_69 = dst_poses[3:] + 1e-2
    results.update({
        'dst_posevec': dst_posevec_69,
        'dst_shape': dst_shape,
        'dst_global_orient': dst_poses[:3],
        'dst_cam_joints': dst_cam_joints,
        # 'dst_valid_joints': dst_valid_joints[:, 0]
        'dst_valid_joints': dst_valid_joints,
        # NOTE: this can not be added to the results dict!
        # 'dst_Th': dst_Th,
    })


    results.update({
        'dst_Rs': np.array([0]), 
        'dst_Ts': np.array([0]), 
        'timestamp': 0,
        'img_type': np.array([0]),
        'hand_type': np.array([1]) if hand_type == 'right' else np.array([0])
    })
    return results