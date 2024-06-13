# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2024) B-
# ytedance Inc..  
# *************************************************************************

import torch
import torch.nn as nn
from core.nets.ohta.component_factory import \
    load_render_mlp, \
    load_deform_mlp
from configs import cfg
from core.utils.network_util import set_requires_grad, trunc_normal_
import third_parties.smplx
from third_parties.smplx.manohd.subdivide import sub_mano
from third_parties.smplx.utils import vertex_normals
from third_parties.smplx.lbs import get_normal_coord_system
from third_parties.pairof.pairof_model import attach_pairof



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # manohd
        smpl_body = third_parties.smplx.create(**cfg.smpl_cfg).requires_grad_(False)
        if not cfg.smpl_cfg.is_rhand:
            smpl_body.shapedirs[:,0,:] *= -1

        if cfg.smpl_cfg['manohd']>0:
            print('MANO-HD in Model')
            smpl_body ,_ ,_ = sub_mano(smpl_body, cfg.smpl_cfg['manohd'])
            lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
            smpl_body.lbs_weights = lbs_weights

        # pairof
        self.smpl_body = attach_pairof(smpl_body, cfg.smpl_cfg)

        #  load pre-trained PairOF
        if cfg.phase == 'train' and not cfg.resume:
            weigt_name = cfg.smpl_cfg.get('pairof_pretrain')
            weight = torch.load(weigt_name)['state_dict']

            new_sd = self.smpl_body.coap.state_dict()  # take the "default" state_dict
            pre_trained_sd = weight  # load the old version pre-trained weights
            # merge information from pre_trained_sd into new_sd
            for k, v in new_sd.items():
                # if k in pre_trained_sd:
                if k in pre_trained_sd and 'encoder' in k:
                    if v.shape == pre_trained_sd[k].shape:
                        new_sd[k] = pre_trained_sd[k]
                    else:
                        print(f'[Warning] <Loading Model> Size mismatch: {k}, {v.shape}, {pre_trained_sd[k].shape}')
                else:
                    print(f'[Warning] <Loading Model> {k} not in pretrained model.')
            weight = new_sd

            # after merging the state dict you can load it:
            self.smpl_body.coap.load_state_dict(weight, strict=False)
            print(f'Load pretrained coap: {weigt_name}')
            if cfg.ignore_smpl_body == 'encoder':
                set_requires_grad(self.smpl_body.coap.part_encoder, False)
                print('freeze coap part encoder')
                if hasattr(self.smpl_body.coap, 'local_encoder'):
                    set_requires_grad(self.smpl_body.coap.local_encoder, False)
                    print('freeze coap local encoder')
            elif cfg.ignore_smpl_body == 'all':
                set_requires_grad(self.smpl_body, False)
                print('freeze coap')


        # deformer 
        self.center_template = torch.mm(self.smpl_body.J_regressor, self.smpl_body.v_template)[4]
        self.deformer = load_deform_mlp(cfg.deform_network.module)(verts=self.smpl_body.v_template[None], **cfg.deform_network)
            
        # Multi-resolution Field
        self.dict_list = cfg.shadow_network.dict_list + cfg.color_network.dict_list 
        self.color_dict_list = nn.ParameterList()
        for _, sample_point_num in enumerate(self.dict_list):
            self.color_dict_list.append(nn.Parameter(
            torch.randn([self.smpl_body.coap.partitioner.__getattr__(f'sample_face_tensor_{sample_point_num}').shape[0], cfg.color_network.code_dim]), requires_grad=True))

        def simple_mlp(in_dim, hidden_dim, out_dim, n_layer):
            assert n_layer >= 2
            net = nn.Sequential()
            for i in range(n_layer):
                if i == 0:
                    net.append(nn.Linear(in_dim, hidden_dim))
                    net.append(nn.ReLU())
                elif i == n_layer - 1:
                    net.append(nn.Linear(hidden_dim, out_dim))
                    net.append(nn.Sigmoid())
                else:
                    net.append(nn.Linear(hidden_dim, hidden_dim))
                    net.append(nn.ReLU())
            return net

        # color fusion
        self.color_index = cfg.color_network.dict_list
        self.color_dict_num = len(self.color_index)
        self.id_net_list = nn.ModuleList([load_render_mlp(cfg.color_network.module)(**cfg.color_network) for _ in range(self.color_dict_num)])
        self.dict_fusion = simple_mlp(cfg.color_network['d_out'] * self.color_dict_num + 33, cfg.color_network['d_out'] * self.color_dict_num, 3, 3)
        
        # shadow fusion
        self.shadow_index = cfg.shadow_network.dict_list
        self.shadow_dict_num = len(self.shadow_index)
        self.shadow_net_list = nn.ModuleList([load_render_mlp(cfg.shadow_network.module)(**cfg.shadow_network) for _ in range(self.shadow_dict_num)])
        self.shadow_dict_fusion = simple_mlp(cfg.color_network['d_out'] * self.shadow_dict_num, cfg.color_network['d_out'] * self.shadow_dict_num, 1, 3)
        
        # id code
        self.id_code = nn.Parameter(torch.zeros(30, 33))
        trunc_normal_(self.id_code, std=.02)


        self.one_shot = cfg.one_shot.enable
        if self.one_shot:
            # parameters for one-shot reconstruction
            self.id_feature = nn.Parameter(torch.zeros(33), requires_grad=True)
            self.color_scale = nn.Parameter(torch.ones(1, 3), requires_grad=True)
            self.color_shift = nn.Parameter(torch.zeros(1, 3), requires_grad=True)
        else:
            self.id_feature = None
            self.color_scale = None 
            self.color_shift = None
        

    def _query_mlp(
            self,
            pos_xyz,
            smpl_output,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        smpl_output=smpl_output,
                        non_rigid_mlp_input=non_rigid_mlp_input,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                        chunk=chunk)

        output = {}
        alpha_flat = result['alpha']
        all_out_flat = result['all_out']
        part_idx = result['part_idx'][:, None]
        output['alpha'] = torch.reshape(
                            alpha_flat, 
                            list(pos_xyz.shape[:-1]) + [alpha_flat.shape[-1]])
        output['all_out'] = torch.reshape(
                            all_out_flat, 
                            list(pos_xyz.shape[:-1]) + [all_out_flat.shape[-1]])
        output['part_idx'] = torch.reshape(
                            part_idx, 
                            list(pos_xyz.shape[:-1]) + [part_idx.shape[-1]])
        
        for k, sample_point_num in enumerate(self.dict_list):
            output[f'feat_{sample_point_num}'] = torch.reshape(
                            result[f'feat_{sample_point_num}'][:, None], 
                            list(pos_xyz.shape[:-1]) + [result[f'feat_{sample_point_num}'].shape[-1]])
            
        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            smpl_output,
            non_rigid_mlp_input,
            non_rigid_pos_embed_fn,
            chunk):
        alpha_list = []
        all_out_list = []
        part_idx_list = []
        list_dict = {}
        for sample_point_num in self.dict_list:
            list_dict[sample_point_num] = []


        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]

            xyz = pos_flat[start:end]
            alpha, part_info = self.smpl_body.coap.query(xyz[None], smpl_output, ret_intermediate=True)
            alpha_list += [alpha[0, :, None]]
            all_out_list += [part_info['all_out'][0, :, None]]
            part_idx_list += [part_info['part_idx']]  # [100000]

            for k, sample_point_num in enumerate(self.dict_list):
                if len(part_info['gdists_list']) != 0:
                    dists = part_info['gdists_list'][k][0]
                    indices = part_info['gindices_list'][k][0]
                    n_point, n_neighbor = indices.shape
                    ws = (1. / dists).unsqueeze(-1)
                    ws = ws / ws.sum(-2, keepdim=True)
                    sel_latent = torch.gather(
                        self.color_dict_list[k], 0, 
                        indices.unsqueeze(-1).view(n_point*n_neighbor, -1).expand(-1, cfg.color_network.code_dim)
                        ).view(n_point, n_neighbor, -1)
                    sel_latent = (sel_latent * ws).sum(dim=-2)
                    list_dict[sample_point_num].append(sel_latent.clone())
                else:
                    list_dict[sample_point_num].append(sel_latent.clone())

        output = {}
        output['alpha'] = torch.cat(alpha_list, dim=0).to(cfg.secondary_gpus[0])
        output['all_out'] = torch.cat(all_out_list, dim=0).to(cfg.secondary_gpus[0])
        output['part_idx'] = torch.cat(part_idx_list, dim=0).to(cfg.secondary_gpus[0])
        for k, sample_point_num in enumerate(self.dict_list):
            output[f'feat_{sample_point_num}'] = torch.cat(list_dict[sample_point_num], dim=0).to(cfg.secondary_gpus[0])
        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            smpl_output, dst_Th=None,
            non_rigid_pos_embed_fn=None,
            non_rigid_mlp_input=None,
            bgcolor=None,
            **_):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        if dst_Th is not None:
            pts -= dst_Th
        dirs = rays_d[:, None, :].expand(pts.shape)
        N_samples = pts.shape[1]

        # occ
        query_result = self._query_mlp(
                                pos_xyz=pts,
                                smpl_output=smpl_output,
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)
        alpha = query_result['alpha'][:, :, 0]
        all_out = query_result['all_out'][:, :, 0]
        part_idx = query_result['part_idx'].long()

        # process id feature
        if self.id_feature is not None:
            id_feat = self.id_feature[None, None].repeat(N_rays, N_samples, 1)
        else:
            if self.id_feat is None:
                id_feat = self.id_code[self.id][None, None].repeat(N_rays, N_samples, 1) # * 0
            else:
                print('[Model Info] Use input identity feature.')
                id_feat = self.id_feat[None, None].repeat(N_rays, N_samples, 1)

        all_feat = []
        for k, sample_point_num in enumerate(self.color_index):
            feat = query_result[f'feat_{sample_point_num}']
            input_feat = torch.cat((feat.reshape(-1, feat.shape[-1]), id_feat.reshape(-1, id_feat.shape[-1])), dim=-1)
            sampled_feat = self.id_net_list[k](None, None, None, input_feat.reshape(-1, input_feat.shape[-1]), False).reshape(N_rays, N_samples, -1)
            all_feat.append(sampled_feat)
        all_feat = torch.cat(all_feat + [id_feat], dim=-1)
        sampled_color = self.dict_fusion(all_feat) 

        # color calibration
        if self.color_scale is not None:
            new_mean = self.color_shift[None]
            new_scale = self.color_scale[None]
            sampled_color = torch.clip(sampled_color * new_scale + new_mean, min=0, max=1)

        if not cfg.ignore_shadow_network:
            # use joint position without global rotation
            pose_feat = self.smpl_output_no_rot.joints.reshape(1, -1)[None].expand(N_rays, N_samples, 63)
            all_feat = []
            for k, sample_point_num in enumerate(self.shadow_index):
                feat = query_result[f'feat_{sample_point_num}']
                input_feat = torch.cat((feat.reshape(-1, feat.shape[-1]), pose_feat.reshape(-1, pose_feat.shape[-1])), dim=-1)
                sampled_feat = self.shadow_net_list[k](None, None, None, input_feat.reshape(-1, input_feat.shape[-1]), False).reshape(N_rays, N_samples, -1)
                all_feat.append(sampled_feat)

            all_feat = torch.cat(all_feat, dim=-1)
            sampled_shadow = self.shadow_dict_fusion(all_feat) 
            sampled_albedo = sampled_color.clone()
            sampled_color = sampled_color * sampled_shadow 

        bgcolor = bgcolor[:sampled_color.shape[-1]]

        # render
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = (sampled_color * weights[:, :, None]).sum(dim=1) # torch.Size([N_rays, 3])
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.-acc_map[..., None]) * (bgcolor[None, :]/255).to(rgb_map.device)

        # fingertip part map.
        # TODO: make it simple
        part_idx = torch.where(part_idx==3, 1, 0) + torch.where(part_idx==6, 1, 0) + torch.where(part_idx==9, 1, 0) + torch.where(part_idx==12, 1, 0) + torch.where(part_idx==15, 1, 0)
        part_map = part_idx.reshape(N_rays, N_samples, -1)
        part_map = (part_map * weights[:, :, None]).sum(dim=1)

        res = {'rgb': rgb_map.to(cfg.primary_gpus[0]),  
                'alpha': acc_map.to(cfg.primary_gpus[0]),
                'part': part_map.to(cfg.primary_gpus[0]),  
                }

        if not cfg.ignore_shadow_network:
            shadow_map = (sampled_shadow[..., 0] * weights).sum(dim=1)
            albedo_map = (sampled_albedo * weights[:, :, None]).sum(dim=1)
            albedo_map = albedo_map + (1.-acc_map[..., None]) * (bgcolor[None, :]/255).to(albedo_map.device)
            res.update({
                'shadow': shadow_map.to(cfg.primary_gpus[0]),
                'albedo': albedo_map.to(cfg.primary_gpus[0]),
                })

        if not self.training and not cfg.ignore_shadow_network:
            shadow_map = (sampled_shadow[..., 0] * weights).sum(dim=1)
            albedo_map = (sampled_albedo * weights[:, :, None]).sum(dim=1)
            albedo_map = albedo_map + (1.-acc_map[..., None]) * (bgcolor[None, :]/255).to(albedo_map.device)
            res.update({
                'shadow': shadow_map.to(cfg.primary_gpus[0]),
                'albedo': albedo_map.to(cfg.primary_gpus[0]),
                'weights': weights.to(cfg.primary_gpus[0]),
                'sampled_albedo': sampled_albedo.to(cfg.primary_gpus[0]),
                'sampled_shadow': sampled_shadow.to(cfg.primary_gpus[0])
                })

        if 'iou3d' in list(cfg.train.lossweights.keys()):
            res.update({
                    'pred_occ': alpha.to(cfg.primary_gpus[0]),
                    'pts': pts.to(cfg.primary_gpus[0]),
                    'all_out': all_out.to(cfg.primary_gpus[0]),
                })

        return res

    def forward(self,
                rays, 
                dst_Rs, dst_Ts,
                dst_posevec, dst_shape, dst_global_orient, dst_Th=None,
                near=None, far=None,
                iter_val=1e7,
                **kwargs):

        if 'id' in kwargs:
            self.id = kwargs['id']
        else:
            self.id = 0

        if 'id_feat' in kwargs:
            self.id_feat = kwargs['id_feat'] # None # kwargs['id_feat']
        else:
            self.id_feat = None

        dst_global_orient=dst_global_orient[None, ...]
        dst_shape=dst_shape[None, ...].reshape(-1, 10)
        dst_posevec=dst_posevec[None, ...]
        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        if dst_Th is not None:
            dst_Th = dst_Th[None]


        if not cfg.ignore_deform:
            id_feat = self.id_code[self.id][None]
            offsets = self.deformer(torch.cat((dst_posevec, id_feat), dim=-1)).to(cfg.primary_gpus[0])
            if cfg.deform_network.get('normal_frame'):
                normals = vertex_normals(self.smpl_body.v_template[None], self.smpl_body.faces_tensor[None])
                normal_coord_sys = get_normal_coord_system(normals.view(-1, 3)).view(1, self.smpl_body.v_template.shape[0], 3, 3)
                offsets = torch.matmul(normal_coord_sys.permute(0, 1, 3, 2), offsets.unsqueeze(-1)).squeeze(-1)            
            shaped_verts = self.smpl_body.v_template[None] + offsets
            center = torch.mm(self.smpl_body.J_regressor, shaped_verts[0])[4]
            shaped_verts = shaped_verts - center[None, None] + self.center_template[None, None].to(shaped_verts.device)
        else:
            shaped_verts = None
        

        smpl_output = self.smpl_body(
            dst_shape, 
            dst_global_orient, 
            dst_posevec, 
            # transl=trans,
            return_verts=True, 
            return_full_pose=True,
            shaped_verts=shaped_verts,
            )

        self.smpl_output_no_rot = self.smpl_body(
            dst_shape, 
            dst_global_orient * 0, 
            dst_posevec, 
            return_verts=True, 
            return_full_pose=True,
            shaped_verts=shaped_verts,
            )

        kwargs.update({
                'smpl_output': smpl_output,
                'dst_Th': dst_Th
            })


        rays_o, rays_d = rays
        rays_shape = rays_d.shape 

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
        for k in all_ret:
            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)

        all_ret.update({'smpl_output': smpl_output, 'shaped_verts': shaped_verts,})
        return all_ret
