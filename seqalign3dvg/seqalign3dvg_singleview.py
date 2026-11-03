# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch

import torch.nn as nn
from mmengine.structures import InstanceData

import torch
import torch.nn.functional as F



from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptInstanceList
from .feat_ext import dinov2_vits14_lc_dpt
from torch.nn import functional as F
from torch import nn

import clip
from .plvf_singleview import ProgressiveTextVoxelFusion


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(torch.meshgrid([
        torch.arange(n_voxels[0]),
        torch.arange(n_voxels[1]),
        torch.arange(n_voxels[2])
    ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject(features, points, projection):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1) # 20, 4, 25600
    points_2d_3 = torch.bmm(projection.float(), points.float()) # projection: 20, 3, 4 points_2d_3: 20, 3, 25600
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long() # 20, 25600
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0) # 20, 25600
    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device)
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid

def _compute_projection(img_meta, stride):
    projection = []
    intrinsic = torch.tensor(img_meta['cam2img'][:3, :3]) # 3x3
    extrinsic = torch.tensor(img_meta['lidar2cam']) # 4x4

    ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
    ratio2 = img_meta['ori_shape'][1] / (img_meta['img_shape'][1] / stride)
    intrinsic[0] /= ratio2
    intrinsic[1] /= ratio

    projection.append(intrinsic @ extrinsic[:3])

    return torch.stack(projection)


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name='ViT-B/32', freeze=True):
        super().__init__()
        self.model, _ = clip.load(model_name, device='cpu', jit=False)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, text_list, return_tokens: bool = False):
        token_ids = clip.tokenize(text_list, truncate=True)
        device = next(self.model.parameters()).device
        token_ids = token_ids.to(device)

        if not return_tokens:
            pooled = self.model.encode_text(token_ids)
            return pooled.float()

        # token embedding: (B,T,d_model)
        x = self.model.token_embedding(token_ids).type(self.model.dtype)
        x = x + self.model.positional_embedding.type(self.model.dtype)

        x = x.permute(1, 0, 2)          # (T,B,d_model)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)          # (B,T,d_model)

        x = self.model.ln_final(x).type(self.model.dtype)

        # project to CLIP embed dim D
        token_embeds = x @ self.model.text_projection  # (B,T,D)

        # pooled at EOT
        eot_pos = token_ids.argmax(dim=-1)  # (B,)
        pooled = token_embeds[torch.arange(token_embeds.shape[0], device=device), eot_pos]  # (B,D)

        return pooled.float(), token_embeds.float(), token_ids

@MODELS.register_module()
class SeqAlign3DVG_SingleView(Base3DDetector):
    """`SeqAlign3DVG Single View`_.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        neck_3d (:obj:`ConfigDict` or dict): The 3D neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        prior_generator (:obj:`ConfigDict` or dict): The prior points
            generator config.
        n_voxels (list): Number of voxels along x, y, z axis.
        coord_type (str): The type of coordinates of points cloud:
            'DEPTH', 'LIDAR', or 'CAMERA'.
        train_cfg (:obj:`ConfigDict` or dict, optional): Config dict of
            training hyper-parameters. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Config dict of test
            hyper-parameters. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (:obj:`ConfigDict` or dict, optional): The initialization
            config. Defaults to None.
    """

    def __init__(self,
                 neck_3d: ConfigType,
                 bbox_head: ConfigType,
                 prior_generator: ConfigType,
                 n_voxels: List,
                 voxel_size: List,
                 coord_type: str,
                 text_encoder_name: str = 'ViT-B/32',
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg) 
        self.neck_3d = MODELS.build(neck_3d)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.coord_type = coord_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.depth_est = dinov2_vits14_lc_dpt(layers=1)  
        self.text_encoder = CLIPTextEncoder(model_name=text_encoder_name, freeze=True)  
        with torch.no_grad():
            dummy_dim = self.text_encoder(["test"]).shape[1]
    
        self.text_dim = dummy_dim
        self.n_voxels = [40, 40, 16]
        self.voxel_size = [0.16, 0.16, 0.16]
        self.label_smoothing = 1.05
        self.epoch_n = 0
        self.inference_time = 0

        voxel_dim = 256

        self.fusion_level = 2 

        self.text_voxel_fuser = ProgressiveTextVoxelFusion(
            voxel_dim=voxel_dim,
            text_dim=self.text_dim,
            n_voxels=tuple(self.n_voxels),
            voxel_size=tuple(self.voxel_size),
            fusion_level=self.fusion_level,
            baseline_method="concat",
            sim_normalize=True,
            use_baseline=False, 
        )

        self.prior_loss_weight = 0.5


        X, Y, Z = self.n_voxels
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(X),
                torch.arange(Y),
                torch.arange(Z),
                indexing="ij",
            )
        ).float()  # (3,X,Y,Z)
        self.register_buffer("voxel_grid", grid, persistent=False)


    @torch.no_grad()
    def _voxel_centers_lidar(self, batch_img_metas, device, dtype) -> torch.Tensor:
        """
        返回 voxel centers in lidar/world frame: (B,3,X,Y,Z)
        与你 backproject 用的 voxel grid 对齐（用 origin / voxel_size / n_voxels）
        """
        B = len(batch_img_metas)
        grid = self.voxel_grid.to(device=device, dtype=dtype)  # (3,X,Y,Z)

        origins = torch.stack(
            [torch.as_tensor(m["origin"], device=device, dtype=dtype) for m in batch_img_metas],
            dim=0,
        )  # (B,3)

        n_vox = torch.tensor(self.n_voxels, device=device, dtype=dtype).view(1, 3, 1, 1, 1)
        vsz  = torch.tensor(self.voxel_size, device=device, dtype=dtype).view(1, 3, 1, 1, 1)

        # 这里按 center = (idx+0.5)*vsz + (origin - 0.5*n_vox*vsz)
        base = origins.view(B, 3, 1, 1, 1) - 0.5 * n_vox * vsz
        # centers = (grid.view(1, 3, *grid.shape[1:]) + 0.5) * vsz + base
        centers = (grid.view(1, 3, *grid.shape[1:])) * vsz + base
        return centers  # (B,3,X,Y,Z)

    @torch.no_grad()
    def _build_gt_target_occ(
        self,
        batch_data_samples,
        batch_img_metas,
        valid_mask: torch.Tensor,   # (B,1,X,Y,Z) bool
    ) -> torch.Tensor:
        """
        从 GT 3D bbox 构造 occupancy: (B,1,X,Y,Z) float in {0,1}
        这里先用 axis-aligned（忽略 yaw），足够给 prior 一个稳定监督。
        """
        device = valid_mask.device
        dtype = torch.float32
        B, _, X, Y, Z = valid_mask.shape

        centers = self._voxel_centers_lidar(batch_img_metas, device=device, dtype=dtype)  # (B,3,X,Y,Z)

        occ = torch.zeros((B, 1, X, Y, Z), device=device, dtype=dtype)

        batch_gt_instances_3d = [data.gt_instances_3d for data in batch_data_samples]
        for b in range(B):
            gt_instance = batch_gt_instances_3d[b]
            if gt_instance is None or len(gt_instance) == 0:
                continue

            # 你给的取法（默认第一个 box）
            gt_bbox = torch.cat(
                (
                    gt_instance.bboxes_3d.gravity_center[0, :3],
                    gt_instance.bboxes_3d.tensor[0, 3:6],
                ),
                dim=0,
            ).to(device=device, dtype=dtype)  # (6,) = (cx,cy,cz, dx,dy,dz)

            c = gt_bbox[:3].view(1, 3, 1, 1, 1)
            d = gt_bbox[3:6].view(1, 3, 1, 1, 1).clamp(min=1e-6)

            inside = (centers[b:b+1] - c).abs() <= (0.5 * d)
            inside = inside.all(dim=1, keepdim=True)  # (1,1,X,Y,Z) bool
            occ[b:b+1] = inside.float()

        occ = occ * valid_mask.float()
        return occ


    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        -> 3d neck -> bbox_head.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            Tuple:
            - torch.Tensor: Features of shape (N, C_out, N_x, N_y, N_z).
            - torch.Tensor: Valid mask of shape (N, 1, N_x, N_y, N_z).
        """

        # Use ViT to extract image features and depth maps.
        img = batch_inputs_dict['imgs']  # b, 3, H, W
        B, _, H, W = img.size()    
        batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]
        descriptions = [img_meta['description'] for img_meta in batch_img_metas]
        
        vision_features, _, _ = self.depth_est(img)
        stride = H / vision_features.shape[2]

        volumes, valids = [], []
        for feature, img_meta in zip(vision_features, batch_img_metas):
            projection = _compute_projection(img_meta, stride).to(vision_features.device)
            points = get_points(
                    n_voxels=torch.tensor(self.n_voxels),
                    voxel_size=torch.tensor(self.voxel_size),
                    origin=torch.tensor(img_meta['origin'])
                ).to(vision_features.device)
            height = int(img_meta['img_shape'][0] // stride)
            width = int(img_meta['img_shape'][1] // stride)         
            volume, valid = backproject(feature.unsqueeze(0)[:, :, :height, :width], points, projection)
            volume = volume.sum(dim=0) #256,40,40,16         
            valid = valid.sum(dim=0) #1,40,40,16
            volume = volume / valid
            valid = valid > 0 #1,40,40,16 many falses
            volume[:, ~valid[0]] = .0 
            volumes.append(volume)
            valids.append(valid)

        x = torch.stack(volumes)
        valids = torch.stack(valids)


        # text_embeds = self.text_encoder(descriptions)  # (B, C_t)
        # x_fused = self._fuse_text_voxel(x, text_embeds)
        with torch.no_grad():
            pooled, token_embeds, token_ids = self.text_encoder(descriptions, return_tokens=True)

        x_fused, fusion_debug = self.text_voxel_fuser(
                voxel_feat=x,
                pooled_text=pooled,
                token_embeds=token_embeds,
                token_ids=token_ids,
                valid_mask=valids,
                return_debug=True,
            )
        
        x = self.neck_3d(x_fused) # x is a list of three tensors: 0 (b, 128, 40, 40, 16), 1 (b, 128, 20, 20, 8), 2 (b, 128, 10, 10, 4)
        return x, valids, fusion_debug
        


    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        x, valid_preds, fusion_debug = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        losses = {}

        losses = self.bbox_head.loss(x, batch_data_samples, valid_preds=valid_preds)
                # prior supervision
        if (fusion_debug is not None) and ("target_logits" in fusion_debug) and (self.prior_loss_weight > 0):
            gt_occ = self._build_gt_target_occ(batch_data_samples, batch_img_metas, valid_preds)  # (B,1,X,Y,Z)

            logits = fusion_debug["target_logits"].to(dtype=torch.float32)  # (B,1,X,Y,Z)
            valid = valid_preds.float()

            # BCEWithLogits per-voxel
            bce = F.binary_cross_entropy_with_logits(logits, gt_occ, reduction="none")
            bce = bce * valid

            pos = (gt_occ > 0.5) & (valid > 0.5)
            neg = (gt_occ <= 0.5) & (valid > 0.5)

            pos_cnt = pos.sum().clamp(min=1)
            neg_cnt = neg.sum().clamp(min=1)
            w_pos = (neg_cnt / pos_cnt).clamp(max=50.0)  # 防止极端不平衡

            loss_pos = bce[pos].mean() if pos.any() else torch.zeros((), device=bce.device)
            loss_neg = bce[neg].mean() if neg.any() else torch.zeros((), device=bce.device)

            loss_prior = w_pos * loss_pos + loss_neg
            losses["loss_prior"] = self.prior_loss_weight * loss_prior

        return losses



    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input images. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        
        x, valid_preds, fusion_debug = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        bbox_list = self.bbox_head.predict(x, batch_data_samples, valid_preds=valid_preds)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  bbox_list)
        
        return predictions



    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type == 'DEPTH':
            x += (valid_preds, )
        results = self.bbox_head.forward(x)
        return results

    def convert_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: OptInstanceList = None,
        data_instances_2d: OptInstanceList = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples
    
