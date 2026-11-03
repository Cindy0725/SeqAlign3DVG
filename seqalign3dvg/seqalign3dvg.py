# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from mmengine.structures import InstanceData

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptInstanceList
from .feat_ext import dinov2_vits14_lc_dpt
from torch.nn import functional as F
from torch import nn

import clip
from .plvf import ProgressiveTextVoxelFusion

from .rovm import TemporalLanguageGuidedVoxelMemory


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
    extrinsics = torch.tensor(img_meta['lidar2cam']) # n_imagesx4x4

    ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0][0] / stride)
    ratio2 = img_meta['ori_shape'][1] / (img_meta['img_shape'][0][1] / stride)
    intrinsic[0] /= ratio2
    intrinsic[1] /= ratio
    for i in range(extrinsics.shape[0]):
        extrinsic = extrinsics[i]
        projection.append(intrinsic.float() @ extrinsic[:3].float())

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
class SeqAlign3DVG_MultiView(Base3DDetector):
    """`SeqAlign3DVG Multi View`_.

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
                 temporal_memory_order: str = 'relevance',
                 temporal_random_order_seed: int = -1,
                 temporal_view_scorer_ablation: str = 'none',
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
        self.temporal_memory_order = temporal_memory_order
        self.temporal_random_order_seed = int(temporal_random_order_seed)
        self.temporal_view_scorer_ablation = temporal_view_scorer_ablation

        self.use_temporal_view_module = True

        self.temporal_view_loss_weight = 0.05

        self.temporal_view_module = TemporalLanguageGuidedVoxelMemory(
            voxel_dim=voxel_dim,
            text_dim=self.text_dim,
            hidden_dim=128,
            top_m=5,                # 关键：先不要再接近“全选”
            score_temperature=1.0,  # 比 0.7 更平一点
            sim_normalize=True,
            prior_gamma=1.5,
            memory_update_order=self.temporal_memory_order,
            random_order_seed=self.temporal_random_order_seed,
            view_scorer_ablation=self.temporal_view_scorer_ablation,
        )


        self.fusion_level = 2

        self.text_voxel_fuser = ProgressiveTextVoxelFusion(
            voxel_dim=voxel_dim,
            text_dim=self.text_dim,
            n_voxels=tuple(self.n_voxels),
            voxel_size=tuple(self.voxel_size),
            fusion_level=self.fusion_level,
            baseline_method="concat",
            sim_normalize=True,
            use_baseline=True,

        )


        self.prior_loss_weight = 0.25

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

    def set_temporal_ablation(
        self,
        memory_order: str = None,
        view_scorer_ablation: str = None,
        random_order_seed: int = None,
    ) -> None:
        """Runtime helper for quick eval/debug sweeps.

        For fair rebuttal ablations, set the same options in the config and
        retrain the corresponding variant. This helper is mainly convenient for
        fixed-checkpoint sanity checks.
        """
        if memory_order is not None:
            self.temporal_memory_order = memory_order
            if hasattr(self.temporal_view_module, "_normalize_memory_update_order"):
                self.temporal_view_module.memory_update_order = (
                    self.temporal_view_module._normalize_memory_update_order(memory_order)
                )
            else:
                self.temporal_view_module.memory_update_order = memory_order

        if view_scorer_ablation is not None:
            self.temporal_view_scorer_ablation = view_scorer_ablation
            if hasattr(self.temporal_view_module, "_normalize_view_scorer_ablation"):
                self.temporal_view_module.view_scorer_ablation = (
                    self.temporal_view_module._normalize_view_scorer_ablation(view_scorer_ablation)
                )
            else:
                self.temporal_view_module.view_scorer_ablation = view_scorer_ablation

        if random_order_seed is not None:
            self.temporal_random_order_seed = int(random_order_seed)
            self.temporal_view_module.random_order_seed = int(random_order_seed)


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


    @torch.no_grad()
    def _build_gt_view_visibility(self, batch_data_samples, batch_img_metas) -> torch.Tensor:
        """
        几何弱监督：
        若 GT target center 投影到该 view 的图像范围内且 z>0，则该 view label=1，否则为0
        返回: (B, V)
        """
        device = next(self.parameters()).device
        dtype = torch.float32

        labels = []
        for data_sample, img_meta in zip(batch_data_samples, batch_img_metas):
            extrinsics = torch.as_tensor(img_meta['lidar2cam'], device=device, dtype=dtype)   # (V,4,4)
            intrinsic = torch.as_tensor(img_meta['cam2img'][:3, :3], device=device, dtype=dtype)  # (3,3)

            V = extrinsics.shape[0]
            label = torch.zeros(V, device=device, dtype=dtype)

            gt_instance = data_sample.gt_instances_3d
            if gt_instance is None or len(gt_instance) == 0:
                labels.append(label)
                continue

            center = gt_instance.bboxes_3d.gravity_center[0, :3].to(device=device, dtype=dtype)  # (3,)
            center_h = torch.cat([center, center.new_tensor([1.0])], dim=0)  # (4,)

            # lidar/world -> camera
            cam_xyz = torch.einsum('vij,j->vi', extrinsics[:, :3, :4], center_h)  # (V,3)
            z = cam_xyz[:, 2]
            z_safe = z.clamp(min=1e-6)

            # camera -> image
            uvw = torch.einsum('ij,vj->vi', intrinsic, cam_xyz)  # (V,3)
            x = uvw[:, 0] / z_safe
            y = uvw[:, 1] / z_safe

            img_h, img_w = img_meta['ori_shape'][:2]
            visible = (z > 0) & (x >= 0) & (x < img_w) & (y >= 0) & (y < img_h)

            label[visible] = 1.0
            labels.append(label)

        return torch.stack(labels, dim=0)  # (B,V)

    @torch.no_grad()
    def _build_gt_view_coverage(
        self,
        batch_data_samples,
        batch_img_metas,
        view_valids: torch.Tensor,   # (B,V,1,X,Y,Z)
    ) -> torch.Tensor:
        """
        soft label:
        每个 view 覆盖 GT target voxel 的比例，返回 (B,V), range in [0,1]
        """
        device = view_valids.device
        B, V, _, X, Y, Z = view_valids.shape

        ones_valid = torch.ones((B, 1, X, Y, Z), device=device, dtype=torch.float32)
        gt_occ = self._build_gt_target_occ(
            batch_data_samples=batch_data_samples,
            batch_img_metas=batch_img_metas,
            valid_mask=ones_valid,
        )  # (B,1,X,Y,Z)

        inter = (view_valids.float() * gt_occ.unsqueeze(1)).sum(dim=(2, 3, 4, 5))  # (B,V)
        denom = gt_occ.sum(dim=(1, 2, 3, 4)).clamp(min=1.0)                         # (B,)

        coverage = inter / denom.unsqueeze(1)
        return coverage.clamp(0.0, 1.0)

    def extract_feat(self,
                    batch_inputs_dict: dict,
                    batch_data_samples: SampleList,
                    return_debug: bool = False):

        img = batch_inputs_dict['imgs']  # (B, V, 3, H, W)
        batch_size, n_views, _, H, W = img.size()

        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        descriptions = [img_meta['description'] for img_meta in batch_img_metas]

        # text encoder frozen
        with torch.no_grad():
            pooled, token_embeds, token_ids = self.text_encoder(descriptions, return_tokens=True)

        # 2D feature extraction
        img = img.reshape([-1] + list(img.shape)[2:])  # (B*V, 3, H, W)
        _, _, H, W = img.size()

        vision_features, _, _ = self.depth_est(img)
        stride = H / vision_features.shape[2]

        # reshape back to (B, V, C, h, w)
        vision_features = vision_features.reshape(
            [batch_size, n_views] + list(vision_features.shape[1:])
        )

        volumes, valids = [], []

        if return_debug:
            temporal_view_logits = []
            temporal_view_scores = []
            temporal_view_weights = []
            temporal_view_valids = []
            temporal_mem_confs = []

        for b, (feature, img_meta) in enumerate(zip(vision_features, batch_img_metas)):
            projection = _compute_projection(img_meta, stride).to(vision_features.device)
            points = get_points(
                n_voxels=torch.tensor(self.n_voxels),
                voxel_size=torch.tensor(self.voxel_size),
                origin=torch.tensor(img_meta['origin'])
            ).to(vision_features.device)

            height = int(img_meta['img_shape'][0][0] // stride)
            width = int(img_meta['img_shape'][0][1] // stride)

            view_volumes, view_valids = backproject(
                feature[:, :, :height, :width], points, projection
            )  # (V,C,X,Y,Z), (V,1,X,Y,Z)

            if self.use_temporal_view_module:
                if return_debug:
                    volume, valid, t_debug = self.temporal_view_module(
                        view_volumes=view_volumes,
                        view_valids=view_valids,
                        pooled_text=pooled[b:b+1],
                        token_embeds=token_embeds[b:b+1],
                        token_ids=token_ids[b:b+1],
                        return_debug=True,
                    )
                    temporal_view_logits.append(t_debug["view_logits"])
                    temporal_view_scores.append(t_debug["view_scores"])
                    temporal_view_weights.append(t_debug["view_weights"])
                    temporal_view_valids.append(view_valids.detach())
                    temporal_mem_confs.append(t_debug["mem_conf"])
                else:
                    volume, valid = self.temporal_view_module(
                        view_volumes=view_volumes,
                        view_valids=view_valids,
                        pooled_text=pooled[b:b+1],
                        token_embeds=token_embeds[b:b+1],
                        token_ids=token_ids[b:b+1],
                        return_debug=False,
                    )
            else:
                valid_cnt = view_valids.sum(dim=0).clamp(min=1.0)
                volume = view_volumes.sum(dim=0) / valid_cnt
                valid = view_valids.sum(dim=0) > 0
                volume[:, ~valid[0]] = 0.0

            volumes.append(volume)
            valids.append(valid)

        x = torch.stack(volumes, dim=0)   # (B,C,X,Y,Z)
        valids = torch.stack(valids, dim=0)  # (B,1,X,Y,Z)

        if return_debug:
            x_fused, fusion_debug = self.text_voxel_fuser(
                voxel_feat=x,
                pooled_text=pooled,
                token_embeds=token_embeds,
                token_ids=token_ids,
                valid_mask=valids,
                return_debug=True,
            )
        else:
            x_fused = self.text_voxel_fuser(
                voxel_feat=x,
                pooled_text=pooled,
                token_embeds=token_embeds,
                token_ids=token_ids,
                valid_mask=valids,
                return_debug=False,
            )
            fusion_debug = None

        x = self.neck_3d(x_fused)

        if return_debug:
            if fusion_debug is None:
                fusion_debug = {}

            if self.use_temporal_view_module and len(temporal_view_logits) > 0:
                fusion_debug["temporal_view_logits"] = torch.cat(temporal_view_logits, dim=0)   # (B,V)
                fusion_debug["temporal_view_scores"] = torch.cat(temporal_view_scores, dim=0)   # (B,V)
                fusion_debug["temporal_view_weights"] = torch.cat(temporal_view_weights, dim=0) # (B,V)
                fusion_debug["temporal_view_valids"] = torch.stack(temporal_view_valids, dim=0) # (B,V,1,X,Y,Z)
                fusion_debug["temporal_mem_conf"] = torch.cat(temporal_mem_confs, dim=0)        # (B,1,X,Y,Z)

            return x, valids, fusion_debug

        return x, valids, None


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
                                           batch_data_samples, return_debug=True)

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

        if (
            fusion_debug is not None
            and ("temporal_view_logits" in fusion_debug)
            and ("temporal_view_valids" in fusion_debug)
            and (self.temporal_view_loss_weight > 0)
        ):
            gt_view = self._build_gt_view_coverage(
                batch_data_samples=batch_data_samples,
                batch_img_metas=batch_img_metas,
                view_valids=fusion_debug["temporal_view_valids"].to(valid_preds.device),
            ).to(fusion_debug["temporal_view_logits"].device)   # (B,V), soft label in [0,1]

            pred_view_logits = fusion_debug["temporal_view_logits"].float()  # (B,V)

            # soft BCE
            bce = F.binary_cross_entropy_with_logits(pred_view_logits, gt_view, reduction="none")

            # 让覆盖率高的 view 权重大一点
            weights = 1.0 + 2.0 * gt_view

            loss_temporal_view = (bce * weights).mean()
            losses["loss_temporal_view"] = self.temporal_view_loss_weight * loss_temporal_view

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
                                           batch_data_samples, return_debug=False)

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
    


@torch.no_grad()
def _flatten_valid(scores, labels, valid_mask):
    """
    scores: (B,1,X,Y,Z) float
    labels: (B,1,X,Y,Z) float/bool
    valid_mask: (B,1,X,Y,Z) bool/float
    return: 1D tensors (scores_1d, labels_1d) only on valid voxels
    """
    if valid_mask is None:
        s = scores.reshape(-1)
        y = labels.reshape(-1)
        return s, y

    v = valid_mask.reshape(-1)
    if v.dtype != torch.bool:
        v = v > 0.5
    s = scores.reshape(-1)[v]
    y = labels.reshape(-1)[v]
    return s, y

@torch.no_grad()
def soft_iou(scores, labels, valid_mask=None, eps=1e-6):
    """
    Continuous IoU:
      intersection = sum(p * g)
      union        = sum(p + g - p*g)
    scores in [0,1], labels in {0,1}
    """
    s, y = _flatten_valid(scores, labels, valid_mask)
    s = s.float()
    y = y.float()

    inter = (s * y).sum()
    union = (s + y - s * y).sum()
    return (inter / (union + eps)).item()

@torch.no_grad()
def hard_iou(scores, labels, valid_mask=None, thr=0.5, eps=1e-6):
    """
    Hard IoU after thresholding scores>=thr
    """
    s, y = _flatten_valid(scores, labels, valid_mask)
    pred = (s >= thr)
    gt   = (y >= 0.5)

    inter = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    return (inter / (union + eps)).item()

@torch.no_grad()
def auroc_fast(scores, labels, valid_mask=None):
    """
    ROC AUC (binary) using sorting + trapezoidal rule.
    Returns NaN if only one class present.
    """
    s, y = _flatten_valid(scores, labels, valid_mask)
    s = s.float()
    y = (y >= 0.5).to(torch.float32)

    pos = y.sum()
    neg = (1.0 - y).sum()
    if pos.item() < 1 or neg.item() < 1:
        return float("nan")

    # sort by score desc
    order = torch.argsort(s, descending=True)
    y_sorted = y[order]

    tps = torch.cumsum(y_sorted, dim=0)
    fps = torch.cumsum(1.0 - y_sorted, dim=0)

    tpr = tps / pos
    fpr = fps / neg

    # Ensure starts at (0,0)
    tpr = torch.cat([torch.zeros(1, device=tpr.device), tpr])
    fpr = torch.cat([torch.zeros(1, device=fpr.device), fpr])

    auc = torch.trapz(tpr, fpr).item()
    return auc