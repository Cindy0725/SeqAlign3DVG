import torch
from torch import nn
from mmcv.cnn import Scale
from mmengine.model import BaseModule, normal_init, bias_init_with_prob
from mmdet.models.utils import multi_apply
from mmdet3d.registry import MODELS
from mmdet3d.structures import DepthInstance3DBoxes
from .seqalign3dvg import get_points
import torch.nn.functional as F

@MODELS.register_module()
class SingleObjectGroundingHead(BaseModule):
    def __init__(self,
                 n_channels=128,
                 n_scales=3,
                 voxel_size=[0.16, 0.16, 0.16], # 假设 Base Voxel Size
                 score_loss=dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0),
                 bbox_loss=dict(type='mmdet.IoULoss', loss_weight=1.0), # 轴对齐框可以直接用 2D IoU 扩展或 3D IoU
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.voxel_size = voxel_size
        self.n_scales = n_scales
        
        # Loss
        self.score_loss = MODELS.build(score_loss)
        self.bbox_loss = MODELS.build(bbox_loss)
        
        # Layers
        # 1. Score Head: 输出 1 个通道 (置信度)
        self.conv_score = nn.Conv3d(n_channels, 1, 3, padding=1)
        # 2. Reg Head: 输出 6 个通道 (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max)
        self.conv_reg = nn.Conv3d(n_channels, 6, 3, padding=1, bias=False)
        
        # Learnable Scales: 每一层有一个可学习的缩放系数，帮助回归收敛
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(n_scales)])

    def init_weights(self):
        # 初始化：分类头 bias 设低一点，防止训练初期 loss 爆炸
        normal_init(self.conv_score, std=0.01, bias=bias_init_with_prob(0.01))
        normal_init(self.conv_reg, std=0.01)

    def forward(self, x):
        """
        x: list of 3 tensors
           0: (B, 128, 40, 40, 16)
           1: (B, 128, 20, 20, 8)
           2: (B, 128, 10, 10, 4)
        """
        return multi_apply(self.forward_single, x, self.scales)

    def forward_single(self, x, scale):
        # x: (B, C, D, H, W) -> 注意 ImVoxelNet 的顺序通常是 (B, C, Z, Y, X) 或类似
        # 假设你的输入已经是 (B, C, D, H, W)
        
        # 1. Score Prediction
        score = self.conv_score(x) # (B, 1, D, H, W)
        
        # 2. Regression Prediction
        reg = self.conv_reg(x)     # (B, 6, D, H, W)
        # 使用 exp(scale(x)) 确保距离非负
        reg = torch.exp(scale(reg))
        
        return score, reg

    # ==================== Loss 计算流程 ====================
    
    def loss(self, x, batch_data_samples, valid_preds=None):
        """
        核心 Loss 计算逻辑
        """
        # 1. 前向传播
        scores_pred, regs_pred = self(x)
        
        batch_gt_instances_3d = [data.gt_instances_3d for data in batch_data_samples]
        batch_input_metas = [data.metainfo for data in batch_data_samples]
        
        # 2. 生成 Targets 并计算 Loss
        return self.loss_by_feat(scores_pred, regs_pred, batch_gt_instances_3d, batch_input_metas, valid_preds)

    @staticmethod
    def distances_to_xyzxyz(points, reg):
        """
        points: (N, 3)  voxel center (x,y,z)
        reg:    (N, 6)  (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max)
        return: (N, 6)  (x1,y1,z1,x2,y2,z2)
        """
        dx_min, dx_max, dy_min, dy_max, dz_min, dz_max = reg.unbind(dim=1)

        x1 = points[:, 0] - dx_min
        x2 = points[:, 0] + dx_max
        y1 = points[:, 1] - dy_min
        y2 = points[:, 1] + dy_max
        z1 = points[:, 2] - dz_min
        z2 = points[:, 2] + dz_max

        return torch.stack([x1, y1, z1, x2, y2, z2], dim=-1)
    
    @staticmethod
    def box6_gravity_to_xyzxyz(box6):
        """
        box6: (..., 6) (x, y, z, w, l, h)
        """
        x, y, z, w, l, h = box6.unbind(dim=-1)
        x1 = x - w * 0.5
        x2 = x + w * 0.5
        y1 = y - l * 0.5
        y2 = y + l * 0.5
        z1 = z - h * 0.5
        z2 = z + h * 0.5
        return torch.stack([x1, y1, z1, x2, y2, z2], dim=-1)

    def loss_by_feat(self, scores_pred, regs_pred, batch_gt_instances_3d, batch_input_metas, valid_preds=None):
        device = scores_pred[0].device
        batch_size = len(batch_gt_instances_3d)
        
        loss_score_list = []
        loss_bbox_list = []

        # 遍历每一个 Scale (Level 0, 1, 2)
        for lvl in range(self.n_scales):
            score_pred = scores_pred[lvl] # (B, 1, D, H, W)
            reg_pred = regs_pred[lvl]     # (B, 6, D, H, W)
            
            feat_shape = score_pred.shape[-3:] # (D, H, W)

            # === 新增逻辑：处理 valid_preds ===
            if valid_preds is not None:
                # valid_preds 通常是 (B, 1, D_orig, H_orig, W_orig)
                # 需要插值到当前 Level 的分辨率 (Nearest 插值保持 mask 性质)
                valid_mask = F.interpolate(valid_preds.float(), size=feat_shape, mode='nearest')
                
                # Flatten 这里的 mask 以匹配 score_pred 的 reshape 方式
                # (B, 1, D, H, W) -> (B, D, H, W, 1) -> (B, N)
                valid_mask = valid_mask.permute(0, 2, 3, 4, 1).reshape(batch_size, -1).bool()
            else:
                valid_mask = torch.ones(batch_size, score_pred.shape[-3:].numel(), device=device).bool()
            
            # 生成当前 Level 每个 Voxel 的物理坐标 (x, y, z)
            # points: (N, 3), N = D*H*W
            # points = self.get_points(feat_shape, batch_input_metas[0]['origin'], device, lvl)
            points = torch.stack([
                self.get_points(feat_shape, batch_input_metas[b]['origin'], device, lvl)
                for b in range(batch_size)
            ], dim=0)   # (B, N, 3)

            N = points.shape[1]
            
            # Flatten predictions for easy indexing
            score_pred = score_pred.permute(0, 2, 3, 4, 1).reshape(batch_size, -1) # (B, N)
            reg_pred = reg_pred.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, 6)  # (B, N, 6)
            
            # 准备 Targets
            target_scores = torch.zeros_like(score_pred) # (B, N)
            # target_bboxes = torch.zeros((batch_size, points.shape[0], 6), device=device) # (B, N, 6)
            target_bboxes = torch.zeros((batch_size, N, 6), device=device)
            weights = torch.zeros_like(score_pred) # (B, N) 用于 bbox loss 的权重

            for b in range(batch_size):
                gt_instance = batch_gt_instances_3d[b]
                if len(gt_instance) == 0:
                    continue
                
                # 获取 GT Box (x, y, z, w, l, h) -> ScanNet 无 yaw
                gt_bbox = gt_instance.bboxes_3d.tensor[0, :6].to(device) # (6,)
                gt_bbox = torch.cat((gt_instance.bboxes_3d.gravity_center[0, :3], gt_instance.bboxes_3d.tensor[0, 3:6]), dim=0)
                gt_center = gt_bbox[:3]
                # gt_center = gt_instance.bboxes_3d.gravity_center[0, :]
                gt_dims = gt_bbox[3:6] # w, l, h
                # --- 核心逻辑：Target Assignment ---
                # 1. 找到距离 GT 中心最近的 Voxel (作为 Positive Sample)
                #    也可以选择 GT 内部的所有点，但在 Grounding 中，Top-1 策略下选中心点最稳健
                # dists = torch.norm(points - gt_center.unsqueeze(0), dim=1)
                dists = torch.norm(points[b] - gt_center.unsqueeze(0), dim=1)  # (N,)
                # min_dist, min_ind = torch.min(dists, dim=0)
                valid_inds = valid_mask[b].nonzero(as_tuple=False).squeeze(1)  # (Nv,)
                if valid_inds.numel() == 0:
                    continue  # 或者跳过该 sample / 该 level

                dists_valid = dists[valid_inds]
                min_ind = valid_inds[dists_valid.argmin()]
                
                # 2. 只有当 GT 确实落在这个 Scale 的感受野内，或者为了简化，
                #    我们通常把 GT 分配给所有层，或者根据大小分配。
                #    这里采用简化策略：每一层都监督，让网络自己学哪一层特征最好。
                
                # 生成 Gaussian Score Target (Heatmap)
                # radius = 2.0 # 示例半径
                min_dim = torch.min(gt_dims)
                radius = max(min_dim.item() / 2.0, 0.16)
                # target = exp(-dist^2 / (2 * sigma^2)) 
                target_scores[b] = torch.exp(-(dists**2) / (2 * (radius ** 2)))
                # 强制中心点为 1.0 (Strict Positive)
                # target_scores[b, min_ind] = 1.0 
                target_scores[b, min_ind] = 1.0
                
                # 3. BBox Target
                # 我们只训练靠近中心的点的回归
                # 设定一个阈值，比如 Heatmap > 0.3 的地方都计算回归 Loss: 得到的大概是1.55*radius范围内的点
                pos_mask = target_scores[b] > 0.3
                
                # gt_bbox is already transformed into (x, y, z, w, l, h) gravity format
                gt_xyzxyz = self.box6_gravity_to_xyzxyz(gt_bbox)            # (6,)
                target_bboxes[b] = gt_xyzxyz.unsqueeze(0).expand(points.shape[1], -1)

                weights[b] = pos_mask.float() * target_scores[b] # 权重可以是 score 本身
                
                # target_scores_b_max = target_scores[b].max()
                # target_scores_b_sum = target_scores[b].sum()
            
        
            # --- 计算 Loss ---
            # 1. 只有 valid 的区域才计算 Score Loss
            # 我们可以构建一个 loss_weight，将 invalid 区域设为 0
            loss_weight_score = valid_mask.float()
            
            # 2. BBox Loss 的权重也要过滤掉 invalid 区域 (虽然通常 target_scores 在 invalid 区域已经是 0 了，但双重保险更好)
            weights = weights * valid_mask.float() 
            # avg_factor = target_scores.sum()
            # avg_factor = max(avg_factor, 1.0) 

            score_pred_sigmoid = score_pred.sigmoid()
            score_pred_sigmoid = score_pred_sigmoid * valid_mask.float()
            target_scores = target_scores * valid_mask.float()

            loss_score = self.score_loss(
                    score_pred_sigmoid, 
                    target_scores,
                    # weight=valid_mask.float(),  # 传入 Mask
                    # avg_factor=batch_size # 传入归一化因子
                )
            loss_score = loss_score / batch_size
            loss_score_list.append(loss_score)

            # 2. BBox Loss
            # 只计算有权重的部分
            pos_inds = weights > 0
            # pos_inds_sum = pos_inds.sum()
            # weights_mean = weights[pos_inds].mean()
            # weights_sum = weights.sum()

            if pos_inds.any():

                pos_reg_pred = reg_pred[pos_inds]           # (Np, 6)
                
                # pos_points   = points.unsqueeze(0).expand(batch_size, -1, 3)[pos_inds]  # (Np, 3)
                pos_points = points[pos_inds]  # (Np, 3)
                pred_xyzxyz  = self.distances_to_xyzxyz(pos_points, pos_reg_pred)       # (Np, 6)

                gt_xyzxyz_pos = target_bboxes[pos_inds]     # (Np, 6)

                loss_bbox = self.bbox_loss(
                    pred_xyzxyz,
                    gt_xyzxyz_pos,
                    weight=weights[pos_inds],
                    avg_factor=weights.sum()
                )
                loss_bbox_list.append(loss_bbox)
                
            else:
                loss_bbox_list.append(reg_pred.sum() * 0)

        return dict(
            loss_score=sum(loss_score_list),
            loss_bbox=sum(loss_bbox_list)
        )

    # ==================== Inference 流程 ====================

    @staticmethod
    def xyzxyz_to_box6_gravity(xyzxyz):
        """
        xyzxyz: (N, 6) (x1,y1,z1,x2,y2,z2)
        return: (N, 6) (x,y,z,w,l,h)
        """
        x1, y1, z1, x2, y2, z2 = xyzxyz.unbind(dim=1)
        w = x2 - x1
        l = y2 - y1
        h = z2 - z1
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        z = (z1 + z2) * 0.5
        return torch.stack([x, y, z, w, l, h], dim=1)

    def predict(self, x, batch_data_samples, valid_preds=None):
        """
        不使用 NMS 直接选 Top-1
        """
        scores_pred, regs_pred = self(x)
        batch_input_metas = [data.metainfo for data in batch_data_samples]
        device = scores_pred[0].device
        batch_size = len(batch_input_metas)
        results = []
        # to be deleted:
        batch_gt_instances_3d = [data.gt_instances_3d for data in batch_data_samples]

        # 预计算所有层的 points
        # mlvl_points = [
        #     self.get_points(scores_pred[i].shape[-3:], batch_input_metas[0]['origin'], device, i)
        #     for i in range(self.n_scales)
        # ]
        mlvl_points = [
            torch.stack([
                self.get_points(scores_pred[lvl].shape[-3:], batch_input_metas[b]['origin'], device, lvl)
                for b in range(batch_size)
            ], dim=0)  # (B, N_lvl, 3)
            for lvl in range(self.n_scales)
        ]
        # mlvl_points: list of (N_i, 3), N_i = 25600, 3200, 400
        for b in range(batch_size):
            best_score = -1e9
            best_bbox = None
            
            # 遍历所有层，寻找全局最大 Score
            for lvl in range(self.n_scales):
                # 获取当前 level 的原始 score (1, 1, D, H, W)
                current_score_feat = scores_pred[lvl][b:b+1] # Keep dim for interpolate
                
                # === 新增逻辑：Mask 掉无效区域 ===
                if valid_preds is not None:
                    # 取出当前 batch 的 valid_mask (1, 1, D_orig, H_orig, W_orig)
                    curr_valid_mask = valid_preds[b:b+1].float()
                    # 插值到当前 level (1, 1, D, H, W)
                    curr_valid_mask = F.interpolate(curr_valid_mask, size=current_score_feat.shape[-3:], mode='nearest').bool()
                    
                    # 将无效区域的分数设为极小值，确保不会被选为 Top-1
                    # 注意：要在 sigmoid 之前还是之后操作取决于你的实现。
                    # 如果你在循环里做 sigmoid (如下面的代码)，建议在这里直接 mask 原始 logit
                    current_score_feat = current_score_feat.masked_fill(~curr_valid_mask, -1e9)
                # =================================


                # 接原来的 Flatten 逻辑
                score = current_score_feat.permute(0, 2, 3, 4, 1).reshape(-1).sigmoid() 
                # score = scores_pred[lvl][b].permute(1, 2, 3, 0).reshape(-1).sigmoid() # (N,)
                reg = regs_pred[lvl][b].permute(1, 2, 3, 0).reshape(-1, 6) # (N, 6)
                # points = mlvl_points[lvl]
                points = mlvl_points[lvl][b]
                
                max_score, max_ind = score.max(dim=0)
                
                if max_score > best_score:
                    best_score = max_score
                    # 解码最佳位置的框
                    best_point = points[max_ind:max_ind+1]
                    best_reg = reg[max_ind:max_ind+1]
                    best_bbox_xyzxyz = self.distances_to_xyzxyz(best_point, best_reg)
                    best_bbox = self.xyzxyz_to_box6_gravity(best_bbox_xyzxyz) 
                
                # TO BE DELETED. here we use the voxel center closest to the ground truth bounding box center as the best point. It is to debug whether the near-0 performance is caused by the wrong center point in the inference stage.
                # gt_bbox = batch_gt_instances_3d[b].bboxes_3d.tensor[0, :6].to(device) # (6,)
                # gt_bbox = torch.cat((batch_gt_instances_3d[b].bboxes_3d.gravity_center[0, :3], batch_gt_instances_3d[b].bboxes_3d.tensor[0, 3:6]), dim=0)
                # # best_point = points[(points - gt_bbox[:3]).norm(dim=1).argmin(): (points - gt_bbox[:3]).norm(dim=1).argmin() + 1]
                # best_bbox = gt_bbox.unsqueeze(0)

            # 封装结果
            res = list()
            # 转换成 DepthInstance3DBoxes (ScanNet 格式)
            # 注意: DepthInstance3DBoxes 需要 7 维 (最后一位是 yaw=0)
            final_box = torch.cat([best_bbox, torch.zeros((1, 1), device=device)], dim=1)
            
            instance_data = dict()
            instance_data['bboxes_3d'] = DepthInstance3DBoxes(final_box, box_dim=7, with_yaw=True, origin=(.5, .5, .5))
            instance_data['scores_3d'] = best_score.unsqueeze(0)
            instance_data['labels_3d'] = torch.zeros(1, dtype=torch.long, device=device)
            
            # 这里为了适配 mmdet3d 结构，通常返回 InstanceData 对象
            from mmengine.structures import InstanceData
            res_instance = InstanceData(**instance_data)
            results.append(res_instance)
            
        return results

    # ==================== 辅助函数 ====================

    @torch.no_grad()
    def get_points(self, feat_shape, origin, device, lvl):
        # 根据 Level 计算当前 Voxel Size
        curr_voxel_size = torch.tensor(self.voxel_size) * (2 ** lvl)
        return get_points(
            n_voxels=torch.tensor(feat_shape),
            voxel_size=curr_voxel_size,
            origin=torch.tensor(origin)
        ).reshape(3, -1).transpose(0, 1).to(device) # (N, 3)

    @staticmethod
    def bbox_pred_to_bbox_no_yaw(points, reg):
        """
        将预测的 6 个距离转换为 (cx, cy, cz, w, l, h)
        reg: (N, 6) -> dx_min, dx_max, dy_min, dy_max, dz_min, dz_max
        points: (N, 3) -> x, y, z
        """
        # 1. 尺寸 (Width, Length, Height)
        w = reg[:, 0] + reg[:, 1]
        l = reg[:, 2] + reg[:, 3]
        h = reg[:, 4] + reg[:, 5]
        
        # 2. 中心点坐标
        # cx = x + (dx_max - dx_min) / 2
        cx = points[:, 0] + (reg[:, 1] - reg[:, 0]) / 2
        cy = points[:, 1] + (reg[:, 3] - reg[:, 2]) / 2
        cz = points[:, 2] + (reg[:, 5] - reg[:, 4]) / 2
        
        return torch.stack([cx, cy, cz, w, l, h], dim=-1)
    


