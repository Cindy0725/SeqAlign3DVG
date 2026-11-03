import torch
import numpy as np
from typing import Dict, Optional, Sequence, Union
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
from mmdet3d.structures import BaseInstance3DBoxes
from typing import Dict, Optional, Sequence, Tuple, Any
from collections import defaultdict

def aabb_iou_3d_single(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Axis-aligned 3D IoU for single boxes.
    box format: (x, y, z, dx, dy, dz, [yaw]) -> we ignore yaw if exists
    Args:
        box1, box2: shape (6,) or (7,)
    Returns:
        iou: scalar tensor
    """
    b1 = box1[:6].float()
    b2 = box2[:6].float()

    c1, s1 = b1[:3], torch.clamp(b1[3:6], min=1e-6)
    c2, s2 = b2[:3], torch.clamp(b2[3:6], min=1e-6)

    min1 = c1 - s1 / 2.0
    max1 = c1 + s1 / 2.0
    min2 = c2 - s2 / 2.0
    max2 = c2 + s2 / 2.0

    inter_min = torch.maximum(min1, min2)
    inter_max = torch.minimum(max1, max2)
    inter = torch.clamp(inter_max - inter_min, min=0.0)
    inter_vol = inter[0] * inter[1] * inter[2]

    v1 = s1[0] * s1[1] * s1[2]
    v2 = s2[0] * s2[1] * s2[2]
    union = v1 + v2 - inter_vol
    return torch.where(union > 0, inter_vol / union, torch.zeros_like(union))

@METRICS.register_module()
class GroundingMetric(BaseMetric):
    """3D Visual Grounding Evaluation Metric (Acc@0.25, Acc@0.5).

    Calculates the IoU between the top-1 scored predicted box and the 
    ground truth target box.

    Args:
        iou_thrs (list[float]): IoU thresholds for accuracy calculation. 
            Defaults to [0.25, 0.5].
        collect_device (str): Device name used for collecting results.
            Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names. Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 iou_thrs: Sequence[float] = [0.25, 0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(prefix=prefix, collect_device=collect_device)
        self.iou_thrs = iou_thrs
        self.ann_file = ann_file
        self.groupby_keys = list(('anchor_bin', 'difficulty_bin', 'relation_bin'))

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # --- 1. 获取预测结果 ---
            if 'pred_instances_3d' not in data_sample:
                continue
            tags = {k: data_sample.get(k, 'unknown') for k in self.groupby_keys}   
            pred_3d = data_sample['pred_instances_3d']
            # 确保有分数和框
            if len(pred_3d['scores_3d']) == 0:
                # 如果没有预测框，存一个 None，计算时视为错误
                self.results.append((None, None))
                continue
            pred_box = pred_3d['bboxes_3d']
            if hasattr(pred_box, 'to'):
                pred_box = pred_box.to('cpu')

            # --- 3. 获取 Ground Truth ---
            
            gt_instances = data_sample['gt_instances_3d']
            if gt_instances is None or len(gt_instances['bboxes_3d']) == 0:
                self.results.append((None, None))
                continue
            
            gt_box = gt_instances['bboxes_3d'] 
            
            if hasattr(gt_box, 'to'):
                gt_box = gt_box.to('cpu')
            # pred_box.tensor和gt_box.tensor都是[1,7]的tensor形式
            iou_matrix = pred_box.overlaps(pred_box, gt_box) 
            iou = iou_matrix[0, 0].item()
            self.results.append(dict(iou=iou, tags=tags))


    def compute_metrics(self, results: list) -> Dict[str, float]:
        total = len(results)
        metrics: Dict[str, float] = {}

        # overall
        for thr in self.iou_thrs:
            hit = 0
            for r in results:
                iou = r['iou']
                if iou is not None and iou >= thr:
                    hit += 1
            metrics[f'Acc@{thr:.2f}'] = hit / total if total > 0 else 0.0

        # groupby
        # group_stats[(key, value)]['total'], group_stats[(key, value)]['hit@thr']
        group_stats: Dict[Tuple[str, Any], dict] = defaultdict(lambda: defaultdict(int))
        for r in results:
            for k, v in r['tags'].items():
                group_stats[(k, v)]['total'] += 1
                iou = r['iou']
                for thr in self.iou_thrs:
                    if iou is not None and iou >= thr:
                        group_stats[(k, v)][f'hit@{thr:.2f}'] += 1

        def _safe(x):
            return str(x).replace(' ', '_').replace('/', '_')

        for (k, v), stat in group_stats.items():
            metrics[f'Count_{k}={_safe(v)}'] = float(stat['total'])
            for thr in self.iou_thrs:
                denom = stat['total']
                num = stat[f'hit@{thr:.2f}']
                metrics[f'Acc@{thr:.2f}_{k}={_safe(v)}'] = (num / denom) if denom > 0 else 0.0

        return metrics