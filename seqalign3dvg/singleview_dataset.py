import numpy as np
import os
import mmengine
from mmdet3d.registry import DATASETS
from mmdet3d.structures import DepthInstance3DBoxes
from mmdet3d.datasets.det3d_dataset import Det3DDataset

@DATASETS.register_module()
class SeqAlign3DVG_Dataset_SingleView(Det3DDataset):
    METAINFO = {
        'classes':
        ('dummy_class', 'wall', 'floor', 'cabinet', 'bed', 'chair', 
        'sofa', 'table', 'door', 'window', 'bookshelf', 
        'picture', 'counter', 'blinds', 'desk', 'shelves', 
        'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 
        'clothes', 'ceiling', 'books', 'refrigerator', 'television', 
        'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 
        'person', 'nightstand', 'toilet', 'sink', 'lamp', 
        'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop')
    }
    def __init__(self, 
                 data_root, 
                 ann_file, 
                 metainfo=None, 
                 pipeline=None, 
                 **kwargs):
        self.data_root = data_root
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            pipeline=pipeline,
            **kwargs)

    def _load_matrix_from_txt(self, path):
        """辅助函数：从txt读取4x4矩阵"""
        try:
            matrix = np.loadtxt(path)
            return matrix
        except Exception as e:
            print(f"Error loading matrix from {path}: {e}")
            return np.eye(4)

    def load_data_list(self):
        """
        重写加载逻辑：读取 List -> 立即解析所有数据 (Eager Loading)
        """
        # 1. 使用 mmengine 加载文件
        annotations = mmengine.load(self.ann_file)

        raw_data_list = []
        
        # 2. 标准化读取逻辑
        if isinstance(annotations, list):
            raw_data_list = annotations
        elif isinstance(annotations, dict) and 'data_list' in annotations:
            if 'metainfo' in annotations:
                self._metainfo.update(annotations['metainfo'])
            raw_data_list = annotations['data_list']
        else:
            raise TypeError(f'Annotation file should be a list or a dict containing "data_list"')

        # ---------------------------------------------------------
        # [关键修改] 3. 在这里直接运行 parse_data_info
        # ---------------------------------------------------------
        print(f"Start parsing {len(raw_data_list)} items in load_data_list (Eager Loading)...")
        processed_data_list = []
        for raw_item in raw_data_list:
            # 主动调用解析，将 'scene_id' 等转换为 'lidar2cam', 'img_path' 等
            parsed_item = self.parse_data_info(raw_item)
            processed_data_list.append(parsed_item)
            
        return processed_data_list

    def parse_data_info(self, raw_data_info):
        """
        将数据转换为 Pipeline 标准格式
        """
        # ---------------------------------------------------------
        # [关键修改] 0. 防重入检查 (Idempotency Check)
        # ---------------------------------------------------------
        # 因为我们在 load_data_list 已经调用过一次，
        # MMEngine 的 __getitem__ 可能会再次调用这个函数。
        # 如果已经存在关键 key (如 lidar2cam)，说明已经处理过，直接返回。
        if 'lidar2cam' in raw_data_info and 'img_path' in raw_data_info:
            return raw_data_info

        data_info = raw_data_info.copy()

        # ---------------------------------------------------------
        # 1. 注入基础路径信息
        # ---------------------------------------------------------
        data_info['img_prefix'] = self.data_root
        
        scene_id = data_info['scene_id']
        view_id = data_info['view_used_for_check']
        
        img_filename = os.path.join('posed_images', scene_id, view_id + '.jpg')
        data_info['img_path'] = img_filename

        intrinsic_path = os.path.join(self.data_root, 'posed_images', scene_id, 'intrinsic.txt')
        pose_path = os.path.join(self.data_root, 'posed_images', scene_id, view_id + '.txt')
        
        # ---------------------------------------------------------
        # 2. 计算 cam2img 和 lidar2cam
        # ---------------------------------------------------------
        intrinsic_matrix = self._load_matrix_from_txt(intrinsic_path)
        data_info['cam2img'] = intrinsic_matrix.astype(np.float32)

        pose_raw = self._load_matrix_from_txt(pose_path)
        
        if 'axis_align_matrix' in data_info:
            axis_align_matrix = data_info['axis_align_matrix']
        else:
            axis_align_matrix = np.eye(4)

        extrinsic_aligned = axis_align_matrix @ pose_raw
        lidar2cam = np.linalg.inv(extrinsic_aligned)
        
        data_info['lidar2cam'] = lidar2cam.astype(np.float32)
        data_info['lidar2img'] = (intrinsic_matrix @ lidar2cam).astype(np.float32)

        ann_info = dict()
        # ---------------------------------------------------------
        # 3. 处理 Ground Truth Bounding Box
        # ---------------------------------------------------------
        if 'box3d' in data_info:
            box_6d = data_info['box3d']
            
            if not isinstance(box_6d, np.ndarray):
                box_6d = np.array(box_6d, dtype=np.float32)

            box_7d = np.concatenate([box_6d, np.zeros(1, dtype=box_6d.dtype)])
            box_7d = box_7d[np.newaxis, :] 

            gt_bboxes_3d = DepthInstance3DBoxes(
                box_7d, 
                box_dim=7, 
                with_yaw=True, 
                origin=(0.5, 0.5, 0.5)
            )
            ann_info['gt_bboxes_3d'] = gt_bboxes_3d
            
            # 删除原始 key
            del data_info['box3d']

        # ---------------------------------------------------------
        # 4. 处理 Labels
        # ---------------------------------------------------------
        if 'label_id' in data_info:
            ann_info['gt_labels_3d'] = np.array([data_info['label_id']], dtype=np.int64)

        data_info['ann_info'] = ann_info
        data_info['description'] = data_info.get('description', '')
        data_info['view_id'] = view_id

        return data_info
    

