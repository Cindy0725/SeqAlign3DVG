# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Union
import random
import mmcv
import mmengine
import numpy as np
from mmcv.transforms import LoadImageFromFile
from mmcv.transforms.base import BaseTransform
from mmdet.datasets.transforms import LoadAnnotations
from mmengine.fileio import get
import os.path as osp
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import get_box_type
from mmdet3d.structures.points import BasePoints, get_points_type


@TRANSFORMS.register_module()
class LoadSingleView3DVG(BaseTransform):
    """Load an image from a specific view for 3D Vision Grounding.
    
    Expected input dict keys: 
        - img_prefix (optional)
        - scene_id
        - cleaning_result -> view_used_for_check
    
    Added output dict keys:
        - img (np.ndarray): Loaded image.
        - img_shape (tuple): (h, w).
        - ori_shape (tuple): (h, w).
        - img_path (str): Path to the loaded image.
    """

    def __init__(self, 
                 data_root=None, 
                 img_folder='color', 
                 file_client_args=dict(backend='disk')):
        """
        Args:
            data_root (str, optional): 数据集的根目录。如果 info 里没有 img_prefix，会用到这个。
            img_folder (str): 存放图片的子文件夹名，比如 ScanNet 通常在 'color' 文件夹下。
            file_client_args (dict): 文件读取后端配置。
        """
        self.data_root = data_root
        self.img_folder = img_folder
        self.file_client_args = file_client_args.copy()

    def transform(self, results):
        """
        Args:
            results (dict): Result dict from loading pipeline.
        
        Returns:
            dict: The dict with loaded image and meta information.
        """
        
        # 1. 提取路径构建所需的信息
        # 优先使用 results 里的 img_prefix，如果没有则使用 init 传入的 data_root
        img_prefix = results.get('img_prefix', self.data_root)
        scene_id = results['scene_id']
        img_path = img_prefix + results['img_path']
        img_byte = get(img_path, backend_args=None)
        img =  mmcv.imfrombytes(img_byte, flag='unchanged')

        # 4. 更新 results 字典 
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['filename'] = img_path
        results['lidar2cam'] = results.get('lidar2cam', np.eye(4))
        results['cam2img'] = results.get('cam2img', np.eye(4)[:3, :4])  
        origin = np.array([.0, .0, .5])
        results['origin'] = origin
        results['description'] = results.get('description', '')
        
        # optional: 记录一些必要的元数据，方便后续 transform 使用 (如 Resize, Normalize)
        # 'pad_shape', 'scale_factor' 等通常在后续 transform 初始化
        
        return results



@TRANSFORMS.register_module()
class RandomOriginShift(BaseTransform):
    """随机平移点云的原点位置，通常用于数据增强。

    Args:
        mean_shift (tuple): 平移的均值，默认为 (0.0, 0.0, 0.0)。
        std_shift (tuple): 平移的标准差，默认为 (0.7, 0.7, 0.0)。
    """

    def __init__(self, 
                 mean_shift=(0.0, 0.0, 0.0), 
                 std_shift=(0.7, 0.7, 0.0)):
        self.mean_shift = np.array(mean_shift)
        self.std_shift = np.array(std_shift)

    def transform(self, results):
        """
        Args:
            results (dict): Result dict from loading pipeline.
        
        Returns:
            dict: The dict with shifted origin.
        """
        shift = np.random.normal(.0, (.7, .7, .0), 3)
        results['origin'] = results['origin'] + shift 
        return results
    


@TRANSFORMS.register_module()
class LoadMultiView3DVG(BaseTransform):
    """Load an image from a specific view for 3D Vision Grounding.
    
    Expected input dict keys: 
        - img_prefix (optional)
        - scene_id
        - cleaning_result -> view_used_for_check
    
    Added output dict keys:
        - img (np.ndarray): Loaded image.
        - img_shape (tuple): (h, w).
        - ori_shape (tuple): (h, w).
        - img_path (str): Path to the loaded image.
    """

    def __init__(self, 
                 data_root=None, 
                 img_folder='color', 
                 file_client_args=dict(backend='disk')):
        """
        Args:
            data_root (str, optional): 数据集的根目录。如果 info 里没有 img_prefix，会用到这个。
            img_folder (str): 存放图片的子文件夹名，比如 ScanNet 通常在 'color' 文件夹下。
            file_client_args (dict): 文件读取后端配置。
        """
        self.data_root = data_root
        self.img_folder = img_folder
        self.file_client_args = file_client_args.copy()

    def transform(self, results):
        """
        Args:
            results (dict): Result dict from loading pipeline.
        
        Returns:
            dict: The dict with loaded image and meta information.
        """
        
        # 1. 提取路径构建所需的信息
        # 优先使用 results 里的 img_prefix，如果没有则使用 init 传入的 data_root
        img_prefix = results.get('img_prefix', self.data_root)
        scene_id = results['scene_id']
        imgs = []
        for i in range(len(results['img_path'])):
            img_path = img_prefix + results['img_path'][i]
            img_byte = get(img_path, backend_args=None)
            img =  mmcv.imfrombytes(img_byte, flag='unchanged')
            imgs.append(img)
        
        img = np.stack(imgs, axis=-1)
        img = img.astype(np.float32)

        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        # 4. 更新 results 字典 
        # results['img'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        results['ori_shape'] = imgs[0].shape[:2]
        results['filename'] = results['img_path']
        results['lidar2cam'] = results.get('lidar2cam', np.eye(4))
        results['cam2img'] = results.get('cam2img', np.eye(4)[:3, :4])  
        origin = np.array([.0, .0, .5])
        results['origin'] = origin
        results['description'] = results.get('description', '')

        
        # optional: 记录一些必要的元数据，方便后续 transform 使用 (如 Resize, Normalize)
        # 'pad_shape', 'scale_factor' 等通常在后续 transform 初始化
        results['ori_lidar2cam'] = results['ori_lidar2cam']
        results['axis_align_matrixs'] = results['axis_align_matrixs']
        return results
