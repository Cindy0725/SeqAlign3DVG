_base_ = [
    '../../../configs/_base_/schedules/mmdet-schedule-1x.py',
    '../../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'projects.SeqAlign3DVG.seqalign3dvg.seqalign3dvg_singleview',
        'projects.SeqAlign3DVG.seqalign3dvg.metric',
        'projects.SeqAlign3DVG.seqalign3dvg.loading',
        'projects.SeqAlign3DVG.seqalign3dvg.processing',
        'projects.SeqAlign3DVG.seqalign3dvg.formating',
        'projects.SeqAlign3DVG.seqalign3dvg.singleview_dataset',
        'projects.SeqAlign3DVG.seqalign3dvg.single_object_grounding_head'
    ],
    allow_failed_imports=False
)


prior_generator = dict(
    type='AlignedAnchor3DRangeGenerator',
    ranges=[[-3.2, -0.2, -2.28, 3.2, 6.2, 0.28]],
    rotations=[.0])
model = dict(
    type='SeqAlign3DVG_SingleView',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=14),
    neck_3d=dict(
        type='IndoorImVoxelNeck',
        in_channels=256,
        out_channels=128,
        n_blocks=[1, 1, 1]),
    bbox_head=dict(
        type='SingleObjectGroundingHead',
        score_loss=dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0, reduction = 'sum'),
        bbox_loss=dict(type='AxisAlignedIoULoss', loss_weight=1.0),
        n_channels=128,
        n_scales=3
        ),
    prior_generator=prior_generator,
    n_voxels=[40, 40, 16],
    voxel_size = [0.16, 0.16, 0.16],
    coord_type='DEPTH',
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.25, score_thr=.01))

dataset_type = 'SeqAlign3DVG_Dataset_SingleView'
data_root = 'data/SeqAlign3DVG/'

backend_args = None


class_names = [
    'dummy_class', 'wall', 'floor', 'cabinet', 'bed', 'chair', 
    'sofa', 'table', 'door', 'window', 'bookshelf', 
    'picture', 'counter', 'blinds', 'desk', 'shelves', 
    'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 
    'clothes', 'ceiling', 'books', 'refrigerator', 'television', 
    'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 
    'person', 'nightstand', 'toilet', 'sink', 'lamp', 
    'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop'
]


metainfo = dict(CLASSES=class_names)


train_pipeline = [
    dict(type='LoadAnnotations3D', backend_args=backend_args),
    dict(type='LoadSingleView3DVG'),
    dict(type='RandomOriginShift'),
    dict(type='SeqAlign3DVG_Resize', scale=(560, 364), keep_ratio=False),
    dict(type='SeqAlign3DVG_Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadAnnotations3D', backend_args=backend_args),
    dict(type='LoadSingleView3DVG'),
    dict(type='SeqAlign3DVG_Resize', scale=(560, 364), keep_ratio=False),
    dict(type='SeqAlign3DVG_Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
    # for the purpose of statistics analysis, we keep gt_bboxes_3d and gt_labels_3d in the test pipeline, but they are not used during inference.
]

train_dataloader = dict(
    batch_size=24,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='scannet_3dvg_trainval_singleview_train.pkl',
            pipeline=train_pipeline,
            test_mode=False,
            metainfo=metainfo)))

val_dataloader = dict(
    batch_size=24,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_3dvg_trainval_singleview_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=metainfo))
test_dataloader = val_dataloader


val_evaluator = dict(
    type='GroundingMetric',
    ann_file=data_root + 'scannet_3dvg_trainval_singleview_val.pkl'
    )

test_evaluator = val_evaluator


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=5e-5, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
    clip_grad=dict(max_norm=35., norm_type=2))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# seed
randomness = dict(seed=126180671)
# hooks
default_hooks = dict(checkpoint=dict(type='CheckpointHook', max_keep_ckpts=2, save_best=['Acc@0.25', 'Acc@0.50'], rule="greater"))

# runtime
find_unused_parameters = True  # only 1 of 4 FPN outputs is used



