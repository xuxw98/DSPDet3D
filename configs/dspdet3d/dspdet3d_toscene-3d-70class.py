voxel_size = .01
n_points = 100000


model = dict(
    type='DSPDet3D',
    voxel_size=voxel_size,
    backbone=dict(type='DSPBackbone', in_channels=3, max_channels=128, depth=34,  pool=False, norm='batch'),
    head=dict(
        type='DSPHead',
        in_channels=(64, 128, 128, 128),
        out_channels=128,
        n_reg_outs=6,
        n_classes=70,
        voxel_size=voxel_size,
        pts_prune_threshold=100000,
        assigner=dict(
            type='DSPAssigner',
            top_pts_threshold=6,
        ),
        assign_type='volume',
        volume_threshold=27,
        r=13,
        prune_threshold=0.5,
        bbox_loss=dict(type='AxisAlignedIoULoss2', mode='diou', reduction='none')),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))


optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=12)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

n_points = 100000
dataset_type = 'ScanNetDataset'  
data_root = '/path/to/.pkl/'  
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                            'window', 'bookshelf','picture', 'counter', 'desk', 'curtain',
                            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin',
                            "bag", "bottle", "bowl", "camera", "can",
                            "cap", "clock", "keyboard", "display", "earphone",
                            "jar", "knife", "lamp", "laptop", "microphone",
                            "microwave", "mug", "printer", "remote control", "phone",
                            "alarm", "book", "cake", "calculator", "candle",
                            "charger", "chessboard", "coffee_machine", "comb", "cutting_board",
                            "dishes", "doll", "eraser", "eye_glasses", "file_box",
                            "fork",  "fruit", "globe", "hat", "mirror",
                            "notebook", "pencil", "plant", "plate", "radio",
                            "ruler", "saucepan", "spoon", "tea_pot", "toaster",
                            "vase", "vegetables")


train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='LoadAnnotations3D'),
    dict(type='GlobalAlignment', rotation_axis=2),
    # we do not sample 100k points for scannet, as very few scenes have
    # significantly more then 100k points. so we sample 33 to 100% of them
    dict(type='PointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=.5,
        flip_ratio_bev_vertical=.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.02, .02],
        scale_ratio_range=[.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # we do not sample 100k points for scannet, as very few scenes have
            # significantly more then 100k points. so it doesn't affect inference
            # time and we ca accept all points
            # dict(type='PointSample', num_points=n_points),
            # dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'toscence_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'toscence_infos_val.pkl',   #scannet_infos_val.pkl
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'toscence_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
