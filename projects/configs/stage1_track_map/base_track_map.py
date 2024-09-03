_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py"]

# Update-2023-06-12: 
# [Enhance] Update some freezing args of UniAD 
# [Bugfix] Reproduce the from-scratch results of stage1
# 1. Remove loss_past_traj in stage1 training
# 2. Unfreeze neck and BN
# --> Reproduced tracking result: AMOTA 0.393

#DownScale: #_
#Hyperparameter Edit1 Value

# Unfreeze neck and BN, the from-scratch results of stage1 could be reproduced
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
) 
_dim_ = 128 #_ 초기값:256 // Transformer의 임베딩 차원 크기, 모델이 입력 데이터에서 학습하는 특징 벡터의 크기
_pos_dim_ = _dim_ // 2 #_ 임베딩 차원의 절반 크기, 위치 임베딩의 차원 크기
_ffn_dim_ = _dim_ * 2 #_ Feed Forward Network의 차원 크기, Transformer의 피드포워드 네트워크 차원
_num_levels_ = 4 #_ 다중 스케일 특징 레벨의 수, 다양한 해상도의 특징을 학습하기 위한 레벨 수
bev_h_ = 50 #_ 초기값: 200 수정log:50 // Bird's Eye View(BEV) 이미지의 높이
bev_w_ = 50 #_ 초기값: 200 수정log:50 // Bird's Eye View(BEV) 이미지의 너비
_feed_dim_ = _ffn_dim_ #_ Feed Forward Network의 차원 크기, 피드포워드 네트워크 차원과 동일
_dim_half_ = _pos_dim_ #_ 임베딩 차원의 절반 크기, 위치 임베딩의 차원과 동일
canvas_size = (bev_h_, bev_w_) #_ 캔버스 크기 설정, BEV 이미지의 크기 설정

# NOTE: You can change queue_length from 5 to 3 to save GPU memory, but at risk of performance drop.
queue_length = 3 #_5  # each sequence contains `queue_length` frames.
                      # 각 시퀀스가 포함하는 프레임 수, 시퀀스 길이 조정

### traj prediction args ###
predict_steps = 12 #_ 모델이 미래를 예측하는 단계 수
predict_modes = 6 #_ 예측 모드 수
fut_steps = 4 #_ 모델이 예측하는 미래 단계 수
past_steps = 4 #_ 모델이 사용하는 과거 단계 수
use_nonlinear_optimizer = True #_ 비선형 최적화 사용 여부

## occflow setting	
occ_n_future = 4 #_ Occupancy Flow에서 사용할 미래 스텝 수, 차선 점유 예측에 사용되는 미래 스텝 수
occ_n_future_plan = 6 #_ 계획 단계에서 사용할 미래 스텝 수, 계획 시 사용되는 미래 스텝 수
occ_n_future_max = max([occ_n_future, occ_n_future_plan])#_ 최대 미래 스텝 수, 두 미래 스텝 수 중 큰 값

### planning ###
planning_steps = 6 #_ planning_steps
use_col_optim = True #_ 충돌 최적화 사용 여부, 충돌을 피하기 위한 최적화 사용

### Occ args ### 
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5], #_ x축 경계 설정, Occupancy Flow의 x축 경계 값
    'ybound': [-50.0, 50.0, 0.5], #_ y축 경계 설정, Occupancy Flow의 y축 경계 값
    'zbound': [-10.0, 10.0, 20.0], #_ z축 경계 설정, Occupancy Flow의 z축 경계 값
}

# Other settings
train_gt_iou_threshold=0.3 #_ 훈련 시 IOU 임계값, IOU가 이 값 이상일 때만 학습에 사용

model = dict(
    type="UniAD",
    gt_iou_threshold=train_gt_iou_threshold,
    queue_length=queue_length,
    use_grid_mask=True,
    video_test_mode=True,
    num_query=500, #_ 수정log:500 // DETR(Deformable DETR) 모델에서 사용하는 쿼리 수
    num_classes=10,
    pc_range=point_cloud_range,
    img_backbone=dict(
        type="ResNet",
        depth=50, #_ 초기값: 101 수정log:50 // ResNet 백본의 깊이, 레이어 수, ResNet 네트워크의 깊이
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        dcn=dict(
            type="DCNv2", deform_groups=1, fallback_on_stride=False
        ),  # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048], #_ 원본: 512, 1024, 2048 / 수정: 256, 512, 1024  #256 -> 128
        out_channels=_dim_,

        # RuntimeError: Given groups=1, weight of size [128, 256, 1, 1], 
        # expected input[6, 512, 116, 200] to have 256 channels, but got 512 channels instead

        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    freeze_img_backbone=True,
    freeze_img_neck=False,
    freeze_bn=False,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    qim_args=dict(
        qim_type="QIMBase", #Track #MOTR
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1,
    ),  # hyper-param for query dropping mentioned in MOTR
    mem_args=dict(
        memory_bank_type="MemoryBank", #Track #MOTR #QIM #TAN
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
    loss_cfg=dict(
        type="ClipMatcher",
        num_classes=10,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type="HungarianAssigner3DTrack",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
            pc_range=point_cloud_range,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_past_traj_weight=0.0,
    ),  # loss cfg for tracking
    pts_bbox_head=dict(
        type="BEVFormerTrackHead",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=500, #_ 수정log:500 // DETR(Deformable DETR) 모델에서 사용하는 쿼리 수
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        past_steps=past_steps,
        fut_steps=fut_steps,
        transformer=dict(
            type="PerceptionTransformer",
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type="BEVFormerEncoder",
                num_layers=4, #_ 초기값:6 // BEVFormerEncoder 인코더의 레이어(층) 수
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayer",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                        ),
                        dict(
                            type="SpatialCrossAttention",
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],


                    # ffn_cfgs=dict
                    #+ : 추가된 명령어
                    #_ _dim_에서 Transformer의 임베딩 차원 크기를 지정해 주었으나 각 모듈의 트랜스포머 레이어에서
                    #_ Feed Forward Network값이 재정의 됌

                    #_ 각 Transformer Layer에서 ffn_cfgs 설정의 embed_dims은 
                    #_ 해당 Feed Forward Network (FFN)가 사용하는 임베딩 차원의 크기를 의미

                    #_ 해당 설정 값 위치: custom_base_transformer_layer.py의 
                    #_                  class MyCustomBaseTransformerLayer(BaseModule)

                    #_ base_track_map.py에 ffn_cfgs 설정을 추가하여 에러 방지

                    ffn_cfgs=dict( #+
                        type='FFN', #+
                        embed_dims=128, #+
                        feedforward_channels=1024, #+
                        num_fcs=2, #+
                        ffn_drop=0., #+
                        act_cfg=dict(type='ReLU', inplace=True), #+
                    ), #+
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=4, #_ 초기값:6 // DetectionTransformerDecoder 디코더의 레이어(층) 수
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8, #_ 초기값:8 수정log:4 // PansegformerHead nhead=4와 맞춤 - 멀티헤드 어텐션 헤드 수
                                            #6으로 하면 embed_dim와 나뉘었을 때 떨어지지 않음
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttention",
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    ffn_cfgs=dict( #+
                        type='FFN', #+
                        embed_dims=128, #+
                        feedforward_channels=1024, #+
                        num_fcs=2, #+
                        ffn_drop=0., #+
                        act_cfg=dict(type='ReLU', inplace=True), #+
                    ), #+
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
        ),
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),
    seg_head=dict(
        type='PansegformerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        canvas_size=canvas_size,
        pc_range=point_cloud_range,
        num_query=150, #_ 수정log: 150 // Segmentation transformer에서 사용하는 쿼리 수
        num_classes=4,
        num_things_classes=3,
        num_stuff_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        with_box_refine=True,
        transformer=dict(
            type='SegDeformableTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=4, #_ 초기값:6 // DetrTransformerEncoder 인코더의 레이어(층) 수
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=_num_levels_,
                         ),
                    ffn_cfgs=dict( #+
                        type='FFN', #+
                        embed_dims=128, #+
                        feedforward_channels=1024, #+
                        num_fcs=2, #+
                        ffn_drop=0., #+
                        act_cfg=dict(type='ReLU', inplace=True), #+
                    ), #+
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=4, #_ 초기값:6 // DetrTransformerEncoder 인코더의 레이어(층) 수
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8, #_ 초기값:8 수정log:4 // 멀티헤드 어텐션 헤드 수
                                            #6으로 하면 embed_dim와 나뉘었을 때 떨어지지 않음
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=_num_levels_,
                        )
                    ],
                    ffn_cfgs=dict( #+
                        type='FFN', #+
                        embed_dims=128, #+
                        feedforward_channels=1024, #+
                        num_fcs=2, #+
                        ffn_drop=0., #+
                        act_cfg=dict(type='ReLU', inplace=True), #+
                    ), #+
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')
                ),
            ),
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_dim_half_,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(type='DiceLoss', loss_weight=2.0),
        thing_transformer_head=dict(type='SegMaskHead',d_model=_dim_,nhead=8,num_decoder_layers=4),
        stuff_transformer_head=dict(type='SegMaskHead',d_model=_dim_,nhead=8,num_decoder_layers=6,self_attn=True),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                ),
            assigner_with_mask=dict(
                type='HungarianAssigner_multi_info',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                mask_cost=dict(type='DiceCost', weight=2.0),
                ),
            sampler =dict(type='PseudoSampler'),
            sampler_with_mask =dict(type='PseudoSampler_segformer'),
        ),
    ),
 
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        )
    ),
)
dataset_type = "NuScenesE2EDataset"
data_root = "/home/hyun/local_storage/code/UniAD/data/nuscenes/" #_"data/nuscenes/" 
info_root = "/home/hyun/local_storage/code/UniAD/data/infos/" #_"data/infos/"
file_client_args = dict(backend="disk")
ann_file_train=info_root + f"nuscenes_infos_temporal_train.pkl"
ann_file_val=info_root + f"nuscenes_infos_temporal_val.pkl"
ann_file_test=info_root + f"nuscenes_infos_temporal_val.pkl"


train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True, file_client_args=file_client_args, img_root=data_root),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D_E2E",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,

        with_future_anns=True, # occ_flow gt
        with_ins_inds_3d=True, # ins_inds 
        ins_inds_add_1=True,   # ins_inds start from 1
    ),

    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
                                    filter_invisible=False),  # NOTE: Currently vis_token is not in pkl 

    dict(type="ObjectRangeFilterTrack", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilterTrack", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=[
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_inds",
            "img",
            "timestamp",
            "l2g_r_mat",
            "l2g_t",
            "gt_fut_traj",
            "gt_fut_traj_mask",
            "gt_past_traj",
            "gt_past_traj_mask",
            "gt_sdc_bbox",
            "gt_sdc_label",
            "gt_sdc_fut_traj",
            "gt_sdc_fut_traj_mask",
            "gt_lane_labels",
            "gt_lane_bboxes",
            "gt_lane_masks",
            # Occ gt
            "gt_segmentation",
            "gt_instance", 
            "gt_centerness", 
            "gt_offset", 
            "gt_flow",
            "gt_backward_flow",
            "gt_occ_has_invalid_frame",
            "gt_occ_img_is_valid",
            # gt future bbox for plan	
            "gt_future_boxes",	
            "gt_future_labels",	
            # planning	
            "sdc_planning",	
            "sdc_planning_mask",	
            "command",
        ],
    ),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True,
            file_client_args=file_client_args, img_root=data_root),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type='LoadAnnotations3D_E2E', 
         with_bbox_3d=False,
         with_label_3d=False, 
         with_attr_label=False,

         with_future_anns=True,
         with_ins_inds_3d=False,
         ins_inds_add_1=True, # ins_inds start from 1
         ),
    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
                                       filter_invisible=False),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(
                type="CustomCollect3D", keys=[
                                            "img",
                                            "timestamp",
                                            "l2g_r_mat",
                                            "l2g_t",
                                            "gt_lane_labels",
                                            "gt_lane_bboxes",
                                            "gt_lane_masks",
                                            "gt_segmentation",
                                            "gt_instance", 
                                            "gt_centerness", 
                                            "gt_offset", 
                                            "gt_flow",
                                            "gt_backward_flow",
                                            "gt_occ_has_invalid_frame",
                                            "gt_occ_img_is_valid",
                                             # planning	
                                            "sdc_planning",	
                                            "sdc_planning_mask",	
                                            "command",
                                        ]
            ),
        ],
    ),
]
data = dict(
    samples_per_gpu=1, #_ 초기값: 1 // GPU 당 샘플 수, batch size
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,

        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,

        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        eval_mod=['det', 'track', 'map'],

        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,
    ),
    test=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        occ_n_future=occ_n_future_max,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        eval_mod=['det', 'map', 'track'],
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)
optimizer = dict( 
    type="AdamW",
    lr=2e-4, #_ 초기값:2e-4, 수정값:1e-6, // Learning Rate
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) 
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 6 #_ 초기값:6 학습 에포크 수
evaluation = dict(interval=6, pipeline=test_pipeline)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
checkpoint_config = dict(interval=1)
load_from = "/home/hyun/local_storage/code/UniAD/ckpts/bevformer_r101_dcn_24ep.pth"
# resume_from = "/home/hyun/local_storage/code/UniAD/projects/work_dirs/stage1_track_map/base_track_map/latest.pth" 
#             #_ 마지막 학습 log 불러옴
find_unused_parameters = True