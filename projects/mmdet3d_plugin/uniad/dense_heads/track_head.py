#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# Modified from bevformer (https://github.com/fundamentalvision/BEVFormer)        #
#---------------------------------------------------------------------------------#

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.runner import force_fp32, auto_fp16

import matplotlib.pyplot as plt
import numpy as np


@HEADS.register_module()
class BEVFormerTrackHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self, # 필요한 매개변수와 변수 설정 # 부모 클래스 DETRHead 초기화
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 past_steps=4,
                 fut_steps=4,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine

        assert as_two_stage is False, 'as_two_stage is not supported yet.'
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        super(BEVFormerTrackHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

    def _init_layers(self): # 분류, 회귀, 궤적 예측 브랜치 레이어 초기화
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        past_traj_reg_branch = []
        for _ in range(self.num_reg_fcs):
            past_traj_reg_branch.append(
                Linear(self.embed_dims, self.embed_dims))
            past_traj_reg_branch.append(nn.ReLU())
        past_traj_reg_branch.append(
            Linear(self.embed_dims, (self.past_steps + self.fut_steps)*2))
        past_traj_reg_branch = nn.Sequential(*past_traj_reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.past_traj_reg_branches = _get_clones(
                past_traj_reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.past_traj_reg_branches = nn.ModuleList(
                [past_traj_reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self): # DeformDETR transformer 가중치 초기화 # Loss에 사용할 바이어스 초기화
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)



    #TrackFormer에 들어가는 값 (Input): #BEV OUTPUT
    def get_bev_features(self, mlvl_feats, img_metas, prev_bev=None): ###### multi level Feature로부터 BEV Features 추출
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype # 리스트의 첫 번째 데이터 타입을 가져옴
        bev_queries = self.bev_embedding.weight.to(dtype) # bs:1, bev_h,w:200, embed_dims:128, 

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), 
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, 
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            img_metas=img_metas,
        )

        # # BEV feature map 형태 확인
        # print("bev_embed shape:", bev_embed.shape)  # bev_embed shape 확인

        # # BEV feature map 시각화
        # bev_embed_np = bev_embed[0].cpu().detach().numpy()  # GPU에 있는 텐서 -> CPU로 이동, numpy 배열로 변환
        # bev_embed_np = bev_embed_np.reshape(self.bev_h, self.bev_w, -1)  # feature map을 bev_h와 bev_w로 reshape, -1: 나머지 차원을 자동으로 맞춤

        # # 각 차원에 대해 최댓값, 합, 평균값 계산
        # bev_max = np.max(bev_embed_np, axis=2)
        # bev_sum = np.sum(bev_embed_np, axis=2)
        # bev_avg = np.mean(bev_embed_np, axis=2)

        # visualizations = {
        #     'max': bev_max,
        #     'sum': bev_sum,
        #     'avg': bev_avg
        # }

        # # 시각화
        # for key, value in visualizations.items():
        #     plt.figure(figsize=(10, 10))  # Figure 생성, 크기 10x10
            
        #     plt.imshow(value, cmap='viridis', interpolation='nearest')  # 값 시각화, 색상: 'viridis', 보간: 'nearest'
        #     plt.colorbar() 
        #     plt.title(f'BEV Feature Map - {key}')
        #     plt.xlabel('BEV Width') 
        #     plt.ylabel('BEV Height') 
        #     plt.savefig(f'/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/track_head-TrackFormer_Visualization/bev_features_{key}.png')  # 결과 저장
        #     plt.close()

        # # 여러 차원을 시각화하기 위한 반복문
        # num_dims = bev_embed_np.shape[-1]  # 차원 수 확인
        # dims_to_plot = [0] + list(range(4, num_dims, 5)) + [num_dims - 1]  # 0번째, 5번째마다, 마지막 차원 포함

        # for i in dims_to_plot:
        #     plt.figure(figsize=(10, 10))  # Figure 생성, 크기 10x10
            
        #     # i번째 차원 시각화
        #     plt.imshow(bev_embed_np[:, :, i], cmap='viridis', interpolation='nearest')  # i번째 차원 시각화, 색상: 'viridis', 보간: 'nearest'
        #     plt.colorbar() 
        #     plt.title(f'BEV Feature Map - dim {i}')
        #     plt.xlabel('BEV Width') 
        #     plt.ylabel('BEV Height') 
        #     plt.savefig(f'/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/track_head-TrackFormer_Visualization/bev_features_dim_{i}.png')  # 결과 저장
        #     plt.close()

        return bev_embed, bev_pos



    # TrackFormer에서 계산되는 과정 (Processing):
    def get_detections( ####### BEV Feature와 Object Query를 기반으로 객체 검출
        self, 
        bev_embed,
        object_query_embeds=None,
        ref_points=None,
        img_metas=None,
    ):
        assert bev_embed.shape[0] == self.bev_h * self.bev_w  # bev_embed의 첫 번째 차원이 bev_h * bev_w인지 확인 (입력 크기 검증), 
        print('bev_embed', bev_embed.shape) #torch.Size([40000, 1, 128])

        # Transformer에서 상태와 ref_point을 가져옴
        hs, init_reference, inter_references = self.transformer.get_states_and_refs(
            bev_embed,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            reference_points=ref_points,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )

        print('hs_B', hs.shape) #torch.Size([4, 901, 1, 128])
        #(num_layers, num_queries, batch_size, embed_dim)

        hs = hs.permute(0, 2, 1, 3)  # 차원 순서 변경. 형태 변환
        print('hs_A', hs.shape) #torch.Size([4, 1, 901, 128])
        #(num_layers, batch_size, num_queries, embed_dim)

        outputs_classes = []  # 클래스 예측 결과를 저장할 리스트 초기화
        outputs_coords = []   # 좌표 예측 결과를 저장할 리스트 초기화
        outputs_trajs = []    # 궤적 예측 결과를 저장할 리스트 초기화

        for lvl in range(hs.shape[0]):  # 각 레이어에 대해 반복
            if lvl == 0:
                reference = ref_points.sigmoid()  # 첫 번째 레이어에서는 ref_point을 sigmoid로 변환
                print('reference_1', reference.shape) #reference_1 torch.Size([901, 3]) #(num_q, dim)
            else:
                reference = inter_references[lvl - 1]  # 그 외의 레이어에서는 이전 레이어의 ref_point 사용
                print('reference_2', reference.shape)
            reference = inverse_sigmoid(reference)  # ref_point을 inverse_sigmoid로 변환
            print('reference_3', reference.shape)

            outputs_class = self.cls_branches[lvl](hs[lvl])  # 현재 레이어의 클래스 예측 계산
            print('outputs_class', outputs_class.shape) #outputs_class torch.Size([1, 901, 10]) 
            #(bs, num_q, num_class)

            tmp = self.reg_branches[lvl](hs[lvl])  # 현재 레이어의 좌표 예측 계산
            print('tmp', tmp.shape) #tmp torch.Size([1, 901, 10]) 
            #(bs, num_q, num_coords)

            # 궤적 예측 계산
            outputs_past_traj = self.past_traj_reg_branches[lvl](hs[lvl]).view(
                tmp.shape[0], -1, self.past_steps + self.fut_steps, 2)
            print('outputs_past_traj', outputs_past_traj.shape) #outputs_past_traj torch.Size([1, 901, 8, 2]) 
            #(bs, num_q, len_traj(past_steps + fut_steps), 2D_traj_point)
            
            # TODO: check the shape of reference
            assert reference.shape[-1] == 3  # ref_point의 마지막 차원이 3인지 확인 (검증)
            print('reference_lastDim', reference.shape) #reference_lastDim torch.Size([901, 3])

            tmp[..., 0:2] += reference[..., 0:2]  # ref_point의 x, y 좌표를 예측값에 더함
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()  # x, y 좌표에 sigmoid 적용
            tmp[..., 4:5] += reference[..., 2:3]  # ref_point의 z 좌표를 예측값에 더함
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()  # z 좌표에 sigmoid 적용

            last_ref_points = torch.cat(  # 마지막 ref_point 계산
                [tmp[..., 0:2], tmp[..., 4:5]], dim=-1,
            )
            print('last_ref_points_lastDim', last_ref_points.shape) #last_ref_points_lastDim torch.Size([1, 901, 3])

            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])  # x 좌표 변환
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])  # y 좌표 변환
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])  # z 좌표 변환

            # tmp[..., 2:4] = tmp[..., 2:4] + ref_size_basse[..., 0:2]  # 너비와 높이 변환 (사용 안 함)
            # tmp[..., 5:6] = tmp[..., 5:6] + ref_size_basse[..., 2:3]  # 깊이 변환 (사용 안 함)

            # TODO: check if using sigmoid
            outputs_coord = tmp  # 변환된 좌표를 outputs_coord에 저장

            outputs_classes.append(outputs_class)  # 클래스 예측 결과를 리스트에 추가
            outputs_coords.append(outputs_coord)   # 좌표 예측 결과를 리스트에 추가
            outputs_trajs.append(outputs_past_traj)  # 궤적 예측 결과를 리스트에 추가

        outputs_classes = torch.stack(outputs_classes)  # 클래스 예측 결과를 텐서로 변환
        print('outputs_class_result', outputs_class.shape) #outputs_class_result torch.Size([1, 901, 10]) 
        #(bs, num_q, num_class)

        outputs_coords = torch.stack(outputs_coords)    # 좌표 예측 결과를 텐서로 변환
        print('outputs_coords_result', outputs_coords.shape) #outputs_coords_result torch.Size([4, 1, 901, 10]) 
        #(num_layers, bs, num_queries, num_coords)

        outputs_trajs = torch.stack(outputs_trajs)      # 궤적 예측 결과를 텐서로 변환
        print('outputs_trajs_result', outputs_trajs.shape) #outputs_trajs_result torch.Size([4, 1, 901, 8, 2])
        #(num_layers, bs, num_queries, len_traj, 2D_traj_point)

        last_ref_points = inverse_sigmoid(last_ref_points)  # 마지막 ref_point에 inverse_sigmoid 적용
        print('last_ref_points_result', last_ref_points.shape) #last_ref_points_result torch.Size([1, 901, 3])
        #(batch_size, num_queries, dim_ref_points)

        # 결과를 딕셔너리로 저장
        outs = {
            'all_cls_scores': outputs_classes,  # 모든 클래스 점수
            'all_bbox_preds': outputs_coords,   # 모든 바운딩 박스 예측
            'all_past_traj_preds': outputs_trajs,  # 모든 궤적 예측
            'enc_cls_scores': None,  # 인코더 클래스 점수 (현재 None)
            'enc_bbox_preds': None,  # 인코더 바운딩 박스 예측 (현재 None)
            'last_ref_points': last_ref_points,  # 마지막 ref_point
            'query_feats': hs,  # 쿼리 피처
        }




        # Attention Heatmap 시각화
        print("hs_outs(query_feats):", hs.shape)
        attention_map = hs.max(dim=3)[0].cpu().detach().numpy()  # max값을 뽑아냄
        attention_map = attention_map.reshape(self.bev_h, self.bev_w, -1)
        
        plt.figure(figsize=(self.bev_h, self.bev_w))
        plt.imshow(attention_map, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Attention Heatmap')
        plt.xlabel('BEV Width')
        plt.ylabel('BEV Height')
        plt.savefig('/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/track_head-TrackFormer_Visualization/attention_heatmap.png')
        plt.close()

        # 궤적 시각화
        print("outputs_trajs:", outputs_trajs.shape)
        trajectories = outputs_trajs.cpu().detach().numpy()
        plt.figure(figsize=(self.bev_h, self.bev_w))
        for traj in trajectories[0]:  # batch size가 1이라고 가정
            for q in traj:  # num_queries
                past_traj = q[:self.past_steps]
                fut_traj = q[self.past_steps:]
                plt.plot(past_traj[:, 0], past_traj[:, 1], 'r-')  # 과거 궤적
                plt.plot(fut_traj[:, 0], fut_traj[:, 1], 'g-')  # 미래 궤적

        plt.title('Trajectories')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig('/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/track_head-TrackFormer_Visualization/trajectories.png')
        plt.close()


        return outs  # 결과 반환




    def _get_target_single(self, # 한 이미지에 대한 회귀 및 분류 타겟 계산
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self, # 예측된 값과 실제 값 비교를 위한 타겟 설정
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self, # 단일 디코더 레이어 출력으로 Loss 계산
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, # 전체 Loss 계산
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict


    #TrackFormer에서 도출되는 값 (Output):
    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False): # Bounding Box 생성
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            bbox_index = preds['bbox_index']
            mask = preds['mask']

            ret_list.append([bboxes, scores, labels, bbox_index, mask])

        return ret_list
