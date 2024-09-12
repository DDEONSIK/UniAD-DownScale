
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os
# import sys #디버깅
# sys.path.append('/home/hyun/local_storage/code/UniAD') #디버깅

#시각화
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16

print("추적 UniAD/projects/mmdet3d_plugin/uniad/modules/spatial_cross_attention.py 지나감")

    
#디버깅
# from projects.mmdet3d_plugin.uniad.modules.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, MultiScaleDeformableAttnFunction_fp16

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

@ATTENTION.register_module() #아래 SpatialCrossAttention 클래스를 ATTENTION 레지스트리에 등록
class SpatialCrossAttention(BaseModule): #BaseModule: 상속받을 부모 클래스. BaseModule기능 사용.
    print("추적 UniAD/projects/mmdet3d_plugin/uniad/modules/spatial_cross_attention.py SpatialCrossAttention 지나감")

    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=128, #_ 원본: 256 / 수정: 128
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=128, #_ 원본: 256 / 수정: 128
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
   
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size() #embed_dims: 128
        # print(f"bs: {bs}, num_query: {num_query}") #디버깅

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes]) #Tensor.__len__:6313~9@@@ #bev_mask: False #

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        # 각 카메라는 해당 BEV 쿼리와만 상호 작용합니다. 이 단계를 통해 GPU 메모리를 크게 절약할 수 있습니다.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros( #reference_points_cam
            [bs, self.num_cams, max_len, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        # print(f"key: {key}") #디버깅
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        # print(f"value: {value}") #디버깅

        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value, #queries 값 보면 2?3?D가 아님 # num_cams 6
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots) #2D로 변환 #_ 차원 섞여 있어서 하나로 맞춰줘야함
        # print(f"slots: {slots}") #디버깅

        # # 시각화 함수
        # def visualize_tensor(tensor, title, filename):
        #     tensor_np = tensor.detach().cpu().numpy() # 텐서를 넘파이 배열로 변환
        #     mean_tensor = np.mean(tensor_np, axis=0) # 텐서의 첫 번째 축을 기준으로 평균 계산
        #     plt.figure(figsize=(10, 8)) # 그래프 크기
        #     sns.heatmap(mean_tensor, cmap="viridis", annot=False) # 히트맵
        #     plt.title(title) # 그래프 제목 
        #     plt.xlabel('Features') # x축 레이블 # 특성
        #     plt.ylabel('Tokens') # y축 레이블 # 토큰
        #     plt.savefig(f'/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/modules/{filename}') # 그래프를 파일로 저장
        #     plt.close()

        # # 개별 텐서 시각화
        # visualize_tensor(query, 'SCA Query Tensor', 'SCA_tensor_query.png')        # query 텐서 시각화
        # visualize_tensor(key, 'SCA Key Tensor', 'SCA_tensor_key.png')              # key 텐서 시각화
        # visualize_tensor(value, 'SCA Value Tensor', 'SCA_tensor_value.png')        # value 텐서 시각화
        # visualize_tensor(slots, 'SCA Slots Tensor', 'SCA_tensor_slots.png')        # slots 텐서 시각화 

        # queries_2d_1 = queries.mean(dim=1)  # queries 텐서의 처음 차원에 대한 평균을 계산, 2차원 텐서로 변환 
        # visualize_tensor(queries_2d_1, 'SCA Queries Tensor 1', 'SCA_tensor_queries_1.png')  # queries_1 텐서 시각화

        # queries_2d__1 = queries.mean(dim=-1)  # queries 텐서의 마지막 차원에 대한 평균을 계산, 2차원 텐서로 변환
        # visualize_tensor(queries_2d__1, 'SCA Queries Tensor -1', 'SCA_tensor_queries_-1.png')  # queries_-1 텐서 시각화

        # queries_2d__2 = queries.mean(dim=2)  # queries 텐서의 중간 차원에 대한 평균을 계산, 2차원 텐서로 변환
        # visualize_tensor(queries_2d__2, 'SCA Queries Tensor 2', 'SCA_tensor_queries_2.png')  # queries_-1 텐서 시각화


        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    print("추적 UniAD/projects/mmdet3d_plugin/uniad/modules/spatial_cross_attention.py MSDeformableAttention3D 지나감")

    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=128, #_ 원본: 256 / 수정: 128
                 num_heads=8, #_
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.

            각 BEV 쿼리마다 높이가 다른 3D 공간에 `num_Z_anchors`를 소유합니다.
            투영 후, 각 BEV 쿼리는 각 2D 이미지에 `num_Z_anchors` 기준점을 갖습니다.
            각 기준점에 대해 `num_points` 샘플링 포인트를 샘플링합니다.
            num_Z_anchors` 기준점의 경우, 전체 `num_points * num_Z_anchors` 샘플링 포인트가 있습니다.            
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2  #_ Deformable
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2) #_output 재배치



        # 시각화 함수 정의
        # def visualize_tensor(tensor, title, filename):
        #     tensor_np = tensor.detach().cpu().numpy() # 텐서를 넘파이 배열로 변환
        #     mean_tensor = np.mean(tensor_np, axis=0) # 텐서의 첫 번째 축을 기준으로 평균 계산
        #     plt.figure(figsize=(10, 8)) # 그래프 크기
        #     sns.heatmap(mean_tensor, cmap="viridis", annot=False) # 히트맵
        #     plt.title(title) # 그래프 제목 
        #     plt.xlabel('Features') # x축 레이블 # 특성
        #     plt.ylabel('Tokens') # y축 레이블 # 토큰
        #     plt.savefig(f'/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/modules/{filename}') # 그래프를 파일로 저장
        #     plt.close()

        # # 개별 텐서 시각화
        # visualize_tensor(query, 'MSDA Query Tensor', 'MSDA_tensor_query.png')        # query 텐서 시각화
        # visualize_tensor(key, 'MSDA Key Tensor', 'MSDA_tensor_key.png')              # key 텐서 시각화
        # value = value.mean(dim=1)  # value 텐서의 처음 차원에 대한 평균을 계산, 2차원 텐서로 변환  
        # visualize_tensor(value, 'MSDA Value Tensor', 'MSDA_tensor_value.png')        # value 텐서 시각화
        # sampling_locations = sampling_locations.mean(dim=2)  # 4차원을 2차원으로 만들어주기 위해 2번째와 3번째 차원으로 평균 # 샘플링 위치 정의
        # visualize_tensor(sampling_locations, 'MSDA sampling_locations Tensor', 'MSDA_sampling_locations.png')  # sampling_locations 텐서 시각화
        # attention_weights = attention_weights.mean(axis=(2, 3))  # 4차원을 2차원으로 만들어주기 위해 2번째와 3번째 차원으로 평균 #각 샘플링 위치의 중요도 결정
        # visualize_tensor(attention_weights, 'MSDA attention_weights Tensor', 'MSDA_attention_weights.png')  # attention_weights 텐서 시각화

        return output


"""
sampling_locations
attention_weights

첫 번째 차원 (dim=0)
예시: tensor.mean(dim=0)
배치 크기에 대한 평균


두 번째 차원 (dim=1)
예시: tensor.mean(dim=1)
각 배치 내의 특성 또는 시퀀스 길이에 대한 평균


마지막 차원 (dim=-1)
예시: tensor.mean(dim=-1)
각 시퀀스 또는 특성의 최종 값에 대한 평균


중간 차원 (예: dim=2)
예시: tensor.mean(dim=2)
특정 특성의 평균
"""