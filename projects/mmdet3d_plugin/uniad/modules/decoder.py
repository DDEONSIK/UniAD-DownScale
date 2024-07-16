# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import mmcv
import cv2 as cv
import copy
import warnings
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


def inverse_sigmoid(x, eps=1e-5):
    """Sigmoid 역함수.
    Args:
        x (Tensor): 역함수를 적용할 텐서.
        eps (float): 수치적 오버플로우를 피하기 위한 작은 값. 기본값 1e-5.
    Returns:
        Tensor: Sigmoid 역함수를 통과한 텐서, 입력과 동일한 shape.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetectionTransformerDecoder(TransformerLayerSequence):
    """DETR3D transformer에서 디코더를 구현.
    Args:
        return_intermediate (bool): 중간 출력을 반환할지 여부.
        coder_norm_cfg (dict): 마지막 정규화 레이어 설정. 기본값: `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """`Detr3DTransformerDecoder`의 forward 함수.
        Args:
            query (Tensor): shape `(num_query, bs, embed_dims)`의 입력 쿼리.
            reference_points (Tensor): 옵셋의 참조 포인트. shape (bs, num_query, 4) as_two_stage, 그렇지 않으면 shape (bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): 회귀 결과를 정제하기 위해 사용됨. with_box_refine이 True일 때만 전달됨.
        Returns:
            Tensor: 결과 shape [1, num_query, bs, embed_dims] (return_intermediate가 `False`일 때), 그렇지 않으면 [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class CustomMSDeformableAttention(BaseModule):
    """Deformable-Detr에서 사용되는 attention 모듈.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): Attention의 임베딩 차원. 기본값: 256.
        num_heads (int): 병렬 attention 헤드 수. 기본값: 64.
        num_levels (int): Attention에서 사용되는 특징 맵 수. 기본값: 4.
        num_points (int): 각 헤드에서 각 쿼리에 대해 샘플링 포인트 수. 기본값: 4.
        im2col_step (int): image_to_column에서 사용되는 스텝. 기본값: 64.
        dropout (float): `inp_identity`에서 드롭아웃 레이어. 기본값: 0.1.
        batch_first (bool): Key, Query 및 Value의 shape이 (batch, n, embed_dim) 또는 (n, batch, embed_dim). 기본값: False.
        norm_cfg (dict): 정규화 레이어에 대한 설정 dict. 기본값: None.
        init_cfg (obj:`mmcv.ConfigDict`): 초기화 설정. 기본값: None.
    """

    def __init__(self,
                 embed_dims=128, #_ 초기값: 256 수정log:128 // Transformer의 임베딩 차원 크기, 모델이 입력 데이터에서 학습하는 특징 벡터의 크기
                 num_heads=4, #_ 초기값:8 수정log:4 // 멀티헤드 어텐션 헤드 수
                                #6으로 하면 embed_dim와 나뉘었을 때 떨어지지 않음
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # CUDA 구현에서 더 효율적인지 확인하려면 dim_per_head를 2의 거듭제곱으로 설정하는 것이 좋습니다
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
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """모듈의 파라미터에 대한 기본 초기화."""
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

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
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
                flag='decoder',
                **kwargs):
        """MultiScaleDeformAttention의 Forward 함수.
        Args:
            query (Tensor): shape (num_query, bs, embed_dims)의 Transformer 쿼리.
            key (Tensor): shape `(num_key, bs, embed_dims)`의 키 텐서.
            value (Tensor): shape `(num_key, bs, embed_dims)`의 value 텐서.
            identity (Tensor): addition에 사용되는 텐서, `query`와 동일한 shape. 기본값 None. None인 경우 `query`가 사용됨.
            query_pos (Tensor): `query`에 대한 positional encoding. 기본값: None.
            key_pos (Tensor): `key`에 대한 positional encoding. 기본값: None.
            reference_points (Tensor): shape (bs, num_query, num_levels, 2)의 정규화된 참조 포인트,
                                        모든 요소는 [0, 1] 범위 내에 있으며, 좌상단 (0,0), 우하단 (1,1), 패딩 영역 포함.
                                        또는 (N, Length_{query}, num_levels, 4)의 추가적인 두 차원 (w, h)을 참조 상자로 형성.
            key_padding_mask (Tensor): shape [bs, num_key]의 `query`에 대한 ByteTensor.
            spatial_shapes (Tensor): 다양한 레벨에서 특징의 공간 shape. shape (num_levels, 2), 마지막 차원은 (h, w).
            level_start_index (Tensor): 각 레벨의 시작 인덱스. shape ``(num_levels, )``의 텐서로 [0, h_0*w_0, h_0*w_0+h_1*w_1, ...]로 표현됨.
        Returns:
             Tensor: shape [num_query, bs, embed_dims]의 forward 결과.
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
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:

            # 다중 합산 작업을 수행하기 때문에 fp16 deformable attention 사용은 불안정
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

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
