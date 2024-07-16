# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, once_differentiable
from mmcv.utils import ext_loader

# 확장 모듈 로드
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


# MultiScaleDeformableAttnFunction_fp16 클래스 정의
class MultiScaleDeformableAttnFunction_fp16(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """멀티 스케일 디포머블 어텐션의 GPU 버전 (fp16)

        Args:
            value (Tensor): (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): 각 피처 맵의 공간 크기, (num_levels, 2)
            sampling_locations (Tensor): 샘플링 포인트의 위치, (bs, num_queries, num_heads, num_levels, num_points, 2)
            attention_weights (Tensor): 샘플링 포인트에 사용되는 가중치, (bs, num_queries, num_heads, num_levels, num_points)
            im2col_step (Tensor): 이미지에서 컬럼으로 변환할 때 사용되는 스텝

        Returns:
            Tensor: (bs, num_queries, embed_dims)
        """
        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU 버전의 backward 함수

        Args:
            grad_output (Tensor): forward 출력의 Gradient

        Returns:
             Tuple[Tensor]: forward 입력 텐서의 Gradient
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


# MultiScaleDeformableAttnFunction_fp32 클래스 정의
class MultiScaleDeformableAttnFunction_fp32(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """멀티 스케일 디포머블 어텐션의 GPU 버전 (fp32)

        Args:
            value (Tensor): (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): 각 피처 맵의 공간 크기, (num_levels, 2)
            sampling_locations (Tensor): 샘플링 포인트의 위치, (bs, num_queries, num_heads, num_levels, num_points, 2)
            attention_weights (Tensor): 샘플링 포인트에 사용되는 가중치, (bs, num_queries, num_heads, num_levels, num_points)
            im2col_step (Tensor): 이미지에서 컬럼으로 변환할 때 사용되는 스텝

        Returns:
            Tensor: (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU 버전의 backward 함수

        Args:
            grad_output (Tensor): forward 출력의 Gradient

        Returns:
             Tuple[Tensor]: forward 입력 텐서의 Gradient
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None
