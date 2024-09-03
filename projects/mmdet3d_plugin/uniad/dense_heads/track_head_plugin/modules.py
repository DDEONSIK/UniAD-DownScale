import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .track_instance import Instances

# MemoryBank 클래스 정의 - 메모리 뱅크 관련 기능을 담당하는 클래스
class MemoryBank(nn.Module):  # QIM, TAN 관련
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 자비에르 유니폼 초기화

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        # 메모리 뱅크 설정 및 네트워크 레이어 구성
        self.save_thresh = args['memory_bank_score_thresh']
        self.save_period = 3  # 저장 주기
        self.max_his_length = args['memory_bank_len']  # 메모리 길이 설정

        # 레이어 정의
        self.save_proj = nn.Linear(dim_in, dim_in)  # 입력 차원을 유지하는 Linear 레이어

        self.temporal_attn = nn.MultiheadAttention(dim_in, 8, dropout=0)  # 멀티 헤드 어텐션 레이어
        self.temporal_fc1 = nn.Linear(dim_in, hidden_dim)  # 첫 번째 선형 레이어
        self.temporal_fc2 = nn.Linear(hidden_dim, dim_in)  # 두 번째 선형 레이어
        self.temporal_norm1 = nn.LayerNorm(dim_in)  # 첫 번째 Layer Normalization
        self.temporal_norm2 = nn.LayerNorm(dim_in)  # 두 번째 Layer Normalization

    def update(self, track_instances):
        # 트랙 인스턴스 업데이트
        embed = track_instances.output_embedding[:, None]  # (N, 1, 256)
        scores = track_instances.scores
        mem_padding_mask = track_instances.mem_padding_mask
        device = embed.device

        save_period = track_instances.save_period
        if self.training:
            saved_idxes = scores > 0  # 훈련 중에는 모든 점수를 저장
        else:
            saved_idxes = (save_period == 0) & (scores > self.save_thresh)
            save_period[save_period > 0] -= 1
            save_period[saved_idxes] = self.save_period

        # # 시각화를 위해 업데이트 전 메모리 뱅크 상태 저장
        # before_mem_bank = track_instances.mem_bank.clone() # 추가

        saved_embed = embed[saved_idxes]
        if len(saved_embed) > 0:
            prev_embed = track_instances.mem_bank[saved_idxes]
            save_embed = self.save_proj(saved_embed)
            mem_padding_mask[saved_idxes] = torch.cat(
                [mem_padding_mask[saved_idxes, 1:], 
                 torch.zeros((len(saved_embed), 1), dtype=torch.bool, device=device)], dim=1)
            track_instances.mem_bank = track_instances.mem_bank.clone()
            track_instances.mem_bank[saved_idxes] = torch.cat([prev_embed[:, 1:], save_embed], dim=1)
        # self.visualize_update(track_instances, saved_idxes, before_mem_bank, track_instances.mem_bank) # 추가


    # def visualize_update(self, track_instances, saved_idxes, before_mem_bank, after_mem_bank):
    #     plt.figure(figsize=(20, 15))

    #     # 1. 점수 분포 시각화
    #     plt.subplot(3, 2, 1)
    #     sns.histplot(track_instances.scores.cpu().detach().numpy(), bins=20, kde=True)
    #     plt.title('Score Distribution')
    #     plt.xlabel('Score')
    #     plt.ylabel('Count')
    #     plt.savefig('/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/track_head-TrackFormer_Visualization/score_distribution.png')  # 결과 저장

    #     # 2. 저장된 인덱스 시각화
    #     plt.subplot(3, 2, 2)
    #     plt.imshow(saved_idxes.cpu().detach().numpy().reshape(1, -1), cmap='binary', aspect='auto')
    #     plt.title('Saved Indexes')
    #     plt.xlabel('Track Instance')
    #     plt.yticks([])
    #     plt.savefig('/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/track_head-TrackFormer_Visualization/saved_indexes.png')  # 결과 저장

    #     # 3. 임베딩 변화 시각화 (첫 번째 저장된 인스턴스에 대해)
    #     if saved_idxes.sum() > 0:
    #         first_saved_idx = saved_idxes.nonzero()[0][0]
    #         plt.subplot(3, 2, 3)
    #         plt.plot(before_mem_bank[first_saved_idx, 0].cpu().detach().numpy(), label='Before')
    #         plt.plot(after_mem_bank[first_saved_idx, 0].cpu().detach().numpy(), label='After')
    #         plt.title(f'Embedding Change for Instance {first_saved_idx}')
    #         plt.xlabel('Dimension')
    #         plt.ylabel('Value')
    #         plt.legend()
    #         plt.savefig(f'/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/track_head-TrackFormer_Visualization/embedding_change_instance_{first_saved_idx}.png')  # 결과 저장

    #     # 4. 메모리 뱅크 변화 시각화
    #     plt.subplot(3, 2, 4)
    #     diff = (after_mem_bank - before_mem_bank).abs().mean(dim=-1)
    #     sns.heatmap(diff.cpu().detach().numpy(), cmap='viridis')
    #     plt.title('Memory Bank Change (Mean Absolute Difference)')
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Track Instance')
    #     plt.savefig('/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/track_head-TrackFormer_Visualization/memory_bank_change.png')  # 결과 저장

    #     # 5. 메모리 패딩 마스크 시각화
    #     plt.subplot(3, 2, 5)
    #     sns.heatmap(track_instances.mem_padding_mask.cpu().detach().numpy(), cmap='binary')
    #     plt.title('Memory Padding Mask')
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Track Instance')
    #     plt.savefig('/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/track_head-TrackFormer_Visualization/memory_padding_mask.png')  # 결과 저장

    #     # 6. 저장 주기 분포 시각화
    #     plt.subplot(3, 2, 6)
    #     sns.histplot(track_instances.save_period.cpu().detach().numpy(), bins=20, kde=True)
    #     plt.title('Save Period Distribution')
    #     plt.xlabel('Save Period')
    #     plt.ylabel('Count')
    #     plt.savefig('/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/track_head-TrackFormer_Visualization/save_period_distribution.png')  # 결과 저장

    #     plt.tight_layout()
    #     plt.close()

    #     # Attention map 시각화 추가
    #     plt.figure(figsize=(10, 10))
    #     bev_embed_np = track_instances.mem_bank.cpu().detach().numpy()
    #     bev_embed_np = bev_embed_np.reshape(self.bev_h, self.bev_w, -1)
    #     bev_max = np.max(bev_embed_np, axis=2)
    #     plt.imshow(bev_max, cmap='viridis', interpolation='nearest')
    #     plt.colorbar()
    #     plt.title('BEV Feature Map - max')
    #     plt.xlabel('BEV Width')
    #     plt.ylabel('BEV Height')
    #     plt.savefig('/home/hyun/local_storage/code/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/track_head-TrackFormer_Visualization/bev_ftr.png')  # 결과 저장
    #     plt.close()



    def _forward_temporal_attn(self, track_instances):
        if len(track_instances) == 0:
            return track_instances

        key_padding_mask = track_instances.mem_padding_mask  # [n_, memory_bank_len]
        valid_idxes = key_padding_mask[:, -1] == 0  # 유효한 인덱스 추출
        embed = track_instances.output_embedding[valid_idxes]  # (n, 256)

        if len(embed) > 0:
            prev_embed = track_instances.mem_bank[valid_idxes]
            key_padding_mask = key_padding_mask[valid_idxes]
            embed2 = self.temporal_attn(
                embed[None],  # (num_track, dim) to (1, num_track, dim)
                prev_embed.transpose(0, 1),  # (num_track, mem_len, dim) to (mem_len, num_track, dim)
                prev_embed.transpose(0, 1),
                key_padding_mask=key_padding_mask,
            )[0][0]

            embed = self.temporal_norm1(embed + embed2)  # 첫 번째 정규화 및 잔차 연결
            embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))  # 활성화 함수와 두 번째 정규화
            embed = self.temporal_norm2(embed + embed2)  # 두 번째 정규화 및 잔차 연결
            track_instances.output_embedding = track_instances.output_embedding.clone()
            track_instances.output_embedding[valid_idxes] = embed

        return track_instances

    def forward_temporal_attn(self, track_instances):
        return self._forward_temporal_attn(track_instances)

    def forward(self, track_instances: Instances, update_bank=True) -> Instances:
        # 메모리 뱅크를 업데이트하고 임베딩을 반환
        track_instances = self._forward_temporal_attn(track_instances)
        if update_bank:
            self.update(track_instances)
        return track_instances

#QIM # MOTR: End-to-End Multiple-Object Tracking with Transformer
class QueryInteractionBase(nn.Module):

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 자비에르 유니폼 초기화

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()

# QueryInteractionModule 클래스 정의 - 객체 상호작용 모듈
class QueryInteractionModule(QueryInteractionBase):  # Query Interaction Module #객체 출입 처리
# 신생 객체는 Detection Queries를 통해 추가, 사라진 객체는 Track Queries를 통해 제거

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args["random_drop"]
        self.fp_ratio = args["fp_ratio"]
        self.update_query_pos = args["update_query_pos"]

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        # 레이어 구성
        dropout = args["merger_dropout"]

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args["update_query_pos"]:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query[:, :dim // 2]
        query_feat = track_instances.query[:, dim // 2:]
        q = k = query_pos + out_embed

        # attention - Self Attention 적용
        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:,0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # feed-forward network (FFN) 적용
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            # query 위치 업데이트
            query_pos2 = self.linear_pos2(
                self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query[:, :dim // 2] = query_pos

        # query 특징 업데이트
        query_feat2 = self.linear_feat2(
            self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query[:, dim // 2:] = query_feat
        # track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        # update ref_pts using track_instances.pred_boxes
        return track_instances

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        # 랜덤으로 트랙을 드롭하는 함수
        drop_probability = self.random_drop
        if drop_probability > 0 and len(track_instances) > 0:
            keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
            track_instances = track_instances[keep_idxes]
        return track_instances

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
                # FP 트랙을 추가하는 함수
        """
        self.fp_ratio is used to control num(add_fp) / num(active)
        """
        inactive_instances = track_instances[track_instances.obj_idxes < 0]
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]
        num_fp = len(selected_active_track_instances)

        if len(inactive_instances) > 0 and num_fp > 0:
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                # randomly select num_fp from inactive_instances
                # fp_indexes = np.random.permutation(len(inactive_instances))
                # fp_indexes = fp_indexes[:num_fp]
                # fp_track_instances = inactive_instances[fp_indexes]

                # v2: select the fps with top scores rather than random selection
                fp_indexes = torch.argsort(inactive_instances.scores)[-num_fp:]
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        # 활성 트랙을 선택하는 함수
        track_instances: Instances = data["track_instances"]
        if self.training:
            active_idxes = (track_instances.obj_idxes >=
                            0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(
                active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def forward(self, data) -> Instances:
        # forward 함수 - 전체 트랙 인스턴스를 반환
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Instances = data["init_track_instances"]
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances
