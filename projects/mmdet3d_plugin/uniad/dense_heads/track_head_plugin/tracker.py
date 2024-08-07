from .track_instance import Instances
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import (
    bbox_overlaps_nearest_3d as iou_3d, )
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

# RuntimeTrackerBase 클래스 정의
class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.5, filter_score_thresh=0.4, miss_tolerance=5):
        self.score_thresh = score_thresh  # 트랙을 유지하는 데 필요한 최소 점수
        self.filter_score_thresh = filter_score_thresh  # 트랙을 필터링하는 데 사용되는 점수
        self.miss_tolerance = miss_tolerance  # 트랙을 제거하기 전 허용되는 최대 사라진 프레임 수
        self.max_obj_id = 0  # 할당된 최대 객체 ID

    # 트래커 초기화 메소드
    def clear(self):
        self.max_obj_id = 0  # 최대 객체 ID를 0으로 재설정

    # 트랙 인스턴스 업데이트 메소드
    def update(self, track_instances: Instances, iou_thre=None):
        # 트랙 인스턴스의 사라진 시간 초기화
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        
        for i in range(len(track_instances)):
            if (
                track_instances.obj_idxes[i] == -1
                and track_instances.scores[i] >= self.score_thresh
            ):  
                # IOU가 설정된 경우 처리
                if iou_thre is not None and track_instances.pred_boxes[track_instances.obj_idxes>=0].shape[0]!=0:
                    iou3ds = iou_3d(denormalize_bbox(track_instances.pred_boxes[i].unsqueeze(0), None)[...,:7], denormalize_bbox(track_instances.pred_boxes[track_instances.obj_idxes>=0], None)[...,:7])
                    if iou3ds.max()>iou_thre:
                        continue
                # new track
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif (
                track_instances.obj_idxes[i] >= 0
                and track_instances.scores[i] < self.filter_score_thresh
            ):
                # sleep time 증가
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    # 트랙을 제거: obj_id를 -1로 설정
                    # TODO: remove it by following functions
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1
