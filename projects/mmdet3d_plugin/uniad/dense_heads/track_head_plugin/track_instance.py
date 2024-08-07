import itertools
from typing import Any, Dict, List, Tuple, Union
import torch

# 이미지 내의 인스턴스를 나타내는 클래스
class Instances:
    """
    이미지 내의 인스턴스를 나타내는 클래스.
    인스턴스의 속성(e.g., boxes, masks, labels, scores)을 "fields"로 저장함.
    모든 필드는 같은 길이(len(__len__))를 가져야 함.
    모든 다른(비필드) 속성은 비공개로 간주: '_'로 시작하고 사용자가 수정할 수 없음.

    주요 사용법:
    1. 필드 설정/가져오기/확인:
       .. code-block:: python
          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # (N, H, W) 형태의 텐서
          print('gt_masks' in instances)
    2. ``len(instances)``는 인스턴스의 수를 반환
    3. 인덱싱: ``instances[indices]``는 모든 필드에 인덱싱을 적용하고 새로운 :class:`Instances`를 반환
       보통 ``indices``는 인덱스의 정수 벡터 또는 길이가 ``num_instances``인 이진 마스크
       .. code-block:: python
          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    # 생성자
    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): 이미지의 공간 크기.
            kwargs: 이 `Instances`에 추가할 필드.
        """
        self._image_size = image_size  # 이미지 크기 저장
        self._fields: Dict[str, Any] = {}  # 필드를 저장할 딕셔너리
        for k, v in kwargs.items():
            self.set(k, v)  # 전달된 필드를 설정

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size  # 이미지 크기 반환

    # 속성 설정 메소드
    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)  # 비공개 속성은 부모 클래스의 setattr 사용
        else:
            self.set(name, val)  # 나머지는 set 메소드 사용

    # 속성 가져오기 메소드
    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("필드 '{}'를 찾을 수 없음!".format(name))  # 필드를 찾을 수 없을 때 예외 발생
        return self._fields[name]  # 필드 반환

    # 필드 설정 메소드
    def set(self, name: str, value: Any) -> None:
        """
        필드 이름 `name`을 `value`로 설정.
        `value`의 길이는 인스턴스의 수와 일치해야 하며, 이 객체의 다른 필드와 일치해야 함.
        """
        data_len = len(value)  # 필드 값의 길이 확인
        if len(self._fields):
            assert len(self) == data_len, "길이가 {}인 필드를 길이가 {}인 Instances에 추가하려고 함".format(data_len, len(self))
        self._fields[name] = value  # 필드 추가

    # 필드 존재 여부 확인 메소드
    def has(self, name: str) -> bool:
        """
        Returns:
            bool: `name`이라는 필드가 존재하는지 여부.
        """
        return name in self._fields  # 필드 존재 여부 반환

    # 필드 제거 메소드
    def remove(self, name: str) -> None:
        """
        `name`이라는 필드 제거.
        """
        del self._fields[name]  # 필드 삭제

    # 필드 값 가져오기 메소드
    def get(self, name: str) -> Any:
        """
        `name`이라는 필드 반환.
        """
        return self._fields[name]  # 필드 값 반환

    # 모든 필드를 딕셔너리 형태로 반환하는 메소드
    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: 이름(str)과 필드 데이터의 딕셔너리 반환
        반환된 딕셔너리를 수정하면 이 인스턴스도 수정됨.
        """
        return self._fields  # 필드 딕셔너리 반환

    # Tensor와 유사한 메소드
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: 모든 필드가 `to(device)`를 호출한 결과.
        """
        ret = Instances(self._image_size)  # 새로운 Instances 객체 생성
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)  # 필드에 to 메소드가 있으면 호출
            ret.set(k, v)  # 필드 설정
        return ret  # 새 Instances 반환

    # 필드를 numpy 배열로 변환하는 메소드
    def numpy(self):
        ret = Instances(self._image_size)  # 새로운 Instances 객체 생성
        for k, v in self._fields.items():
            if hasattr(v, "numpy"):
                v = v.numpy()  # 필드에 numpy 메소드가 있으면 호출
            ret.set(k, v)  # 필드 설정
        return ret  # 새 Instances 반환

    # 인덱싱 메소드
    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: 모든 필드를 인덱싱하는 데 사용할 인덱스 객체.
        Returns:
            `item`이 문자열인 경우 해당 필드의 데이터를 반환.
            그렇지 않으면, 모든 필드가 `item`에 의해 인덱싱된 `Instances` 반환.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances 인덱스가 범위를 벗어남!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)  # 새로운 Instances 객체 생성
        for k, v in self._fields.items():
            if k == 'kalman_models' and isinstance(item, torch.Tensor):
                ret_list = [self.kalman_models[i] for i, if_true in enumerate(item) if if_true]
                ret.set(k, ret_list)  # kalman_models 필드를 별도로 처리
            else:
                ret.set(k, v[item])  # 나머지 필드는 일반적으로 인덱싱
        return ret  # 새 Instances 반환

    # 인스턴스 수 반환
    def __len__(self) -> int:
        for v in self._fields.values():
            return v.__len__()  # 필드의 길이 반환
        raise NotImplementedError("빈 Instances는 __len__을 지원하지 않음!")

    # 반복자 지원 메소드
    def __iter__(self):
        raise NotImplementedError("`Instances` 객체는 반복할 수 없음!")

    # 여러 Instances 객체를 하나로 병합하는 메소드
    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])
        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)  # 모든 요소가 Instances인지 확인
        assert len(instance_lists) > 0  # 빈 리스트가 아닌지 확인
        if len(instance_lists) == 1:
            return instance_lists[0]  # 요소가 하나면 그대로 반환

        image_size = instance_lists[0].image_size  # 첫 번째 요소의 이미지 크기
        for i in instance_lists[1:]:
            assert i.image_size == image_size  # 모든 요소의 이미지 크기가 같은지 확인
        ret = Instances(image_size)  # 새로운 Instances 객체 생성
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)  # 텐서 병합
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))  # 리스트 병합
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)  # 사용자 정의 타입 병합
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)  # 필드 설정
        return ret  # 새 Instances 반환

    # 인스턴스 문자열 표현 메소드
    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))  # 인스턴스 수
        s += "image_height={}, ".format(self._image_size[0])  # 이미지 높이
        s += "image_width={}, ".format(self._image_size[1])  # 이미지 너비
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))  # 필드 정보
        return s  # 문자열 반환

    __repr__ = __str__  # repr 메소드를 str 메소드와 동일하게 설정
