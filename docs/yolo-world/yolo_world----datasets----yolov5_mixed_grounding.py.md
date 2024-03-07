# `.\YOLO-World\yolo_world\datasets\yolov5_mixed_grounding.py`

```
# 导入必要的模块
import os.path as osp
from typing import List, Union

# 导入自定义模块
from mmengine.fileio import get_local_path, join_path
from mmengine.utils import is_abs
from mmdet.datasets.coco import CocoDataset
from mmyolo.registry import DATASETS
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset

# 注册YOLOv5MixedGroundingDataset类为DATASETS
@DATASETS.register_module()
class YOLOv5MixedGroundingDataset(BatchShapePolicyDataset, CocoDataset):
    """Mixed grounding dataset."""

    # 定义元信息
    METAINFO = {
        'classes': ('object',),
        'palette': [(220, 20, 60)]}

    # 加载数据列表
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        # 使用get_local_path函数获取本地路径
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            # 使用COCOAPI加载本地路径的数据
            self.coco = self.COCOAPI(local_path)

        # 获取图像ID列表
        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            # 加载原始图像信息
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            # 获取图像对应的注释ID列表
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            # 解析数据信息
            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        # 检查注释ID是否唯一
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        # 删除self.coco对象
        del self.coco
        # 返回数据列表
        return data_list
    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        # 如果处于测试模式，则直接返回原始数据列表
        if self.test_mode:
            return self.data_list

        # 如果没有设置过滤配置，则直接返回原始数据列表
        if self.filter_cfg is None:
            return self.data_list

        # 获取过滤空标注和最小尺寸的配置参数
        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # 获取包含标注的图片的 ID 集合
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)

        valid_data_infos = []
        # 遍历数据列表，筛选符合条件的数据信息
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = int(data_info['width'])
            height = int(data_info['height'])
            # 如果设置了过滤空标注并且当前图片没有标注，则跳过
            if filter_empty_gt and img_id not in ids_with_ann:
                continue
            # 如果图片宽高中的最小值大于等于最小尺寸，则将该数据信息添加到有效数据列表中
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        # 返回筛选后的有效数据信息列表
        return valid_data_infos
    # 将 self.data_root 与 self.data_prefix 和 self.ann_file 连接起来
    def _join_prefix(self):
        """Join ``self.data_root`` with ``self.data_prefix`` and
        ``self.ann_file``.
        """
        # 如果 self.ann_file 不是绝对路径且 self.data_root 存在，则自动将注释文件路径与 self.root 连接起来
        if self.ann_file and not is_abs(self.ann_file) and self.data_root:
            self.ann_file = join_path(self.data_root, self.ann_file)
        # 如果 self.data_prefix 中的路径值不是绝对路径，则自动将数据目录与 self.root 连接起来
        for data_key, prefix in self.data_prefix.items():
            if isinstance(prefix, (list, tuple)):
                abs_prefix = []
                for p in prefix:
                    if not is_abs(p) and self.data_root:
                        abs_prefix.append(join_path(self.data_root, p))
                    else:
                        abs_prefix.append(p)
                self.data_prefix[data_key] = abs_prefix
            elif isinstance(prefix, str):
                if not is_abs(prefix) and self.data_root:
                    self.data_prefix[data_key] = join_path(
                        self.data_root, prefix)
                else:
                    self.data_prefix[data_key] = prefix
            else:
                raise TypeError('prefix should be a string, tuple or list,'
                                f'but got {type(prefix)}')
```