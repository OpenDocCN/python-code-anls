# `.\PaddleOCR\ppocr\metrics\e2e_metric.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”提供的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。

# 导入必要的模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 定义模块的公开接口
__all__ = ['E2EMetric']

# 从不同模块导入函数
from ppocr.utils.e2e_metric.Deteval import get_socre_A, get_socre_B, combine_results
from ppocr.utils.e2e_utils.extract_textpoint_slow import get_dict

# 定义 E2EMetric 类
class E2EMetric(object):
    # 初始化函数
    def __init__(self,
                 mode,
                 gt_mat_dir,
                 character_dict_path,
                 main_indicator='f_score_e2e',
                 **kwargs):
        # 初始化属性
        self.mode = mode
        self.gt_mat_dir = gt_mat_dir
        # 获取字符字典
        self.label_list = get_dict(character_dict_path)
        self.max_index = len(self.label_list)
        self.main_indicator = main_indicator
        # 重置指标
        self.reset()
    # 定义一个方法，用于计算模型的预测结果和真实标签之间的评估指标
    def __call__(self, preds, batch, **kwargs):
        # 如果模式为'A'
        if self.mode == 'A':
            # 获取真实多边形数据
            gt_polyons_batch = batch[2]
            # 获取临时的真实标签数据
            temp_gt_strs_batch = batch[3][0]
            # 获取忽略标签数据
            ignore_tags_batch = batch[4]
            # 初始化真实标签列表
            gt_strs_batch = []

            # 遍历临时真实标签数据
            for temp_list in temp_gt_strs_batch:
                t = ""
                # 根据索引获取真实标签
                for index in temp_list:
                    if index < self.max_index:
                        t += self.label_list[index]
                gt_strs_batch.append(t)

            # 遍历预测结果、真实多边形数据、真实标签数据和忽略标签数据
            for pred, gt_polyons, gt_strs, ignore_tags in zip(
                [preds], gt_polyons_batch, [gt_strs_batch], ignore_tags_batch):
                # 准备真实标签信息列表
                gt_info_list = [{
                    'points': gt_polyon,
                    'text': gt_str,
                    'ignore': ignore_tag
                } for gt_polyon, gt_str, ignore_tag in
                                zip(gt_polyons, gt_strs, ignore_tags)]
                # 准备预测结果信息列表
                e2e_info_list = [{
                    'points': det_polyon,
                    'texts': pred_str
                } for det_polyon, pred_str in
                                 zip(pred['points'], pred['texts'])]

                # 获取评分结果
                result = get_socre_A(gt_info_list, e2e_info_list)
                self.results.append(result)
        # 如果模式不为'A'
        else:
            # 获取图像ID
            img_id = batch[5][0]
            # 准备预测结果信息列表
            e2e_info_list = [{
                'points': det_polyon,
                'texts': pred_str
            } for det_polyon, pred_str in zip(preds['points'], preds['texts'])]
            # 获取评分结果
            result = get_socre_B(self.gt_mat_dir, img_id, e2e_info_list)
            self.results.append(result)

    # 获取评估指标
    def get_metric(self):
        # 合并所有结果
        metrics = combine_results(self.results)
        # 重置结果列表
        self.reset()
        return metrics

    # 重置结果列表
    def reset(self):
        self.results = []  # clear results
```