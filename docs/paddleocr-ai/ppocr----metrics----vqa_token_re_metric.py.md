# `.\PaddleOCR\ppocr\metrics\vqa_token_re_metric.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import paddle

# 定义模块中公开的所有内容
__all__ = ['KIEMetric']

# 定义 VQAReTokenMetric 类
class VQAReTokenMetric(object):
    # 初始化函数，设置主要指标为 'hmean'
    def __init__(self, main_indicator='hmean', **kwargs):
        self.main_indicator = main_indicator
        # 重置指标
        self.reset()

    # 调用函数，用于处理预测结果和批次数据
    def __call__(self, preds, batch, **kwargs):
        # 解包预测结果
        pred_relations, relations, entities = preds
        # 将预测的关系、关系和实体添加到列表中
        self.pred_relations_list.extend(pred_relations)
        self.relations_list.extend(relations)
        self.entities_list.extend(entities)
    # 获取指标
    def get_metric(self):
        # 初始化一个空列表用于存储真实关系
        gt_relations = []
        # 遍历关系列表
        for b in range(len(self.relations_list)):
            # 初始化一个空列表用于存储关系句子
            rel_sent = []
            # 获取当前关系列表和实体列表
            relation_list = self.relations_list[b]
            entitie_list = self.entities_list[b]
            # 获取头部长度
            head_len = relation_list[0, 0]
            # 如果头部长度大于0
            if head_len > 0:
                # 获取实体起始位置列表
                entitie_start_list = entitie_list[1:entitie_list[0, 0] + 1, 0]
                # 获取实体结束位置列表
                entitie_end_list = entitie_list[1:entitie_list[0, 1] + 1, 1]
                # 获取实体标签列表
                entitie_label_list = entitie_list[1:entitie_list[0, 2] + 1, 2]
                # 遍历头部和尾部关系
                for head, tail in zip(relation_list[1:head_len + 1, 0],
                                      relation_list[1:head_len + 1, 1]):
                    # 创建一个关系字典
                    rel = {}
                    rel["head_id"] = head
                    rel["head"] = (entitie_start_list[head],
                                   entitie_end_list[head])
                    rel["head_type"] = entitie_label_list[head]

                    rel["tail_id"] = tail
                    rel["tail"] = (entitie_start_list[tail],
                                   entitie_end_list[tail])
                    rel["tail_type"] = entitie_label_list[tail]

                    rel["type"] = 1
                    rel_sent.append(rel)
            # 将当前句子的关系列表添加到真实关系列表中
            gt_relations.append(rel_sent)
        # 计算关系得分
        re_metrics = self.re_score(
            self.pred_relations_list, gt_relations, mode="boundaries")
        # 构建指标字典
        metrics = {
            "precision": re_metrics["ALL"]["p"],
            "recall": re_metrics["ALL"]["r"],
            "hmean": re_metrics["ALL"]["f1"],
        }
        # 重置对象状态
        self.reset()
        # 返回指标
        return metrics

    # 重置对象状态
    def reset(self):
        # 重置预测关系列表、关系列表和实体列表
        self.pred_relations_list = []
        self.relations_list = []
        self.entities_list = []
```