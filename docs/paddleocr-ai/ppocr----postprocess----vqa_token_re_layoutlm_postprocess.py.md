# `.\PaddleOCR\ppocr\postprocess\vqa_token_re_layoutlm_postprocess.py`

```
# 版权声明和许可证信息
# 2021年PaddlePaddle作者保留所有权利。
# 根据Apache许可证2.0版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和限制。
# 导入Paddle库
import paddle

# 定义VQAReTokenLayoutLMPostProcess类，用于在文本标签和文本索引之间进行转换
class VQAReTokenLayoutLMPostProcess(object):
    def __init__(self, **kwargs):
        super(VQAReTokenLayoutLMPostProcess, self).__init__()

    # 定义__call__方法，用于处理预测结果和标签
    def __call__(self, preds, label=None, *args, **kwargs):
        # 获取预测的关系
        pred_relations = preds['pred_relations']
        # 如果预测的关系是Paddle张量，则转换为NumPy数组
        if isinstance(preds['pred_relations'], paddle.Tensor):
            pred_relations = pred_relations.numpy()
        # 解码预测的关系
        pred_relations = self.decode_pred(pred_relations)

        # 如果存在标签，则返回评估指标
        if label is not None:
            return self._metric(pred_relations, label)
        # 否则进行推理
        else:
            return self._infer(pred_relations, *args, **kwargs)

    # 定义评估指标方法，返回预测关系、标签的倒数第一个元素和倒数第二个元素
    def _metric(self, pred_relations, label):
        return pred_relations, label[-1], label[-2]
    # 推断函数，根据预测的关系和参数，返回结果
    def _infer(self, pred_relations, *args, **kwargs):
        # 从关键字参数中获取序列化结果
        ser_results = kwargs['ser_results']
        # 从关键字参数中获取实体索引字典批次
        entity_idx_dict_batch = kwargs['entity_idx_dict_batch']

        # 合并关系和 OCR 信息
        results = []
        # 遍历预测的关系、序列化结果和实体索引字典批次
        for pred_relation, ser_result, entity_idx_dict in zip(
                pred_relations, ser_results, entity_idx_dict_batch):
            result = []
            used_tail_id = []
            # 遍历预测的关系
            for relation in pred_relation:
                # 如果尾部 ID 已经被使用过，则跳过
                if relation['tail_id'] in used_tail_id:
                    continue
                used_tail_id.append(relation['tail_id'])
                # 获取头部 OCR 信息和尾部 OCR 信息
                ocr_info_head = ser_result[entity_idx_dict[relation['head_id']]]
                ocr_info_tail = ser_result[entity_idx_dict[relation['tail_id']]]
                result.append((ocr_info_head, ocr_info_tail))
            results.append(result)
        return results

    # 解码预测的关系
    def decode_pred(self, pred_relations):
        pred_relations_new = []
        # 遍历预测的关系
        for pred_relation in pred_relations:
            pred_relation_new = []
            # 从第一个元素到第一个元素的值加一，获取新的预测关系
            pred_relation = pred_relation[1:pred_relation[0, 0, 0] + 1]
            # 遍历新的预测关系
            for relation in pred_relation:
                relation_new = dict()
                relation_new['head_id'] = relation[0, 0]
                relation_new['head'] = tuple(relation[1])
                relation_new['head_type'] = relation[2, 0]
                relation_new['tail_id'] = relation[3, 0]
                relation_new['tail'] = tuple(relation[4])
                relation_new['tail_type'] = relation[5, 0]
                relation_new['type'] = relation[6, 0]
                pred_relation_new.append(relation_new)
            pred_relations_new.append(pred_relation_new)
        return pred_relations_new
# 定义一个名为DistillationRePostProcess的类，继承自VQAReTokenLayoutLMPostProcess类
class DistillationRePostProcess(VQAReTokenLayoutLMPostProcess):
    """
    DistillationRePostProcess
    """

    # 初始化方法，接受model_name、key和其他关键字参数
    def __init__(self, model_name=["Student"], key=None, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果model_name不是列表，则将其转换为列表
        if not isinstance(model_name, list):
            model_name = [model_name]
        # 将model_name和key赋值给实例变量
        self.model_name = model_name
        self.key = key

    # 调用方法，接受预测结果preds和其他参数
    def __call__(self, preds, *args, **kwargs):
        # 创建一个空字典用于存储输出结果
        output = dict()
        # 遍历model_name列表中的每个模型名
        for name in self.model_name:
            # 获取对应模型的预测结果
            pred = preds[name]
            # 如果key不为None，则从预测结果中获取指定key的值
            if self.key is not None:
                pred = pred[self.key]
            # 调用父类的__call__方法处理预测结果，并将结果存储到output字典中
            output[name] = super().__call__(pred, *args, **kwargs)
        # 返回处理后的输出结果字典
        return output
```