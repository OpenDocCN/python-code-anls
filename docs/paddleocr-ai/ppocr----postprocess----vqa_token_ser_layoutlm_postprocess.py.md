# `.\PaddleOCR\ppocr\postprocess\vqa_token_ser_layoutlm_postprocess.py`

```
# 导入所需的库
import numpy as np
import paddle
# 导入自定义的函数 load_vqa_bio_label_maps
from ppocr.utils.utility import load_vqa_bio_label_maps

# 定义一个类 VQASerTokenLayoutLMPostProcess，用于文本标签和文本索引之间的转换
class VQASerTokenLayoutLMPostProcess(object):
    """ Convert between text-label and text-index """

    # 初始化方法，接受 class_path 和其他关键字参数
    def __init__(self, class_path, **kwargs):
        # 调用父类的初始化方法
        super(VQASerTokenLayoutLMPostProcess, self).__init__()
        # 调用 load_vqa_bio_label_maps 函数，获取 label2id_map 和 id2label_map
        label2id_map, self.id2label_map = load_vqa_bio_label_maps(class_path)

        # 初始化 label2id_map_for_draw 字典
        self.label2id_map_for_draw = dict()
        # 遍历 label2id_map 中的键值对
        for key in label2id_map:
            # 如果键以 "I-" 开头，则将对应的值存入 label2id_map_for_draw 中
            if key.startswith("I-"):
                self.label2id_map_for_draw[key] = label2id_map["B" + key[1:]]
            else:
                self.label2id_map_for_draw[key] = label2id_map[key]

        # 初始化 id2label_map_for_show 字典
        self.id2label_map_for_show = dict()
        # 遍历 label2id_map_for_draw 中的键值对
        for key in self.label2id_map_for_draw:
            val = self.label2id_map_for_draw[key]
            # 根据不同的情况将键值对存入 id2label_map_for_show 中
            if key == "O":
                self.id2label_map_for_show[val] = key
            if key.startswith("B-") or key.startswith("I-"):
                self.id2label_map_for_show[val] = key[2:]
            else:
                self.id2label_map_for_show[val] = key
    # 定义一个方法，用于对预测结果进行处理
    def __call__(self, preds, batch=None, *args, **kwargs):
        # 如果 preds 是一个元组，则取第一个元素
        if isinstance(preds, tuple):
            preds = preds[0]
        # 如果 preds 是一个 paddle.Tensor 对象，则转换成 numpy 数组
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()

        # 如果 batch 不为 None，则调用 _metric 方法
        if batch is not None:
            return self._metric(preds, batch[5])
        # 否则调用 _infer 方法
        else:
            return self._infer(preds, **kwargs)

    # 定义一个方法，用于计算指标
    def _metric(self, preds, label):
        # 获取预测结果中每个样本的最大概率值对应的索引
        pred_idxs = preds.argmax(axis=2)
        # 初始化解码输出列表和标签解码输出列表
        decode_out_list = [[] for _ in range(pred_idxs.shape[0])]
        label_decode_out_list = [[] for _ in range(pred_idxs.shape[0])]

        # 遍历每个样本的预测结果
        for i in range(pred_idxs.shape[0]):
            for j in range(pred_idxs.shape[1]):
                # 如果标签不为 -100，则将对应的标签和预测结果添加到解码输出列表中
                if label[i, j] != -100:
                    label_decode_out_list[i].append(self.id2label_map[label[i, j]])
                    decode_out_list[i].append(self.id2label_map[pred_idxs[i, j]])
        # 返回解码输出列表和标签解码输出列表
        return decode_out_list, label_decode_out_list
    # 推断函数，根据预测结果、段落偏移ID和OCR信息生成结果列表
    def _infer(self, preds, segment_offset_ids, ocr_infos):
        # 初始化结果列表
        results = []

        # 遍历预测结果、段落偏移ID和OCR信息
        for pred, segment_offset_id, ocr_info in zip(preds, segment_offset_ids,
                                                     ocr_infos):
            # 获取预测结果中概率最大的类别索引
            pred = np.argmax(pred, axis=1)
            # 将类别索引转换为标签
            pred = [self.id2label_map[idx] for idx in pred]

            # 遍历段落偏移ID
            for idx in range(len(segment_offset_id)):
                # 根据索引计算起始位置和结束位置
                if idx == 0:
                    start_id = 0
                else:
                    start_id = segment_offset_id[idx - 1]

                end_id = segment_offset_id[idx]

                # 获取当前段落的预测结果
                curr_pred = pred[start_id:end_id]
                # 将预测结果转换为用于显示的标签ID
                curr_pred = [self.label2id_map_for_draw[p] for p in curr_pred]

                # 如果当前段落预测结果为空，则将预测ID设为0
                if len(curr_pred) <= 0:
                    pred_id = 0
                else:
                    # 统计当前段落预测结果中出现次数最多的标签ID
                    counts = np.bincount(curr_pred)
                    pred_id = np.argmax(counts)
                # 更新OCR信息中的预测ID和预测标签
                ocr_info[idx]["pred_id"] = int(pred_id)
                ocr_info[idx]["pred"] = self.id2label_map_for_show[int(pred_id)]
            # 将更新后的OCR信息添加到结果列表中
            results.append(ocr_info)
        # 返回最终结果列表
        return results
# 定义一个名为DistillationSerPostProcess的类，继承自VQASerTokenLayoutLMPostProcess类
class DistillationSerPostProcess(VQASerTokenLayoutLMPostProcess):
    """
    DistillationSerPostProcess
    """

    # 初始化方法，接受class_path、model_name、key和kwargs参数
    def __init__(self, class_path, model_name=["Student"], key=None, **kwargs):
        # 调用父类的初始化方法
        super().__init__(class_path, **kwargs)
        # 如果model_name不是列表，则将其转换为列表
        if not isinstance(model_name, list):
            model_name = [model_name]
        # 将model_name和key赋值给实例变量
        self.model_name = model_name
        self.key = key

    # 调用方法，接受preds、batch和kwargs参数
    def __call__(self, preds, batch=None, *args, **kwargs):
        # 创建一个空字典用于存储输出
        output = dict()
        # 遍历model_name列表
        for name in self.model_name:
            # 获取对应模型的预测结果
            pred = preds[name]
            # 如果key不为None，则从预测结果中获取指定key的值
            if self.key is not None:
                pred = pred[self.key]
            # 将处理后的预测结果传递给父类的调用方法，并将结果存储到output字典中
            output[name] = super().__call__(pred, batch=batch, *args, **kwargs)
        # 返回存储了处理后预测结果的output字典
        return output
```