# `.\PaddleOCR\ppocr\postprocess\rec_postprocess.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
import numpy as np
import paddle
from paddle.nn import functional as F
import re

# 定义一个类，用于文本标签和文本索引之间的转换
class BaseRecLabelDecode(object):
    def __init__(self, character_dict_path=None, use_space_char=False):
        # 定义起始标记和结束标记
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        # 如果未提供字符字典路径，则使用默认字符集
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            # 从字符字典文件中读取字符集
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            # 如果使用空格字符，则添加空格到字符集中
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            # 如果字符字典路径中包含'arabic'，则设置为反转字符集
            if 'arabic' in character_dict_path:
                self.reverse = True

        # 添加特殊字符到字符集中
        dict_character = self.add_special_char(dict_character)
        # 创建字符到索引的映射字典
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
    # 反转预测文本
    def pred_reverse(self, pred):
        # 初始化反转后的预测文本列表
        pred_re = []
        # 初始化当前字符
        c_current = ''
        # 遍历预测文本中的每个字符
        for c in pred:
            # 如果字符不是字母、数字或特定符号
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                # 如果当前字符不为空，则将其添加到反转后的预测文本列表中
                if c_current != '':
                    pred_re.append(c_current)
                # 将当前字符添加到反转后的预测文本列表中
                pred_re.append(c)
                # 重置当前字符为空
                c_current = ''
            else:
                # 将当前字符添加到当前字符中
                c_current += c
        # 如果当前字符不为空，则将其添加到反转后的预测文本列表中
        if c_current != '':
            pred_re.append(c_current)

        # 返回反转后的预测文本
        return ''.join(pred_re[::-1])

    # 添加特殊字符
    def add_special_char(self, dict_character):
        return dict_character

    # 解码文本索引为文本标签
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        # 初始化结果列表
        result_list = []
        # 获取需要忽略的标记
        ignored_tokens = self.get_ignored_tokens()
        # 获取批处理大小
        batch_size = len(text_index)
        # 遍历每个批次
        for batch_idx in range(batch_size):
            # 初始化选择标记
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            # 如果需要移除重复字符
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            # 过滤需要忽略的标记
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            # 获取字符列表
            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            # 如果存在文本概率，则获取置信度列表
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            # 如果置信度列表为空，则置信度为0
            if len(conf_list) == 0:
                conf_list = [0]

            # 将字符列表连接成文本
            text = ''.join(char_list)

            # 如果需要反转文本（用于阿拉伯语识别）
            if self.reverse:
                text = self.pred_reverse(text)

            # 将文本及其平均置信度添加到结果列表中
            result_list.append((text, np.mean(conf_list).tolist()))
        # 返回结果列表
        return result_list

    # 获取需要忽略的标记
    def get_ignored_tokens(self):
        return [0]  # 用于CTC空白标记
class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类的构造函数，初始化 CTCLabelDecode 类
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        # 如果 preds 是元组或列表，则取最后一个元素
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        # 如果 preds 是 paddle.Tensor 类型，则转换为 numpy 数组
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        # 获取预测结果中概率最大的索引
        preds_idx = preds.argmax(axis=2)
        # 获取预测结果中概率最大的值
        preds_prob = preds.max(axis=2)
        # 解码预测结果，去除重复字符
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        # 如果 label 为空，则返回解码后的文本
        if label is None:
            return text
        # 解码标签
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        # 在字符字典中添加特殊字符 'blank'
        dict_character = ['blank'] + dict_character
        return dict_character


class DistillationCTCLabelDecode(CTCLabelDecode):
    """
    Convert 
    Convert between text-label and text-index
    """

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 model_name=["student"],
                 key=None,
                 multi_head=False,
                 **kwargs):
        # 调用父类的构造函数，初始化 DistillationCTCLabelDecode 类
        super(DistillationCTCLabelDecode, self).__init__(character_dict_path,
                                                         use_space_char)
        # 如果 model_name 不是列表，则转换为列表
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name

        self.key = key
        self.multi_head = multi_head
    # 定义一个类的方法，用于对模型的预测结果进行处理
    def __call__(self, preds, label=None, *args, **kwargs):
        # 初始化一个空字典用于存储处理后的结果
        output = dict()
        # 遍历模型名称列表
        for name in self.model_name:
            # 获取对应模型的预测结果
            pred = preds[name]
            # 如果指定了关键字，则从预测结果中提取对应键的值
            if self.key is not None:
                pred = pred[self.key]
            # 如果是多头模型且预测结果是字典形式，则取出 'ctc' 键对应的值
            if self.multi_head and isinstance(pred, dict):
                pred = pred['ctc']
            # 调用父类的 __call__ 方法对预测结果进行处理，并将结果存入输出字典中
            output[name] = super().__call__(pred, label=label, *args, **kwargs)
        # 返回处理后的结果字典
        return output
class AttnLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类的构造函数，初始化字符字典路径和是否使用空格字符
        super(AttnLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def add_special_char(self, dict_character):
        # 设置起始和结束特殊字符
        self.beg_str = "sos"
        self.end_str = "eos"
        # 更新字符字典，添加起始和结束特殊字符
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        # 获取需要忽略的标记
        ignored_tokens = self.get_ignored_tokens()
        [beg_idx, end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # 只用于预测，去除重复字符
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list
    # 定义一个方法，用于处理模型的预测结果和标签
    def __call__(self, preds, label=None, *args, **kwargs):
        """
        # 对预测结果进行解码
        text = self.decode(text)
        # 如果标签为空，则返回解码后的文本
        if label is None:
            return text
        else:
            # 对标签进行解码，不去除重复字符
            label = self.decode(label, is_remove_duplicate=False)
            return text, label
        """
        # 如果预测结果是 paddle.Tensor 类型，则转换为 numpy 数组
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()

        # 获取预测结果中概率最大的索引
        preds_idx = preds.argmax(axis=2)
        # 获取预测结果中概率最大的值
        preds_prob = preds.max(axis=2)
        # 根据预测结果的索引和概率进行解码，不去除重复字符
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        # 如果标签为空，则返回解码后的文本
        if label is None:
            return text
        # 对标签进行解码，不去除重复字符
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    # 获取被忽略的标记
    def get_ignored_tokens(self):
        # 获取起始标记的索引
        beg_idx = self.get_beg_end_flag_idx("beg")
        # 获取结束标记的索引
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    # 获取起始或结束标记的索引
    def get_beg_end_flag_idx(self, beg_or_end):
        # 如果是起始标记
        if beg_or_end == "beg":
            # 获取起始标记在字典中的索引
            idx = np.array(self.dict[self.beg_str])
        # 如果是结束标记
        elif beg_or_end == "end":
            # 获取结束标记在字典中的索引
            idx = np.array(self.dict[self.end_str])
        else:
            # 抛出异常，表示不支持的类型
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx
class RFLLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类的构造函数，初始化字符字典路径和是否使用空格字符
        super(RFLLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def add_special_char(self, dict_character):
        # 设置起始和结束特殊字符
        self.beg_str = "sos"
        self.end_str = "eos"
        # 更新字符字典，添加起始和结束特殊字符
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        # 获取需要忽略的标记
        ignored_tokens = self.get_ignored_tokens()
        [beg_idx, end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # 只用于预测，去除重复字符
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list
    # 定义一个方法，用于对模型的预测结果进行处理
    def __call__(self, preds, label=None, *args, **kwargs):
        # 判断是否有序列输出
        if isinstance(preds, tuple) or isinstance(preds, list):
            # 如果有序列输出，则分别获取分类输出和序列输出
            cnt_outputs, seq_outputs = preds
            # 如果序列输出是 PaddlePaddle 的 Tensor 对象，则转换为 numpy 数组
            if isinstance(seq_outputs, paddle.Tensor):
                seq_outputs = seq_outputs.numpy()
            # 获取预测结果中概率最大的索引
            preds_idx = seq_outputs.argmax(axis=2)
            # 获取预测结果中概率最大的值
            preds_prob = seq_outputs.max(axis=2)
            # 对预测结果进行解码，生成文本
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

            # 如果没有标签，则返回文本结果
            if label is None:
                return text
            # 对标签进行解码
            label = self.decode(label, is_remove_duplicate=False)
            return text, label

        else:
            # 如果没有序列输出，则获取分类输出
            cnt_outputs = preds
            # 如果分类输出是 PaddlePaddle 的 Tensor 对象，则转换为 numpy 数组
            if isinstance(cnt_outputs, paddle.Tensor):
                cnt_outputs = cnt_outputs.numpy()
            # 计算分类输出中每个样本的长度
            cnt_length = []
            for lens in cnt_outputs:
                length = round(np.sum(lens))
                cnt_length.append(length)
            # 如果没有标签，则返回长度结果
            if label is None:
                return cnt_length
            # 对标签进行解码
            label = self.decode(label, is_remove_duplicate=False)
            # 获取标签中每个样本的长度
            length = [len(res[0]) for res in label]
            return cnt_length, length

    # 获取被忽略的标记
    def get_ignored_tokens(self):
        # 获取起始标记的索引
        beg_idx = self.get_beg_end_flag_idx("beg")
        # 获取结束标记的索引
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    # 获取起始或结束标记的索引
    def get_beg_end_flag_idx(self, beg_or_end):
        # 根据参数判断是获取起始标记还是结束标记的索引
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            # 如果参数不是"beg"或"end"，则抛出异常
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx
class SEEDLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类的构造函数，初始化 SEEDLabelDecode 类
        super(SEEDLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def add_special_char(self, dict_character):
        # 设置特殊字符
        self.padding_str = "padding"
        self.end_str = "eos"
        self.unknown = "unknown"
        # 将特殊字符添加到字符字典中
        dict_character = dict_character + [
            self.end_str, self.padding_str, self.unknown
        ]
        return dict_character

    def get_ignored_tokens(self):
        # 获取结束标记的索引
        end_idx = self.get_beg_end_flag_idx("eos")
        return [end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        # 根据参数获取开始或结束标记的索引
        if beg_or_end == "sos":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "eos":
            idx = np.array(self.dict[self.end_str])
        else:
            # 如果参数不是"sos"或"eos"，则抛出异常
            assert False, "unsupport type %s in get_beg_end_flag_idx" % beg_or_end
        return idx
    # 将文本索引转换为文本标签
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        # 初始化结果列表
        result_list = []
        # 获取忽略标记的结束索引
        [end_idx] = self.get_ignored_tokens()
        # 获取批处理大小
        batch_size = len(text_index)
        # 遍历每个批次
        for batch_idx in range(batch_size):
            # 初始化字符列表和置信度列表
            char_list = []
            conf_list = []
            # 遍历每个文本索引
            for idx in range(len(text_index[batch_idx])):
                # 如果遇到结束索引，则停止
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                # 如果需要去除重复字符
                if is_remove_duplicate:
                    # 仅用于预测
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                # 将字符添加到字符列表中
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                # 如果存在文本概率，则将概率添加到置信度列表中
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            # 将字符列表转换为文本
            text = ''.join(char_list)
            # 计算置信度的平均值，并添加到结果列表中
            result_list.append((text, np.mean(conf_list).tolist()))
        # 返回结果列表
        return result_list
    # 定义一个调用函数，接受预测结果、标签以及其他参数
    def __call__(self, preds, label=None, *args, **kwargs):
        """
        # 对文本进行解码处理
        text = self.decode(text)
        # 如果标签为空，则返回解码后的文本
        if label is None:
            return text
        else:
            # 对标签进行解码处理，不去除重复字符
            label = self.decode(label, is_remove_duplicate=False)
            # 返回解码后的文本和标签
            return text, label
        """
        # 获取预测结果中的"rec_pred"字段
        preds_idx = preds["rec_pred"]
        # 如果"rec_pred"是Paddle张量，则转换为NumPy数组
        if isinstance(preds_idx, paddle.Tensor):
            preds_idx = preds_idx.numpy()
        # 如果预测结果中包含"rec_pred_scores"字段
        if "rec_pred_scores" in preds:
            # 获取"rec_pred"和"rec_pred_scores"字段的值
            preds_idx = preds["rec_pred"]
            preds_prob = preds["rec_pred_scores"]
        else:
            # 如果没有"rec_pred_scores"字段，则计算"rec_pred"的最大值索引和概率
            preds_idx = preds["rec_pred"].argmax(axis=2)
            preds_prob = preds["rec_pred"].max(axis=2)
        # 对预测结果进行解码处理，不去除重复字符
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        # 如果标签为空，则返回解码后的文本
        if label is None:
            return text
        # 对标签进行解码处理，不去除重复字符
        label = self.decode(label, is_remove_duplicate=False)
        # 返回解码后的文本和标签
        return text, label
class SRNLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    # 初始化函数，用于初始化SRNLabelDecode类的实例
    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类BaseRecLabelDecode的初始化函数
        super(SRNLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)
        # 获取参数中的最大文本长度，默认为25
        self.max_text_length = kwargs.get('max_text_length', 25)

    # 调用函数，用于将预测结果转换为文本
    def __call__(self, preds, label=None, *args, **kwargs):
        # 获取预测结果
        pred = preds['predict']
        # 计算字符数量
        char_num = len(self.character_str) + 2
        # 如果预测结果是Paddle张量，则转换为NumPy数组
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        # 重塑预测结果的形状
        pred = np.reshape(pred, [-1, char_num])

        # 获取预测结果中概率最大的索引
        preds_idx = np.argmax(pred, axis=1)
        # 获取预测结果中概率最大的值
        preds_prob = np.max(pred, axis=1)

        # 重塑预测结果索引的形状
        preds_idx = np.reshape(preds_idx, [-1, self.max_text_length])

        # 重塑预测结果概率的形状
        preds_prob = np.reshape(preds_prob, [-1, self.max_text_length])

        # 解码预测结果，得到文本
        text = self.decode(preds_idx, preds_prob)

        # 如果没有标签，则直接返回解码后的文本
        if label is None:
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            return text
        # 解码标签
        label = self.decode(label)
        # 返回解码后的文本和标签
        return text, label
    # 将文本索引转换为文本标签
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        # 初始化结果列表
        result_list = []
        # 获取需要忽略的标记
        ignored_tokens = self.get_ignored_tokens()
        # 获取批处理大小
        batch_size = len(text_index)

        # 遍历每个批次
        for batch_idx in range(batch_size):
            # 初始化字符列表和置信度列表
            char_list = []
            conf_list = []
            # 遍历每个文本索引
            for idx in range(len(text_index[batch_idx])):
                # 如果文本索引在忽略标记中，则跳过
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                # 如果需要去重
                if is_remove_duplicate:
                    # 仅用于预测
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                # 将字符添加到字符列表中
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                # 如果存在文本置信度，则添加到置信度列表中
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            # 将字符列表转换为文本
            text = ''.join(char_list)
            # 计算置信度的平均值，并添加到结果列表中
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    # 添加特殊字符到字符字典中
    def add_special_char(self, dict_character):
        dict_character = dict_character + [self.beg_str, self.end_str]
        return dict_character

    # 获取需要忽略的标记
    def get_ignored_tokens(self):
        # 获取起始标记的索引
        beg_idx = self.get_beg_end_flag_idx("beg")
        # 获取结束标记的索引
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    # 获取起始或结束标记的索引
    def get_beg_end_flag_idx(self, beg_or_end):
        # 根据参数选择起始或结束标记
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" % beg_or_end
        return idx
class SARLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    # SARLabelDecode 类的初始化方法，接受字符字典路径和是否使用空格字符作为参数
    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(SARLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

        # 获取是否移除符号的参数
        self.rm_symbol = kwargs.get('rm_symbol', False)

    # 添加特殊字符到字符字典中
    def add_special_char(self, dict_character):
        # 定义特殊字符
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        # 将未知字符添加到字符字典中
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        # 将起始和结束字符添加到字符字典中
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        # 将填充字符添加到字符字典中
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1
        return dict_character
    # 将文本索引转换为文本标签
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        # 初始化结果列表
        result_list = []
        # 获取忽略的标记
        ignored_tokens = self.get_ignored_tokens()

        # 获取批处理大小
        batch_size = len(text_index)
        # 遍历每个批次
        for batch_idx in range(batch_size):
            # 初始化字符列表和置信度列表
            char_list = []
            conf_list = []
            # 遍历每个文本索引
            for idx in range(len(text_index[batch_idx])):
                # 如果文本索引在忽略的标记中，则跳过
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                # 如果文本索引为结束索引且没有文本概率，则跳过
                if int(text_index[batch_idx][idx]) == int(self.end_idx):
                    if text_prob is None and idx == 0:
                        continue
                    else:
                        break
                # 如果需要去除重复字符
                if is_remove_duplicate:
                    # 仅用于预测
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                # 添加字符到字符列表
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                # 如果有文本概率，则添加到置信度列表
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            # 将字符列表连接成文本
            text = ''.join(char_list)
            # 如果需要移除符号
            if self.rm_symbol:
                # 编译正则表达式，用于匹配非字母、数字和中文字符
                comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
                # 将文本转换为小写
                text = text.lower()
                # 使用正则表达式替换文本中的符号
                text = comp.sub('', text)
            # 添加文本和平均置信度到结果列表
            result_list.append((text, np.mean(conf_list).tolist()))
        # 返回结果列表
        return result_list

    # 调用函数
    def __call__(self, preds, label=None, *args, **kwargs):
        # 如果预测结果是张量，则转换为 numpy 数组
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        # 获取预测结果的最大值索引和概率
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        # 解码预测结果
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

        # 如果没有标签，则返回解码的文本
        if label is None:
            return text
        # 解码标签
        label = self.decode(label, is_remove_duplicate=False)
        # 返回解码的文本和标签
        return text, label
    # 返回一个包含 padding_idx 的列表，用于表示被忽略的标记
    def get_ignored_tokens(self):
        return [self.padding_idx]
class SATRNLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类的构造函数，初始化字符字典路径和是否使用空格字符
        super(SATRNLabelDecode, self).__init__(character_dict_path,
                                               use_space_char)

        # 获取是否需要移除特殊符号的参数
        self.rm_symbol = kwargs.get('rm_symbol', False)

    def add_special_char(self, dict_character):
        # 定义特殊字符
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        # 将未知字符添加到字符字典中
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        # 将起始和结束字符添加到字符字典中
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        # 将填充字符添加到字符字典中
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1
        return dict_character
    # 将文本索引转换为文本标签
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        # 初始化结果列表
        result_list = []
        # 获取忽略的标记
        ignored_tokens = self.get_ignored_tokens()

        # 获取批处理大小
        batch_size = len(text_index)
        # 遍历每个批次
        for batch_idx in range(batch_size):
            # 初始化字符列表和置信度列表
            char_list = []
            conf_list = []
            # 遍历每个文本索引
            for idx in range(len(text_index[batch_idx])):
                # 如果文本索引在忽略的标记中，则跳过
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                # 如果文本索引为结束索引且没有文本概率，则跳过
                if int(text_index[batch_idx][idx]) == int(self.end_idx):
                    if text_prob is None and idx == 0:
                        continue
                    else:
                        break
                # 如果需要去除重复字符
                if is_remove_duplicate:
                    # 仅用于预测
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                # 添加字符到字符列表
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                # 如果有文本概率，则添加到置信度列表
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            # 将字符列表连接成文本
            text = ''.join(char_list)
            # 如果需要移除符号
            if self.rm_symbol:
                # 编译正则表达式，用于匹配非字母、数字和中文字符
                comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
                # 将文本转换为小写
                text = text.lower()
                # 使用正则表达式替换文本中的符号
                text = comp.sub('', text)
            # 添加文本和平均置信度到结果列表
            result_list.append((text, np.mean(conf_list).tolist()))
        # 返回结果列表
        return result_list

    # 调用函数
    def __call__(self, preds, label=None, *args, **kwargs):
        # 如果预测结果是张量，则转换为 numpy 数组
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        # 获取预测结果的最大值索引和概率
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        # 解码预测结果
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

        # 如果没有标签，则返回解码的文本
        if label is None:
            return text
        # 解码标签
        label = self.decode(label, is_remove_duplicate=False)
        # 返回解码的文本和标签
        return text, label
    # 返回一个包含 padding_idx 的列表，用于表示被忽略的标记
    def get_ignored_tokens(self):
        return [self.padding_idx]
class DistillationSARLabelDecode(SARLabelDecode):
    """
    Convert 
    Convert between text-label and text-index
    """

    # 初始化函数，用于初始化对象的属性
    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 model_name=["student"],
                 key=None,
                 multi_head=False,
                 **kwargs):
        # 调用父类的初始化函数
        super(DistillationSARLabelDecode, self).__init__(character_dict_path,
                                                         use_space_char)
        # 如果model_name不是列表，则转换为列表
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name

        self.key = key
        self.multi_head = multi_head

    # 调用对象时执行的函数
    def __call__(self, preds, label=None, *args, **kwargs):
        output = dict()
        # 遍历model_name列表
        for name in self.model_name:
            pred = preds[name]
            # 如果指定了key，则获取对应的预测结果
            if self.key is not None:
                pred = pred[self.key]
            # 如果是多头模型且预测结果是字典，则获取'sar'对应的结果
            if self.multi_head and isinstance(pred, dict):
                pred = pred['sar']
            # 将预测结果转换为文本并存储在output字典中
            output[name] = super().__call__(pred, label=label, *args, **kwargs)
        return output


class PRENLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    # 初始化函数，用于初始化对象的属性
    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类的初始化函数
        super(PRENLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    # 添加特殊字符到字典中
    def add_special_char(self, dict_character):
        padding_str = '<PAD>'  # 0 
        end_str = '<EOS>'  # 1
        unknown_str = '<UNK>'  # 2

        # 将特殊字符添加到字典中
        dict_character = [padding_str, end_str, unknown_str] + dict_character
        self.padding_idx = 0
        self.end_idx = 1
        self.unknown_idx = 2

        return dict_character
    # 将文本索引转换为文本标签
    def decode(self, text_index, text_prob=None):
        # 初始化结果列表
        result_list = []
        # 获取批处理大小
        batch_size = len(text_index)

        # 遍历每个批次
        for batch_idx in range(batch_size):
            # 初始化字符列表和置信度列表
            char_list = []
            conf_list = []
            # 遍历每个文本索引
            for idx in range(len(text_index[batch_idx])):
                # 如果遇到结束索引，则停止
                if text_index[batch_idx][idx] == self.end_idx:
                    break
                # 如果索引为填充索引或未知索引，则跳过
                if text_index[batch_idx][idx] in [self.padding_idx, self.unknown_idx]:
                    continue
                # 将字符添加到字符列表中
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                # 如果存在文本概率，则添加到置信度列表中
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            # 将字符列表转换为文本
            text = ''.join(char_list)
            # 如果文本长度大于0，则添加到结果列表中
            if len(text) > 0:
                result_list.append((text, np.mean(conf_list).tolist()))
            else:
                # 如果识别结果为空，则置信度为1
                result_list.append(('', 1))
        # 返回结果列表
        return result_list

    # 定义调用函数
    def __call__(self, preds, label=None, *args, **kwargs):
        # 如果预测结果为张量，则转换为numpy数组
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        # 获取预测结果的最大值索引和概率
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        # 解码预测结果
        text = self.decode(preds_idx, preds_prob)
        # 如果没有标签，则返回解码的文本
        if label is None:
            return text
        # 解码标签
        label = self.decode(label)
        # 返回解码的文本和标签
        return text, label
class NRTRLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    # 初始化函数，用于创建 NRTRLabelDecode 对象
    def __init__(self, character_dict_path=None, use_space_char=True, **kwargs):
        # 调用父类的初始化函数
        super(NRTRLabelDecode, self).__init__(character_dict_path, use_space_char)

    # 对象调用函数，用于将预测结果转换为文本
    def __call__(self, preds, label=None, *args, **kwargs):

        # 判断预测结果的长度是否为2
        if len(preds) == 2:
            # 获取预测结果的id和概率
            preds_id = preds[0]
            preds_prob = preds[1]
            # 将 paddle.Tensor 类型转换为 numpy 数组
            if isinstance(preds_id, paddle.Tensor):
                preds_id = preds_id.numpy()
            if isinstance(preds_prob, paddle.Tensor):
                preds_prob = preds_prob.numpy()
            # 判断预测结果的第一个元素是否为2
            if preds_id[0][0] == 2:
                preds_idx = preds_id[:, 1:]
                preds_prob = preds_prob[:, 1:]
            else:
                preds_idx = preds_id
            # 解码预测结果，不去除重复字符
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            # 如果没有标签，则返回文本结果
            if label is None:
                return text
            # 解码标签，去除第一个元素
            label = self.decode(label[:, 1:])
        else:
            # 将 paddle.Tensor 类型转换为 numpy 数组
            if isinstance(preds, paddle.Tensor):
                preds = preds.numpy()
            # 获取预测结果的最大值索引和概率
            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)
            # 解码预测结果，不去除重复字符
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            # 如果没有标签，则返回文本结果
            if label is None:
                return text
            # 解码标签，去除第一个元素
            label = self.decode(label[:, 1:])
        # 返回文本结果和标签结果
        return text, label

    # 添加特殊字符到字符字典中
    def add_special_char(self, dict_character):
        # 添加特殊字符到字符字典中
        dict_character = ['blank', '<unk>', '<s>', '</s>'] + dict_character
        return dict_character
    # 将文本索引转换为文本标签
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        # 初始化结果列表
        result_list = []
        # 获取批处理大小
        batch_size = len(text_index)
        # 遍历每个批次
        for batch_idx in range(batch_size):
            # 初始化字符列表和置信度列表
            char_list = []
            conf_list = []
            # 遍历每个文本索引
            for idx in range(len(text_index[batch_idx])):
                try:
                    # 尝试获取字符索引对应的字符
                    char_idx = self.character[int(text_index[batch_idx][idx])]
                except:
                    # 如果出现异常则跳过
                    continue
                # 如果字符索引为结束符号，则结束循环
                if char_idx == '</s>':  # end
                    break
                # 将字符索引添加到字符列表中
                char_list.append(char_idx)
                # 如果存在文本置信度，则将置信度添加到置信度列表中
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    # 否则默认置信度为1
                    conf_list.append(1)
            # 将字符列表转换为文本
            text = ''.join(char_list)
            # 计算置信度的平均值，并转换为列表形式，将文本和置信度添加到结果列表中
            result_list.append((text, np.mean(conf_list).tolist()))
        # 返回结果列表
        return result_list
# 定义一个类 ViTSTRLabelDecode，继承自 NRTRLabelDecode 类，用于文本标签和文本索引之间的转换
class ViTSTRLabelDecode(NRTRLabelDecode):
    """ Convert between text-label and text-index """

    # 初始化方法，接受字符字典路径和是否使用空格字符作为参数
    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(ViTSTRLabelDecode, self).__init__(character_dict_path,
                                                use_space_char)

    # 定义 __call__ 方法，用于将预测结果转换为文本
    def __call__(self, preds, label=None, *args, **kwargs):
        # 如果 preds 是 paddle.Tensor 类型，则将其转换为 numpy 数组
        if isinstance(preds, paddle.Tensor):
            preds = preds[:, 1:].numpy()
        else:
            preds = preds[:, 1:]
        # 获取预测结果中概率最大的索引
        preds_idx = preds.argmax(axis=2)
        # 获取预测结果中概率最大的值
        preds_prob = preds.max(axis=2)
        # 调用 decode 方法将索引转换为文本
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        # 如果 label 为空，则返回文本结果
        if label is None:
            return text
        # 将 label 转换为文本
        label = self.decode(label[:, 1:])
        return text, label

    # 定义 add_special_char 方法，用于添加特殊字符到字符字典中
    def add_special_char(self, dict_character):
        # 在字符字典中添加特殊字符 '<s>' 和 '</s>'
        dict_character = ['<s>', '</s>'] + dict_character
        return dict_character


# 定义一个类 ABINetLabelDecode，继承自 NRTRLabelDecode 类，用于文本标签和文本索引之间的转换
class ABINetLabelDecode(NRTRLabelDecode):
    """ Convert between text-label and text-index """

    # 初始化方法，接受字符字典路径和是否使用空格字符作为参数
    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(ABINetLabelDecode, self).__init__(character_dict_path,
                                                use_space_char)

    # 定义 __call__ 方法，用于将预测结果转换为文本
    def __call__(self, preds, label=None, *args, **kwargs):
        # 如果 preds 是字典类型，则取出字典中的 'align' 键对应的值并转换为 numpy 数组
        if isinstance(preds, dict):
            preds = preds['align'][-1].numpy()
        # 如果 preds 是 paddle.Tensor 类型，则将其转换为 numpy 数组
        elif isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        else:
            preds = preds

        # 获取预测结果中概率最大的索引
        preds_idx = preds.argmax(axis=2)
        # 获取预测结果中概率最大的值
        preds_prob = preds.max(axis=2)
        # 调用 decode 方法将索引转换为文本
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        # 如果 label 为空，则返回文本结果
        if label is None:
            return text
        # 将 label 转换为文本
        label = self.decode(label)
        return text, label

    # 定义 add_special_char 方法，用于添加特殊字符到字符字典中
    def add_special_char(self, dict_character):
        # 在字符字典中添加特殊字符 '</s>'
        dict_character = ['</s>'] + dict_character
        return dict_character


# 定义一个类 SPINLabelDecode，继承自 AttnLabelDecode 类
class SPINLabelDecode(AttnLabelDecode):
    """ Convert between text-label and text-index """

    # 初始化函数，用于将文本标签和文本索引之间进行转换
    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类的初始化函数
        super(SPINLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    # 添加特殊字符到字符字典中
    def add_special_char(self, dict_character):
        # 定义起始字符和结束字符
        self.beg_str = "sos"
        self.end_str = "eos"
        # 将起始字符、结束字符和原字符字典合并
        dict_character = [self.beg_str] + [self.end_str] + dict_character
        # 返回更新后的字符字典
        return dict_character
# 定义 VLLabelDecode 类，用于文本标签和文本索引之间的转换
class VLLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    # 初始化方法，接受字符字典路径和是否使用空格字符等参数
    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(VLLabelDecode, self).__init__(character_dict_path, use_space_char)
        # 设置最大文本长度和类别数
        self.max_text_length = kwargs.get('max_text_length', 25)
        self.nclass = len(self.character) + 1

    # 将文本索引转换为文本标签的方法
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        # 初始化结果列表和忽略的标记
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        # 遍历每个批次
        for batch_idx in range(batch_size):
            # 创建一个全为 True 的选择数组
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            # 如果需要去除重复的标记
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            # 过滤掉忽略的标记
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            # 根据选择数组获取字符列表和置信度列表
            char_list = [
                self.character[text_id - 1]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            # 将字符列表拼接成文本，并计算平均置信度
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

# 定义 CANLabelDecode 类，用于 LaTeX 符号和符号索引之间的转换
class CANLabelDecode(BaseRecLabelDecode):
    """ Convert between latex-symbol and symbol-index """

    # 初始化方法，接受字符字典路径和是否使用空格字符等参数
    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(CANLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)
    # 解码函数，将模型输出的索引转换为文本结果
    def decode(self, text_index, preds_prob=None):
        # 初始化结果列表
        result_list = []
        # 获取批量大小
        batch_size = len(text_index)
        # 遍历每个样本
        for batch_idx in range(batch_size):
            # 获取当前样本的序列结束位置
            seq_end = text_index[batch_idx].argmin(0)
            # 获取当前样本的索引列表并转换为列表
            idx_list = text_index[batch_idx][:seq_end].tolist()
            # 根据索引列表获取对应的符号列表
            symbol_list = [self.character[idx] for idx in idx_list]
            # 初始化概率列表
            probs = []
            # 如果预测概率不为空，则获取对应的概率列表
            if preds_prob is not None:
                probs = preds_prob[batch_idx][:len(symbol_list)].tolist()

            # 将符号列表和概率列表添加到结果列表中
            result_list.append([' '.join(symbol_list), probs])
        # 返回结果列表
        return result_list

    # 调用函数，根据模型输出和标签获取文本结果
    def __call__(self, preds, label=None, *args, **kwargs):
        # 获取模型预测的概率、无用变量、无用变量、无用变量
        pred_prob, _, _, _ = preds
        # 获取预测的索引
        preds_idx = pred_prob.argmax(axis=2)

        # 解码预测结果
        text = self.decode(preds_idx)
        # 如果标签为空，则直接返回解码的文本结果
        if label is None:
            return text
        # 解码标签
        label = self.decode(label)
        # 返回解码的文本结果和标签
        return text, label
```