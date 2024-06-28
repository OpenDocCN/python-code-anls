# `.\models\mgp_str\processing_mgp_str.py`

```
# coding=utf-8
# 定义字符编码类型枚举，包括字符级编码、BPE编码和WordPiece编码
from transformers import AutoTokenizer
from transformers.utils import is_torch_available
from transformers.utils.generic import ExplicitEnum
from ...processing_utils import ProcessorMixin

# 检查是否安装了torch，以便条件导入
if is_torch_available():
    import torch

# 枚举不同的解码类型：字符级、BPE、WordPiece
class DecodeType(ExplicitEnum):
    CHARACTER = "char"
    BPE = "bpe"
    WORDPIECE = "wp"

# 支持的注释格式，包括字符级、BPE和WordPiece
SUPPORTED_ANNOTATION_FORMATS = (DecodeType.CHARACTER, DecodeType.BPE, DecodeType.WORDPIECE)

# MGP-STR处理器类，继承自ProcessorMixin
class MgpstrProcessor(ProcessorMixin):
    """
    构建MGP-STR处理器，将图像处理器和MGP-STR分词器封装到一个单独的处理器中。

    [`MgpstrProcessor`] 提供了`ViTImageProcessor`和`MgpstrTokenizer`的所有功能。查看[`~MgpstrProcessor.__call__`]和
    [`~MgpstrProcessor.batch_decode`]获取更多信息。

    Args:
        image_processor (`ViTImageProcessor`, *可选*):
            `ViTImageProcessor`的实例。图像处理器是必需的输入。
        tokenizer ([`MgpstrTokenizer`], *可选*):
            分词器是必需的输入。
    """

    # 类属性定义
    attributes = ["image_processor", "char_tokenizer"]
    image_processor_class = "ViTImageProcessor"
    char_tokenizer_class = "MgpstrTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 弃用警告：`feature_extractor`参数将在v5中移除，请使用`image_processor`
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 设置图像处理器，如果没有提供则使用`feature_extractor`
        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        
        # 检查是否提供了分词器，如果没有则引发异常
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 初始化MGP-STR处理器实例，设置字符级分词器
        self.char_tokenizer = tokenizer
        # 使用预训练模型创建BPE编码的分词器
        self.bpe_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        # 使用预训练模型创建WordPiece编码的分词器
        self.wp_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

        # 调用父类ProcessorMixin的构造函数，传递图像处理器和分词器
        super().__init__(image_processor, tokenizer)
    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        """
        当以普通模式使用时，此方法将所有参数转发到 ViTImageProcessor 的 [`~ViTImageProcessor.__call__`] 并返回其输出。
        如果 `text` 不为 `None`，此方法还将 `text` 和 `kwargs` 参数转发到 MgpstrTokenizer 的 [`~MgpstrTokenizer.__call__`] 以编码文本。
        更多信息请参考上述方法的文档字符串。
        """
        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        if images is not None:
            # 使用图像处理器处理图像输入
            inputs = self.image_processor(images, return_tensors=return_tensors, **kwargs)
        if text is not None:
            # 使用字符标记器编码文本输入
            encodings = self.char_tokenizer(text, return_tensors=return_tensors, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            # 将标记化的文本作为标签添加到图像输入中
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, sequences):
        """
        将一组标记 id 的列表转换为字符串列表，通过调用 decode 方法实现。

        Args:
            sequences (`torch.Tensor`):
                标记化输入 id 的列表。

        Returns:
            `Dict[str, any]`: 解码结果的所有输出字典。
                generated_text (`List[str]`): 融合字符、bpe 和 wp 后的最终结果。
                scores (`List[float]`): 融合字符、bpe 和 wp 后的最终分数。
                char_preds (`List[str]`): 字符解码后的句子列表。
                bpe_preds (`List[str]`): bpe 解码后的句子列表。
                wp_preds (`List[str]`): wp 解码后的句子列表。

        此方法将其所有参数转发到 PreTrainedTokenizer 的 [`~PreTrainedTokenizer.batch_decode`]。更多信息请参考此方法的文档字符串。
        """
        char_preds, bpe_preds, wp_preds = sequences
        batch_size = char_preds.size(0)

        # 分别调用 `_decode_helper` 方法解码字符、bpe 和 wp
        char_strs, char_scores = self._decode_helper(char_preds, "char")
        bpe_strs, bpe_scores = self._decode_helper(bpe_preds, "bpe")
        wp_strs, wp_scores = self._decode_helper(wp_preds, "wp")

        final_strs = []
        final_scores = []
        for i in range(batch_size):
            scores = [char_scores[i], bpe_scores[i], wp_scores[i]]
            strs = [char_strs[i], bpe_strs[i], wp_strs[i]]
            max_score_index = scores.index(max(scores))
            final_strs.append(strs[max_score_index])
            final_scores.append(scores[max_score_index])

        out = {}
        out["generated_text"] = final_strs
        out["scores"] = final_scores
        out["char_preds"] = char_strs
        out["bpe_preds"] = bpe_strs
        out["wp_preds"] = wp_strs
        return out
    def _decode_helper(self, pred_logits, format):
        """
        Convert a list of lists of bpe token ids into a list of strings by calling bpe tokenizer.

        Args:
            pred_logits (`torch.Tensor`):
                List of model prediction logits.
            format (`Union[DecoderType, str]`):
                Type of model prediction. Must be one of ['char', 'bpe', 'wp'].
        Returns:
            `tuple`:
                dec_strs(`str`): The decode strings of model prediction.
                conf_scores(`List[float]`): The confidence score of model prediction.
        """
        # 根据不同的解码类型选择相应的解码器和结束标记
        if format == DecodeType.CHARACTER:
            decoder = self.char_decode
            eos_token = 1  # 结束标记为1
            eos_str = "[s]"  # 结束字符串为"[s]"
        elif format == DecodeType.BPE:
            decoder = self.bpe_decode
            eos_token = 2  # 结束标记为2
            eos_str = "#"  # 结束字符串为"#"
        elif format == DecodeType.WORDPIECE:
            decoder = self.wp_decode
            eos_token = 102  # 结束标记为102
            eos_str = "[SEP]"  # 结束字符串为"[SEP]"
        else:
            raise ValueError(f"Format {format} is not supported.")  # 如果格式不支持，则抛出异常

        dec_strs, conf_scores = [], []  # 初始化解码字符串列表和置信度分数列表
        batch_size = pred_logits.size(0)  # 获取批次大小
        batch_max_length = pred_logits.size(1)  # 获取每个样本的最大长度
        _, preds_index = pred_logits.topk(1, dim=-1, largest=True, sorted=True)  # 获取每个位置上概率最大的预测索引
        preds_index = preds_index.view(-1, batch_max_length)[:, 1:]  # 去除开始标记，保留有效预测部分
        preds_str = decoder(preds_index)  # 使用对应解码器对预测索引进行解码成字符串
        preds_max_prob, _ = torch.nn.functional.softmax(pred_logits, dim=2).max(dim=2)  # 获取每个位置上的最大概率及其索引
        preds_max_prob = preds_max_prob[:, 1:]  # 去除开始位置的概率

        # 遍历每个样本
        for index in range(batch_size):
            pred_eos = preds_str[index].find(eos_str)  # 查找结束字符串在预测字符串中的位置
            pred = preds_str[index][:pred_eos]  # 截取到结束字符串前的部分作为最终预测
            pred_index = preds_index[index].cpu().tolist()  # 将预测索引转换为CPU上的列表
            pred_eos_index = pred_index.index(eos_token) if eos_token in pred_index else -1  # 查找结束标记的位置
            pred_max_prob = preds_max_prob[index][: pred_eos_index + 1]  # 获取对应的最大概率
            confidence_score = pred_max_prob.cumprod(dim=0)[-1] if pred_max_prob.nelement() != 0 else 0.0  # 计算置信度分数
            dec_strs.append(pred)  # 将预测字符串添加到结果列表
            conf_scores.append(confidence_score)  # 将置信度分数添加到结果列表

        return dec_strs, conf_scores  # 返回解码字符串列表和置信度分数列表

    def char_decode(self, sequences):
        """
        Convert a list of lists of char token ids into a list of strings by calling char tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of char decoded sentences.
        """
        # 使用字符级解码器对字符级标记序列进行解码成字符串
        decode_strs = [seq.replace(" ", "") for seq in self.char_tokenizer.batch_decode(sequences)]
        return decode_strs  # 返回解码后的字符串列表

    def bpe_decode(self, sequences):
        """
        Convert a list of lists of bpe token ids into a list of strings by calling bpe tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of bpe decoded sentences.
        """
        return self.bpe_tokenizer.batch_decode(sequences)  # 使用BPE解码器对BPE级标记序列进行解码成字符串并返回
    def wp_decode(self, sequences):
        """
        Convert a list of lists of word piece token ids into a list of strings by calling word piece tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of wp decoded sentences.
        """
        # 对每个序列进行批量解码，并去除解码后字符串中的空格
        decode_strs = [seq.replace(" ", "") for seq in self.wp_tokenizer.batch_decode(sequences)]
        # 返回解码后的字符串列表
        return decode_strs
```