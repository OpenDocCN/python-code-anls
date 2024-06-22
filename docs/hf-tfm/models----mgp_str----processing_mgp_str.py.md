# `.\transformers\models\mgp_str\processing_mgp_str.py`

```py
# 导入必要的库和模块
import warnings
from transformers import AutoTokenizer
from transformers.utils import is_torch_available
from transformers.utils.generic import ExplicitEnum
from ...processing_utils import ProcessorMixin

# 如果 PyTorch 可用，则导入 torch 库
if is_torch_available():
    import torch

# 定义字符、BPE 和 WordPiece 分词的编码类型
class DecodeType(ExplicitEnum):
    CHARACTER = "char"
    BPE = "bpe"
    WORDPIECE = "wp"

# 支持的注释格式
SUPPORTED_ANNOTATION_FORMATS = (DecodeType.CHARACTER, DecodeType.BPE, DecodeType.WORDPIECE)

# 定义 MGP-STR 处理器类
class MgpstrProcessor(ProcessorMixin):
    """
    构建 MGP-STR 处理器，包装图像处理器和 MGP-STR 分词器。
    
    [`MgpstrProcessor`] 提供 `ViTImageProcessor` 和 `MgpstrTokenizer` 的所有功能。
    查看 [`~MgpstrProcessor.__call__`] 和 [`~MgpstrProcessor.batch_decode`] 以获取更多信息。
    
    参数:
        image_processor (`ViTImageProcessor`, *可选*):
            ViTImageProcessor 的实例。需要指定图像处理器。
        tokenizer ([`MgpstrTokenizer`], *可选*):
            需要指定分词器。
    """

    attributes = ["image_processor", "char_tokenizer"]
    image_processor_class = "ViTImageProcessor"
    char_tokenizer_class = "MgpstrTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 处理弃用的 feature_extractor 参数
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 使用 image_processor 参数，如果未提供则使用 feature_extractor
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 检查是否提供了图像处理器和分词器
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 保存 character tokenizer
        self.char_tokenizer = tokenizer
        # 加载 BPE 和 WordPiece tokenizer
        self.bpe_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.wp_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # 初始化父类
        super().__init__(image_processor, tokenizer)
    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        """
        当在普通模式下使用时，该方法将所有参数转发给 ViTImageProcessor 的
        [`~ViTImageProcessor.__call__`]，并返回其输出。如果 `text` 不是 `None`，则该方法还将 `text` 和 `kwargs`
        参数转发给 MgpstrTokenizer 的 [`~MgpstrTokenizer.__call__`] 以对文本进行编码。更多信息请参考上述方法的文档。
        """
        如果 images 和 text 都是 None：
            抛出 ValueError 异常，提示需要指定 `images` 或 `text` 中的一个输入来处理。
            
        如果 images 不是 None：
            调用 image_processor 处理 images，并将结果存储在 inputs 中
        如果 text 不是 None：
            调用 char_tokenizer 处理 text，并将结果存储在 encodings 中

        如果 text 是 None：
            返回 inputs
        否则如果 images 是 None：
            返回 encodings
        否则：
            将 encodings 中的 "input_ids" 存储在 inputs 的 "labels" 中
            返回 inputs

    def batch_decode(self, sequences):
        """
        将一组列表的标记 id 转换为一组字符串，通过调用 decode 来完成。

        参数：
            sequences (`torch.Tensor`)：
                标记化输入 id 的列表。

        返回：
            `Dict[str, any]`: 解码结果的所有输出的字典。
                generated_text (`List[str]`): 合并 char、bpe 和 wp 后的最终结果。
                scores (`List[float]`): 合并 char、bpe 和 wp 后的最终分数。
                char_preds (`List[str]`): 字符解码句子的列表。
                bpe_preds (`List[str]`): bpe 解码句子的列表。
                wp_preds (`List[str]`): wp 解码句子的列表。

        该方法将所有参数转发给 PreTrainedTokenizer 的 [`~PreTrainedTokenizer.batch_decode`]。更多信息请参考该方法的文档。
        """
        从 sequences 中获取 char_preds、bpe_preds 和 wp_preds
        获取 batch 的大小为 batch_size

        分别使用 _decode_helper 方法对 char_preds、bpe_preds 和 wp_preds 进行解码，得到字符、bpe 和 wp 的字符串及其对应的分数
        char_strs, char_scores = self._decode_helper(char_preds, "char")
        bpe_strs, bpe_scores = self._decode_helper(bpe_preds, "bpe")
        wp_strs, wp_scores = self._decode_helper(wp_preds, "wp")

        初始化 final_strs 和 final_scores 列表
        遍历每个 batch：
            对于每个 batch 中的字符、bpe 和 wp 的分数和字符串，选择分数最高的作为最终结果
            将最终结果的字符串和分数存储在 final_strs 和 final_scores 中

        创建一个字典 out 来存储输出结果
        将 final_strs、final_scores、char_strs、bpe_strs、wp_strs 存储在 out 中并返回
    def _decode_helper(self, pred_logits, format):
        """
        将一组一组的 BPE 标记 ID 转换为字符串列表，通过调用 BPE 分词器进行转换。
    
        参数：
            pred_logits（`torch.Tensor`）：
                存储模型预测的逻辑回归值的列表。
            format（`Union[DecoderType, str]`）：
                模型预测的类型。必须是 ['char', 'bpe', 'wp'] 中的一个。
    
        返回：
            `tuple`：
                dec_strs（`str`）：模型预测的解码字符串。
                conf_scores（`List[float]`）：模型预测的置信度得分。
        """
        # 根据模型预测的格式选择相应的解码器和终止符号
        if format == DecodeType.CHARACTER: 
            decoder = self.char_decode
            eos_token = 1
            eos_str = "[s]"
        elif format == DecodeType.BPE:
            decoder = self.bpe_decode
            eos_token = 2
            eos_str = "#"
        elif format == DecodeType.WORDPIECE:
            decoder = self.wp_decode
            eos_token = 102
            eos_str = "[SEP]"
        else:
            raise ValueError(f"Format {format} is not supported.")
    
        dec_strs, conf_scores = [], [] # 创建空列表，用于存储解码后的字符串和置信度得分
        batch_size = pred_logits.size(0) # 获取预测逻辑回归值的批次大小
        batch_max_length = pred_logits.size(1) # 获取预测逻辑回归值的最大长度
        _, preds_index = pred_logits.topk(1, dim=-1, largest=True, sorted=True) # 获取每个预测逻辑回归值的最大值及其索引
        preds_index = preds_index.view(-1, batch_max_length)[:, 1:] # 调整形状，去除开始符号
        preds_str = decoder(preds_index) # 将预测的索引转换成字符串
        preds_max_prob, _ = torch.nn.functional.softmax(pred_logits, dim=2).max(dim=2) # 获取每个预测逻辑回归值的最大概率及其索引
        preds_max_prob = preds_max_prob[:, 1:] # 调整形状，去除开始符号
    
        for index in range(batch_size): # 遍历每个样本
            pred_eos = preds_str[index].find(eos_str) # 查找终止符号的索引
            pred = preds_str[index][:pred_eos] # 根据终止符号的索引，获取解码字符串
            pred_index = preds_index[index].cpu().tolist() # 将预测的索引转换为列表
            pred_eos_index = pred_index.index(eos_token) if eos_token in pred_index else -1 # 查找终止符号的索引
            pred_max_prob = preds_max_prob[index][: pred_eos_index + 1] # 根据终止符号的索引，获取概率值
            confidence_score = pred_max_prob.cumprod(dim=0)[-1] if pred_max_prob.nelement() != 0 else 0.0 # 计算置信度得分
            dec_strs.append(pred) # 将解码字符串添加到列表中
            conf_scores.append(confidence_score) # 将置信度得分添加到列表中
    
        return dec_strs, conf_scores
    
    def char_decode(self, sequences):
        """
        将一组一组的���符标记 ID 转换为字符串列表，通过调用字符分词器进行转换。
    
        参数：
            sequences（`torch.Tensor`）：
                存储令牌化输入的 ID 的列表。
    
        返回：
            `List[str]`：字符解码后的句子列表。
        """
        decode_strs = [seq.replace(" ", "") for seq in self.char_tokenizer.batch_decode(sequences)] # 对每个句子进行解码
        return decode_strs
    
    def bpe_decode(self, sequences):
        """
        将一组一组的 BPE 标记 ID 转换为字符串列表，通过调用 BPE 分词器进行转换。
    
        参数：
            sequences（`torch.Tensor`）：
                存储令牌化输入的 ID 的列表。
    
        返回：
            `List[str]`：BPE 解码后的句子列表。
        """
        return self.bpe_tokenizer.batch_decode(sequences) # 对每个句子进行解码
    def wp_decode(self, sequences):
        """
        Convert a list of lists of word piece token ids into a list of strings by calling word piece tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.  # 输入的标记化 ID 的列表
        Returns:
            `List[str]`: The list of wp decoded sentences.  # 解码后的字符串列表
        """
        # 将每个子列表中的标记化 ID 转换为字符串，同时调用词片段标记器
        decode_strs = [seq.replace(" ", "") for seq in self.wp_tokenizer.batch_decode(sequences)]
        # 返回解码后的字符串列表
        return decode_strs
```