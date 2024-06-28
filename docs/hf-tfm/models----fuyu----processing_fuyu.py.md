# `.\models\fuyu\processing_fuyu.py`

```
# 指定 Python 源文件的编码格式为 UTF-8
# 版权声明，此代码版权归 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证的要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据许可证分发的软件是基于“原样”提供的，
# 不附带任何明示或暗示的保证或条件
# 请参阅许可证以了解特定语言的权限和限制
"""
GIT 的图像/文本处理器类
"""
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# 导入通用处理函数
from ...processing_utils import ProcessorMixin
# 导入标记化工具的基类
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
# 导入实用函数
from ...utils import TensorType, is_torch_available, logging, requires_backends

# 如果 Torch 可用，则导入相关模块
if is_torch_available():
    from .image_processing_fuyu import FuyuBatchFeature

# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果 Torch 可用，则导入 Torch 模块
if is_torch_available():
    import torch

# 定义用于表示文本中边界框和点的标记
TEXT_REPR_BBOX_OPEN = "<box>"
TEXT_REPR_BBOX_CLOSE = "</box>"
TEXT_REPR_POINT_OPEN = "<point>"
TEXT_REPR_POINT_CLOSE = "</point>"

# 定义用于标记化的特殊字符串
TOKEN_BBOX_OPEN_STRING = "<0x00>"  # <bbox>
TOKEN_BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
TOKEN_POINT_OPEN_STRING = "<0x02>"  # <point>
TOKEN_POINT_CLOSE_STRING = "<0x03>"  # </point>
BEGINNING_OF_ANSWER_STRING = "<0x04>"  # <boa>

def full_unpacked_stream_to_tensor(
    all_bi_tokens_to_place: List[int],
    full_unpacked_stream: List["torch.Tensor"],
    fill_value: int,
    batch_size: int,
    new_seq_len: int,
    offset: int,
) -> "torch.Tensor":
    """将解压的令牌流（即批次中每个项目的张量列表）进行必要的填充，以创建一个形状为 batch_size x new_seq_len 的单个张量。
    """

    # 确保 all_bi_tokens_to_place 的长度等于批次大小
    assert len(all_bi_tokens_to_place) == batch_size
    # 确保 full_unpacked_stream 的长度等于批次大小
    assert len(full_unpacked_stream) == batch_size

    # 创建一个填充后的批次张量
    new_padded_tensor = torch.full(
        [batch_size, new_seq_len],
        fill_value=fill_value,
        dtype=full_unpacked_stream[0].dtype,
        device=full_unpacked_stream[0].device,
    )

    # 将每个批次项放入批次张量中
    for bi in range(batch_size):
        tokens_to_place = all_bi_tokens_to_place[bi]
        # 将解压流中的每个项目放入填充张量的相应位置
        new_padded_tensor[bi, :tokens_to_place] = full_unpacked_stream[bi][offset : tokens_to_place + offset]

    return new_padded_tensor

def construct_full_unpacked_stream(
    num_real_text_tokens: Union[List[List[int]], "torch.Tensor"],
    input_stream: "torch.Tensor",
    image_tokens: List[List["torch.Tensor"]],
    batch_size: int,
    num_sub_sequences: int,
) -> List["torch.Tensor"]:
    """接受形状为 B x S x ? 的 input_stream 张量。对于每个子序列，添加所需的
    """
    # 存储所有子序列流的列表
    all_bi_stream = []

    # 遍历每个批次中的索引
    for batch_index in range(batch_size):
        # 存储每个子序列流的列表
        all_si_stream = []

        # 首先，构建完整的标记流（包括图像占位符标记）和每个子序列的损失掩码，并添加到列表中。
        # 我们使用列表而不是张量，因为每个子序列的大小是可变的。
        # TODO 在后续的版本中删除此逻辑，因为不支持子序列。
        
        # 获取图像调整后的标记流
        image_adjustment = image_tokens[batch_index][0]
        
        # 将图像调整后的标记流和输入流的第一个子序列连接起来
        subsequence_stream = torch.cat([image_adjustment, input_stream[batch_index, 0]], dim=0)
        
        # 计算真实标记的数量
        num_real_tokens = image_adjustment.shape[0] + num_real_text_tokens[batch_index][0]
        
        # 只保留真实标记的部分，并添加到子序列流列表中
        all_si_stream.append(subsequence_stream[:num_real_tokens])
        
        # 将所有子序列流连接成一个张量，并添加到所有子序列流的列表中
        all_bi_stream.append(torch.cat(all_si_stream, dim=0))

    # 返回所有批次的标记流列表
    return all_bi_stream
def _replace_string_repr_with_token_tags(prompt: str) -> str:
    # 替换字符串中的特定文本表示符号为对应的标记化标签
    prompt = prompt.replace(TEXT_REPR_POINT_OPEN, TOKEN_POINT_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_POINT_CLOSE, TOKEN_POINT_CLOSE_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_OPEN, TOKEN_BBOX_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_CLOSE, TOKEN_BBOX_CLOSE_STRING)
    return prompt


def _segment_prompt_into_text_token_conversions(prompt: str) -> List:
    """
    Given a string prompt, converts the prompt into a list of TextTokenConversions.
    """
    # 初始化空列表用于存储分段后的文本与标记转换
    prompt_text_list: List = []
    # 创建正则表达式模式，用于匹配文本中的特定标记
    regex_pattern = re.compile(
        f"({TOKEN_BBOX_OPEN_STRING}|{TOKEN_BBOX_CLOSE_STRING}|{TOKEN_POINT_OPEN_STRING}|{TOKEN_POINT_CLOSE_STRING})"
    )
    # 使用正则表达式模式分割文本
    prompt_split = regex_pattern.split(prompt)
    for i, elem in enumerate(prompt_split):
        # 跳过空字符串和特定标记的文本片段
        if len(elem) == 0 or elem in [
            TOKEN_BBOX_OPEN_STRING,
            TOKEN_BBOX_CLOSE_STRING,
            TOKEN_POINT_OPEN_STRING,
            TOKEN_POINT_CLOSE_STRING,
        ]:
            continue
        # 添加文本片段及其是否位于特定标记之内的信息到列表中
        prompt_text_list.append(
            (elem, i > 1 and prompt_split[i - 1] in [TOKEN_BBOX_OPEN_STRING, TOKEN_POINT_OPEN_STRING])
        )
    return prompt_text_list


def _transform_coordinates_and_tokenize(prompt: str, scale_factor: float, tokenizer) -> List[int]:
    """
    This function transforms the prompt in the following fashion:
    - <box> <point> and </box> </point> to their respective token mappings
    - extract the coordinates from the tag
    - transform the coordinates into the transformed image space
    - return the prompt tokens with the transformed coordinates and new tags

    Bounding boxes and points MUST be in the following format: <box>y1, x1, y2, x2</box> <point>x, y</point> The spaces
    and punctuation added above are NOT optional.
    """
    # 使用指定的标记替换文本中的特定字符串表示符号
    prompt = _replace_string_repr_with_token_tags(prompt)
    # 将文本分段为文本与标记转换的列表
    prompt_text_list = _segment_prompt_into_text_token_conversions(prompt)
    transformed_prompt_tokens: List[int] = []
    for elem in prompt_text_list:
        if elem[1]:
            # 如果文本位于特定标记内，需对其进行进一步的标记化处理
            within_tag_tokenized = _transform_within_tags(elem[0], scale_factor, tokenizer)
            # 将处理后的标记化结果扩展到转换后的提示标记列表中
            transformed_prompt_tokens.extend(within_tag_tokenized)
        else:
            # 否则，按照正常方式对文本进行标记化处理
            transformed_prompt_tokens.extend(tokenizer(elem[0], add_special_tokens=False).input_ids)
    # 返回经过转换的提示标记列表
    return transformed_prompt_tokens
def _transform_within_tags(text: str, scale_factor: float, tokenizer) -> List[int]:
    """
    Given a bounding box of the fashion <box>1, 2, 3, 4</box> | <point>1, 2</point> This function is responsible for
    converting 1, 2, 3, 4 into tokens of 1 2 3 4 without any commas.
    """
    # 将文本按逗号分隔成字符串列表
    num_int_strs = text.split(",")
    
    if len(num_int_strs) == 2:
        # 如果有开启或关闭标签，移除它们
        token_space_open_string = tokenizer.vocab[TOKEN_POINT_OPEN_STRING]
        token_space_close_string = tokenizer.vocab[TOKEN_POINT_CLOSE_STRING]
    else:
        token_space_open_string = tokenizer.vocab[TOKEN_BBOX_OPEN_STRING]
        token_space_close_string = tokenizer.vocab[TOKEN_BBOX_CLOSE_STRING]

    # 移除所有数字字符串中的空格并转换为浮点数
    num_ints = [float(num.strip()) for num in num_int_strs]
    
    # 根据坐标数量调整到变换后的图像大小
    if len(num_ints) == 2:
        num_ints_translated = scale_point_to_transformed_image(x=num_ints[0], y=num_ints[1], scale_factor=scale_factor)
    elif len(num_ints) == 4:
        num_ints_translated = scale_bbox_to_transformed_image(
            top=num_ints[0],
            left=num_ints[1],
            bottom=num_ints[2],
            right=num_ints[3],
            scale_factor=scale_factor,
        )
    else:
        raise ValueError(f"Invalid number of ints: {len(num_ints)}")
    
    # 将坐标转换为对应的标记，并加入开启和关闭标记
    tokens = [tokenizer.vocab[str(num)] for num in num_ints_translated]
    return [token_space_open_string] + tokens + [token_space_close_string]


def _tokenize_prompts_with_image_and_batch(
    tokenizer,
    prompts: List[List[str]],
    scale_factors: Optional[List[List["torch.Tensor"]]],
    max_tokens_to_generate: int,
    max_position_embeddings: int,
    add_BOS: bool,
    add_beginning_of_answer_token: bool,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them into a 3D tensor.
    """

    # 如果存在缩放因子，同时转换坐标并进行标记化
    if scale_factors is not None:
        transformed_prompt_tokens = []
        for prompt_seq, scale_factor_seq in zip(prompts, scale_factors):
            transformed_prompt_tokens.append(
                [
                    _transform_coordinates_and_tokenize(prompt, scale_factor.item(), tokenizer)
                    for prompt, scale_factor in zip(prompt_seq, scale_factor_seq)
                ]
            )
    else:
        # 否则，仅对提示进行标记化
        transformed_prompt_tokens = [[tokenizer.tokenize(prompt) for prompt in prompt_seq] for prompt_seq in prompts]

    prompts_tokens = transformed_prompt_tokens

    if add_BOS:
        # 如果需要添加起始标记，获取起始标记的词汇表索引
        bos_token = tokenizer.vocab["<s>"]
    # 如果不需要在答案开头添加特定的开始标记，则使用文本生成器的结束标记作为开始标记
    else:
        bos_token = tokenizer.vocab["|ENDOFTEXT|"]

    # 将每个提示序列的每个子序列的开头加上开始标记，并形成三重嵌套列表
    prompts_tokens = [[[bos_token] + x for x in prompt_seq] for prompt_seq in prompts_tokens]

    # 如果需要在答案开头添加开始标记，则将其加入到每个提示序列的最后一个子序列中
    if add_beginning_of_answer_token:
        boa = tokenizer.vocab[BEGINNING_OF_ANSWER_STRING]
        # 只将开始答案标记添加到最后一个子序列中，因为这是将要生成的部分
        for token_seq in prompts_tokens:
            token_seq[-1].append(boa)

    # 现在我们有了一个嵌套列表的列表，每个子列表代表不同长度的序列
    # 我们希望扩展这些列表以：
    #   - 包含需要生成的标记
    #   - 使所有序列长度相等
    # 获取提示序列的长度
    prompts_length = [[len(x) for x in prompts_tokens_seq] for prompts_tokens_seq in prompts_tokens]

    # 获取最大的提示序列长度
    max_prompt_len: int = np.max(prompts_length)

    # 每个样本的长度，为最大提示长度加上最大可生成的标记数，但不超过最大位置嵌入长度
    samples_length = min(max_prompt_len + max_tokens_to_generate, max_position_embeddings)

    # 如果提示长度加上最大可生成的标记数超过了最大位置嵌入长度，则发出警告并生成尽可能多的标记
    if max_prompt_len + max_tokens_to_generate > max_position_embeddings:
        logger.warning(
            f"Max subsequence prompt length of {max_prompt_len} + max tokens to generate {max_tokens_to_generate}",
            f"exceeds context length of {max_position_embeddings}. Will generate as many tokens as possible.",
        )

    # 现在更新嵌套列表，使其所有子列表长度相等：samples_length
    for prompt_tokens_seq, prompts_length_seq in zip(prompts_tokens, prompts_length):
        for prompt_tokens, prompt_length in zip(prompt_tokens_seq, prompts_length_seq):
            if len(prompt_tokens) > samples_length:
                raise ValueError("Length of subsequence prompt exceeds sequence length.")
            padding_size = samples_length - prompt_length
            # 添加结束文本标记来填充使子序列长度达到 samples_length
            prompt_tokens.extend([tokenizer.vocab["|ENDOFTEXT|"]] * padding_size)

    # 现在我们有了结构化的格式，可以将其转换为张量
    prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.int64)
    prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.int64)

    # 返回处理后的张量
    return prompts_tokens_tensor, prompts_length_tensor
# 定义一个函数，将原始的水平坐标转换为变换后的水平坐标
def original_to_transformed_h_coords(original_coords, scale_h):
    return np.round(original_coords * scale_h).astype(np.int32)

# 定义一个函数，将原始的垂直坐标转换为变换后的垂直坐标
def original_to_transformed_w_coords(original_coords, scale_w):
    return np.round(original_coords * scale_w).astype(np.int32)

# 定义一个函数，根据缩放因子将点的坐标缩放到变换后的图像上，并返回整数列表
def scale_point_to_transformed_image(x: float, y: float, scale_factor: float) -> List[int]:
    # 将 x 坐标缩放并转换为整数
    x_scaled = original_to_transformed_w_coords(np.array([x / 2]), scale_factor)[0]
    # 将 y 坐标缩放并转换为整数
    y_scaled = original_to_transformed_h_coords(np.array([y / 2]), scale_factor)[0]
    return [x_scaled, y_scaled]

# 定义一个函数，根据缩放因子将边界框的坐标缩放到变换后的图像上，并返回整数列表
def scale_bbox_to_transformed_image(
    top: float, left: float, bottom: float, right: float, scale_factor: float
) -> List[int]:
    # 将 top 坐标缩放并转换为整数
    top_scaled = original_to_transformed_w_coords(np.array([top / 2]), scale_factor)[0]
    # 将 left 坐标缩放并转换为整数
    left_scaled = original_to_transformed_h_coords(np.array([left / 2]), scale_factor)[0]
    # 将 bottom 坐标缩放并转换为整数
    bottom_scaled = original_to_transformed_w_coords(np.array([bottom / 2]), scale_factor)[0]
    # 将 right 坐标缩放并转换为整数
    right_scaled = original_to_transformed_h_coords(np.array([right / 2]), scale_factor)[0]
    return [top_scaled, left_scaled, bottom_scaled, right_scaled]

class FuyuProcessor(ProcessorMixin):
    r"""
    构造一个 Fuyu 处理器，将 Fuyu 图像处理器和 Llama 分词器封装为单个处理器。

    [`FuyuProcessor`] 提供了 [`FuyuImageProcessor`] 和 [`LlamaTokenizerFast`] 的所有功能。查看 [`~FuyuProcessor.__call__`] 和 [`~FuyuProcessor.decode`] 获取更多信息。

    Args:
        image_processor ([`FuyuImageProcessor`]):
            必需的图像处理器输入。
        tokenizer ([`LlamaTokenizerFast`]):
            必需的分词器输入。
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FuyuImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor  # 设置图像处理器
        self.tokenizer = tokenizer  # 设置分词器
        self.max_tokens_to_generate = 10  # 最大生成的令牌数量
        self.max_position_embeddings = 16384  # TODO 无法从模型文件中推断出来：在何处设置它？
        self.pad_token_id = 0  # 填充令牌的ID
        self.dummy_image_index = -1  # 虚拟图像索引
    # 将输入序列和注意力掩码填充为相同长度
    def _left_pad_inputs_with_attention_mask(self, model_inputs: List[Dict], return_attention_mask: bool):
        # 计算输入序列中最长的 input_ids 的长度
        max_length_input_ids = max(entry["input_ids"].shape[1] for entry in model_inputs)
        # 计算输入序列中最长的 image_patches_indices 的长度
        max_length_image_patch_indices = max(entry["image_patches_indices"].shape[1] for entry in model_inputs)

        # 初始化批处理后的输入字典
        batched_inputs = {"input_ids": [], "image_patches": [], "image_patches_indices": [], "attention_mask": []}

        # 遍历每个输入条目
        for entry in model_inputs:
            for key, tensor in entry.items():
                if key == "input_ids":
                    # 计算需要填充的 token 数量
                    num_padding_tokens = max_length_input_ids - tensor.shape[1]
                    # 在序列开头填充 pad_token_id，使得所有序列长度一致
                    padded_input_ids = torch.cat(
                        [
                            torch.full((tensor.shape[0], num_padding_tokens), self.pad_token_id, dtype=torch.long),
                            tensor,
                        ],
                        dim=1,
                    )
                    batched_inputs[key].append(padded_input_ids)

                    # 创建相同形状的 attention_mask，用于指示哪些 token 是填充的
                    attention_mask = torch.cat(
                        [torch.zeros(tensor.shape[0], num_padding_tokens, dtype=torch.long), torch.ones_like(tensor)],
                        dim=1,
                    )
                    batched_inputs["attention_mask"].append(attention_mask)

                elif key == "image_patches":
                    # 对于 image_patches，直接将其添加到列表中，不进行填充处理
                    batched_inputs[key].append(tensor)

                else:  # 对于 image_patches_indices
                    # 计算需要填充的 indices 数量
                    num_padding_indices = max_length_image_patch_indices - tensor.shape[1]
                    # 在序列开头填充 dummy_image_index，使得所有序列长度一致
                    padded_indices = torch.cat(
                        [
                            torch.full(
                                (tensor.shape[0], num_padding_indices), self.dummy_image_index, dtype=torch.long
                            ),
                            tensor,
                        ],
                        dim=1,
                    )
                    batched_inputs[key].append(padded_indices)

        # 确定最终的批处理键值，准备进行拼接
        batched_keys = ["input_ids", "image_patches_indices"]
        if return_attention_mask:
            batched_keys.append("attention_mask")

        # 将所有列表中的 tensor 沿着第 0 维度（批次维度）进行拼接
        for key in batched_keys:
            batched_inputs[key] = torch.cat(batched_inputs[key], dim=0)

        # 返回批处理后的输入字典
        return batched_inputs
        ):
        # 创建一个包含单个值为1的张量，用于表示图像是否存在的标志
        image_present = torch.ones(1, 1, 1)
        # 使用图像处理器预处理图像数据，并结合标记信息进行预处理
        model_image_input = self.image_processor.preprocess_with_tokenizer_info(
            image_input=tensor_batch_images,
            image_present=image_present,
            image_unpadded_h=image_unpadded_heights,
            image_unpadded_w=image_unpadded_widths,
            image_placeholder_id=image_placeholder_id,
            image_newline_id=image_newline_id,
            variable_sized=True,
        )
        # FIXME max_tokens_to_generate 被嵌入到此处理器的调用中。FIXME 是用来指示待修复的问题或改进的注释。
        # 使用给定的tokenizer对提示语进行标记化处理，包括图像和批处理信息
        prompt_tokens, prompts_length = _tokenize_prompts_with_image_and_batch(
            tokenizer=self.tokenizer,
            prompts=prompts,
            scale_factors=scale_factors,
            max_tokens_to_generate=self.max_tokens_to_generate,
            max_position_embeddings=self.max_position_embeddings,
            add_BOS=True,
            add_beginning_of_answer_token=True,
        )
        # 构建完整的解包流，包括图像输入的标记化和文本提示标记
        image_padded_unpacked_tokens = construct_full_unpacked_stream(
            num_real_text_tokens=prompts_length,
            input_stream=prompt_tokens,
            image_tokens=model_image_input["image_input_ids"],
            batch_size=1,
            num_sub_sequences=self.subsequence_length,
        )
        # 构建图像补丁索引的输入
        unpacked_image_patch_indices_per_batch = construct_full_unpacked_stream(
            num_real_text_tokens=prompts_length,
            input_stream=torch.full_like(prompt_tokens, -1),
            image_tokens=model_image_input["image_patch_indices_per_batch"],
            batch_size=1,
            num_sub_sequences=self.subsequence_length,
        )
        # 计算最长提示长度
        max_prompt_length = max(x.shape[-1] for x in image_padded_unpacked_tokens)
        # 计算批处理中的最大序列长度
        max_seq_len_batch = min(max_prompt_length + self.max_tokens_to_generate, self.max_position_embeddings)
        # 确定要放置的标记数量
        tokens
    def __call__(
        self,
        text=None,
        images=None,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        """
        这个方法用于调用 LlamaTokenizerFast 的 `PreTrainedTokenizer.__call__` 方法，接收多种参数并处理。
        请参考 LlamaTokenizerFast 的 `PreTrainedTokenizer.__call__` 方法的文档了解更多信息。
        """
        return self.tokenizer.__call__(
            text=text,
            images=images,
            add_special_tokens=add_special_tokens,
            return_attention_mask=return_attention_mask,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_token_type_ids=return_token_type_ids,
            return_length=return_length,
            verbose=verbose,
            return_tensors=return_tensors,
            **kwargs,
        )

    def batch_decode(self, *args, **kwargs):
        """
        这个方法将其所有参数转发给 LlamaTokenizerFast 的 `PreTrainedTokenizer.batch_decode` 方法。
        请参考 LlamaTokenizerFast 的 `PreTrainedTokenizer.batch_decode` 方法的文档了解更多信息。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        这个方法将其所有参数转发给 LlamaTokenizerFast 的 `PreTrainedTokenizer.decode` 方法。
        请参考 LlamaTokenizerFast 的 `PreTrainedTokenizer.decode` 方法的文档了解更多信息。
        """
        return self.tokenizer.decode(*args, **kwargs)
```