# `.\models\fuyu\processing_fuyu.py`

```
# 设置文件编码、版权声明及许可
# 为 GIT 的图像/文本处理类创建注释

# 从 typing 模块中导入字典、列表、可选项、元组和联合
import re
from typing import Dict, List, Optional, Tuple, Union

# 从 numpy 模块中导入数组处理工具
import numpy as np

# 从 processing_utils 模块中导入 ProcessorMixin
from ...processing_utils import ProcessorMixin
# 从 tokenization_utils_base 模块中导入 PaddingStrategy 和 TruncationStrategy
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
# 从 utils 模块中导入 TensorType, is_torch_available, logging 和 requires_backends
from ...utils import TensorType, is_torch_available, logging, requires_backends

# 如果 Torch 可用，则从 image_processing_fuyu 模块中导入 FuyuBatchFeature
if is_torch_available():
    from .image_processing_fuyu import FuyuBatchFeature

# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果 Torch 可用，则从 Torch 模块中导入 Torch
if is_torch_available():
    import torch

# 设置图像/文本处理类所需的特殊标记
TEXT_REPR_BBOX_OPEN = "<box>"
TEXT_REPR_BBOX_CLOSE = "</box>"
TEXT_REPR_POINT_OPEN = "<point>"
TEXT_REPR_POINT_CLOSE = "</point>"
TOKEN_BBOX_OPEN_STRING = "<0x00>"  # <bbox>
TOKEN_BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
TOKEN_POINT_OPEN_STRING = "<0x02>"  # <point>
TOKEN_POINT_CLOSE_STRING = "<0x03>"  # </point>
BEGINNING_OF_ANSWER_STRING = "<0x04>"  # <boa>

# 定义 full_unpacked_stream_to_tensor 函数
def full_unpacked_stream_to_tensor(
    all_bi_tokens_to_place: List[int],
    full_unpacked_stream: List["torch.Tensor"],
    fill_value: int,
    batch_size: int,
    new_seq_len: int,
    offset: int,
) -> "torch.Tensor":
    """Takes an unpacked stream of tokens (i.e. a list of tensors, one for each item in the batch) and does
    the required padding to create a single tensor for the batch of shape batch_size x new_seq_len.
    """
    # 确保所有 bi_tokens_to_place 的长度等于 batch_size
    assert len(all_bi_tokens_to_place) == batch_size
    assert len(full_unpacked_stream) == batch_size

    # 创建填充后的 batch 张量
    new_padded_tensor = torch.full(
        [batch_size, new_seq_len],
        fill_value=fill_value,
        dtype=full_unpacked_stream[0].dtype,
        device=full_unpacked_stream[0].device,
    )

    # 将每个 batch 条目放入 batch 张量中
    for bi in range(batch_size):
        tokens_to_place = all_bi_tokens_to_place[bi]
        new_padded_tensor[bi, :tokens_to_place] = full_unpacked_stream[bi][offset : tokens_to_place + offset]

    # 返回填充后的 batch 张量
    return new_padded_tensor

# 定义 construct_full_unpacked_stream 函数
def construct_full_unpacked_stream(
    num_real_text_tokens: Union[List[List[int]], "torch.Tensor"],
    input_stream: "torch.Tensor",
    image_tokens: List[List["torch.Tensor"]],
    batch_size: int,
    num_sub_sequences: int,
) -> List["torch.Tensor"]:
    """Takes an input_stream tensor of shape B x S x ?. For each subsequence, adds any required
    padding to account for images and then unpacks the subsequences to create a single sequence per item in the batch.
    Returns a list of tensors, one for each item in the batch."""

    # 初始化空列表，用于存储所有的单一序列
    all_bi_stream = []
    
    # 遍历 batch 中的每个索引
    for batch_index in range(batch_size):
        # 初始化空列表，用于存储每个子序列
        all_si_stream = []

        # 首先，构建完整的标记流（包括图像占位符标记）和每个子序列的损失掩码，并附加到列表中。
        # 我们使用列表而不是张量，因为每个子序列的大小不固定。
        # TODO 在以后的版本中删除此逻辑，因为不支持子序列。
        image_adjustment = image_tokens[batch_index][0]
        subsequence_stream = torch.cat([image_adjustment, input_stream[batch_index, 0]], dim=0)
        num_real_tokens = image_adjustment.shape[0] + num_real_text_tokens[batch_index][0]
        all_si_stream.append(subsequence_stream[:num_real_tokens])
        # 将所有子序列合并成一个张量，并添加到 all_bi_stream 列表中
        all_bi_stream.append(torch.cat(all_si_stream, dim=0))
    
    # 返回所有的单一序列列表
    return all_bi_stream
def _replace_string_repr_with_token_tags(prompt: str) -> str:
    # 将文本中的特定字符串替换为对应的token标签
    prompt = prompt.replace(TEXT_REPR_POINT_OPEN, TOKEN_POINT_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_POINT_CLOSE, TOKEN_POINT_CLOSE_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_OPEN, TOKEN_BBOX_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_CLOSE, TOKEN_BBOX_CLOSE_STRING)
    return prompt


def _segment_prompt_into_text_token_conversions(prompt: str) -> List:
    """
    Given a string prompt, converts the prompt into a list of TextTokenConversions.
    """
    # 将字符串提示分割成文本标记转换的列表
    prompt_text_list: List = []
    regex_pattern = re.compile(
        f"({TOKEN_BBOX_OPEN_STRING}|{TOKEN_BBOX_CLOSE_STRING}|{TOKEN_POINT_OPEN_STRING}|{TOKEN_POINT_CLOSE_STRING})"
    )
    # 通过正则表达式模式进行分割
    prompt_split = regex_pattern.split(prompt)
    for i, elem in enumerate(prompt_split):
        if len(elem) == 0 or elem in [
            TOKEN_BBOX_OPEN_STRING,
            TOKEN_BBOX_CLOSE_STRING,
            TOKEN_POINT_OPEN_STRING,
            TOKEN_POINT_CLOSE_STRING,
        ]:
            continue
        # 将非空的文本和标记标记为元组加入到列表中
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
    # 创建一个存储“text”和“is_bbox”的命名元组

    # 我们想要做到以下几点：正常对代码进行标记化 -> 当我们看到一个点或框时，使用tokenize_within_tag函数进行标记化
    # 当点或框关闭标签时，继续正常进行标记化
    # 首先，我们将点和框标签替换为它们各自的tokens
    prompt = _replace_string_repr_with_token_tags(prompt)
    # 对prompt进行标记化
    # 将prompt转换为分割列表
    prompt_text_list = _segment_prompt_into_text_token_conversions(prompt)
    transformed_prompt_tokens: List[int] = []
    for elem in prompt_text_list:
        if elem[1]:
            # 这是一个位置，我们需要对其进行标记化
            within_tag_tokenized = _transform_within_tags(elem[0], scale_factor, tokenizer)
            # 使用开放和关闭标签包围文本
            transformed_prompt_tokens.extend(within_tag_tokenized)
        else:
            transformed_prompt_tokens.extend(tokenizer(elem[0], add_special_tokens=False).input_ids)
    # 返回经过转换后的提示标记
    return transformed_prompt_tokens
# 定义一个内部函数，将给定文本中的坐标转换为无逗号的令牌
def _transform_within_tags(text: str, scale_factor: float, tokenizer) -> List[int]:
    """
    Given a bounding box of the fashion <box>1, 2, 3, 4</box> | <point>1, 2</point> This function is responsible for
    converting 1, 2, 3, 4 into tokens of 1 2 3 4 without any commas.
    """

    # 将文本转换为字符串列表
    num_int_strs = text.split(",")
    if len(num_int_strs) == 2:
        # 如果存在开放或关闭标签，则移除它们
        token_space_open_string = tokenizer.vocab[TOKEN_POINT_OPEN_STRING]
        token_space_close_string = tokenizer.vocab[TOKEN_POINT_CLOSE_STRING]
    else:
        token_space_open_string = tokenizer.vocab[TOKEN_BBOX_OPEN_STRING]
        token_space_close_string = tokenizer.vocab[TOKEN_BBOX_CLOSE_STRING]

    # 将所有数字字符串中的空格移除
    num_ints = [float(num.strip()) for num in num_int_strs]
    
    # 根据长度不同对坐标进行转换
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
    
    # 通过令牌化文本，跳过空格
    tokens = [tokenizer.vocab[str(num)] for num in num_ints_translated]
    return [token_space_open_string] + tokens + [token_space_close_string]


def _tokenize_prompts_with_image_and_batch(
    tokenizer,
    prompts: List[List[str]],
    scale_factors: Optional[List[List["torch.Tensor"]]],
    max_tokens_to_generate: int,
    max_position_embeddings: int,
    add_BOS: bool,  # 和上面的类型问题一样
    add_beginning_of_answer_token: bool,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them into a 3D tensor.
    """

    # 如果不使用工具，则在令牌化时转换坐标
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
        transformed_prompt_tokens = [[tokenizer.tokenize(prompt) for prompt in prompt_seq] for prompt_seq in prompts]

    prompts_tokens = transformed_prompt_tokens

    if add_BOS:
        bos_token = tokenizer.vocab["<s>"]
    # 如果不需要在输入序列的开头添加特殊标记，则获取结束标记的索引
    else:
        bos_token = tokenizer.vocab["|ENDOFTEXT|"]
    # 每个提示序列的开头添加开始标记，并组成新的列表
    prompts_tokens = [[[bos_token] + x for x in prompt_seq] for prompt_seq in prompts_tokens]
    if add_beginning_of_answer_token:
        # 获取回答开始标记的索引
        boa = tokenizer.vocab[BEGINNING_OF_ANSWER_STRING]
        # 只在最后一个子序列添加回答开头标记，因为那是将要完成的部分
        for token_seq in prompts_tokens:
            token_seq[-1].append(boa)

    # 现在我们有一个列表的列表，每个列表的大小都不同
    # 我们想要扩展这个列表来：
    #   - 包含需要生成的标记
    #   - 让所有序列长度相等
    # 获取提示序列的长度
    prompts_length = [[len(x) for x in prompts_tokens_seq] for prompts_tokens_seq in prompts_tokens]
    # 获取最大提示长度
    max_prompt_len: int = np.max(prompts_length)
    # 每个样本的标记数量
    samples_length = min(max_prompt_len + max_tokens_to_generate, max_position_embeddings)
    if max_prompt_len + max_tokens_to_generate > max_position_embeddings:
        logger.warning(
            f"最大子序列提示长度为{max_prompt_len} + 最大要生成的标记数{max_tokens_to_generate}",
            f"超过上下文长度{max_position_embeddings}。将尽可能生成尽可能多的标记。",
        )
    # 现在将列表的列表更新为相同的大小：samples_length。
    for prompt_tokens_seq, prompts_length_seq in zip(prompts_tokens, prompts_length):
        for prompt_tokens, prompt_length in zip(prompt_tokens_seq, prompts_length_seq):
            if len(prompt_tokens) > samples_length:
                raise ValueError("子序列提示长度超出序列长度。")
            padding_size = samples_length - prompt_length
            prompt_tokens.extend([tokenizer.vocab["|ENDOFTEXT|"]] * padding_size)

    # 现在我们有一个结构化的格式，可以转换为张量。
    prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.int64)
    prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.int64)

    return prompts_tokens_tensor, prompts_length_tensor
# 通过原始坐标和高度缩放比例计算转换后的高度坐标，四舍五入取整，转换成整型数组
def original_to_transformed_h_coords(original_coords, scale_h):
    return np.round(original_coords * scale_h).astype(np.int32)


# 通过原始坐标和宽度缩放比例计算转换后的宽度坐标，四舍五入取整，转换成整型数组
def original_to_transformed_w_coords(original_coords, scale_w):
    return np.round(original_coords * scale_w).astype(np.int32)


# 缩放点到转换后的图像，根据给定的缩放因子计算转换后的x、y坐标，返回坐标的整型列表
def scale_point_to_transformed_image(x: float, y: float, scale_factor: float) -> List[int]:
    x_scaled = original_to_transformed_w_coords(np.array([x / 2]), scale_factor)[0]
    y_scaled = original_to_transformed_h_coords(np.array([y / 2]), scale_factor)[0]
    return [x_scaled, y_scaled]


# 缩放边界框到转换后的图像，根据给定的缩放因子计算转换后的top、left、bottom、right坐标，返回坐标的整型列表
def scale_bbox_to_transformed_image(
    top: float, left: float, bottom: float, right: float, scale_factor: float
) -> List[int]:
    top_scaled = original_to_transformed_w_coords(np.array([top / 2]), scale_factor)[0]
    left_scaled = original_to_transformed_h_coords(np.array([left / 2]), scale_factor)[0]
    bottom_scaled = original_to_transformed_w_coords(np.array([bottom / 2]), scale_factor)[0]
    right_scaled = original_to_transformed_h_coords(np.array([right / 2]), scale_factor)[0]
    return [top_scaled, left_scaled, bottom_scaled, right_scaled]


class FuyuProcessor(ProcessorMixin):
    r"""
    构建一个Fuyu处理器，将一个Fuyu图像处理器和一个Llama分词器封装成一个处理器。

    [`FuyuProcessor`] 提供了 [`FuyuImageProcessor`] 和 [`LlamaTokenizerFast`] 的所有功能。查看 [`~FuyuProcessor.__call__`] 和 [`~FuyuProcessor.decode`] 获取更多信息。

    Args:
        image_processor ([`FuyuImageProcessor`]):
            图像处理器是必需的输入。
        tokenizer ([`LlamaTokenizerFast`]):
            分词器是必需的输入。
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FuyuImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_tokens_to_generate = 10
        self.max_position_embeddings = 16384  # TODO 无法从模型文件中推导出：在哪里设置？
        self.pad_token_id = 0
        self.dummy_image_index = -1
    def _left_pad_inputs_with_attention_mask(self, model_inputs: List[Dict], return_attention_mask: bool):
        # 获取 model_inputs 中 input_ids 对应的最大长度
        max_length_input_ids = max(entry["input_ids"].shape[1] for entry in model_inputs)
        # 获取 model_inputs 中 image_patches_indices 对应的最大长度
        max_length_image_patch_indices = max(entry["image_patches_indices"].shape[1] for entry in model_inputs)

        # 创建空字典 batched_inputs
        batched_inputs = {"input_ids": [], "image_patches": [], "image_patches_indices": [], "attention_mask": []}

        # 遍历 model_inputs 中的每个 entry
        for entry in model_inputs:
            # 遍历 entry 的每个键值对
            for key, tensor in entry.items():
                # 当键为 "input_ids" 时
                if key == "input_ids":
                    # 计算需要填充的 padding 的数量
                    num_padding_tokens = max_length_input_ids - tensor.shape[1]
                    # 使用填充的 padding 构建 padded_input_ids
                    padded_input_ids = torch.cat(
                        [
                            torch.full((tensor.shape[0], num_padding_tokens), self.pad_token_id, dtype=torch.long),
                            tensor,
                        ],
                        dim=1,
                    )
                    # 将 padded_input_ids 加入到 batched_inputs 字典中的 "input_ids" 对应的列表中
                    batched_inputs[key].append(padded_input_ids)

                    # 构建 attention_mask
                    attention_mask = torch.cat(
                        [torch.zeros(tensor.shape[0], num_padding_tokens, dtype=torch.long), torch.ones_like(tensor)],
                        dim=1,
                    )
                    # 将 attention_mask 加入到 batched_inputs 字典中的 "attention_mask" 对应的列表中
                    batched_inputs["attention_mask"].append(attention_mask)

                # 当键为 "image_patches" 时，不进行填充，直接将 tensor 添加到 batched_inputs 字典中的相应键对应的列表中
                elif key == "image_patches":
                    # 对于 image_patches，不需要进行填充，直接将其添加到 batched_inputs 字典中的 "image_patches" 对应的列表中
                    batched_inputs[key].append(tensor)

                # 当键为 "image_patches_indices" 时
                else:  
                    # 计算需要填充的 padding 的数量
                    num_padding_indices = max_length_image_patch_indices - tensor.shape[1]
                    # 使用填充的 padding 构建 padded_indices
                    padded_indices = torch.cat(
                        [
                            torch.full(
                                (tensor.shape[0], num_padding_indices), self.dummy_image_index, dtype=torch.long
                            ),
                            tensor,
                        ],
                        dim=1,
                    )
                    # 将 padded_indices 加入到 batched_inputs 字典中的 "image_patches_indices" 对应的列表中
                    batched_inputs[key].append(padded_indices)
        # 创建要返回的 batched_keys 列表
        batched_keys = ["input_ids", "image_patches_indices"]
        # 当需要返回 attention_mask 时，将 "attention_mask" 加入到 batched_keys 中
        if return_attention_mask:
            batched_keys.append("attention_mask")
        # 将 batched_keys 列表中的键对应的列表都合并成一个大的 tensor，并保存在 batched_inputs 字典中对应的键对应的位置
        for key in batched_keys:
            batched_inputs[key] = torch.cat(batched_inputs[key], dim=0)

        # 返回合并后的 batched_inputs
        return batched_inputs

    def get_sample_encoding(
        self,
        prompts,
        scale_factors,
        image_unpadded_heights,
        image_unpadded_widths,
        image_placeholder_id,
        image_newline_id,
        tensor_batch_images,


注释：
    ):  # 开始函数定义
        # 初始化一个全是1的张量，用于表示图片存在
        image_present = torch.ones(1, 1, 1)
        # 使用图像处理器对图像进行预处理，并返回处理后的结果
        model_image_input = self.image_processor.preprocess_with_tokenizer_info(
            image_input=tensor_batch_images,
            image_present=image_present,
            image_unpadded_h=image_unpadded_heights,
            image_unpadded_w=image_unpadded_widths,
            image_placeholder_id=image_placeholder_id,
            image_newline_id=image_newline_id,
            variable_sized=True,
        )
        # 根据图片和批处理的提示信息进行标记处理，并返回处理后的标记和长度信息
        prompt_tokens, prompts_length = _tokenize_prompts_with_image_and_batch(
            tokenizer=self.tokenizer,
            prompts=prompts,
            scale_factors=scale_factors,
            max_tokens_to_generate=self.max_tokens_to_generate,
            max_position_embeddings=self.max_position_embeddings,
            add_BOS=True,
            add_beginning_of_answer_token=True,
        )
        # 构建完全解包的标记流，用于处理图像的输入
        image_padded_unpacked_tokens = construct_full_unpacked_stream(
            num_real_text_tokens=prompts_length,
            input_stream=prompt_tokens,
            image_tokens=model_image_input["image_input_ids"],
            batch_size=1,
            num_sub_sequences=self.subsequence_length,
        )
        # 构建用于图像补丁索引的输入
        unpacked_image_patch_indices_per_batch = construct_full_unpacked_stream(
            num_real_text_tokens=prompts_length,
            input_stream=torch.full_like(prompt_tokens, -1),
            image_tokens=model_image_input["image_patch_indices_per_batch"],
            batch_size=1,
            num_sub_sequences=self.subsequence_length,
        )
        # 计算最大提示长度和最大序列长度
        max_prompt_length = max(x.shape[-1] for x in image_padded_unpacked_tokens)
        max_seq_len_batch = min(max_prompt_length + self.max_tokens_to_generate, self.max_position_embeddings)
        tokens_to_place = min(max_seq_len_batch, max(0, image_padded_unpacked_tokens[0].shape[0]))
        
        # 使用相同的打包逻辑处理图像补丁索引
        image_patch_input_indices = full_unpacked_stream_to_tensor(
            all_bi_tokens_to_place=[tokens_to_place],
            full_unpacked_stream=unpacked_image_patch_indices_per_batch,
            fill_value=-1,
            batch_size=1,
            new_seq_len=max_seq_len_batch,
            offset=0,
        )
        # 打包成张量，用于表示图像片段
        image_patches_tensor = torch.stack([img[0] for img in model_image_input["image_patches"]])
        # 构建批量编码结果
        batch_encoding = {
            "input_ids": image_padded_unpacked_tokens[0].unsqueeze(0),
            "image_patches": image_patches_tensor,
            "image_patches_indices": image_patch_input_indices,
        }
        return batch_encoding  # 返回批量编码结果
    # 定义一个方法，用于对给定文本和图片进行tokenize处理
    def __call__(
        self,
        text=None,  # 文本内容，默认为None
        images=None,  # 图片内容，默认为None
        add_special_tokens: bool = True,  # 是否添加特殊标记，默认为True
        return_attention_mask: bool = True,  # 是否返回注意力掩码，默认为True
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略，默认为False
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断策略，默认为None
        max_length: Optional[int] = None,  # 最大长度，默认为None
        stride: int = 0,  # 步长，默认为0
        pad_to_multiple_of: Optional[int] = None,  # 填充到的倍数，默认为None
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token，默认为False
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码，默认为False
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，默认为False
        return_token_type_ids: bool = False,  # 是否返回token类型id，默认为False
        return_length: bool = False,  # 是否返回长度，默认为False
        verbose: bool = True,  # 是否详细输出，默认为True
        return_tensors: Optional[Union[str, TensorType]] = None,  # 是否返回张量，默认为None
        **kwargs,  # 其他参数
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 批量解码，将所有参数转发给LlamaTokenizerFast的`~PreTrainedTokenizer.batch_decode`
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        # 解码，将所有参数转发给LlamaTokenizerFast的`~PreTrainedTokenizer.decode`
        return self.tokenizer.decode(*args, **kwargs)
```