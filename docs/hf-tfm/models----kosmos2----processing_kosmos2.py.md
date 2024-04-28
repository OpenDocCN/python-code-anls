# `.\models\kosmos2\processing_kosmos2.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Microsoft Research 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言
"""KOSMOS-2""" 的处理器类
import copy
import math
import re
from typing import List, Optional, Tuple, Union

# 导入相关模块和类
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, is_batched
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import AddedToken
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType

# 定义 BboxInput 类型
BboxInput = Union[
    List[Tuple[int, int]],
    List[Tuple[float, float, float, float]],
    List[List[Tuple[int, int]]],
    List[List[Tuple[float, float, float]]],
]

# 定义 Kosmos2Processor 类，继承 ProcessorMixin 类
class Kosmos2Processor(ProcessorMixin):
    r"""
    构建一个 KOSMOS-2 处理器，将 KOSMOS-2 图像处理器和 KOSMOS-2 分词器封装成一个单一处理器。

    [`Kosmos2Processor`] 提供了 [`CLIPImageProcessor`] 的所有功能以及 [`XLMRobertaTokenizerFast`] 的一些功能。
    有关更多信息，请参阅 [`~Kosmos2Processor.__call__`] 和 [`~Kosmos2Processor.decode`] 的文档字符串。

    Args:
        image_processor (`CLIPImageProcessor`):
            [`CLIPImageProcessor`] 的实例。图像处理器是必需的输入。
        tokenizer (`XLMRobertaTokenizerFast`):
            [`XLMRobertaTokenizerFast`] 的实例。分词器是必需的输入。
        num_patch_index_tokens (`int`, *optional*, 默认为 1024):
            表示补丁索引的令牌数量。
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("XLMRobertaTokenizer", "XLMRobertaTokenizerFast")
    # 初始化函数，接受图像处理器、分词器和补丁索引标记数作为参数
    def __init__(self, image_processor, tokenizer, num_patch_index_tokens=1024):
        # 设置分词器不返回 token 类型 ID
        tokenizer.return_token_type_ids = False

        # 定义特殊标记
        self.eod_token = "</doc>"
        self.boi_token = "<image>"
        self.eoi_token = "</image>"
        self.eoc_token = "</chunk>"
        self.eol_token = "</line>"
        self.bop_token = "<phrase>"
        self.eop_token = "</phrase>"
        self.boo_token = "<object>"
        self.eoo_token = "</object>"
        self.dom_token = "<|delimiter_of_multi_objects|>"
        self.grd_token = "<grounding>"

        # 将特殊标记组成列表
        self.tag_tokens = [
            self.eod_token,
            self.boi_token,
            self.eoi_token,
            self.eoc_token,
            self.eol_token,
            self.bop_token,
            self.eop_token,
            self.boo_token,
            self.eoo_token,
            self.dom_token,
            self.grd_token,
        ]

        # 设置补丁索引标记数
        self.num_patch_index_tokens = num_patch_index_tokens
        # 生成补丁索引标记列表
        patch_index_tokens = [f"<patch_index_{str(x).zfill(4)}>" for x in range(self.num_patch_index_tokens)]

        # 创建要添加的 token 列表
        tokens_to_add = []
        # 将特殊标记和补丁索引标记添加到 tokens_to_add 列表中
        for token in self.tag_tokens + patch_index_tokens:
            tokens_to_add.append(AddedToken(token, lstrip=True, rstrip=False, normalized=False))
        # 将 tokens_to_add 列表中的 token 添加到分词器中
        tokenizer.add_tokens(tokens_to_add)

        # 调用父类的初始化函数，传入图像处理器和分词器
        super().__init__(image_processor, tokenizer)

    # 调用函数，接受图像、文本、边界框等参数
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, List[TextInput]] = None,
        bboxes: BboxInput = None,
        num_image_tokens: Optional[int] = 64,
        first_image_token_id: Optional[int] = None,
        add_special_tokens: bool = True,
        add_eos_token: bool = False,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    def _check_bboxes_for_single_text(self, bboxes):
        """
        Check `bboxes` for a single text example. It could be
            - `None`: no bounding box associated to a text.
            - A list with each element being the bounding boxes associated to one `<phrase> ... </phrase>` pair found
              in a text. This could be:
                  - `None`: no bounding box associated to a `<phrase> ... </phrase>` pair.
                  - A tuple of 2 integers: A single bounding box specified by patch indices.
                  - A tuple of 4 float point number: A single bounding box specified by (normalized) coordinates.
                  - A list containing the above 2 tuple types: Multiple bounding boxes for a
                   `<phrase> ... </phrase>` pair.
        """
        # 如果 bboxes 为 None，则直接返回
        if bboxes is None:
            return
        # 如果 bboxes 不是列表类型，则抛出数值错误
        elif not isinstance(bboxes, list):
            raise ValueError("`bboxes` (for a single text example) should be `None` or a list.")

        # `bbox` 是单个 <phrase> </phrase> 对应的边界框
        for bbox in bboxes:
            # 如果 bbox 为 None，则继续下一次循环
            if bbox is None:
                continue
            # 如果 bbox 不是列表类型，则转换为列表
            elif not isinstance(bbox, list):
                bbox = [bbox]
            for element in bbox:
                # 如果 element 不是元组类型，或者元组长度不是 2 或 4，或者元组元素类型不符合要求，则抛出数值错误
                if not isinstance(element, tuple) or not (
                    (len(element) == 2 and all(isinstance(x, int) for x in element))
                    or (len(element) == 4 and all(isinstance(x, float) for x in element))
                ):
                    raise ValueError(
                        "Each element in `bboxes` (for a single text example) should be either `None`, a tuple containing "
                        "2 integers or 4 float point numbers, or a list containing such tuples. Also "
                        "make sure the arguments `texts` and `bboxes` passed to `preprocess_text` are both in "
                        "batches or both for a single example."
                    )

    def _preprocess_single_example(self, text, image, bboxes, img_info_tokens):
        # 去除文本两端的空格
        text = text.strip()
        if image is not None:
            # 在文本前添加 `<image> ... (fake) image tokens ... </image>`
            text = f"{img_info_tokens} {text}"

        # 在 `<phrase> phrase text </phrase>` 后添加 `<object> <patch_idx_xxxx> <patch_idx_yyy> </object>`
        text = self._insert_patch_index_tokens(text, bboxes)
        return text

    def preprocess_examples(
        self,
        texts: Union[TextInput, List[TextInput]],
        images: ImageInput = None,
        bboxes: BboxInput = None,
        num_image_tokens: Optional[int] = 64,
    ) -> Union[str, List[str]]:
        """Add image and bounding box information to `texts` as image and patch index tokens.

        Args:
            texts (`Union[TextInput, List[TextInput]]`): The texts to be processed.
            images (`ImageInput`, *optional*): The images associated to `texts`.
            bboxes (`Union[List[Tuple[int]], List[Tuple[float]], List[List[Tuple[int]]], List[List[Tuple[float]]]]`, *optional*):
                The bounding bboxes associated to `texts`.
            num_image_tokens (`int`, *optional*, defaults to 64):
                The number of image tokens (used as latent queries). This should corresponds to the `latent_query_num`
                attribute in `Kosmos2Config`.

        Returns:
            `Union[TextInput, List[TextInput]]`: The processed texts with image and patch index tokens.
        """
        # These are fake `<image>` tokens enclosed between (the actual) `<image>` token and `</image>`.
        img_tokens = [self.boi_token] * num_image_tokens  # 创建一个包含指定数量的 `<image>` tokens 的列表
        img_info_tokens = " ".join([self.boi_token] + img_tokens + [self.eoi_token])  # 将 `<image>` token 和 `</image>` token 以及 img_tokens 中的内容连接成字符串

        # make batch to simplify processing logic
        batched = True  # 初始化 batched 为 True
        if isinstance(texts, str):  # 如果 texts 是字符串
            batched = False  # 将 batched 设置为 False
            texts = [texts]  # 将单个字符串放入列表中进行处理

        if images is None:  # 如果 images 为 None
            images = [None] * len(texts)  # 创建一个与 texts 长度相同的包含 None 的列表
        elif not is_batched(images):  # 如果 images 不是 batched
            images = [images]  # 将单个图片放入列表中进行处理
        if len(texts) != len(images):  # 如果 texts 和 images 长度不相等
            raise ValueError(
                f"The number of examples in `texts` and `images` should be the same. Got {len(texts)} v.s. {len(images)} instead."
            )  # 抛出数值错误异常

        if not batched:  # 如果不是 batched 的话
            self._check_bboxes_for_single_text(bboxes)  # 检查单个文本的边界框
            bboxes = [bboxes]  # 将单个边界框放入列表中进行处理
        elif bboxes is not None:  # 如果 bboxes 不为 None
            if not isinstance(bboxes, list):  # 如果 bboxes 不是列表
                raise ValueError("`bboxes` should be `None` or a list (as a batch) when `texts` is passed as a batch.")  # 抛出数值错误异常
            for x in bboxes:  # 遍历 bboxes
                self._check_bboxes_for_single_text(x)  # 检查单个文本的边界框
        else:  # 如果 bboxes 为 None
            bboxes = [None] * len(texts)  # 创建一个与 texts 长度相同的包含 None 的列表

        if len(bboxes) != len(texts):  # 如果 bboxes 和 texts 长度不相等
            raise ValueError(
                f"The number of examples in `texts` and `bboxes` should be the same. Got {len(texts)} v.s. {len(bboxes)} instead."
            )  # 抛出数值错误异常

        result = [
            self._preprocess_single_example(text, image, bbox, img_info_tokens)
            for text, image, bbox in zip(texts, images, bboxes)
        ]  # 对每个文本、图片和边界框进行预处理并组成列表
        # un-batch if necessary
        if not batched:  # 如果不是 batched 的话
            result = result[0]  # 将结果设为列表中的第一个元素

        return result  # 返回结果

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.batch_decode with BertTokenizerFast->PreTrainedTokenizer
    # 将所有参数转发给 PreTrainedTokenizer 的 batch_decode 方法
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 从 transformers.models.blip.processing_blip.BlipProcessor.decode 中拷贝，使用 BertTokenizerFast 替换为 PreTrainedTokenizer
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 对生成的文本进行后处理，清理并提取实体
    def post_process_generation(self, text, cleanup_and_extract=True):
        caption = text.split(self.eoi_token)[-1]
        if cleanup_and_extract:
            return clean_text_and_extract_entities_with_bboxes(caption)
        return caption

    @property
    # 从 transformers.models.blip.processing_blip.BlipProcessor.model_input_names 中拷贝
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
    # 将patch索引标记插入文本中的指定位置
    def _insert_patch_index_tokens(self, text: str, bboxes: Union[List[Tuple[int]], List[Tuple[float]]]) -> str:
        # 如果bboxes为None或长度为0，则返回原文本
        if bboxes is None or len(bboxes) == 0:
            return text

        # 查找文本中所有匹配的<phrase> ... </phrase>对
        matched_phrases = list(re.finditer(r"<phrase>.+?</phrase>", string=text))
        # 如果匹配的数量不等于bboxes的数量，则抛出异常
        if len(matched_phrases) != len(bboxes):
            raise ValueError(
                f"The number of elements in `bboxes` should be the same as the number of `<phrase> ... </phrase>` pairs in `text`. Got {len(matched_phrases)} v.s. {len(bboxes)} instead."
            )

        # 插入对象的patch索引标记到匹配的<phrase> ... </phrase>对中
        curr_pos = 0
        buffer = []
        for matched, bbox in zip(matched_phrases, bboxes):
            _, end = matched.span()
            buffer.append(text[curr_pos:end])
            curr_pos = end
            # 如果bbox为None，则跳过当前短语
            if bbox is None:
                continue
            # 如果bbox是元组，则将其转换为列表
            if isinstance(bbox, tuple):
                bbox = [bbox]
            patch_index_strings = []
            # 检查bbox中是否有None值
            if not all(box is not None for box in bbox):
                raise ValueError(
                    "The multiple bounding boxes for a single phrase should not contain any `None` value."
                )
            for box in bbox:
                # 将bbox转换为patch索引标记
                patch_index_1, patch_index_2 = self._convert_bbox_to_patch_index_tokens(box)
                patch_index_strings.append(f"{patch_index_1} {patch_index_2}")
            # 如果patch索引标记为空，则继续下一个短语
            if len(patch_index_strings) == 0:
                continue
            position_str = " <|delimiter_of_multi_objects|> ".join(patch_index_strings)
            buffer.append(f"<object> {position_str} </object>")
        # 处理剩余文本
        if curr_pos < len(text):
            buffer.append(text[curr_pos:])

        # 合并buffer中的文本片段为最终文本
        text = "".join(buffer)
        return text

    # 将bbox转换为patch索引标记
    def _convert_bbox_to_patch_index_tokens(
        self, bbox: Union[Tuple[int, int], Tuple[float, float, float, float]]
    ) -> Tuple[str, str]:
        # 如果bbox长度为2，则bbox已经是patch索引标记
        if len(bbox) == 2:
            idx_1, idx_2 = bbox
        # 如果bbox的长度不为2，则将其转换为patch索引标记
        else:
            # 使用self.tokenizer获取num_patches_per_side
            num_patches_per_side = int(math.sqrt(self.num_patch_index_tokens))
            idx_1, idx_2 = coordinate_to_patch_index(bbox, num_patches_per_side)

        # 根据patch索引生成token
        token_1 = f"<patch_index_{str(idx_1).zfill(4)}>"
        token_2 = f"<patch_index_{str(idx_2).zfill(4)}>"

        return token_1, token_2
# 将坐标转换为补丁索引的函数
def coordinate_to_patch_index(bbox: Tuple[float, float, float, float], num_patches_per_side: int) -> Tuple[int, int]:
    """
    将边界框转换为补丁索引的函数

    参数:
        bbox (`Tuple[float, float, float, float]`):
            边界框的四个坐标，格式为(x1，y1，x2，y2)，表示边框的左上角和右下角。需满足x2 > x1和y2 > y1。
        num_patches_per_side (`int`): 每边的补丁数量。

    返回:
        `Tuple[int, int]`: 代表左上角补丁和右下角补丁的补丁索引的成对数值。
    """
    # 解构输入的边界框坐标
    (x1, y1, x2, y2) = bbox

    # 如果bbox坐标格式不正确，抛出值错误异常
    if not (x2 > x1 and y2 > y1):
        raise ValueError("The coordinates in `bbox` should be `(x1, y1, x2, y2)` with `x2 > x1` and `y2 > y1`.")

    # 计算左上角补丁索引的x和y坐标
    ul_x = math.floor(x1 * num_patches_per_side)
    ul_y = math.floor(y1 * num_patches_per_side)

    # 计算右下角补丁索引的x和y坐标
    lr_x = math.ceil(x2 * num_patches_per_side - 1)
    lr_y = math.ceil(y2 * num_patches_per_side - 1)

    # 计算左上角和右下角补丁的索引
    ul_idx = ul_y * num_patches_per_side + ul_x
    lr_idx = lr_y * num_patches_per_side + lr_x

    # 返回左上角和右下角补丁的索引
    return ul_idx, lr_idx


# 从https://github.com/microsoft/unilm/blob/97e4923e97d3ee10b57e97013556e3fd0d207a9b/kosmos-2/demo/decode_string.py#L35C1-L75C38复制而来（格式修改）
def patch_index_to_coordinate(ul_idx: int, lr_idx: int, num_patches_per_side: int):
    """
    给定一个边界框的左上角和右下角补丁的网格索引，返回边界框的归一化坐标。

    参数:
        ul_idx (`int`): 与边界框左上角对应的网格单元的索引。
        lr_idx (`int`): 与边界框右下角对应的网格单元的索引。
        num_patches_per_side (`int`): 每边的补丁数量。

    返回:
        `Tuple[float]`: 边界框的归一化坐标，格式为(x1, y1, x2, y2)。
    """
    # 计算网格单元的大小
    cell_size = 1.0 / num_patches_per_side

    # 计算边界框左上角和右下角的x和y坐标索引
    ul_x = ul_idx % num_patches_per_side
    ul_y = ul_idx // num_patches_per_side

    lr_x = lr_idx % num_patches_per_side
    lr_y = lr_idx // num_patches_per_side

    # 计算边界框的归一化坐标
    if ul_idx == lr_idx:
        x1 = ul_x * cell_size
        y1 = ul_y * cell_size
        x2 = lr_x * cell_size + cell_size
        y2 = lr_y * cell_size + cell_size
    elif ul_x == lr_x or ul_y == lr_y:
        x1 = ul_x * cell_size
        y1 = ul_y * cell_size
        x2 = lr_x * cell_size + cell_size
        y2 = lr_y * cell_size + cell_size
    # 如果不是第一种情况，则计算矩形的中心坐标
    else:
        # 计算矩形左上角点的中心坐标
        x1 = ul_x * cell_size + cell_size / 2
        y1 = ul_y * cell_size + cell_size / 2
        # 计算矩形右下角点的中心坐标
        x2 = lr_x * cell_size + cell_size / 2
        y2 = lr_y * cell_size + cell_size / 2

    # 返回矩形的四个坐标
    return x1, y1, x2, y2
# 从给定的文本中提取包含在其中的实体，并返回其边界框以图块索引的形式给出
# 此函数仅用于 `clean_text_and_extract_entities_with_bboxes` 中，包括进一步处理，包括转换为标准化坐标和清理空格字符
def extract_entities_with_patch_indices(text):
    # 匹配所需格式的正则表达式模式
    pattern = r"(?:(<phrase>([^<]+)</phrase>))?<object>((?:<patch_index_\d+><patch_index_\d+><|delimiter_of_multi_objects|>)*<patch_index_\d+><patch_index_\d+>)</object>"

    # 在给定字符串中查找所有匹配项
    matches = re.finditer(pattern, text)
    
    # 初始化一个空列表，用于存储有效的 patch_index 组合
    entities_with_patch_indices = []

    for match in matches:
        # `phrase` 的跨度，在 <phrase> 和 </phrase> 之间的部分
        span = match.span(2)
        phrase_tag, phrase, match_content = match.groups()
        if not phrase_tag:
            phrase = None
            # 我们取得 `<object>` 的起始位置
            span = (match.span(0)[0], match.span(0)[0])

        # 通过分隔符拆分 match_content 以获取单独的 patch_index 对
        patch_index_pairs = match_content.split("<|delimiter_of_multi_objects|>")

        entity_bboxes = []
        for pair in patch_index_pairs:
            # 从 patch_index 对中提取 xxxx 和 yyyy 值
            x = re.search(r"<patch_index_(\d+)>", pair)
            y = re.search(r"<patch_index_(\d+)>", pair[1:])

            if x and y:
                if phrase:
                    entity_bboxes.append((int(x.group(1)), int(y.group(1)))
                else:
                    entity_bboxes.append((int(x.group(1)), int(y.group(1)))

        if phrase:
            entities_with_patch_indices.append((phrase, span, entity_bboxes))
        else:
            for bbox in entity_bboxes:
                # 伪造的实体名称
                entity = f"<patch_index_{bbox[0]}><patch_index_{bbox[1]}>"
                entities_with_patch_indices.append((entity, span, [bbox]))

    return entities_with_patch_indices


# 调整 `text` 中实体的位置，使其相对于移除特殊字段后的文本
def adjust_entity_positions(entity, text):
    entity_name, (start, end) = entity
    # 计算去除特定字段（标签标记，补丁索引标记等）的字符串长度
    # 计算调整后的起始位置，去除特定字段后的文本长度
    adjusted_start = len(re.sub("<.*?>", "", text[:start]))
    # 计算调整后的结束位置，去除特定字段后的文本长度
    adjusted_end = len(re.sub("<.*?>", "", text[:end]))
    # 组成调整后的实体信息，包括实体名称和调整后的起始/结束位置
    adjusted_entity = (entity_name, (adjusted_start, adjusted_end))
    # 返回调整后的实体信息
    return adjusted_entity
# 清理文本周围和其中的实体的空格
def _cleanup_spaces(text, entities):
    """Remove the spaces around the text and the entities in it."""
    # 去除文本两侧的空格
    new_text = text.strip()
    # 计算文本开头的空格数量
    leading_spaces = len(text) - len(text.lstrip())

    new_entities = []
    for entity_name, (start, end), bboxes in entities:
        # 计算实体名称开头的空格数量
        entity_name_leading_spaces = len(entity_name) - len(entity_name.lstrip())
        # 计算实体名称末尾的空格数量
        entity_name_trailing_spaces = len(entity_name) - len(entity_name.rstrip())

        # 调整实体起始和结束位置，并去除实体名两侧的空格
        start = start - leading_spaces + entity_name_leading_spaces
        end = end - leading_spaces - entity_name_trailing_spaces
        entity_name = entity_name.strip()

        new_entities.append((entity_name, (start, end), bboxes))

    return new_text, new_entities


# 从https://github.com/microsoft/unilm/blob/97e4923e97d3ee10b57e97013556e3fd0d207a9b/kosmos-2/demo/decode_string.py#L77-L87 复制过来（格式进行了修改）
def clean_text_and_extract_entities_with_bboxes(text, num_patches_per_side=32):
    """Remove the tag tokens from `text`, extract entities in it with some cleaning up of white characters.

    Examples:

    ```python
    >>> text = "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>."
    >>> clean_text, entities = clean_text_and_extract_entities_with_bboxes(text)
    >>> clean_text
    'An image of a snowman warming himself by a fire.'

    >>> entities
    [('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]
    ```py"""
    # 去除特殊字段（标签令牌、补丁索引令牌等）
    processed_text = re.sub("<.*?>", "", text)

    entities_with_patch_indices = extract_entities_with_patch_indices(text)
    entities = []
    for item in entities_with_patch_indices:
        entity, bboxes = item[0:2], item[2]
        adjusted_entity = adjust_entity_positions(entity, text)
        bboxes_in_coords = [patch_index_to_coordinate(bbox[0], bbox[1], num_patches_per_side) for bbox in bboxes]

        entities.append(adjusted_entity + (bboxes_in_coords,))

    return _cleanup_spaces(processed_text, entities)
```