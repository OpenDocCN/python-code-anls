# `.\models\kosmos2\processing_kosmos2.py`

```py
# coding=utf-8
# 上面是指定文件编码为 UTF-8

# 版权声明，版权归 Microsoft Research 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证版本 2.0 使用本文件，除非符合许可证，否则不得使用本文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件，不提供任何形式的担保或条件
# 有关特定语言的权限，请参阅许可证文档

"""KOSMOS-2 的处理器类。"""

import copy  # 导入 copy 模块，用于复制对象
import math  # 导入 math 模块，用于数学运算
import re  # 导入 re 模块，用于正则表达式操作
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的模块

from ...image_processing_utils import BatchFeature  # 导入批量特征处理工具
from ...image_utils import ImageInput, is_batched  # 导入图像输入和批处理检查函数
from ...processing_utils import ProcessorMixin  # 导入处理器混合类
from ...tokenization_utils import AddedToken  # 导入添加的标记类
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy  # 导入批编码相关类和策略
from ...utils import TensorType  # 导入张量类型

BboxInput = Union[
    List[Tuple[int, int]],  # BboxInput 可以是 (int, int) 元组的列表
    List[Tuple[float, float, float, float]],  # 或者是浮点数 (float, float, float, float) 元组的列表
    List[List[Tuple[int, int]]],  # 或者是 (int, int) 元组列表的列表
    List[List[Tuple[float, float, float]]],  # 或者是浮点数 (float, float, float) 元组列表的列表
]

class Kosmos2Processor(ProcessorMixin):
    """
    构造一个 KOSMOS-2 处理器，将 KOSMOS-2 图像处理器和 KOSMOS-2 分词器封装成一个单一的处理器。

    [`Kosmos2Processor`] 提供了 [`CLIPImageProcessor`] 的所有功能以及 [`XLMRobertaTokenizerFast`] 的一些功能。
    更多信息请参阅 [`~Kosmos2Processor.__call__`] 和 [`~Kosmos2Processor.decode`] 的文档字符串。

    Args:
        image_processor (`CLIPImageProcessor`):
            一个 [`CLIPImageProcessor`] 实例。图像处理器是必需的输入。
        tokenizer (`XLMRobertaTokenizerFast`):
            一个 [`XLMRobertaTokenizerFast`] 实例。分词器是必需的输入。
        num_patch_index_tokens (`int`, *optional*, 默认为 1024):
            表示补丁索引的标记数。
    """

    attributes = ["image_processor", "tokenizer"]  # 定义类的属性列表
    image_processor_class = "CLIPImageProcessor"  # 图像处理器的类名
    tokenizer_class = ("XLMRobertaTokenizer", "XLMRobertaTokenizerFast")  # 分词器的类名
        def __init__(self, image_processor, tokenizer, num_patch_index_tokens=1024):
            # 设置 tokenizer 不返回 token 类型 ID
            tokenizer.return_token_type_ids = False

            # 定义结束文档标记
            self.eod_token = "</doc>"

            # 定义图像开始标记和结束标记
            self.boi_token = "<image>"
            self.eoi_token = "</image>"

            # 定义块结束和行结束标记
            self.eoc_token = "</chunk>"
            self.eol_token = "</line>"

            # 定义短语开始和结束标记
            self.bop_token = "<phrase>"
            self.eop_token = "</phrase>"

            # 定义对象开始和结束标记
            self.boo_token = "<object>"
            self.eoo_token = "</object>"

            # 定义多对象分隔符结束标记
            self.dom_token = "<|delimiter_of_multi_objects|>"

            # 定义图像 grounding 标记
            self.grd_token = "<grounding>"

            # 将所有标记放入列表中
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

            # 设置索引 token 的数量
            self.num_patch_index_tokens = num_patch_index_tokens
            # 生成索引 token 列表，格式为 "<patch_index_0000>" 到 "<patch_index_1023>"
            patch_index_tokens = [f"<patch_index_{str(x).zfill(4)}>" for x in range(self.num_patch_index_tokens)]

            # 创建要添加的 token 列表
            tokens_to_add = []
            # 将所有标记和索引 token 添加为 AddedToken 对象到 tokenizer 中
            for token in self.tag_tokens + patch_index_tokens:
                tokens_to_add.append(AddedToken(token, lstrip=True, rstrip=False, normalized=False))
            tokenizer.add_tokens(tokens_to_add)

            # 调用父类初始化方法，传递图像处理器和 tokenizer
            super().__init__(image_processor, tokenizer)

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
        if bboxes is None:
            return  # 如果 bboxes 是 None，直接返回，表示没有边界框与文本关联

        elif not isinstance(bboxes, list):
            raise ValueError("`bboxes` (for a single text example) should be `None` or a list.")
            # 如果 bboxes 不是 list 类型，则引发 ValueError 异常，说明它应该是 None 或者一个列表

        # `bbox` is the bounding boxes for a single <phrase> </phrase> pair
        for bbox in bboxes:
            if bbox is None:
                continue  # 如果 bbox 是 None，则跳过当前循环，继续下一个 bbox

            elif not isinstance(bbox, list):
                bbox = [bbox]  # 如果 bbox 不是 list 类型，则转换成单元素的列表

            for element in bbox:
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
                    # 检查每个 element 是否符合要求，如果不符合，则引发 ValueError 异常

    def _preprocess_single_example(self, text, image, bboxes, img_info_tokens):
        text = text.strip()  # 去除 text 的首尾空白字符
        if image is not None:
            # 在文本前添加 `<image> ... (fake) image tokens ... </image>`
            text = f"{img_info_tokens} {text}"

        # 在 `<phrase> phrase text </phrase>` 后面添加 `<object> <patch_idx_xxxx> <patch_idx_yyy> </object>`
        text = self._insert_patch_index_tokens(text, bboxes)
        return text
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
        # 创建一个包含指定数量 `<image>` token 的列表，并将它们用空格分隔成一个字符串
        img_tokens = [self.boi_token] * num_image_tokens
        img_info_tokens = " ".join([self.boi_token] + img_tokens + [self.eoi_token])

        # make batch to simplify processing logic
        # 如果 texts 是单个字符串，则转换成单元素列表
        batched = True
        if isinstance(texts, str):
            batched = False
            texts = [texts]

        # 如果 images 为 None，则将其初始化为与 texts 相同长度的 None 列表
        if images is None:
            images = [None] * len(texts)
        # 如果 images 不是批量输入，则转换为单元素列表
        elif not is_batched(images):
            images = [images]
        # 检查 texts 和 images 的数量是否相同，否则引发 ValueError
        if len(texts) != len(images):
            raise ValueError(
                f"The number of examples in `texts` and `images` should be the same. Got {len(texts)} v.s. {len(images)} instead."
            )

        # 如果 texts 不是批量输入，则检查单个文本的 bboxes 格式
        if not batched:
            self._check_bboxes_for_single_text(bboxes)
            bboxes = [bboxes]
        # 如果 texts 是批量输入且 bboxes 不为 None，则检查 bboxes 的格式
        elif bboxes is not None:
            if not isinstance(bboxes, list):
                raise ValueError("`bboxes` should be `None` or a list (as a batch) when `texts` is passed as a batch.")
            for x in bboxes:
                self._check_bboxes_for_single_text(x)
        # 如果 bboxes 为 None，则初始化为与 texts 相同长度的 None 列表
        else:
            bboxes = [None] * len(texts)

        # 检查 texts 和 bboxes 的数量是否相同，否则引发 ValueError
        if len(bboxes) != len(texts):
            raise ValueError(
                f"The number of examples in `texts` and `bboxes` should be the same. Got {len(texts)} v.s. {len(bboxes)} instead."
            )

        # 对每个文本、对应的图片和边界框进行预处理，返回结果列表
        result = [
            self._preprocess_single_example(text, image, bbox, img_info_tokens)
            for text, image, bbox in zip(texts, images, bboxes)
        ]
        # 如果 texts 不是批量输入，则将结果转换为单个元素
        # 反之，如果是批量输入，则保持结果列表形式
        if not batched:
            result = result[0]

        # 返回处理后的结果列表或单个文本
        return result

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.batch_decode with BertTokenizerFast->PreTrainedTokenizer
    # 将所有参数转发给 PreTrainedTokenizer 的 batch_decode 方法
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 从 transformers.models.blip.processing_blip.BlipProcessor.decode 复制代码，并将 BertTokenizerFast->PreTrainedTokenizer
    # 将所有参数转发给 PreTrainedTokenizer 的 decode 方法
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 对生成的文本进行后处理，可以清理文本并提取实体及其边界框
    def post_process_generation(self, text, cleanup_and_extract=True):
        caption = text.split(self.eoi_token)[-1]
        if cleanup_and_extract:
            return clean_text_and_extract_entities_with_bboxes(caption)
        return caption

    @property
    # 从 transformers.models.blip.processing_blip.BlipProcessor.model_input_names 复制代码
    # 返回模型输入的名称列表，包括 Tokenizer 和图像处理器的输入名称，并去除重复项
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
    # 将补丁索引标记插入文本中的指定短语区域
    def _insert_patch_index_tokens(self, text: str, bboxes: Union[List[Tuple[int]], List[Tuple[float]]]) -> str:
        # 如果未提供边界框或边界框列表为空，则直接返回原始文本
        if bboxes is None or len(bboxes) == 0:
            return text

        # 找出文本中所有匹配的 `<phrase>...</phrase>` 对
        matched_phrases = list(re.finditer(r"<phrase>.+?</phrase>", string=text))
        # 检查匹配到的短语对数与边界框数量是否相等，若不相等则引发异常
        if len(matched_phrases) != len(bboxes):
            raise ValueError(
                f"The number of elements in `bboxes` should be the same as the number of `<phrase> ... </phrase>` pairs in `text`. Got {len(matched_phrases)} v.s. {len(bboxes)} instead."
            )

        # 插入对象的补丁索引标记到找到的 `<phrase>...</phrase>` 对中
        curr_pos = 0
        buffer = []
        for matched, bbox in zip(matched_phrases, bboxes):
            _, end = matched.span()
            buffer.append(text[curr_pos:end])
            curr_pos = end

            # 如果边界框为 None，则跳过当前短语的处理
            if bbox is None:
                continue

            # 如果边界框是单个元组，则转换为列表以便处理
            if isinstance(bbox, tuple):
                bbox = [bbox]

            # 检查边界框列表中是否有 None 值，若有则引发异常
            if not all(box is not None for box in bbox):
                raise ValueError(
                    "The multiple bounding boxes for a single phrase should not contain any `None` value."
                )

            patch_index_strings = []
            # 对于每个边界框，将其转换为补丁索引标记，并构建标记字符串列表
            for box in bbox:
                patch_index_1, patch_index_2 = self._convert_bbox_to_patch_index_tokens(box)
                patch_index_strings.append(f"{patch_index_1} {patch_index_2}")

            # 如果标记字符串列表为空，则跳过当前短语的处理
            if len(patch_index_strings) == 0:
                continue

            # 将补丁索引标记字符串插入到 `<object>...</object>` 标签中
            position_str = " <|delimiter_of_multi_objects|> ".join(patch_index_strings)
            buffer.append(f"<object> {position_str} </object>")

        # 处理剩余的文本部分并将其添加到缓冲区中
        if curr_pos < len(text):
            buffer.append(text[curr_pos:])

        # 将缓冲区中的文本片段合并为最终的修改后的文本
        text = "".join(buffer)
        return text

    # 将边界框转换为对应的补丁索引标记
    def _convert_bbox_to_patch_index_tokens(
        self, bbox: Union[Tuple[int, int], Tuple[float, float, float, float]]
    ) -> Tuple[str, str]:
        # 如果边界框长度为 2，则表示已经是补丁索引标记，直接使用
        if len(bbox) == 2:
            idx_1, idx_2 = bbox
        # 否则，根据 (归一化的) 坐标计算对应的补丁索引标记
        else:
            # 使用 self.tokenizer 获取 num_patches_per_side
            num_patches_per_side = int(math.sqrt(self.num_patch_index_tokens))
            idx_1, idx_2 = coordinate_to_patch_index(bbox, num_patches_per_side)

        # 构建补丁索引标记字符串并返回
        token_1 = f"<patch_index_{str(idx_1).zfill(4)}>"
        token_2 = f"<patch_index_{str(idx_2).zfill(4)}>"

        return token_1, token_2
# 将边界框转换为一对补丁索引。
def coordinate_to_patch_index(bbox: Tuple[float, float, float, float], num_patches_per_side: int) -> Tuple[int, int]:
    """Convert a bounding box to a pair of patch indices.

    Args:
        bbox (`Tuple[float, float, float, float]`):
            The 4 coordinates of the bounding box, with the format being (x1, y1, x2, y2) specifying the upper-left and
            lower-right corners of the box. It should have x2 > x1 and y2 > y1.
        num_patches_per_side (`int`): the number of patches along each side.

    Returns:
        `Tuple[int, int]`: A pair of patch indices representing the upper-left patch and lower-right patch.
    """
    (x1, y1, x2, y2) = bbox  # 解包边界框坐标

    if not (x2 > x1 and y2 > y1):  # 检查边界框坐标是否有效
        raise ValueError("The coordinates in `bbox` should be `(x1, y1, x2, y2)` with `x2 > x1` and `y2 > y1`.")

    ul_x = math.floor(x1 * num_patches_per_side)  # 计算上左角补丁的 x 索引
    ul_y = math.floor(y1 * num_patches_per_side)  # 计算上左角补丁的 y 索引

    lr_x = math.ceil(x2 * num_patches_per_side - 1)  # 计算下右角补丁的 x 索引
    lr_y = math.ceil(y2 * num_patches_per_side - 1)  # 计算下右角补丁的 y 索引

    ul_idx = ul_y * num_patches_per_side + ul_x  # 计算上左角补丁的索引
    lr_idx = lr_y * num_patches_per_side + lr_x  # 计算下右角补丁的索引

    return ul_idx, lr_idx  # 返回补丁索引对


# 从 https://github.com/microsoft/unilm/blob/97e4923e97d3ee10b57e97013556e3fd0d207a9b/kosmos-2/demo/decode_string.py#L35C1-L75C38 复制（格式修改）
def patch_index_to_coordinate(ul_idx: int, lr_idx: int, num_patches_per_side: int):
    """
    Given a grid of length `num_patches_per_side` and the indices of the upper-left and lower-right corners of a
    bounding box, returns the normalized coordinates of the bounding box, in the form (x1, y1, x2, y2).

    Args:
        ul_idx (`int`): the index of the grid cell that corresponds to the upper-left corner of the bounding box.
        lr_idx (`int`): the index of the grid cell that corresponds to the lower-right corner of the bounding box.
        num_patches_per_side (`int`): the number of patches along each side.

    Returns:
        `Tuple[float]`: the normalized coordinates of the bounding box, in the form (x1, y1, x2, y2).
    """
    # Compute the size of each cell in the grid
    cell_size = 1.0 / num_patches_per_side  # 计算网格中每个单元格的大小

    # Compute the x and y indices of the upper-left and lower-right corners of the bounding box
    ul_x = ul_idx % num_patches_per_side  # 计算上左角补丁的 x 索引
    ul_y = ul_idx // num_patches_per_side  # 计算上左角补丁的 y 索引

    lr_x = lr_idx % num_patches_per_side  # 计算下右角补丁的 x 索引
    lr_y = lr_idx // num_patches_per_side  # 计算下右角补丁的 y 索引

    # Compute the normalized coordinates of the bounding box
    if ul_idx == lr_idx:
        x1 = ul_x * cell_size  # 左上角 x 坐标
        y1 = ul_y * cell_size  # 左上角 y 坐标
        x2 = lr_x * cell_size + cell_size  # 右下角 x 坐标
        y2 = lr_y * cell_size + cell_size  # 右下角 y 坐标
    elif ul_x == lr_x or ul_y == lr_y:
        x1 = ul_x * cell_size  # 左上角 x 坐标
        y1 = ul_y * cell_size  # 左上角 y 坐标
        x2 = lr_x * cell_size + cell_size  # 右下角 x 坐标
        y2 = lr_y * cell_size + cell_size  # 右下角 y 坐标
    # 如果条件不满足，执行以下语句
    else:
        # 计算矩形左上角点的 x 坐标
        x1 = ul_x * cell_size + cell_size / 2
        # 计算矩形左上角点的 y 坐标
        y1 = ul_y * cell_size + cell_size / 2
        # 计算矩形右下角点的 x 坐标
        x2 = lr_x * cell_size + cell_size / 2
        # 计算矩形右下角点的 y 坐标
        y2 = lr_y * cell_size + cell_size / 2

    # 返回计算得到的四个坐标值作为元组
    return x1, y1, x2, y2
# 从给定的文本中提取带有补丁索引的实体信息
def extract_entities_with_patch_indices(text):
    """Extract entities contained in `text`. The bounding bboxes is given in the form of patch indices.

    This function is only intended to be used within `clean_text_and_extract_entities_with_bboxes` where further
    processing happens, including converting to normalized coordinates and whitespace character cleaning up.

    Examples:

    ```
    >>> text = "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>."
    >>> entities = extract_entities_with_patch_indices(text)
    >>> entities
    [(' a snowman', (31, 41), [(44, 863)]), (' a fire', (130, 137), [(5, 911)])]
    ```"""

    # 匹配所需格式的正则表达式模式
    pattern = r"(?:(<phrase>([^<]+)</phrase>))?<object>((?:<patch_index_\d+><patch_index_\d+><|delimiter_of_multi_objects|>)*<patch_index_\d+><patch_index_\d+>)</object>"

    # 在给定的文本中找到所有匹配项
    matches = re.finditer(pattern, text)

    # 初始化一个空列表，用于存储有效的补丁索引组合
    entities_with_patch_indices = []

    for match in matches:
        # 获取 `<phrase>` 标签之间的文本范围
        span = match.span(2)
        phrase_tag, phrase, match_content = match.groups()
        if not phrase_tag:
            phrase = None
            # 如果没有 `<phrase>` 标签，使用 `<object>` 的起始位置作为文本范围起始
            span = (match.span(0)[0], match.span(0)[0])

        # 使用特定分隔符拆分 match_content 以获取单个补丁索引对
        patch_index_pairs = match_content.split("<|delimiter_of_multi_objects|>")

        entity_bboxes = []
        for pair in patch_index_pairs:
            # 从补丁索引对中提取 xxxx 和 yyyy 的值
            x = re.search(r"<patch_index_(\d+)>", pair)
            y = re.search(r"<patch_index_(\d+)>", pair[1:])

            if x and y:
                if phrase:
                    entity_bboxes.append((int(x.group(1)), int(y.group(1))))
                else:
                    entity_bboxes.append((int(x.group(1)), int(y.group(1))))

        if phrase:
            entities_with_patch_indices.append((phrase, span, entity_bboxes))
        else:
            for bbox in entity_bboxes:
                # 构造一个虚假的实体名称
                entity = f"<patch_index_{bbox[0]}><patch_index_{bbox[1]}>"
                entities_with_patch_indices.append((entity, span, [bbox]))

    return entities_with_patch_indices


def adjust_entity_positions(entity, text):
    """Adjust the positions of the entities in `text` to be relative to the text with special fields removed."""
    entity_name, (start, end) = entity
    # 计算去除特殊字段（标签标记、补丁索引标记等）后的字符串长度
    adjusted_start = len(re.sub("<.*?>", "", text[:start]))
    # 计算去除特殊字段后，起始到结束位置之间的字符串长度
    adjusted_end = len(re.sub("<.*?>", "", text[:end]))
    # 构建调整后的实体信息元组，包括实体名称和调整后的起始、结束位置
    adjusted_entity = (entity_name, (adjusted_start, adjusted_end))
    # 返回调整后的实体信息元组
    return adjusted_entity
# 从文本中清除周围的空格和其中的实体
def _cleanup_spaces(text, entities):
    # 去除文本两侧的空格
    new_text = text.strip()
    # 计算文本开头的空格数量
    leading_spaces = len(text) - len(text.lstrip())

    # 处理实体列表中的每一个实体
    new_entities = []
    for entity_name, (start, end), bboxes in entities:
        # 计算实体名称开头的空格数量
        entity_name_leading_spaces = len(entity_name) - len(entity_name.lstrip())
        # 计算实体名称末尾的空格数量
        entity_name_trailing_spaces = len(entity_name) - len(entity_name.rstrip())

        # 调整实体的起始和结束位置，考虑到文本开头的空格
        start = start - leading_spaces + entity_name_leading_spaces
        end = end - leading_spaces - entity_name_trailing_spaces
        # 去除实体名称两侧的空格
        entity_name = entity_name.strip()

        # 将处理后的实体添加到新的实体列表中
        new_entities.append((entity_name, (start, end), bboxes))

    # 返回处理后的文本和实体列表
    return new_text, new_entities


# 从 https://github.com/microsoft/unilm/blob/97e4923e97d3ee10b57e97013556e3fd0d207a9b/kosmos-2/demo/decode_string.py#L77-L87 处复制并稍作格式修改
def clean_text_and_extract_entities_with_bboxes(text, num_patches_per_side=32):
    """从 `text` 中删除标签标记，提取其中的实体并清除一些空白字符。

    示例：

    ```
    >>> text = "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>."
    >>> clean_text, entities = clean_text_and_extract_entities_with_bboxes(text)
    >>> clean_text
    'An image of a snowman warming himself by a fire.'

    >>> entities
    [('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]
    ```"""
    # 删除特殊字段（标签标记、补丁索引标记等）
    processed_text = re.sub("<.*?>", "", text)

    # 提取带有补丁索引的实体
    entities_with_patch_indices = extract_entities_with_patch_indices(text)
    entities = []
    for item in entities_with_patch_indices:
        # 获取实体和其边界框
        entity, bboxes = item[0:2], item[2]
        # 调整实体在文本中的位置
        adjusted_entity = adjust_entity_positions(entity, text)
        # 将边界框的补丁索引转换为坐标
        bboxes_in_coords = [patch_index_to_coordinate(bbox[0], bbox[1], num_patches_per_side) for bbox in bboxes]

        # 将调整后的实体和坐标添加到实体列表中
        entities.append(adjusted_entity + (bboxes_in_coords,))

    # 返回清理空格后的文本和处理后的实体列表
    return _cleanup_spaces(processed_text, entities)
```