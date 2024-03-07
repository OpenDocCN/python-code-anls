# `.\marker\marker\schema.py`

```py
# 导入 Counter 类，用于计数
# 导入 List、Optional、Tuple 类型，用于类型提示
from collections import Counter
from typing import List, Optional, Tuple

# 导入 BaseModel、field_validator 类，用于定义数据模型和字段验证
# 导入 ftfy 模块，用于修复文本中的 Unicode 错误
from pydantic import BaseModel, field_validator
import ftfy

# 导入 boxes_intersect_pct、multiple_boxes_intersect 函数，用于计算两个框的交集比例和多个框的交集情况
# 导入 settings 模块，用于获取配置信息
from marker.bbox import boxes_intersect_pct, multiple_boxes_intersect
from marker.settings import settings

# 定义函数 find_span_type，用于查找给定 span 在页面块中的类型
def find_span_type(span, page_blocks):
    # 默认块类型为 "Text"
    block_type = "Text"
    # 遍历页面块列表
    for block in page_blocks:
        # 如果 span 的边界框与页面块的边界框有交集
        if boxes_intersect_pct(span.bbox, block.bbox):
            # 更新块类型为页面块的类型
            block_type = block.block_type
            break
    # 返回块类型
    return block_type

# 定义类 BboxElement，继承自 BaseModel 类，表示具有边界框的元素
class BboxElement(BaseModel):
    bbox: List[float]

    # 验证 bbox 字段是否包含 4 个元素
    @field_validator('bbox')
    @classmethod
    def check_4_elements(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError('bbox must have 4 elements')
        return v

    # 计算元素的高度、宽度、起始 x 坐标、起始 y 坐标、面积
    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def x_start(self):
        return self.bbox[0]

    @property
    def y_start(self):
        return self.bbox[1]

    @property
    def area(self):
        return self.width * self.height

# 定义类 BlockType，继承自 BboxElement 类，表示具有块类型的元素
class BlockType(BboxElement):
    block_type: str

# 定义类 Span，继承自 BboxElement 类，表示具有文本内容的元素
class Span(BboxElement):
    text: str
    span_id: str
    font: str
    color: int
    ascender: Optional[float] = None
    descender: Optional[float] = None
    block_type: Optional[str] = None
    selected: bool = True

    # 修复文本中的 Unicode 错误
    @field_validator('text')
    @classmethod
    def fix_unicode(cls, text: str) -> str:
        return ftfy.fix_text(text)

# 定义类 Line，继承自 BboxElement 类，表示具有多个 Span 的行元素
class Line(BboxElement):
    spans: List[Span]

    # 获取行的预备文本，即所有 Span 的文本拼接而成
    @property
    def prelim_text(self):
        return "".join([s.text for s in self.spans])

    # 获取行的起始 x 坐标
    @property
    def start(self):
        return self.spans[0].bbox[0]

# 定义类 Block，继承自 BboxElement 类，表示具有多个 Line 的块元素
class Block(BboxElement):
    lines: List[Line]
    pnum: int

    # 获取块的预备文本，即所有 Line 的预备文本拼接而成
    @property
    def prelim_text(self):
        return "\n".join([l.prelim_text for l in self.lines])
    # 检查文本块是否包含公式，通过检查每个 span 的 block_type 是否为 "Formula" 来确定
    def contains_equation(self, equation_boxes=None):
        # 生成一个包含每个 span 的 block_type 是否为 "Formula" 的条件列表
        conditions = [s.block_type == "Formula" for l in self.lines for s in l.spans]
        # 如果提供了 equation_boxes 参数，则添加一个条件，检查文本块的边界框是否与给定框相交
        if equation_boxes:
            conditions += [multiple_boxes_intersect(self.bbox, equation_boxes)]
        # 返回条件列表中是否有任何一个条件为 True
        return any(conditions)

    # 过滤掉包含在 bad_span_ids 中的 span
    def filter_spans(self, bad_span_ids):
        new_lines = []
        for line in self.lines:
            new_spans = []
            for span in line.spans:
                # 如果 span 的 span_id 不在 bad_span_ids 中，则保留该 span
                if not span.span_id in bad_span_ids:
                    new_spans.append(span)
            # 更新 line 的 spans 属性为过滤后的 new_spans
            line.spans = new_spans
            # 如果 line 中仍有 spans，则将其添加到 new_lines 中
            if len(new_spans) > 0:
                new_lines.append(line)
        # 更新 self.lines 为过滤后的 new_lines
        self.lines = new_lines

    # 过滤掉包含在 settings.BAD_SPAN_TYPES 中的 span 的 block_type
    def filter_bad_span_types(self):
        new_lines = []
        for line in self.lines:
            new_spans = []
            for span in line.spans:
                # 如果 span 的 block_type 不在 BAD_SPAN_TYPES 中，则保留该 span
                if span.block_type not in settings.BAD_SPAN_TYPES:
                    new_spans.append(span)
            # 更新 line 的 spans 属性为过滤后的 new_spans
            line.spans = new_spans
            # 如果 line 中仍有 spans，则将其添加到 new_lines 中
            if len(new_spans) > 0:
                new_lines.append(line)
        # 更新 self.lines 为过滤后的 new_lines
        self.lines = new_lines

    # 返回文本块中出现频率最高的 block_type
    def most_common_block_type(self):
        # 统计每个 span 的 block_type 出现的次数
        counter = Counter([s.block_type for l in self.lines for s in l.spans])
        # 返回出现次数最多的 block_type
        return counter.most_common(1)[0][0]

    # 设置文本块中所有 span 的 block_type 为给定的 block_type
    def set_block_type(self, block_type):
        for line in self.lines:
            for span in line.spans:
                # 将 span 的 block_type 设置为给定的 block_type
                span.block_type = block_type
# 定义一个名为 Page 的类，继承自 BboxElement 类
class Page(BboxElement):
    # 类属性：blocks 为 Block 对象列表，pnum 为整数，column_count 和 rotation 可选整数，默认为 None
    blocks: List[Block]
    pnum: int
    column_count: Optional[int] = None
    rotation: Optional[int] = None # 页面的旋转角度

    # 获取页面中非空行的方法
    def get_nonblank_lines(self):
        # 获取页面中所有行
        lines = self.get_all_lines()
        # 过滤出非空行
        nonblank_lines = [l for l in lines if l.prelim_text.strip()]
        return nonblank_lines

    # 获取页面中所有行的方法
    def get_all_lines(self):
        # 获取页面中所有行的列表
        lines = [l for b in self.blocks for l in b.lines]
        return lines

    # 获取页面中非空跨度的方法，返回 Span 对象列表
    def get_nonblank_spans(self) -> List[Span]:
        # 获取页面中所有行
        lines = [l for b in self.blocks for l in b.lines]
        # 过滤出非空跨度
        spans = [s for l in lines for s in l.spans if s.text.strip()]
        return spans

    # 添加块类型到行的方法
    def add_block_types(self, page_block_types):
        # 如果检测到的块类型数量与页面行数不匹配，则打印警告信息
        if len(page_block_types) != len(self.get_all_lines()):
            print(f"Warning: Number of detected lines {len(page_block_types)} does not match number of lines {len(self.get_all_lines())}")

        i = 0
        for block in self.blocks:
            for line in block.lines:
                if i < len(page_block_types):
                    line_block_type = page_block_types[i].block_type
                else:
                    line_block_type = "Text"
                i += 1
                for span in line.spans:
                    span.block_type = line_block_type

    # 获取页面中字体统计信息的方法
    def get_font_stats(self):
        # 获取页面中非空跨度的字体信息
        fonts = [s.font for s in self.get_nonblank_spans()]
        # 统计字体出现次数
        font_counts = Counter(fonts)
        return font_counts

    # 获取页面中行高统计信息的方法
    def get_line_height_stats(self):
        # 获取页面中非空行的行高信息
        heights = [l.bbox[3] - l.bbox[1] for l in self.get_nonblank_lines()]
        # 统计行高出现次数
        height_counts = Counter(heights)
        return height_counts

    # 获取页面中行起始位置统计信息的方法
    def get_line_start_stats(self):
        # 获取页面中非空行的起始位置信息
        starts = [l.bbox[0] for l in self.get_nonblank_lines()]
        # 统计起始位置出现次数
        start_counts = Counter(starts)
        return start_counts
    # 获取文本块中非空行的起始位置列表
    def get_min_line_start(self):
        # 通过列表推导式获取非空行的起始位置，并且该行为文本类型
        starts = [l.bbox[0] for l in self.get_nonblank_lines() if l.spans[0].block_type == "Text"]
        # 如果没有找到非空行，则抛出索引错误
        if len(starts) == 0:
            raise IndexError("No lines found")
        # 返回起始位置列表中的最小值
        return min(starts)

    # 获取文本块中每个文本块的 prelim_text 属性，并用换行符连接成字符串
    @property
    def prelim_text(self):
        return "\n".join([b.prelim_text for b in self.blocks])
# 定义一个继承自BboxElement的MergedLine类，包含文本和字体列表属性
class MergedLine(BboxElement):
    text: str
    fonts: List[str]

    # 返回该行中出现频率最高的字体
    def most_common_font(self):
        # 统计字体列表中各个字体出现的次数
        counter = Counter(self.fonts)
        # 返回出现频率最高的字体
        return counter.most_common(1)[0][0]


# 定义一个继承自BboxElement的MergedBlock类，包含行列表、段落号和块类型列表属性
class MergedBlock(BboxElement):
    lines: List[MergedLine]
    pnum: int
    block_types: List[str]

    # 返回该块中出现频率最高的块类型
    def most_common_block_type(self):
        # 统计块类型列表中各个类型出现的次数
        counter = Counter(self.block_types)
        # 返回出现频率最高的块类型
        return counter.most_common(1)[0][0]


# 定义一个继承自BaseModel的FullyMergedBlock类，包含文本和块类型属性
class FullyMergedBlock(BaseModel):
    text: str
    block_type: str
```