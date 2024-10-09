# `.\MinerU\magic_pdf\integrations\rag\type.py`

```
# 导入 Enum 类，用于定义枚举类型
from enum import Enum

# 导入 Pydantic 库中的 BaseModel 和 Field，用于数据模型的定义和验证
from pydantic import BaseModel, Field


# 定义文档中可能出现的类别类型的枚举
class CategoryType(Enum):  # py310 not support StrEnum
    # 文本类型
    text = 'text'
    # 标题类型
    title = 'title'
    # 行间公式类型
    interline_equation = 'interline_equation'
    # 图片类型
    image = 'image'
    # 图片主体类型
    image_body = 'image_body'
    # 图片说明类型
    image_caption = 'image_caption'
    # 表格类型
    table = 'table'
    # 表格主体类型
    table_body = 'table_body'
    # 表格说明类型
    table_caption = 'table_caption'
    # 表格脚注类型
    table_footnote = 'table_footnote'


# 定义元素关系类型的枚举
class ElementRelType(Enum):
    # 兄弟元素关系
    sibling = 'sibling'


# 定义页面信息的数据模型
class PageInfo(BaseModel):
    # 页面索引，从零开始
    page_no: int = Field(description='the index of page, start from zero',
                         ge=0)
    # 页面高度，必须大于零
    height: int = Field(description='the height of page', gt=0)
    # 页面宽度，必须大于或等于零
    width: int = Field(description='the width of page', ge=0)
    # 页面图像路径，类型为字符串或 None
    image_path: str | None = Field(description='the image of this page',
                                   default=None)


# 定义内容对象的数据模型
class ContentObject(BaseModel):
    # 内容对象的类别类型
    category_type: CategoryType = Field(description='类别')
    # 该对象的坐标列表，需转换回 PDF 坐标系
    poly: list[float] = Field(
        description=('Coordinates, need to convert back to PDF coordinates,'
                     ' order is top-left, top-right, bottom-right, bottom-left'
                     ' x,y coordinates'))
    # 是否忽略此对象，默认值为 False
    ignore: bool = Field(description='whether ignore this object',
                         default=False)
    # 对象的文本内容，类型为字符串或 None
    text: str | None = Field(description='text content of the object',
                             default=None)
    # 嵌入图像的路径，类型为字符串或 None
    image_path: str | None = Field(description='path of embedded image',
                                   default=None)
    # 对象在页面中的顺序，默认值为 -1
    order: int = Field(description='the order of this object within a page',
                       default=-1)
    # 对象的唯一 ID，默认值为 -1
    anno_id: int = Field(description='unique id', default=-1)
    # LaTeX 结果，类型为字符串或 None
    latex: str | None = Field(description='latex result', default=None)
    # HTML 结果，类型为字符串或 None
    html: str | None = Field(description='html result', default=None)


# 定义元素关系的数据模型
class ElementRelation(BaseModel):
    # 源对象的唯一 ID，默认值为 -1
    source_anno_id: int = Field(description='unique id of the source object',
                                default=-1)
    # 目标对象的唯一 ID，默认值为 -1
    target_anno_id: int = Field(description='unique id of the target object',
                                default=-1)
    # 源对象和目标对象之间的关系类型
    relation: ElementRelType = Field(
        description='the relation between source and target element')


# 定义布局元素附加信息的数据模型
class LayoutElementsExtra(BaseModel):
    # 源对象与目标对象之间的关系列表
    element_relation: list[ElementRelation] = Field(
        description='the relation between source and target element')


# 定义布局元素的数据模型
class LayoutElements(BaseModel):
    # 布局元素的详细信息列表
    layout_dets: list[ContentObject] = Field(
        description='layout element details')
    # 页面信息
    page_info: PageInfo = Field(description='page info')
    # 附加信息
    extra: LayoutElementsExtra = Field(description='extra information')


# 定义节点数据格式的数据模型
class Node(BaseModel):
    # 节点的类别类型
    category_type: CategoryType = Field(description='类别')
    # 节点的文本内容，类型为字符串或 None
    text: str | None = Field(description='text content of the object',
                             default=None)
    # 嵌入图像的路径，类型为字符串或 None
    image_path: str | None = Field(description='path of embedded image',
                                   default=None)
    # 定义一个整数类型的字段，表示唯一标识符，默认为 -1
        anno_id: int = Field(description='unique id', default=-1)
        # 定义一个可选字符串字段，用于存储 LaTeX 结果，默认为 None
        latex: str | None = Field(description='latex result', default=None)
        # 定义一个可选字符串字段，用于存储 HTML 结果，默认为 None
        html: str | None = Field(description='html result', default=None)
```