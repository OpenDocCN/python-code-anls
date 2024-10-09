# `.\MinerU\magic_pdf\para\raw_processor.py`

```
# RawBlockProcessor 类用于处理原始文本块
class RawBlockProcessor:
    # 初始化方法，设置容差和 PDF 字典
    def __init__(self) -> None:
        self.y_tolerance = 2  # y 方向容差设为 2
        self.pdf_dic = {}  # 初始化 PDF 字典为空

    # 私有方法，用于分解字体标志
    def __span_flags_decomposer(self, span_flags):
        """
        将字体标志转换为人类可读的格式。

        参数
        ----------
        self : object
            类的实例。

        span_flags : int
            字体标志

        返回
        -------
        l : dict
            分解后的标志
        """

        # 初始化字典，用于存储分解后的标志
        l = {
            "is_superscript": False,  # 是否上标
            "is_italic": False,  # 是否斜体
            "is_serifed": False,  # 是否衬线字体
            "is_sans_serifed": False,  # 是否非衬线字体
            "is_monospaced": False,  # 是否等宽字体
            "is_proportional": False,  # 是否比例字体
            "is_bold": False,  # 是否粗体
        }

        # 检查上标标志并更新字典
        if span_flags & 2**0:
            l["is_superscript"] = True  # 表示上标

        # 检查斜体标志并更新字典
        if span_flags & 2**1:
            l["is_italic"] = True  # 表示斜体

        # 检查衬线字体标志并更新字典
        if span_flags & 2**2:
            l["is_serifed"] = True  # 表示衬线字体
        else:
            l["is_sans_serifed"] = True  # 表示非衬线字体

        # 检查等宽字体标志并更新字典
        if span_flags & 2**3:
            l["is_monospaced"] = True  # 表示等宽字体
        else:
            l["is_proportional"] = True  # 表示比例字体

        # 检查粗体标志并更新字典
        if span_flags & 2**4:
            l["is_bold"] = True  # 表示粗体

        # 返回分解后的标志字典
        return l
    # 定义一个私有方法，用于处理原始行并生成新行
    def __make_new_lines(self, raw_lines):
        """
        该函数用于生成新行。

        参数
        ----------
        self : object
            类的实例。

        raw_lines : list
            原始行的列表。

        返回
        -------
        new_lines : list
            新行的列表。
        """
        # 初始化一个空列表，用于存储新行
        new_lines = []
        # 初始化新行变量为空
        new_line = None

        # 遍历每一行的原始数据
        for raw_line in raw_lines:
            # 获取原始行的边界框
            raw_line_bbox = raw_line["bbox"]
            # 获取原始行的跨度信息
            raw_line_spans = raw_line["spans"]
            # 合并所有跨度的文本为一行文本
            raw_line_text = "".join([span["text"] for span in raw_line_spans])
            # 获取原始行的方向，默认为 None
            raw_line_dir = raw_line.get("dir", None)

            # 初始化一个空列表，用于存储分解后的行跨度
            decomposed_line_spans = []
            # 遍历每个跨度
            for span in raw_line_spans:
                # 获取跨度的标志
                raw_flags = span["flags"]
                # 分解标志以便后续使用
                decomposed_flags = self.__span_flags_decomposer(raw_flags)
                # 将分解后的标志添加到跨度中
                span["decomposed_flags"] = decomposed_flags
                # 将已分解的跨度添加到列表中
                decomposed_line_spans.append(span)

            # 如果新行变量为空，则创建新的行
            if new_line is None:
                new_line = {
                    # 设置新行的边界框为当前行的边界框
                    "bbox": raw_line_bbox,
                    # 设置新行的文本为当前行的文本
                    "text": raw_line_text,
                    # 设置新行的方向，若为空则使用默认值
                    "dir": raw_line_dir if raw_line_dir else (0, 0),
                    # 将分解后的跨度添加到新行中
                    "spans": decomposed_line_spans,
                }
            else:
                # 检查当前行和新行的边界框是否在允许的容差范围内
                if (
                    abs(raw_line_bbox[1] - new_line["bbox"][1]) <= self.y_tolerance
                    and abs(raw_line_bbox[3] - new_line["bbox"][3]) <= self.y_tolerance
                ):
                    # 更新新行的边界框，合并当前行的边界框
                    new_line["bbox"] = (
                        min(new_line["bbox"][0], raw_line_bbox[0]),  # 左边界
                        new_line["bbox"][1],  # 上边界保持不变
                        max(new_line["bbox"][2], raw_line_bbox[2]),  # 右边界
                        raw_line_bbox[3],  # 下边界保持当前行的下边界
                    )
                    # 合并文本，加上一个空格
                    new_line["text"] += " " + raw_line_text
                    # 扩展新行的跨度信息
                    new_line["spans"].extend(raw_line_spans)
                    # 更新新行的方向
                    new_line["dir"] = (
                        new_line["dir"][0] + raw_line_dir[0],
                        new_line["dir"][1] + raw_line_dir[1],
                    )
                else:
                    # 如果不在容差范围内，将当前新行添加到新行列表中
                    new_lines.append(new_line)
                    # 创建新的行
                    new_line = {
                        # 设置新行的边界框为当前行的边界框
                        "bbox": raw_line_bbox,
                        # 设置新行的文本为当前行的文本
                        "text": raw_line_text,
                        # 设置新行的方向，若为空则使用默认值
                        "dir": raw_line_dir if raw_line_dir else (0, 0),
                        # 设置新行的跨度为当前行的跨度
                        "spans": raw_line_spans,
                    }
        # 如果最后的 new_line 不为空，将其添加到新行列表中
        if new_line:
            new_lines.append(new_line)

        # 返回生成的新行列表
        return new_lines
    # 定义一个私有方法，用于创建新的块
        def __make_new_block(self, raw_block):
            """
            此函数创建一个新块。
    
            参数
            ----------
            self : object
                类的实例。
            ----------
            raw_block : dict
                一个原始块
    
            返回
            -------
            new_block : dict
    
            new_block 的结构：
            {
                "block_id": "block_1",
                "bbox": [0, 0, 100, 100],
                "text": "这是一个块。",
                "lines": [
                    {
                        "bbox": [0, 0, 100, 100],
                        "text": "这是一行。",
                        "spans": [
                            {
                                "text": "这是一个跨度。",
                                "font": "Times New Roman",
                                "size": 12,
                                "color": "#000000",
                            }
                        ],
                    }
                ],
            }
            """
            # 初始化一个新的块字典
            new_block = {}
    
            # 获取原始块的编号
            block_id = raw_block["number"]
            # 获取原始块的边界框
            block_bbox = raw_block["bbox"]
            # 从原始块的每一行中提取文本并连接成一个字符串
            block_text = " ".join(span["text"] for line in raw_block["lines"] for span in line["spans"])
            # 获取原始块中的行
            raw_lines = raw_block["lines"]
            # 调用方法生成新行
            block_lines = self.__make_new_lines(raw_lines)
    
            # 将块的编号存入新块字典
            new_block["block_id"] = block_id
            # 将块的边界框存入新块字典
            new_block["bbox"] = block_bbox
            # 将块的文本存入新块字典
            new_block["text"] = block_text
            # 将块的行存入新块字典
            new_block["lines"] = block_lines
    
            # 返回新的块字典
            return new_block
    
        # 定义一个方法，用于批量处理块
        def batch_process_blocks(self, pdf_dic):
            """
            此函数批量处理块。
    
            参数
            ----------
            self : object
                类的实例。
            ----------
            blocks : list
                输入块是一个原始块的列表。结构可以参考键"preproc_blocks"的值，示例文件为app/pdf_toolbox/tests/preproc_2_parasplit_example.json。
    
            返回
            -------
            result_dict : dict
                结果字典
            """
    
            # 遍历 PDF 字典中的每一页
            for page_id, blocks in pdf_dic.items():
                # 检查页 ID 是否以 "page_" 开头
                if page_id.startswith("page_"):
                    # 初始化一个空列表以存储段落块
                    para_blocks = []
                    # 检查块字典中是否包含预处理块
                    if "preproc_blocks" in blocks.keys():
                        # 获取输入块
                        input_blocks = blocks["preproc_blocks"]
                        # 遍历每个原始块
                        for raw_block in input_blocks:
                            # 创建新块并添加到段落块列表中
                            new_block = self.__make_new_block(raw_block)
                            para_blocks.append(new_block)
    
                    # 将生成的段落块存入块字典
                    blocks["para_blocks"] = para_blocks
    
            # 返回处理后的 PDF 字典
            return pdf_dic
```