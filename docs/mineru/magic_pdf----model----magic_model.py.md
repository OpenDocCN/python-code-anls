# `.\MinerU\magic_pdf\model\magic_model.py`

```
# 导入 json 库，用于处理 JSON 数据格式
import json

# 从 magic_pdf.libs.boxbase 导入多个函数，用于处理框和重叠区域相关的计算
from magic_pdf.libs.boxbase import (_is_in, _is_part_overlap, bbox_distance,
                                    bbox_relative_pos, box_area, calculate_iou,
                                    calculate_overlap_area_in_bbox1_area_ratio,
                                    get_overlap_area)
# 从 magic_pdf.libs.commons 导入 fitz 和 join_path，fitz 可能用于处理 PDF 文档，join_path 用于路径拼接
from magic_pdf.libs.commons import fitz, join_path
# 从 magic_pdf.libs.coordinate_transform 导入获取缩放比的函数
from magic_pdf.libs.coordinate_transform import get_scale_ratio
# 从 magic_pdf.libs.local_math 导入比较浮点数大小的函数
from magic_pdf.libs.local_math import float_gt
# 从 magic_pdf.libs.ModelBlockTypeEnum 导入模型块类型枚举
from magic_pdf.libs.ModelBlockTypeEnum import ModelBlockTypeEnum
# 从 magic_pdf.libs.ocr_content_type 导入分类和内容类型的常量
from magic_pdf.libs.ocr_content_type import CategoryId, ContentType
# 从 magic_pdf.rw 导入抽象的读写器和磁盘读写器
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

# 定义常量，表示重叠区域的比率阈值
CAPATION_OVERLAP_AREA_RATIO = 0.6
# 定义常量，表示合并框的重叠区域比率阈值
MERGE_BOX_OVERLAP_AREA_RATIO = 1.1


class MagicModel:
    """每个函数没有得到元素的时候返回空list."""

    def __fix_axis(self):
        # 遍历模型列表中的每一页信息
        for model_page_info in self.__model_list:
            # 创建一个待删除的列表
            need_remove_list = []
            # 获取当前页面的页码
            page_no = model_page_info['page_info']['page_no']
            # 获取当前页面的横向和纵向缩放比
            horizontal_scale_ratio, vertical_scale_ratio = get_scale_ratio(
                model_page_info, self.__docs[page_no]
            )
            # 获取当前页面的布局检测结果
            layout_dets = model_page_info['layout_dets']
            # 遍历每个布局检测结果
            for layout_det in layout_dets:

                if layout_det.get('bbox') is not None:
                    # 兼容直接输出bbox的模型数据，如paddle
                    x0, y0, x1, y1 = layout_det['bbox']
                else:
                    # 兼容直接输出poly的模型数据，如xxx
                    x0, y0, _, _, x1, y1, _, _ = layout_det['poly']

                # 根据缩放比调整边界框的坐标
                bbox = [
                    int(x0 / horizontal_scale_ratio),
                    int(y0 / vertical_scale_ratio),
                    int(x1 / horizontal_scale_ratio),
                    int(y1 / vertical_scale_ratio),
                ]
                # 将计算后的 bbox 更新到布局检测结果中
                layout_det['bbox'] = bbox
                # 如果边界框的宽度或高度小于等于0，标记为待删除
                if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                    need_remove_list.append(layout_det)
            # 从布局检测结果中删除标记为待删除的元素
            for need_remove in need_remove_list:
                layout_dets.remove(need_remove)

    def __fix_by_remove_low_confidence(self):
        # 遍历模型列表中的每一页信息
        for model_page_info in self.__model_list:
            # 创建一个待删除的列表
            need_remove_list = []
            # 获取当前页面的布局检测结果
            layout_dets = model_page_info['layout_dets']
            # 遍历每个布局检测结果
            for layout_det in layout_dets:
                # 如果置信度低于阈值，则标记为待删除
                if layout_det['score'] <= 0.05:
                    need_remove_list.append(layout_det)
                else:
                    # 置信度高则继续
                    continue
            # 从布局检测结果中删除标记为待删除的元素
            for need_remove in need_remove_list:
                layout_dets.remove(need_remove)
    # 定义一个私有方法，用于根据 IOU 和置信度调整检测结果
    def __fix_by_remove_high_iou_and_low_confidence(self):
        # 遍历模型列表中的每一项
        for model_page_info in self.__model_list:
            # 初始化需要移除的布局检测列表
            need_remove_list = []
            # 获取当前模型页面信息中的布局检测结果
            layout_dets = model_page_info['layout_dets']
            # 遍历布局检测结果中的每一个检测框
            for layout_det1 in layout_dets:
                # 对每一个检测框进行二次遍历
                for layout_det2 in layout_dets:
                    # 如果两个检测框是同一个，则跳过
                    if layout_det1 == layout_det2:
                        continue
                    # 检查两个检测框的类别是否在指定范围内
                    if layout_det1['category_id'] in [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                    ] and layout_det2['category_id'] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                        # 计算两个检测框的 IOU 值
                        if (
                            calculate_iou(layout_det1['bbox'], layout_det2['bbox'])
                            > 0.9
                        ):
                            # 如果 layout_det1 的分数低于 layout_det2，则需要移除 layout_det1
                            if layout_det1['score'] < layout_det2['score']:
                                layout_det_need_remove = layout_det1
                            # 否则，移除 layout_det2
                            else:
                                layout_det_need_remove = layout_det2

                            # 如果需要移除的检测框不在列表中，则添加它
                            if layout_det_need_remove not in need_remove_list:
                                need_remove_list.append(layout_det_need_remove)
                        else:
                            # 如果 IOU 不大于 0.9，则继续
                            continue
                    else:
                        # 如果类别不在范围内，则继续
                        continue
            # 遍历需要移除的检测框列表
            for need_remove in need_remove_list:
                # 从布局检测结果中移除检测框
                layout_dets.remove(need_remove)

    # 初始化方法，接收模型列表和文档对象
    def __init__(self, model_list: list, docs: fitz.Document):
        # 将传入的模型列表存储到实例变量中
        self.__model_list = model_list
        # 将传入的文档对象存储到实例变量中
        self.__docs = docs
        """为所有模型数据添加bbox信息(缩放，poly->bbox)"""
        # 调用方法修正坐标轴
        self.__fix_axis()
        """删除置信度特别低的模型数据(<0.05),提高质量"""
        # 调用方法移除置信度低于 0.05 的检测框
        self.__fix_by_remove_low_confidence()
        """删除高iou(>0.9)数据中置信度较低的那个"""
        # 调用方法移除 IOU 大于 0.9 的检测框中置信度较低的那个
        self.__fix_by_remove_high_iou_and_low_confidence()
        # 调用方法修正脚注
        self.__fix_footnote()
    # 修复脚注与图像、表格之间的关联
    def __fix_footnote(self):
        # 分类 ID 说明：3 表示图像，5 表示表格，7 表示脚注
        for model_page_info in self.__model_list:
            # 初始化脚注、图像和表格的列表
            footnotes = []
            figures = []
            tables = []

            # 遍历页面布局中的每个对象
            for obj in model_page_info['layout_dets']:
                # 将脚注添加到脚注列表
                if obj['category_id'] == 7:
                    footnotes.append(obj)
                # 将图像添加到图像列表
                elif obj['category_id'] == 3:
                    figures.append(obj)
                # 将表格添加到表格列表
                elif obj['category_id'] == 5:
                    tables.append(obj)
                # 如果脚注或图像列表为空，则跳过后续处理
                if len(footnotes) * len(figures) == 0:
                    continue
            # 存储图像与脚注的距离信息
            dis_figure_footnote = {}
            # 存储表格与脚注的距离信息
            dis_table_footnote = {}

            # 计算图像与脚注之间的距离
            for i in range(len(footnotes)):
                for j in range(len(figures)):
                    # 计算脚注与图像之间的相对位置标志数量
                    pos_flag_count = sum(
                        list(
                            map(
                                lambda x: 1 if x else 0,
                                bbox_relative_pos(
                                    footnotes[i]['bbox'], figures[j]['bbox']
                                ),
                            )
                        )
                    )
                    # 如果位置标志数量大于 1，则跳过
                    if pos_flag_count > 1:
                        continue
                    # 记录脚注与图像之间的最小距离
                    dis_figure_footnote[i] = min(
                        bbox_distance(figures[j]['bbox'], footnotes[i]['bbox']),
                        dis_figure_footnote.get(i, float('inf')),
                    )
            # 计算表格与脚注之间的距离
            for i in range(len(footnotes)):
                for j in range(len(tables)):
                    # 计算脚注与表格之间的相对位置标志数量
                    pos_flag_count = sum(
                        list(
                            map(
                                lambda x: 1 if x else 0,
                                bbox_relative_pos(
                                    footnotes[i]['bbox'], tables[j]['bbox']
                                ),
                            )
                        )
                    )
                    # 如果位置标志数量大于 1，则跳过
                    if pos_flag_count > 1:
                        continue

                    # 记录脚注与表格之间的最小距离
                    dis_table_footnote[i] = min(
                        bbox_distance(tables[j]['bbox'], footnotes[i]['bbox']),
                        dis_table_footnote.get(i, float('inf')),
                    )
            # 更新脚注的类别 ID，如果图像距离小于表格距离
            for i in range(len(footnotes)):
                if i not in dis_figure_footnote:
                    continue
                if dis_table_footnote.get(i, float('inf')) > dis_figure_footnote[i]:
                    footnotes[i]['category_id'] = CategoryId.ImageFootnote

    # 去除重叠的边界框
    def __reduct_overlap(self, bboxes):
        N = len(bboxes)  # 边界框数量
        keep = [True] * N  # 初始化保留标志
        # 遍历所有边界框
        for i in range(N):
            for j in range(N):
                # 如果是同一个边界框，则跳过
                if i == j:
                    continue
                # 如果边界框 i 在边界框 j 内部，则标记为不保留
                if _is_in(bboxes[i]['bbox'], bboxes[j]['bbox']):
                    keep[i] = False
        # 返回所有未被标记为不保留的边界框
        return [bboxes[i] for i in range(N) if keep[i]]

    # 根据距离将类别进行绑定
    def __tie_up_category_by_distance(
        self, page_no, subject_category_id, object_category_id
    # 获取指定页面的图像信息
        def get_imgs(self, page_no: int):
            # 根据页面号获取带有标题的类别信息，返回图像信息和其他信息
            with_captions, _ = self.__tie_up_category_by_distance(page_no, 3, 4)
            # 根据页面号获取带有脚注的类别信息
            with_footnotes, _ = self.__tie_up_category_by_distance(
                page_no, 3, CategoryId.ImageFootnote
            )
            # 初始化返回结果列表
            ret = []
            # 获取带标题和脚注的数量
            N, M = len(with_captions), len(with_footnotes)
            # 确保标题和脚注数量一致
            assert N == M
            # 遍历所有图像
            for i in range(N):
                # 创建记录字典，包含分数和边界框信息
                record = {
                    'score': with_captions[i]['score'],
                    'img_caption_bbox': with_captions[i].get('object_body', None),
                    'img_body_bbox': with_captions[i]['subject_body'],
                    'img_footnote_bbox': with_footnotes[i].get('object_body', None),
                }
    
                # 计算最小和最大边界框坐标
                x0 = min(with_captions[i]['all'][0], with_footnotes[i]['all'][0])
                y0 = min(with_captions[i]['all'][1], with_footnotes[i]['all'][1])
                x1 = max(with_captions[i]['all'][2], with_footnotes[i]['all'][2])
                y1 = max(with_captions[i]['all'][3], with_footnotes[i]['all'][3])
                # 将边界框坐标添加到记录中
                record['bbox'] = [x0, y0, x1, y1]
                # 将记录添加到结果列表
                ret.append(record)
            # 返回所有图像的记录
            return ret
    
        # 获取指定页面的表格信息
        def get_tables(
            self, page_no: int
        ) -> list:  # 3个坐标， caption, table主体，table-note
            # 根据页面号获取带有标题的表格类别信息
            with_captions, _ = self.__tie_up_category_by_distance(page_no, 5, 6)
            # 根据页面号获取带有脚注的表格类别信息
            with_footnotes, _ = self.__tie_up_category_by_distance(page_no, 5, 7)
            # 初始化返回结果列表
            ret = []
            # 获取带标题和脚注的数量
            N, M = len(with_captions), len(with_footnotes)
            # 确保标题和脚注数量一致
            assert N == M
            # 遍历所有表格
            for i in range(N):
                # 创建记录字典，包含分数和边界框信息
                record = {
                    'score': with_captions[i]['score'],
                    'table_caption_bbox': with_captions[i].get('object_body', None),
                    'table_body_bbox': with_captions[i]['subject_body'],
                    'table_footnote_bbox': with_footnotes[i].get('object_body', None),
                }
    
                # 计算最小和最大边界框坐标
                x0 = min(with_captions[i]['all'][0], with_footnotes[i]['all'][0])
                y0 = min(with_captions[i]['all'][1], with_footnotes[i]['all'][1])
                x1 = max(with_captions[i]['all'][2], with_footnotes[i]['all'][2])
                y1 = max(with_captions[i]['all'][3], with_footnotes[i]['all'][3])
                # 将边界框坐标添加到记录中
                record['bbox'] = [x0, y0, x1, y1]
                # 将记录添加到结果列表
                ret.append(record)
            # 返回所有表格的记录
            return ret
    
        # 获取指定页面的方程信息
        def get_equations(self, page_no: int) -> list:  # 有坐标，也有字
            # 获取页面中行内方程块
            inline_equations = self.__get_blocks_by_type(
                ModelBlockTypeEnum.EMBEDDING.value, page_no, ['latex']
            )
            # 获取页面中行间方程块
            interline_equations = self.__get_blocks_by_type(
                ModelBlockTypeEnum.ISOLATED.value, page_no, ['latex']
            )
            # 获取页面中孤立方程块
            interline_equations_blocks = self.__get_blocks_by_type(
                ModelBlockTypeEnum.ISOLATE_FORMULA.value, page_no
            )
            # 返回所有方程块信息
            return inline_equations, interline_equations, interline_equations_blocks
    
        # 获取被丢弃的块信息
        def get_discarded(self, page_no: int) -> list:  # 自研模型，只有坐标
            # 根据页面号获取被丢弃的块信息
            blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.ABANDON.value, page_no)
            # 返回所有被丢弃的块
            return blocks
    # 获取指定页码的文本块，返回文本块列表
    def get_text_blocks(self, page_no: int) -> list:  # 自研模型搞的，只有坐标，没有字
        # 根据块类型和页码获取文本块
        blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.PLAIN_TEXT.value, page_no)
        # 返回获取到的文本块列表
        return blocks

    # 获取指定页码的标题块，返回标题块列表
    def get_title_blocks(self, page_no: int) -> list:  # 自研模型，只有坐标，没字
        # 根据块类型和页码获取标题块
        blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.TITLE.value, page_no)
        # 返回获取到的标题块列表
        return blocks

    # 获取指定页码的OCR文本，返回文本及其位置信息的列表
    def get_ocr_text(self, page_no: int) -> list:  # paddle 搞的，有字也有坐标
        # 初始化文本跨度列表
        text_spans = []
        # 获取指定页码的模型信息
        model_page_info = self.__model_list[page_no]
        # 提取布局检测结果
        layout_dets = model_page_info['layout_dets']
        # 遍历每个布局检测结果
        for layout_det in layout_dets:
            # 检查类别是否为文本
            if layout_det['category_id'] == '15':
                # 创建文本跨度字典，包含边界框和内容
                span = {
                    'bbox': layout_det['bbox'],
                    'content': layout_det['text'],
                }
                # 将文本跨度添加到列表中
                text_spans.append(span)
        # 返回所有文本跨度
        return text_spans

    # 获取指定页码的所有跨度，去重后返回
    def get_all_spans(self, page_no: int) -> list:

        # 定义去重函数，返回无重复的跨度列表
        def remove_duplicate_spans(spans):
            # 初始化新的跨度列表
            new_spans = []
            # 遍历每个跨度
            for span in spans:
                # 如果跨度不在新列表中，则添加
                if not any(span == existing_span for existing_span in new_spans):
                    new_spans.append(span)
            # 返回去重后的新跨度列表
            return new_spans

        # 初始化所有跨度列表
        all_spans = []
        # 获取指定页码的模型信息
        model_page_info = self.__model_list[page_no]
        # 提取布局检测结果
        layout_dets = model_page_info['layout_dets']
        # 定义允许的类别ID列表
        allow_category_id_list = [3, 5, 13, 14, 15]
        """当成span拼接的"""
        #  3: 'image', # 图片
        #  5: 'table',       # 表格
        #  13: 'inline_equation',     # 行内公式
        #  14: 'interline_equation',      # 行间公式
        #  15: 'text',      # ocr识别文本
        # 遍历每个布局检测结果
        for layout_det in layout_dets:
            # 获取类别ID
            category_id = layout_det['category_id']
            # 如果类别ID在允许的列表中
            if category_id in allow_category_id_list:
                # 创建跨度字典，包含边界框和得分
                span = {'bbox': layout_det['bbox'], 'score': layout_det['score']}
                # 根据类别ID设置不同的类型
                if category_id == 3:
                    span['type'] = ContentType.Image
                elif category_id == 5:
                    # 获取表格模型结果
                    latex = layout_det.get('latex', None)
                    html = layout_det.get('html', None)
                    # 如果有LaTeX格式，添加到跨度中
                    if latex:
                        span['latex'] = latex
                    # 如果有HTML格式，添加到跨度中
                    elif html:
                        span['html'] = html
                    span['type'] = ContentType.Table
                elif category_id == 13:
                    # 添加行内公式的内容和类型
                    span['content'] = layout_det['latex']
                    span['type'] = ContentType.InlineEquation
                elif category_id == 14:
                    # 添加行间公式的内容和类型
                    span['content'] = layout_det['latex']
                    span['type'] = ContentType.InterlineEquation
                elif category_id == 15:
                    # 添加OCR识别文本的内容和类型
                    span['content'] = layout_det['text']
                    span['type'] = ContentType.Text
                # 将跨度添加到所有跨度列表中
                all_spans.append(span)
        # 返回去重后的所有跨度列表
        return remove_duplicate_spans(all_spans)
    # 获取指定页面的宽高
        def get_page_size(self, page_no: int):  # 获取页面宽高
            # 根据页码获取当前页的页面对象
            page = self.__docs[page_no]
            # 获取当前页的宽度
            page_w = page.rect.width
            # 获取当前页的高度
            page_h = page.rect.height
            # 返回当前页的宽度和高度
            return page_w, page_h
    
        # 根据类型和页码获取特定块的列表
        def __get_blocks_by_type(
            self, type: int, page_no: int, extra_col: list[str] = []
        ) -> list:
            # 初始化块列表
            blocks = []
            # 遍历模型列表中的每个页面字典
            for page_dict in self.__model_list:
                # 获取布局细节，如果没有则为空列表
                layout_dets = page_dict.get('layout_dets', [])
                # 获取页面信息
                page_info = page_dict.get('page_info', {})
                # 获取当前页面编号，默认是-1
                page_number = page_info.get('page_no', -1)
                # 如果当前页面编号不匹配，则跳过
                if page_no != page_number:
                    continue
                # 遍历布局细节中的每个项
                for item in layout_dets:
                    # 获取类别ID，默认是-1
                    category_id = item.get('category_id', -1)
                    # 获取边界框信息
                    bbox = item.get('bbox', None)
    
                    # 如果类别ID匹配指定类型
                    if category_id == type:
                        # 创建一个包含边界框和评分的块字典
                        block = {
                            'bbox': bbox,
                            'score': item.get('score'),
                        }
                        # 遍历额外列并将其加入块字典
                        for col in extra_col:
                            block[col] = item.get(col, None)
                        # 将块添加到块列表
                        blocks.append(block)
            # 返回符合条件的块列表
            return blocks
    
        # 获取指定页码的模型列表
        def get_model_list(self, page_no):
            # 返回指定页码的模型列表
            return self.__model_list[page_no]
# 如果该模块是主程序，则执行以下代码
if __name__ == '__main__':
    # 创建一个 DiskReaderWriter 对象，用于读取和写入指定路径的文件
    drw = DiskReaderWriter(r'D:/project/20231108code-clean')
    # 条件判断，不会执行，因为条件为 0
    if 0:
        # 定义 PDF 文件和模型文件的路径
        pdf_file_path = r'linshixuqiu\19983-00.pdf'
        model_file_path = r'linshixuqiu\19983-00_new.json'
        # 读取 PDF 文件的二进制内容
        pdf_bytes = drw.read(pdf_file_path, AbsReaderWriter.MODE_BIN)
        # 读取模型文件的文本内容
        model_json_txt = drw.read(model_file_path, AbsReaderWriter.MODE_TXT)
        # 将 JSON 格式的文本解析为 Python 列表
        model_list = json.loads(model_json_txt)
        # 定义写入的路径
        write_path = r'D:\project\20231108code-clean\linshixuqiu\19983-00'
        # 定义图片存储的路径
        img_bucket_path = 'imgs'
        # 创建一个新的 DiskReaderWriter 对象，用于写入图片
        img_writer = DiskReaderWriter(join_path(write_path, img_bucket_path))
        # 打开 PDF 文档
        pdf_docs = fitz.open('pdf', pdf_bytes)
        # 创建一个 MagicModel 对象，传入模型列表和 PDF 文档
        magic_model = MagicModel(model_list, pdf_docs)

    # 条件判断，执行此部分，因为条件为 1
    if 1:
        # 从指定路径读取 JSON 文件并解析为列表
        model_list = json.loads(
            drw.read('/opt/data/pdf/20240418/j.chroma.2009.03.042.json')
        )
        # 读取指定路径的 PDF 文件的二进制内容
        pdf_bytes = drw.read(
            '/opt/data/pdf/20240418/j.chroma.2009.03.042.pdf', AbsReaderWriter.MODE_BIN
        )
        # 打开 PDF 文档
        pdf_docs = fitz.open('pdf', pdf_bytes)
        # 创建一个 MagicModel 对象，传入模型列表和 PDF 文档
        magic_model = MagicModel(model_list, pdf_docs)
        # 遍历 7 次，打印每次获取的图片
        for i in range(7):
            print(magic_model.get_imgs(i))
```