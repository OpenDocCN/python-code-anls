# `.\MinerU\tests\test_metascan_classify\test_classify.py`

```
# 导入操作系统模块，用于路径和文件操作
import os

# 导入 pytest 测试框架
import pytest

# 从 magic_pdf.filter 导入多个分类函数
from magic_pdf.filter.pdf_classify_by_type import classify_by_area, classify_by_text_len, classify_by_avg_words, \
    classify_by_img_num, classify_by_text_layout, classify_by_img_narrow_strips
# 从 magic_pdf.filter 导入获取 PDF 页面的大小、每页文本长度和每页图像数量的函数
from magic_pdf.filter.pdf_meta_scan import get_pdf_page_size_pts, get_pdf_textlen_per_page, get_imgs_per_page
# 从测试公共模块导入获取测试 PDF 文档和测试 JSON 数据的函数
from tests.test_commons import get_docs_from_test_pdf, get_test_json_data

# 获取当前文件的目录路径
current_directory = os.path.dirname(os.path.abspath(__file__))

'''
根据图片尺寸占页面面积的比例，判断是否为扫描版
'''
# 使用 pytest 的参数化装饰器定义测试用例，包含书名和期望的分类结果
@pytest.mark.parametrize("book_name, expected_bool_classify_by_area",
                         [
                             ("the_eye/the_eye_cdn_00391653", True),  # 特殊文字版1，图像占比小
                             ("scihub/scihub_08400000/libgen.scimag08489000-08489999.zip_10.1016/0370-1573(90)90070-i", False),  # 特殊扫描版2，图像占比大
                             ("zlib/zlib_17216416", False),  # 特殊扫描版3，包含大图和小图
                             ("the_eye/the_eye_wtl_00023799", False),  # 特殊扫描版4，小图拼成大图
                             ("the_eye/the_eye_cdn_00328381", False),  # 特殊扫描版5，小图重复使用
                             ("scihub/scihub_25800000/libgen.scimag25889000-25889999.zip_10.2307/4153991", False),  # 特殊扫描版6，含有扫描页
                             ("scanned_detection/llm-raw-scihub-o.O-0584-8539%2891%2980165-f", False),  # 特殊扫描版7，单页大图
                             ("scanned_detection/llm-raw-scihub-o.O-bf01427123", False),  # 特殊扫描版8，全部为大图
                             ("scihub/scihub_41200000/libgen.scimag41253000-41253999.zip_10.1080/00222938709460256", False),  # 特殊扫描版12，包含文字版和扫描版
                             ("scihub/scihub_37000000/libgen.scimag37068000-37068999.zip_10.1080/0015587X.1936.9718622", False)  # 特殊扫描版13，类似情况
                         ])
# 定义测试函数，使用参数 book_name 和 expected_bool_classify_by_area
def test_classify_by_area(book_name, expected_bool_classify_by_area):
    # 获取测试 JSON 数据
    test_data = get_test_json_data(current_directory, "test_metascan_classify_data.json")
    # 获取指定书名的文档数据
    docs = get_docs_from_test_pdf(book_name)
    # 获取 PDF 页面的中位数宽度和高度
    median_width, median_height = get_pdf_page_size_pts(docs)
    # 将中位数宽度转换为整数
    page_width = int(median_width)
    # 将中位数高度转换为整数
    page_height = int(median_height)
    # 获取期望的图像信息
    img_sz_list = test_data[book_name]["expected_image_info"]
    # 获取文档总页数
    total_page = len(docs)
    # 获取每页的文本长度
    text_len_list = get_pdf_textlen_per_page(docs)
    # 根据分类函数判断是否为扫描版
    bool_classify_by_area = classify_by_area(total_page, page_width, page_height, img_sz_list, text_len_list)
    # 断言结果与期望一致
    assert bool_classify_by_area == expected_bool_classify_by_area


'''
广义上的文字版检测，任何一页大于100字，都认为为文字版
'''
# 使用 pytest 参数化，测试文字版分类基于文本长度的函数
@pytest.mark.parametrize("book_name, expected_bool_classify_by_text_len",
                         [
                             # 测试样例：文件名及预期结果（少于50页）
                             ("scihub/scihub_67200000/libgen.scimag67237000-67237999.zip_10.1515/crpm-2017-0020", True),  # 文字版，少于50页
                             # 测试样例：文件名及预期结果（多于50页）
                             ("scihub/scihub_83300000/libgen.scimag83306000-83306999.zip_10.1007/978-3-658-30153-8", True),  # 文字版，多于50页
                             # 测试样例：文件名及预期结果（无字的宣传册）
                             ("zhongwenzaixian/zhongwenzaixian_65771414", False),  # 完全无字的宣传册
                         ])
# 定义测试函数，参数为书名和预期布尔分类结果
def test_classify_by_text_len(book_name, expected_bool_classify_by_text_len):
    # 从测试 PDF 中获取文档
    docs = get_docs_from_test_pdf(book_name)
    # 获取每页文本长度
    text_len_list = get_pdf_textlen_per_page(docs)
    # 计算文档总页数
    total_page = len(docs)
    # 根据每页文本长度和总页数进行分类
    bool_classify_by_text_len = classify_by_text_len(text_len_list, total_page)
    # 断言分类结果与预期相符
    assert bool_classify_by_text_len == expected_bool_classify_by_text_len


'''
狭义上的文字版检测，需要平均每页字数大于200字
'''
# 使用 pytest 参数化，测试文字版分类基于平均每页字数的函数
@pytest.mark.parametrize("book_name, expected_bool_classify_by_avg_words",
                         [
                             # 测试样例：扫描版，书末尾几页有大纲文字
                             ("zlib/zlib_21207669", False),  # 扫描版，书末尾几页有大纲文字
                             # 测试样例：扫描版，好几本扫描书的集合
                             ("zlib/zlib_19012845", False),  # 扫描版，好几本扫描书的集合，每本书末尾有一页文字页
                             # 测试样例：正常文字版
                             ("scihub/scihub_67200000/libgen.scimag67237000-67237999.zip_10.1515/crpm-2017-0020", True),# 正常文字版
                             # 测试样例：宣传册
                             ("zhongwenzaixian/zhongwenzaixian_65771414", False),  # 宣传册
                             # 测试样例：图解书或无字书
                             ("zhongwenzaixian/zhongwenzaixian_351879", False),  # 图解书/无字or少字
                             # 测试样例：书法集
                             ("zhongwenzaixian/zhongwenzaixian_61357496_pdfvector", False),  # 书法集
                             # 测试样例：设计图
                             ("zhongwenzaixian/zhongwenzaixian_63684541", False),  # 设计图
                             # 测试样例：绘本
                             ("zhongwenzaixian/zhongwenzaixian_61525978", False),  # 绘本
                             # 测试样例：摄影集
                             ("zhongwenzaixian/zhongwenzaixian_63679729", False),  # 摄影集
                         ])
# 定义测试函数，参数为书名和预期布尔分类结果
def test_classify_by_avg_words(book_name, expected_bool_classify_by_avg_words):
    # 从测试 PDF 中获取文档
    docs = get_docs_from_test_pdf(book_name)
    # 获取每页文本长度
    text_len_list = get_pdf_textlen_per_page(docs)
    # 根据平均每页字数进行分类
    bool_classify_by_avg_words = classify_by_avg_words(text_len_list)
    # 断言分类结果与预期相符
    assert bool_classify_by_avg_words == expected_bool_classify_by_avg_words


'''
这个规则只针对特殊扫描版1，因为扫描版1的图片信息都由于junk_list的原因被舍弃了，只能通过图片数量来判断
'''
# 使用 pytest 参数化，测试特殊扫描版分类基于图片数量的函数
@pytest.mark.parametrize("book_name, expected_bool_classify_by_img_num",
                         [
                             # 测试样例：特殊扫描版1，图占比大
                             ("zlib/zlib_21370453", False),  # 特殊扫描版1，每页都有所有扫描页图片，特点是图占比大，每页展示1至n张
                             # 测试样例：特殊扫描版2，类似特1，但每页数量不完全相等
                             ("zlib/zlib_22115997", False),  # 特殊扫描版2，类似特1，但是每页数量不完全相等
                             # 测试样例：特殊扫描版3，类似特1，但每页数量不完全相等
                             ("zlib/zlib_21814957", False),  # 特殊扫描版3，类似特1，但是每页数量不完全相等
                             # 测试样例：特殊扫描版4，类似特1，但每页数量不完全相等
                             ("zlib/zlib_21814955", False),  # 特殊扫描版4，类似特1，但是每页数量不完全相等
                         ])
# 定义测试函数，参数为书名和预期布尔分类结果
def test_classify_by_img_num(book_name, expected_bool_classify_by_img_num):
    # 从 JSON 文件中获取测试数据
    test_data = get_test_json_data(current_directory, "test_metascan_classify_data.json")
    # 从测试 PDF 中获取文档
    docs = get_docs_from_test_pdf(book_name)
    # 获取每页图片数量
    img_num_list = get_imgs_per_page(docs)
    # 从测试数据中提取指定书籍的预期图像信息
        img_sz_list = test_data[book_name]["expected_image_info"]
        # 根据图像大小列表和图像数量列表判断分类是否正确
        bool_classify_by_img_num = classify_by_img_num(img_sz_list, img_num_list)
        # 断言分类结果是否与预期结果一致
        assert bool_classify_by_img_num == expected_bool_classify_by_img_num
'''
# 排除纵向排版的pdf
'''
# 使用 pytest 的参数化功能，提供多个测试案例
@pytest.mark.parametrize("book_name, expected_bool_classify_by_text_layout",
                         [
                             # 测试竖排版本1，预期分类结果为 False
                             ("vertical_detection/三国演义_繁体竖排版", False),  # 竖排版本1
                             # 测试竖排版本2，预期分类结果为 False
                             ("vertical_detection/净空法师_大乘无量寿", False),  # 竖排版本2
                             # 测试横排版本1，预期分类结果为 True
                             ("vertical_detection/om3006239", True),  # 横排版本1
                             # 测试横排版本2，预期分类结果为 True
                             ("vertical_detection/isit.2006.261791", True),  # 横排版本2
                         ])
# 定义测试函数，接收书名和预期分类结果作为参数
def test_classify_by_text_layout(book_name, expected_bool_classify_by_text_layout):
    # 从 JSON 文件中获取测试数据
    test_data = get_test_json_data(current_directory, "test_metascan_classify_data.json")
    # 获取特定书名的预期文本排版信息
    text_layout_per_page = test_data[book_name]["expected_text_layout"]
    # 根据页面的文本排版进行分类
    bool_classify_by_text_layout = classify_by_text_layout(text_layout_per_page)
    # 断言实际分类结果与预期结果相符
    assert bool_classify_by_text_layout == expected_bool_classify_by_text_layout


'''
# 通过检测页面是否由多个窄长条图像组成，来过滤特殊的扫描版
# 这个规则只对窄长条组成的pdf进行识别，而不会识别常规的大图扫描pdf
'''
# 使用 pytest 的参数化功能，提供多个测试案例
@pytest.mark.parametrize("book_name, expected_bool_classify_by_img_narrow_strips",
                         [
                             # 测试特殊扫描版，预期分类结果为 False
                             ("scihub/scihub_25900000/libgen.scimag25991000-25991999.zip_10.2307/40066695", False),  # 特殊扫描版
                             # 测试特殊扫描版4，预期分类结果为 False
                             ("the_eye/the_eye_wtl_00023799", False),  # 特殊扫描版4，每一页都是一张张小图拼出来的，检测图片占比之前需要先按规则把小图拼成大图
                             # 测试特殊扫描版5，预期分类结果为 False
                             ("the_eye/the_eye_cdn_00328381", False),  # 特殊扫描版5，每一页都是一张张小图拼出来的，存在多个小图多次重复使用情况，检测图片占比之前需要先按规则把小图拼成大图
                             # 测试特殊扫描版7，预期分类结果为 False
                             ("scanned_detection/llm-raw-scihub-o.O-0584-8539%2891%2980165-f", False),  # 特殊扫描版7，只有一页且由小图拼成大图
                             # 测试特殊扫描版6，预期分类结果为 True
                             ("scihub/scihub_25800000/libgen.scimag25889000-25889999.zip_10.2307/4153991", True),  # 特殊扫描版6，只有三页，其中两页是扫描版
                             # 测试特殊扫描版8，预期分类结果为 True
                             ("scanned_detection/llm-raw-scihub-o.O-bf01427123", True),  # 特殊扫描版8，只有3页且全是大图扫描版
                             # 测试特殊文本版，预期分类结果为 True
                             ("scihub/scihub_53700000/libgen.scimag53724000-53724999.zip_10.1097/00129191-200509000-00018", True),  # 特殊文本版，有一长条，但是只有一条
                         ])
# 定义测试函数，接收书名和预期分类结果作为参数
def test_classify_by_img_narrow_strips(book_name, expected_bool_classify_by_img_narrow_strips):
    # 从 JSON 文件中获取测试数据
    test_data = get_test_json_data(current_directory, "test_metascan_classify_data.json")
    # 获取特定书名的预期图像信息
    img_sz_list = test_data[book_name]["expected_image_info"]
    # 从 PDF 中获取文档数据
    docs = get_docs_from_test_pdf(book_name)
    # 获取 PDF 页面的中位数宽度和高度
    median_width, median_height = get_pdf_page_size_pts(docs)
    # 将中位数宽度和高度转为整数
    page_width = int(median_width)
    page_height = int(median_height)
    # 根据页面宽度、高度和图像信息进行分类
    bool_classify_by_img_narrow_strips = classify_by_img_narrow_strips(page_width, page_height, img_sz_list)
    # 断言实际分类结果与预期结果相符
    assert bool_classify_by_img_narrow_strips == expected_bool_classify_by_img_narrow_strips
```