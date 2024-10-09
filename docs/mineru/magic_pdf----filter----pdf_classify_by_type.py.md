# `.\MinerU\magic_pdf\filter\pdf_classify_by_type.py`

```
"""
根据利用meta_scan得到的结果，对pdf是否为文字版进行分类。
定义标准：
一、什么pdf会是文字pdf，只要满足以下任意一条
  1. 随机抽取N页，如果有任何一页文字数目大于100
  2. 只要存在一个页面，图片的数量为0
二、什么是扫描版pdf，只要满足以下任意一条
  1. ~~80%页面上的最大图大小一样并且面积超过页面面积0.6~~
  2. 大部分页面上文字的长度都是相等的。
"""
# 导入json库，用于处理JSON数据
import json
# 导入sys库，用于访问与Python解释器交互的变量和函数
import sys
# 从collections模块导入Counter，用于计数器功能
from collections import Counter

# 导入click库，用于创建命令行界面
import click
# 导入numpy库，常用于数值计算
import numpy as np
# 导入loguru库，用于记录日志
from loguru import logger

# 从自定义模块导入mymax和get_top_percent_list函数
from magic_pdf.libs.commons import mymax, get_top_percent_list
# 从pdf_meta_scan模块导入scan_max_page和junk_limit_min
from magic_pdf.filter.pdf_meta_scan import scan_max_page, junk_limit_min

# 定义文字长度的阈值，超过此值认为是文字pdf
TEXT_LEN_THRESHOLD = 100
# 定义平均文字长度的阈值
AVG_TEXT_LEN_THRESHOLD = 100
# 定义抽样比例，0.1表示抽取10%的页面进行文字长度统计
TEXT_LEN_SAMPLE_RATIO = 0.1  # 抽取0.1的页面进行文字长度统计

# 一个拼接图片的方案，将某些特殊扫描版本的拆图拼成一张整图
def merge_images(image_list, page_width, page_height, max_offset=5, max_gap=2):
    # 先通过set去除所有bbox重叠的图片数据
    image_list_result = []  # 初始化结果列表
    for page_images in image_list:  # 遍历每一页的图片
        page_result = []  # 存储去重后的图片
        dedup = set()  # 用于记录已经处理的bbox
        for img in page_images:  # 遍历当前页的每一张图片
            x0, y0, x1, y1, img_bojid = img  # 解包图片的坐标和ID
            if (x0, y0, x1, y1) in dedup:  # 检查当前bbox是否已经处理过
                continue  # 如果已处理，跳过
            else:
                dedup.add((x0, y0, x1, y1))  # 将当前bbox添加到去重集合中
                page_result.append([x0, y0, x1, y1, img_bojid])  # 将当前图片添加到结果中
        image_list_result.append(page_result)  # 将当前页的结果添加到最终结果中

    # 接下来，将同一页可拼接的图片进行合并
    merged_images = []  # 初始化合并后的图片列表
    for page_images in image_list_result:  # 遍历去重后的图片列表
        if not page_images:  # 如果当前页没有图片
            continue  # 跳过

        # 先将同一页的图片从上到下，从左到右进行排序
        page_images.sort(key=lambda img: (img[1], img[0]))  # 根据y坐标和x坐标排序

        merged = [page_images[0]]  # 初始化合并列表，以第一张图片为起始

        for img in page_images[1:]:  # 遍历第二张及以后的图片
            x0, y0, x1, y1, imgid = img  # 解包图片的坐标和ID

            last_img = merged[-1]  # 获取最后一张合并的图片
            last_x0, last_y0, last_x1, last_y1, last_imgid = last_img  # 解包最后一张图片的坐标和ID

            # 单张图片宽或者高覆盖页面宽高的9成以上是拼图的一个前置条件
            full_width = abs(x1 - x0) >= page_width * 0.9  # 检查图片宽度是否达到页面宽度的90%
            full_height = abs(y1 - y0) >= page_height * 0.9  # 检查图片高度是否达到页面高度的90%

            # 如果宽达标，检测是否能竖着拼
            if full_width:
                # 竖着拼需要满足两个前提，左右边界各偏移不能超过 max_offset，第一张图的下边界和第二张图的上边界偏移不能超过 max_gap
                close1 = (last_x0 - max_offset) <= x0 <= (last_x0 + max_offset) and (last_x1 - max_offset) <= x1 <= (
                            last_x1 + max_offset) and (last_y1 - max_gap) <= y0 <= (last_y1 + max_gap)

            # 如果高达标，检测是否可以横着拼
            if full_height:
                # 横着拼需要满足两个前提，上下边界各偏移不能超过 max_offset，第一张图的右边界和第二张图的左边界偏移不能超过 max_gap
                close2 = (last_y0 - max_offset) <= y0 <= (last_y0 + max_offset) and (last_y1 - max_offset) <= y1 <= (
                            last_y1 + max_offset) and (last_x1 - max_gap) <= x0 <= (last_x1 + max_gap)

            # Check if the image can be merged with the last image
            if (full_width and close1) or (full_height and close2):  # 检查是否满足合并条件
                # Merge the image with the last image
                merged[-1] = [min(x0, last_x0), min(y0, last_y0),
                              max(x1, last_x1), max(y1, last_y1), imgid]  # 更新合并的图片信息
            else:
                # Add the image as a new image
                merged.append(img)  # 将当前图片添加到合并列表中

        merged_images.append(merged)  # 将合并后的结果添加到最终列表中

    return merged_images  # 返回合并后的图片列表
# 根据页面总数、页面宽度和高度、图像大小列表及文本长度列表进行分类
def classify_by_area(total_page: int, page_width, page_height, img_sz_list, text_len_list: list):
    """
    80%页面上的最大图大小一样并且面积超过页面面积0.6则返回False，否则返回True
    :param pdf_path:
    :param total_page:
    :param page_width:
    :param page_height:
    :param img_sz_list:
    :return:
    """
    # 检查是否存在没有图片的页面
    # if any([len(img_sz) == 0 for img_sz in img_sz_list]):  # 含有不含图片的页面
    #     # 找到没有图片的页面索引
    #     empty_page_index = [i for i, img_sz in enumerate(img_sz_list) if len(img_sz) == 0]
    #     # 检查这些页面是否有文字
    #     text_len_at_page_idx = [text_len for i, text_len in enumerate(text_len_list) if i in empty_page_index and text_len > 0]
    #     if len(text_len_at_page_idx) > TEXT_LEN_THRESHOLD:  # 页面没有图片但有足够文字量，可能是文字版
    #         return True

    # 统计每个图片的objid出现次数，去除重复出现超过10次的透明图片
    objid_cnt = Counter([objid for page_img_sz in img_sz_list for _, _, _, _, objid in page_img_sz])
    # 限制总页数，若超过scan_max_page，则只扫描前scan_max_page页
    if total_page >= scan_max_page:  # 新的meta_scan只扫描前 scan_max_page 页
        total_page = scan_max_page

    repeat_threshold = 2  # 设置bad_image的出现阈值为2
    # bad_image_objid为出现次数超过阈值的objid集合
    bad_image_objid = set([objid for objid, cnt in objid_cnt.items() if cnt >= repeat_threshold])
    # bad_image_page_idx = [i for i, page_img_sz in enumerate(img_sz_list) if any([objid in bad_image_objid for _, _, _, _, objid in page_img_sz])]
    # text_len_at_bad_image_page_idx = [text_len for i, text_len in enumerate(text_len_list) if i in bad_image_page_idx and text_len > 0]

    # 处理特殊情况，检查是否存在覆盖大透明图片的文字版PDF
    # fake_image_ids = [objid for objid in bad_image_objid if
    #                   any([abs((x1 - x0) * (y1 - y0) / page_width * page_height) > 0.9 for images in img_sz_list for
    #                        x0, y0, x1, y1, _ in images])]  # 原代码逻辑检查

    # if len(fake_image_ids) > 0 and any([l > TEXT_LEN_THRESHOLD for l in text_len_at_bad_image_page_idx]):  # 透明图片页面有足够文字
    #     return True

    # 过滤掉出现重复的图片
    img_sz_list = [[img_sz for img_sz in page_img_sz if img_sz[-1] not in bad_image_objid] for page_img_sz in
                   img_sz_list]  # 过滤掉重复出现的图片

    # 合并拆分的图片，确保计算面积时完整
    img_sz_list = merge_images(img_sz_list, page_width, page_height)

    # 计算每个页面上最大图片的面积及其占页面面积的比例
    max_image_area_per_page = [mymax([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1, _ in page_img_sz]) for page_img_sz in
                               img_sz_list]
    page_area = page_width * page_height
    # 将 max_image_area_per_page 中的每个面积除以 page_area，更新每个元素为比例值
    max_image_area_per_page = [area / page_area for area in max_image_area_per_page]
    # 过滤 max_image_area_per_page，保留大于 0.5 的比例值
    max_image_area_per_page = [area for area in max_image_area_per_page if area > 0.5]

    # 检查 max_image_area_per_page 的长度是否大于或等于总页数的一半
    if len(max_image_area_per_page) >= 0.5 * total_page:  # 阈值从0.8改到0.5，适配3页里面有两页和两页里面有一页的情况
        # 如果条件成立，意味着反复出现的隐藏透明图层图片已被移除，这些图片的 id 是相同的
        return False
    else:
        # 如果条件不成立，返回 True
        return True
# 根据页面文字长度分类PDF文档类型
def classify_by_text_len(text_len_list: list, total_page: int):
    # 随机抽取10%的页面，如果页面数量少于5，则取全部页面
    select_page_cnt = int(total_page * TEXT_LEN_SAMPLE_RATIO)  # 选取10%的页面
    if select_page_cnt < 5:  # 如果选取的页面数量少于5
        select_page_cnt = total_page  # 则选择全部页面

    # # 排除头尾各10页
    # if total_page > 20:  # 如果总页数大于20
    #     page_range = list(range(10, total_page - 10))  # 从第11页到倒数第11页
    # else:
    #     page_range = list(range(total_page))  # 否则选择所有页面
    # page_num = np.random.choice(page_range, min(select_page_cnt, len(page_range)), replace=False)
    # 排除前后10页对只有21，22页的pdf很尴尬，如果选出来的中间那一两页恰好没字容易误判，有了avg_words规则，这个规则可以忽略
    page_num = np.random.choice(total_page, select_page_cnt, replace=False)  # 随机选择页面编号
    text_len_lst = [text_len_list[i] for i in page_num]  # 根据选中的页面编号获取对应的文字长度
    is_text_pdf = any([text_len > TEXT_LEN_THRESHOLD for text_len in text_len_lst])  # 判断是否存在文字长度超过阈值的页面
    return is_text_pdf  # 返回是否为文字PDF的判断结果


# 根据每页平均字数分类PDF文档类型
def classify_by_avg_words(text_len_list: list):
    # 补充规则，如果平均每页字数少于 AVG_TEXT_LEN_THRESHOLD，就不是文字pdf
    sum_words = sum(text_len_list)  # 计算总字数
    count_of_numbers = len(text_len_list)  # 计算页面数量
    if count_of_numbers == 0:  # 如果没有页面
        is_text_pdf = False  # 则不是文字PDF
    else:
        avg_words = round(sum_words / count_of_numbers)  # 计算平均字数
        if avg_words > AVG_TEXT_LEN_THRESHOLD:  # 如果平均字数超过阈值
            is_text_pdf = True  # 则为文字PDF
        else:
            is_text_pdf = False  # 否则不是文字PDF

    return is_text_pdf  # 返回判断结果


# 根据图片数量分类PDF文档类型
def classify_by_img_num(img_sz_list: list, img_num_list: list):
    # 补充规则，检测扫描版本的PDF
    count_img_sz_list_not_none = sum(1 for item in img_sz_list if item)  # 计算img_sz_list中非空元素的个数
    top_eighty_percent = get_top_percent_list(img_num_list, 0.8)  # 获取前80%的元素
    # 判断条件：非空元素数量小于等于1，前80%元素相等且最大值满足条件
    if count_img_sz_list_not_none <= 1 and len(set(top_eighty_percent)) == 1 and max(img_num_list) >= junk_limit_min:
        return False  # 如果满足条件，返回False，说明不是文字版PDF
    else:
        return True  # 不满足条件，返回True，可能是文字版PDF


# 根据文本布局分类PDF文档类型
def classify_by_text_layout(text_layout_per_page: list):
    # 判断文本布局是否以竖排为主
    count_vertical = sum(1 for item in text_layout_per_page if item == 'vertical')  # 统计竖排数量
    # 统计text_layout_per_page中横排的个数
    # 计算text_layout_per_page中横排的数量
    count_horizontal = sum(1 for item in text_layout_per_page if item == 'horizontal')
    # 计算已知布局的总数（竖排加横排）
    known_layout_cnt = count_vertical + count_horizontal
    # 如果已知布局数量不为零
    if known_layout_cnt != 0:
        # 计算竖排占已知布局的比例
        ratio = count_vertical / known_layout_cnt
        # 如果竖排比例大于等于0.5，认为不是文字版pdf
        if ratio >= 0.5:  # 阈值设为0.5，适配3页里面有2页和两页里有一页的情况
            return False  # 文本布局以竖排为主，认为不是文字版pdf
        else:
            return True  # 文本布局以横排为主，认为是文字版pdf
    else:
        # 如果布局未知，默认认为不是文字版pdf
        return False  # 文本布局未知，默认认为不是文字版pdf
# 判断一页是否由细长条组成的函数，接收页面宽度、高度和图片尺寸列表作为参数
def classify_by_img_narrow_strips(page_width, page_height, img_sz_list):
    """
    判断一页是否由细长条组成，有两个条件：
    1. 图片的宽或高达到页面宽或高的90%，且长边需要是窄边长度的数倍以上
    2. 整个页面所有的图片有80%以上满足条件1

    Args:
        page_width (float): 页面宽度
        page_height (float): 页面高度
        img_sz_list (list): 图片尺寸列表，每个元素为一个元组，表示图片的矩形区域和尺寸，形如(x0, y0, x1, y1, size)，其中(x0, y0)为矩形区域的左上角坐标，(x1, y1)为矩形区域的右下角坐标，size为图片的尺寸

    Returns:
        bool: 如果满足条件的页面的比例小于0.5，返回True，否则返回False
    """

    # 定义内部函数，判断单个图片是否为细长条
    def is_narrow_strip(img):
        # 解包图片的坐标和尺寸
        x0, y0, x1, y1, _ = img
        # 计算图片的宽度和高度
        width, height = x1 - x0, y1 - y0
        # 返回是否满足细长条的两个条件之一
        return any([
            # 图片宽度大于等于页面宽度的90%，且宽度大于等于高度4倍
            width >= page_width * 0.9 and width >= height * 4,
            # 图片高度大于等于页面高度的90%，且高度大于等于宽度4倍
            height >= page_height * 0.9 and height >= width * 4,
        ])

    # 初始化满足条件的页面数量
    narrow_strip_pages_count = 0

    # 遍历所有页面的图片列表
    for page_img_list in img_sz_list:
        # 忽略空页面
        if not page_img_list:
            continue

        # 计算页面中的图片总数
        total_images = len(page_img_list)

        # 计算页面中细长条图片的数量
        narrow_strip_images_count = 0
        for img in page_img_list:
            # 如果该图片是细长条，则计数加一
            if is_narrow_strip(img):
                narrow_strip_images_count += 1
        # 如果细长条图片的数量少于5，跳过该页面
        if narrow_strip_images_count < 5:
            continue
        else:
            # 如果细长条图片的比例大于或等于0.8，增加满足条件的页面数量
            if narrow_strip_images_count / total_images >= 0.8:
                narrow_strip_pages_count += 1

    # 计算满足条件的页面的比例
    narrow_strip_pages_ratio = narrow_strip_pages_count / len(img_sz_list)

    # 返回满足条件的页面比例是否小于0.5
    return narrow_strip_pages_ratio < 0.5


# 分类函数，根据各种条件进行分类
def classify(total_page: int, page_width, page_height, img_sz_list: list, text_len_list: list, img_num_list: list,
             text_layout_list: list, invalid_chars: bool):
    """
    这里的图片和页面长度单位是pts
    :param total_page:
    :param text_len_list:
    :param page_width:
    :param page_height:
    :param img_sz_list:
    :param pdf_path:
    :return:
    """
    # 结果字典，存储各种分类结果
    results = {
        # 根据图片面积进行分类
        'by_image_area': classify_by_area(total_page, page_width, page_height, img_sz_list, text_len_list),
        # 根据文本长度进行分类
        'by_text_len': classify_by_text_len(text_len_list, total_page),
        # 根据平均单词数进行分类
        'by_avg_words': classify_by_avg_words(text_len_list),
        # 根据图片数量进行分类
        'by_img_num': classify_by_img_num(img_sz_list, img_num_list),
        # 根据文本布局进行分类
        'by_text_layout': classify_by_text_layout(text_layout_list),
        # 根据细长条图片进行分类
        'by_img_narrow_strips': classify_by_img_narrow_strips(page_width, page_height, img_sz_list),
        # 根据无效字符进行分类
        'by_invalid_chars': invalid_chars,
    }

    # 如果所有分类结果都为真，返回真和结果字典
    if all(results.values()):
        return True, results
    # 如果所有分类结果都为假，返回假和结果字典
    elif not any(results.values()):
        return False, results
    # 如果不满足分类条件，执行以下代码
        else:
            # 记录警告信息，指出PDF未能根据区域和文本长度进行分类，打印各个分类结果
            logger.warning(
                f"pdf is not classified by area and text_len, by_image_area: {results['by_image_area']},"
                f" by_text: {results['by_text_len']}, by_avg_words: {results['by_avg_words']}, by_img_num: {results['by_img_num']},"
                f" by_text_layout: {results['by_text_layout']}, by_img_narrow_strips: {results['by_img_narrow_strips']},"
                f" by_invalid_chars: {results['by_invalid_chars']}",
                file=sys.stderr)  # 将警告信息输出到标准错误流，便于快速识别特殊PDF并修正分类算法
            # 返回分类结果为假，并附带处理结果
            return False, results
# 定义命令行接口命令 main
@click.command()
# 定义命令行选项，指定 JSON 文件
@click.option("--json-file", type=str, help="pdf信息")
# 主函数，接收 JSON 文件名作为参数
def main(json_file):
    # 检查 JSON 文件名是否为 None
    if json_file is None:
        # 输出错误信息到标准错误流
        print("json_file is None", file=sys.stderr)
        # 退出程序
        exit(0)
    try:
        # 以只读方式打开 JSON 文件
        with open(json_file, "r") as f:
            # 遍历文件中的每一行
            for l in f:
                # 跳过空行
                if l.strip() == "":
                    continue
                # 解析 JSON 行为字典对象
                o = json.loads(l)
                # 获取 PDF 总页数
                total_page = o["total_page"]
                # 获取页面宽度（单位为点）
                page_width = o["page_width_pts"]
                # 获取页面高度（单位为点）
                page_height = o["page_height_pts"]
                # 获取每页的图像信息列表
                img_sz_list = o["image_info_per_page"]
                # 获取每页的文本长度列表
                text_len_list = o['text_len_per_page']
                # 获取每页的文本布局列表
                text_layout_list = o['text_layout_per_page']
                # 获取 PDF 文件路径
                pdf_path = o['pdf_path']
                # 获取 PDF 是否加密的标志
                is_encrypted = o['is_encrypted']
                # 获取 PDF 是否需要密码的标志
                is_needs_password = o['is_needs_password']
                # 如果 PDF 加密或没有页面或需要密码，则跳过处理
                if is_encrypted or total_page == 0 or is_needs_password:
                    continue
                # 根据 PDF 信息进行分类
                tag = classify(total_page, page_width, page_height, img_sz_list, text_len_list, text_layout_list)
                # 将分类结果存入字典中
                o['is_text_pdf'] = tag
                # 将字典转换为 JSON 格式并打印
                print(json.dumps(o, ensure_ascii=False))
    except Exception as e:
        # 捕获异常并输出错误信息到标准错误流
        print("ERROR: ", e, file=sys.stderr)


# 如果当前模块是主程序，执行 main 函数
if __name__ == "__main__":
    main()
    # 以下是注释掉的代码示例，不会被执行
    # false = False
    # true = True
    # null = None
    # o = json.loads(json.dumps(o))
    # total_page = o["total_page"]
    # page_width = o["page_width_pts"]
    # page_height = o["page_height_pts"]
    # img_sz_list = o["image_info_per_page"]
    # text_len_list = o['text_len_per_page']
    # pdf_path = o['pdf_path']
    # is_encrypted = o['is_encrypted']
    # is_needs_password = o['is_needs_password']
    # if is_encrypted or total_page == 0 or is_needs_password:  # 加密的，需要密码的，没有页面的，都不处理
    #     print("加密的")
    #     exit(0)
    # tag = classify(pdf_path, total_page, page_width, page_height, img_sz_list, text_len_list)
    # o['is_text_pdf'] = tag
    # print(json.dumps(o, ensure_ascii=False))
```