# `.\MinerU\magic_pdf\filter\pdf_meta_scan.py`

```
# 输入： s3路径，每行一个
# 输出： pdf文件元信息，包括每一页上的所有图片的长宽高，bbox位置
"""
import sys  # 导入系统相关模块
import click  # 导入命令行工具模块

from magic_pdf.libs.commons import read_file, mymax, get_top_percent_list  # 从commons模块导入函数
from magic_pdf.libs.commons import fitz  # 导入fitz库以处理PDF
from loguru import logger  # 导入logger用于日志记录
from collections import Counter  # 从collections模块导入Counter以计数

from magic_pdf.libs.drop_reason import DropReason  # 导入DropReason模块
from magic_pdf.libs.language import detect_lang  # 导入语言检测模块
from magic_pdf.libs.pdf_check import detect_invalid_chars  # 导入PDF检查模块

scan_max_page = 50  # 设置扫描最大页面数为50
junk_limit_min = 10  # 设置垃圾图像的最小限制为10


def calculate_max_image_area_per_page(result: list, page_width_pts, page_height_pts):
    # 计算每页图片最大面积的比例
    max_image_area_per_page = [mymax([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1, _ in page_img_sz]) for page_img_sz in
                               result]  # 计算每页图片的最大面积
    page_area = int(page_width_pts) * int(page_height_pts)  # 计算页面的总面积
    max_image_area_per_page = [area / page_area for area in max_image_area_per_page]  # 计算每页图片面积占比
    max_image_area_per_page = [area for area in max_image_area_per_page if area > 0.6]  # 筛选占比大于0.6的图片
    return max_image_area_per_page  # 返回每页图片面积占比列表


def process_image(page, junk_img_bojids=[]):
    # 处理每个页面的图像，返回包含图像信息的列表
    page_result = []  # 存每个页面里的多张图四元组信息
    items = page.get_images()  # 获取页面中的所有图像
    dedup = set()  # 用于去重的集合
    for img in items:  # 遍历每个图像
        img_bojid = img[0]  # 获取图像的全局唯一标识
        if img_bojid in junk_img_bojids:  # 如果是垃圾图像，就跳过
            continue
        recs = page.get_image_rects(img, transform=True)  # 获取图像在页面上的矩形区域
        if recs:  # 如果矩形区域存在
            rec = recs[0][0]  # 获取第一个矩形区域
            x0, y0, x1, y1 = map(int, rec)  # 将矩形区域坐标转换为整数
            width = x1 - x0  # 计算图像宽度
            height = y1 - y0  # 计算图像高度
            if (x0, y0, x1, y1, img_bojid) in dedup:  # 如果已经记录过该图像的区域，跳过
                continue
            if not all([width, height]):  # 长和宽任何一个都不能是0
                continue
            dedup.add((x0, y0, x1, y1, img_bojid))  # 将图像区域记录到去重集合中
            page_result.append([x0, y0, x1, y1, img_bojid])  # 添加图像信息到结果列表
    return page_result  # 返回每个页面的图像信息


def get_image_info(doc: fitz.Document, page_width_pts, page_height_pts) -> list:
    # 返回每个页面里的图片的四元组，每个页面多个图片。
    :param doc: PDF文档对象
    :return: 每个页面的图像信息
    """
    img_bojid_counter = Counter(img[0] for page in doc for img in page.get_images())  # 统计每个图像标识的出现次数
    junk_limit = max(len(doc) * 0.5, junk_limit_min)  # 计算垃圾图像的阈值

    junk_img_bojids = [img_bojid for img_bojid, count in img_bojid_counter.items() if count >= junk_limit]  # 筛选出垃圾图像标识

    #todo 加个判断，用前十页就行，这些垃圾图片需要满足两个条件，不止出现的次数要足够多，而且图片占书页面积的比例要足够大，且图与图大小都差不多
    # 有两种扫描版，一种文字版，这里可能会有误判
    # 扫描版1：每页都有所有扫描页图片，特点是图占比大，每页展示1张
    # 扫描版2，每页存储的扫描页图片数量递增，特点是图占比大，每页展示1张，需要清空junklist跑前50页图片信息用于分类判断
    # 文字版1.每页存储所有图片，特点是图片占页面比例不大，每页展示可能为0也可能不止1张 这种pdf需要拿前10页抽样检测img大小和个数，如果符合需要清空junklist
    imgs_len_list = [len(page.get_images()) for page in doc]  # 统计每个页面的图像数量

    special_limit_pages = 10  # 设置特殊限制页面数量为10

    # 统一用前十页结果做判断
    result = []  # 初始化结果列表
    break_loop = False  # 初始化循环退出标志
    # 遍历文档中的每一页，并获取当前索引和页面对象
    for i, page in enumerate(doc):
        # 如果需要中断循环，直接退出
        if break_loop:
            break
        # 如果索引超出特殊限制页数，直接退出
        if i >= special_limit_pages:
            break
        # 处理当前页面的图像信息，返回处理结果
        page_result = process_image(page)  # 这里不传junk_img_bojids，拿前十页所有图片信息用于后续分析
        # 将当前页面的处理结果添加到结果列表中
        result.append(page_result)
        # 遍历结果列表中的每个项目
        for item in result:
            # 如果当前项目没有任何元素，说明该页为纯文本版，需要进一步判断
            if not any(item):  # 如果任何一页没有图片，说明是个文字版，需要判断是否为特殊文字版
                # 判断图片长度列表的最大值是否等于最小值且大于垃圾图片的最小限制
                if max(imgs_len_list) == min(imgs_len_list) and max(
                        imgs_len_list) >= junk_limit_min:  # 如果是特殊文字版，就把junklist置空并break
                    # 清空垃圾图片列表
                    junk_img_bojids = []
                else:  # 不是特殊文字版，是个普通文字版，但是存在垃圾图片，不置空junklist
                    # 对于普通文本版，保持垃圾图片列表不变
                    pass
                # 设置中断循环标志为 True
                break_loop = True
                break
    # 如果没有中断循环，则继续进行判断
    if not break_loop:
        # 获取图片长度列表的前80%元素
        top_eighty_percent = get_top_percent_list(imgs_len_list, 0.8)
        # 检查前80%的元素是否都相等
        if len(set(top_eighty_percent)) == 1 and max(imgs_len_list) >= junk_limit_min:

            # 前10页都有图，根据每页图片数量是否相等判断是否需要清除junklist
            # if max(imgs_len_list) == min(imgs_len_list) and max(imgs_len_list) >= junk_limit_min:

            # 前10页都有图，且每页数量一致，需要检测图片大小占页面的比例判断是否需要清除junklist
            # 计算每页最大图像面积
            max_image_area_per_page = calculate_max_image_area_per_page(result, page_width_pts, page_height_pts)
            # 如果前10页不全是大图，则清空垃圾图片列表
            if len(max_image_area_per_page) < 0.8 * special_limit_pages:  # 前10页不全是大图，说明可能是个文字版pdf，把垃圾图片list置空
                junk_img_bojids = []
            else:  # 前10页都有图，而且80%都是大图，且每页图片数量一致并都很多，说明是扫描版1，不需要清空junklist
                # 保持垃圾图片列表不变
                pass
        else:  # 每页图片数量不一致，需要清掉junklist全量跑前50页图片
            # 清空垃圾图片列表
            junk_img_bojids = []

    # 正式进入取前50页图片的信息流程
    result = []
    # 遍历文档中的每一页，限制在最大扫描页数内
    for i, page in enumerate(doc):
        # 如果索引超出最大扫描页数，直接退出
        if i >= scan_max_page:
            break
        # 处理当前页面的图像信息，并传入垃圾图片列表
        page_result = process_image(page, junk_img_bojids)
        # 记录当前页面的图片长度
        # logger.info(f"page {i} img_len: {len(page_result)}")
        # 将处理结果添加到结果列表中
        result.append(page_result)

    # 返回处理结果和垃圾图片列表
    return result, junk_img_bojids
# 获取PDF文档每一页的大小，单位为点（pts）
def get_pdf_page_size_pts(doc: fitz.Document):
    # 获取文档的页数
    page_cnt = len(doc)
    # 取页数与50的较小值，以限制处理的页数
    l: int = min(page_cnt, 50)
    # 初始化存放宽度和高度的列表
    page_width_list = []
    page_height_list = []
    # 遍历前l页，获取每页的宽度和高度
    for i in range(l):
        page = doc[i]
        page_rect = page.rect
        # 将当前页的宽度添加到宽度列表
        page_width_list.append(page_rect.width)
        # 将当前页的高度添加到高度列表
        page_height_list.append(page_rect.height)

    # 对宽度列表进行排序
    page_width_list.sort()
    # 对高度列表进行排序
    page_height_list.sort()

    # 获取宽度列表的中位数
    median_width = page_width_list[len(page_width_list) // 2]
    # 获取高度列表的中位数
    median_height = page_height_list[len(page_height_list) // 2]

    # 返回中位数宽度和高度
    return median_width, median_height


# 获取每一页的文本长度
def get_pdf_textlen_per_page(doc: fitz.Document):
    # 初始化存放文本长度的列表
    text_len_lst = []
    # 遍历文档中的每一页
    for page in doc:
        # 拿包含img和text的所有blocks
        # text_block = page.get_text("blocks")
        # 拿所有text的blocks
        # text_block = page.get_text("words")
        # text_block_len = sum([len(t[4]) for t in text_block])
        # 拿所有text的str
        text_block = page.get_text("text")
        # 计算文本长度
        text_block_len = len(text_block)
        # logger.info(f"page {page.number} text_block_len: {text_block_len}")
        # 将文本长度添加到列表
        text_len_lst.append(text_block_len)

    # 返回每页文本长度的列表
    return text_len_lst


# 获取每一页的文本布局
def get_pdf_text_layout_per_page(doc: fitz.Document):
    """
    根据PDF文档的每一页文本布局，判断该页的文本布局是横向、纵向还是未知。

    Args:
        doc (fitz.Document): PDF文档对象。

    Returns:
        List[str]: 每一页的文本布局（横向、纵向、未知）。

    """
    # 初始化存放文本布局的列表
    text_layout_list = []

    # 返回文本布局列表
    return text_layout_list


'''定义一个自定义异常用来抛出单页svg太多的pdf'''


# 自定义异常类，用于处理SVG数量过多的情况
class PageSvgsTooManyError(Exception):
    # 初始化异常信息
    def __init__(self, message="Page SVGs are too many"):
        self.message = message
        # 调用父类初始化
        super().__init__(self.message)


# 获取每一页的SVG数量
def get_svgs_per_page(doc: fitz.Document):
    # 初始化存放每页SVG数量的列表
    svgs_len_list = []
    # 遍历文档中的每一页
    for page_id, page in enumerate(doc):
        # svgs = page.get_drawings()
        # 获取当前页的SVG绘图对象，切换成get_cdrawings以提高效率
        svgs = page.get_cdrawings()  # 切换成get_cdrawings，效率更高
        # 计算当前页SVG数量
        len_svgs = len(svgs)
        # 如果当前页SVG数量大于等于3000，则抛出自定义异常
        if len_svgs >= 3000:
            raise PageSvgsTooManyError()
        else:
            # 将SVG数量添加到列表
            svgs_len_list.append(len_svgs)
        # logger.info(f"page_id: {page_id}, svgs_len: {len(svgs)}")
    # 返回每页SVG数量的列表
    return svgs_len_list


# 获取每一页的图片数量
def get_imgs_per_page(doc: fitz.Document):
    # 初始化存放每页图片数量的列表
    imgs_len_list = []
    # 遍历文档中的每一页
    for page_id, page in enumerate(doc):
        # 获取当前页的图片列表
        imgs = page.get_images()
        # 将当前页图片数量添加到列表
        imgs_len_list.append(len(imgs))
        # logger.info(f"page_id: {page}, imgs_len: {len(imgs)}")

    # 返回每页图片数量的列表
    return imgs_len_list


# 获取PDF文档的语言
def get_language(doc: fitz.Document):
    """
    获取PDF文档的语言。
    Args:
        doc (fitz.Document): PDF文档对象。
    Returns:
        str: 文档语言，如 "en-US"。
    """
    # 初始化存放语言的列表
    language_lst = []
    # 遍历文档中的每一页
    for page_id, page in enumerate(doc):
        # 如果当前页超过最大扫描页数，则停止处理
        if page_id >= scan_max_page:
            break
        # 拿所有text的str
        text_block = page.get_text("text")
        # 检测当前页语言
        page_language = detect_lang(text_block)
        # 将当前页语言添加到列表
        language_lst.append(page_language)

        # logger.info(f"page_id: {page_id}, page_language: {page_language}")

    # 统计每种语言的出现次数
    count_dict = Counter(language_lst)
    # 输出出现次数最多的语言
    # 从计数字典中找到具有最大计数的键（语言），并将其赋值给变量 language
    language = max(count_dict, key=count_dict.get)
    # 返回找到的语言
    return language
# 检查 PDF 字节中的无效字符
def check_invalid_chars(pdf_bytes):
    """
    乱码检测
    """
    # 调用检测无效字符的函数，传入 PDF 字节数据
    return detect_invalid_chars(pdf_bytes)


# 扫描 PDF 文件的元数据
def pdf_meta_scan(pdf_bytes: bytes):
    """
    :param s3_pdf_path:
    :param pdf_bytes: pdf文件的二进制数据
    几个维度来评价：是否加密，是否需要密码，纸张大小，总页数，是否文字可提取
    """
    # 使用 PyMuPDF 打开 PDF 文档
    doc = fitz.open("pdf", pdf_bytes)
    # 检查 PDF 是否需要密码
    is_needs_password = doc.needs_pass
    # 检查 PDF 是否被加密
    is_encrypted = doc.is_encrypted
    # 获取 PDF 的总页数
    total_page = len(doc)
    # 如果 PDF 总页数为 0
    if total_page == 0:
        # 记录警告日志，说明该 PDF 被丢弃
        logger.warning(f"drop this pdf, drop_reason: {DropReason.EMPTY_PDF}")
        # 返回丢弃的原因
        result = {"_need_drop": True, "_drop_reason": DropReason.EMPTY_PDF}
        return result
    else:
        # 获取页面的宽度和高度（单位：点）
        page_width_pts, page_height_pts = get_pdf_page_size_pts(doc)
        # logger.info(f"page_width_pts: {page_width_pts}, page_height_pts: {page_height_pts}")

        # svgs_per_page = get_svgs_per_page(doc)
        # logger.info(f"svgs_per_page: {svgs_per_page}")
        # 获取每页的图片数量
        imgs_per_page = get_imgs_per_page(doc)
        # logger.info(f"imgs_per_page: {imgs_per_page}")

        # 获取每页的图像信息及无用图像的标识符
        image_info_per_page, junk_img_bojids = get_image_info(doc, page_width_pts, page_height_pts)
        # logger.info(f"image_info_per_page: {image_info_per_page}, junk_img_bojids: {junk_img_bojids}")
        # 获取每页文本的长度
        text_len_per_page = get_pdf_textlen_per_page(doc)
        # logger.info(f"text_len_per_page: {text_len_per_page}")
        # 获取每页文本的布局信息
        text_layout_per_page = get_pdf_text_layout_per_page(doc)
        # logger.info(f"text_layout_per_page: {text_layout_per_page}")
        # 获取文本的语言
        text_language = get_language(doc)
        # logger.info(f"text_language: {text_language}")
        # 检查 PDF 字节中的无效字符
        invalid_chars = check_invalid_chars(pdf_bytes)
        # logger.info(f"invalid_chars: {invalid_chars}")

        # 最后输出一条包含 PDF 元数据的字典
        res = {
            "is_needs_password": is_needs_password,  # PDF 是否需要密码
            "is_encrypted": is_encrypted,            # PDF 是否加密
            "total_page": total_page,                # PDF 总页数
            "page_width_pts": int(page_width_pts),   # 页面宽度（整数）
            "page_height_pts": int(page_height_pts), # 页面高度（整数）
            "image_info_per_page": image_info_per_page,  # 每页的图像信息
            "text_len_per_page": text_len_per_page,  # 每页文本长度
            "text_layout_per_page": text_layout_per_page,  # 每页文本布局信息
            "text_language": text_language,          # 文本语言
            # "svgs_per_page": svgs_per_page,
            "imgs_per_page": imgs_per_page,          # 每页图片数量
            "junk_img_bojids": junk_img_bojids,      # 垃圾图片的标识符列表
            "invalid_chars": invalid_chars,          # 无效字符信息
            "metadata": doc.metadata                  # PDF 的元数据
        }
        # logger.info(json.dumps(res, ensure_ascii=False))
        # 返回 PDF 的元数据
        return res


# CLI 命令定义
@click.command()
# 定义命令行选项，获取 S3 中 PDF 文件的路径
@click.option('--s3-pdf-path', help='s3上pdf文件的路径')
# 定义命令行选项，获取 S3 上的配置文件
@click.option('--s3-profile', help='s3上的profile')
def main(s3_pdf_path: str, s3_profile: str):
    """
    
    """
    try:
        # 读取 S3 上 PDF 文件的内容
        file_content = read_file(s3_pdf_path, s3_profile)
        # 扫描 PDF 文件的元数据
        pdf_meta_scan(file_content)
    except Exception as e:
        # 捕获异常并输出错误信息到标准错误流
        print(f"ERROR: {s3_pdf_path}, {e}", file=sys.stderr)
        # 记录异常日志
        logger.exception(e)


# 如果该脚本是主程序，执行主函数
if __name__ == '__main__':
    main()
    # "D:\project/20231108code-clean\pdf_cost_time\竖排例子\净空法师-大乘无量寿.pdf"
    # PDF 文件路径示例，包含繁体竖排版的《三国演义》文件
    # PDF 文件路径示例，包含 SCIHUB 中的一个压缩包内的 PDF 文件
    # PDF 文件路径示例，包含另一个 SCIHUB 中的 PDF 文件
    # 调用 read_file 函数读取指定 PDF 文件内容，并传入空字符串作为参数
    # 调用 read_file 函数读取另一个指定 PDF 文件的内容，并传入空字符串作为参数
    # 使用 fitz 库打开读取到的 PDF 文件内容，创建文档对象
    # 获取 PDF 文档每一页的文本布局信息，返回一个列表
    # 打印文本布局信息列表
```