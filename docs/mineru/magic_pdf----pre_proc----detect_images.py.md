# `.\MinerU\magic_pdf\pre_proc\detect_images.py`

```
# 导入统计相关库
import collections      # 统计库
import re               # 正则表达式库
from magic_pdf.libs.commons import fitz             # 导入pyMuPDF库中的fitz模块


#--------------------------------------- Tool Functions --------------------------------------#
# 正则化，输入文本，输出只保留a-z,A-Z,0-9
def remove_special_chars(s: str) -> str:
    # 定义正则表达式模式，匹配非字母和数字字符
    pattern = r"[^a-zA-Z0-9]"
    # 使用正则表达式替换字符串中的特殊字符
    res = re.sub(pattern, "", s)
    # 返回处理后的字符串
    return res

def check_rect1_sameWith_rect2(L1: float, U1: float, R1: float, D1: float, L2: float, U2: float, R2: float, D2: float) -> bool:
    # 判断rect1和rect2是否一模一样
    return L1 == L2 and U1 == U2 and R1 == R2 and D1 == D2

def check_rect1_contains_rect2(L1: float, U1: float, R1: float, D1: float, L2: float, U2: float, R2: float, D2: float) -> bool:
    # 判断rect1包含了rect2
    return (L1 <= L2 <= R2 <= R1) and (U1 <= U2 <= D2 <= D1)

def check_rect1_overlaps_rect2(L1: float, U1: float, R1: float, D1: float, L2: float, U2: float, R2: float, D2: float) -> bool:
    # 判断rect1与rect2是否存在重叠（只有一条边重叠，也算重叠）
    return max(L1, L2) <= min(R1, R2) and max(U1, U2) <= min(D1, D2)

def calculate_overlapRatio_between_rect1_and_rect2(L1: float, U1: float, R1: float, D1: float, L2: float, U2: float, R2: float, D2: float) -> (float, float):
    # 计算两个rect，重叠面积各占2个rect面积的比例
    if min(R1, R2) < max(L1, L2) or min(D1, D2) < max(U1, U2):
        # 如果没有重叠，返回0
        return 0, 0
    # 计算rect1的面积
    square_1 = (R1 - L1) * (D1 - U1)
    # 计算rect2的面积
    square_2 = (R2 - L2) * (D2 - U2)
    if square_1 == 0 or square_2 == 0:
        # 如果任意一个面积为0，返回0
        return 0, 0
    # 计算重叠面积
    square_overlap = (min(R1, R2) - max(L1, L2)) * (min(D1, D2) - max(U1, U2))
    # 返回重叠面积相对于两个矩形面积的比例
    return square_overlap / square_1, square_overlap / square_2

def calculate_overlapRatio_between_line1_and_line2(L1: float, R1: float, L2: float, R2: float) -> (float, float):
    # 计算两个line，重叠区间各占2个line长度的比例
    if max(L1, L2) > min(R1, R2):
        # 如果没有重叠，返回0
        return 0, 0
    if L1 == R1 or L2 == R2:
        # 如果任意一条线的长度为0，返回0
        return 0, 0
    # 计算重叠长度
    overlap_line = min(R1, R2) - max(L1, L2)
    # 返回重叠长度相对于两条线长度的比例
    return overlap_line / (R1 - L1), overlap_line / (R2 - L2)


# 判断rect其实是一条line
def check_rect_isLine(L: float, U: float, R: float, D: float) -> bool:
    # 计算矩形的宽度和高度
    width = R - L
    height = D - U
    if width <= 3 or height <= 3:
        # 如果宽度或高度小于等于3，返回True
        return True
    if width / height >= 30 or height / width >= 30:
        # 如果宽高比大于等于30，返回True
        return True



def parse_images(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict, junk_img_bojids=[]):
    """
    :param page_ID: int类型，当前page在当前pdf文档中是第page_D页。
    :param page :fitz读取的当前页的内容
    :param res_dir_path: str类型，是每一个pdf文档，在当前.py文件的目录下生成一个与pdf文档同名的文件夹，res_dir_path就是文件夹的dir
    :param json_from_DocXchain_obj: dict类型，把pdf文档送入DocXChain模型中后，提取bbox，结果保存到pdf文档同名文件夹下的 page_ID.json文件中了。json_from_DocXchain_obj就是打开后的dict
    """
    #### 通过fitz获取page信息
    ## 超越边界
    DPI = 72  # use this resolution
    # 获取当前页面的像素图，指定DPI
    pix = page.get_pixmap(dpi=DPI)
    # 初始化页面的左边界
    pageL = 0
    # 获取页面的右边界
    pageR = int(pix.w)
    # 初始化页面的上边界
    pageU = 0
    # 获取页面的下边界
    pageD = int(pix.h)
    
    #----------------- 保存每一个文本块的LURD ------------------#
    # 初始化存储文本块的列表
    textLine_blocks = []
    # 获取页面文本内容，并以字典形式返回，其中包含文本块信息
    blocks = page.get_text(
            "dict",                              # 指定返回类型为字典
            flags=fitz.TEXTFLAGS_TEXT,          # 设置标志，表示获取文本内容
            #clip=clip,                          # 这里注释掉的代码可能是用于剪切文本区域
        )["blocks"]                              # 从返回的字典中提取“blocks”字段

    # 遍历所有文本块
    for i in range(len(blocks)):
        bbox = blocks[i]['bbox']              # 获取当前文本块的边界框（bbox）
        # print(bbox)                          # 可选：打印当前文本块的边界框
        
        # 遍历当前文本块中的所有行
        for tt in blocks[i]['lines']:
            # 当前行的边界框初始化为 None
            cur_line_bbox = None                            # 当前行，最右侧的section的bbox
            
            # 遍历当前行中的所有文本段
            for xf in tt['spans']:
                L, U, R, D = xf['bbox']                   # 获取文本段的边界框
                L, R = min(L, R), max(L, R)               # 确保 L 和 R 赋值正确
                U, D = min(U, D), max(U, D)               # 确保 U 和 D 赋值正确
                
                # 将文本段的边界框添加到文本行块列表中
                textLine_blocks.append((L, U, R, D))    

    # 根据文本行块的上边界和左边界对列表进行排序
    textLine_blocks.sort(key = lambda LURD: (LURD[1], LURD[0]))  

    #---------------------------------------------- 保存img --------------------------------------------------#
    # 获取当前页面中的所有图片信息
    raw_imgs = page.get_images()                    # 获取所有的图片
    imgs = []                                       # 存储图片数据的列表
    img_names = []                                  # 保存图片的名字，方便在md中插入引用
    img_bboxs = []                                  # 保存图片的位置信息
    img_visited = []                                # 记录该图片是否在md中已插入过
    img_ID = 0                                      # 图片ID初始化

    # 获取并保存每张图片的位置信息（左上和右下坐标）
    for i in range(len(raw_imgs)):
        # 如果图片ID在垃圾图片列表中则跳过
        if raw_imgs[i][0] in junk_img_bojids:
            continue                                 # 跳过当前循环，进入下一张图片
        else:
            try:
                # 获取当前图片的矩形区域（带变换）
                tt = page.get_image_rects(raw_imgs[i][0], transform = True)

                rec = tt[0][0]                        # 取出第一个矩形区域
                L, U, R, D = int(rec[0]), int(rec[1]), int(rec[2]), int(rec[3])  # 获取边界框的四个坐标

                # 确保 L 和 R 赋值正确
                L, R = min(L, R), max(L, R)         
                # 确保 U 和 D 赋值正确
                U, D = min(U, D), max(U, D)         

                # 检查当前图片的边界框是否在有效页面范围内
                if not(pageL <= L < R <= pageR and pageU <= U < D <= pageD):
                    continue                             # 跳过不在页面范围的图片
                
                # 检查图片是否是页面的完整边界
                if pageL == L and R == pageR:
                    continue                             # 跳过完整的左边缘图片
                if pageU == U and D == pageD:
                    continue                             # 跳过完整的上边缘图片
                
                # pix1 = page.get_Pixmap(clip=(L,U,R,D))  # 可选：获取图片的像素图
                
                # 为图片生成新的文件名
                new_img_name = "{}_{}.png".format(page_ID, i)      # 图片名称
                
                # pix1.save(res_dir_path + '/' + new_img_name)  # 可选：将图片保存到指定目录并命名
                
                # 将图片名、边界框和访问状态添加到对应列表
                img_names.append(new_img_name)               
                img_bboxs.append((L, U, R, D))               
                img_visited.append(False)                    
                imgs.append(raw_imgs[i])                      # 将当前图片数据添加到 imgs 列表
            except:
                continue                                     # 遇到异常时跳过当前图片
    
    #-------- 如果图片之间有重叠，说明获取的图片大小或位置有问题，标记为无效 --------#
    imgs_ok = [True for _ in range(len(imgs))]        # 初始化图片有效性列表
    # 遍历所有图片的索引
    for i in range(len(imgs)):
        # 解包当前图片的边界框坐标
        L1, U1, R1, D1 = img_bboxs[i]
        # 遍历当前图片之后的所有图片
        for j in range(i + 1, len(imgs)):
            # 解包第二张图片的边界框坐标
            L2, U2, R2, D2 = img_bboxs[j]
            # 计算两张图片的重叠比率
            ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
            # 计算当前图片的面积
            s1 = abs(R1 - L1) * abs(D1 - U1)
            # 计算第二张图片的面积
            s2 = abs(R2 - L2) * abs(D2 - U2)
            # 如果两张图片都有重叠
            if ratio_1 > 0 and ratio_2 > 0:
                # 如果第一张图片完全包含第二张且第二张占比大于0.8
                if ratio_1 == 1 and ratio_2 > 0.8:
                    imgs_ok[i] = False  # 标记第一张图片为不合适
                # 如果第二张图片完全包含第一张且第一张占比大于0.8
                elif ratio_1 > 0.8 and ratio_2 == 1:
                    imgs_ok[j] = False  # 标记第二张图片为不合适
                # 如果两张图片都很大且重叠比率较高
                elif s1 > 20000 and s2 > 20000 and ratio_1 > 0.4 and ratio_2 > 0.4:
                    imgs_ok[i] = False  # 标记第一张图片为不合适
                    imgs_ok[j] = False  # 标记第二张图片为不合适
                # 如果第一张图片相对第二张图片面积大且第二张重叠比率较高
                elif s1 / s2 > 5 and ratio_2 > 0.5:
                    imgs_ok[j] = False  # 标记第二张图片为不合适
                # 如果第二张图片相对第一张图片面积大且第一张重叠比率较高
                elif s2 / s1 > 5 and ratio_1 > 0.5:
                    imgs_ok[i] = False  # 标记第一张图片为不合适
                    
    # 根据 imgs_ok 的标记筛选有效的图片
    imgs = [imgs[i] for i in range(len(imgs)) if imgs_ok[i] == True]
    # 根据 imgs_ok 的标记筛选有效的图片名称
    img_names = [img_names[i] for i in range(len(imgs)) if imgs_ok[i] == True]
    # 根据 imgs_ok 的标记筛选有效的边界框
    img_bboxs = [img_bboxs[i] for i in range(len(imgs)) if imgs_ok[i] == True]
    # 根据 imgs_ok 的标记筛选有效的访问标记
    img_visited = [img_visited[i] for i in range(len(imgs)) if imgs_ok[i] == True]
    #*******************************************************************************#
    
    #---------------------------------------- 通过fitz提取svg的信息 -----------------------------------------#
    #
    # 获取页面的所有绘图信息
    svgs = page.get_drawings()
    #------------ preprocess, check一些大框，看是否是合理的 ----------#
    ## 去重。有时候会遇到rect1和rect2是完全一样的情形。
    # 创建一个集合用于存储已访问的 SVG 边界框
    svg_rect_visited = set()
    # 存储有效 SVG 的索引
    available_svgIdx = []
    # 遍历所有 SVG 的索引
    for i in range(len(svgs)):
        # 解包当前 SVG 的边界框坐标
        L, U, R, D = svgs[i]['rect'].irect
        # 确保左、右、上、下坐标的顺序正确
        L, R = min(L, R), max(L, R)
        U, D = min(U, D), max(U, D)
        # 创建边界框的元组
        tt = (L, U, R, D)
        # 如果该边界框未被访问过
        if tt not in svg_rect_visited:
            # 将其加入访问集合
            svg_rect_visited.add(tt)
            # 记录该索引为有效
            available_svgIdx.append(i)
        
    # 根据有效索引筛选去重后的 SVG
    svgs = [svgs[i] for i in available_svgIdx]                  # 去重后，有效的svgs
    # 初始化每个 SVG 的子 SVG 列表
    svg_childs = [[] for _ in range(len(svgs))]
    # 初始化每个 SVG 的父 SVG 列表
    svg_parents = [[] for _ in range(len(svgs))]
    # 初始化每个 SVG 的重叠 SVG 列表
    svg_overlaps = [[] for _ in range(len(svgs))]            #svg_overlaps[i]是一个list，存的是与svg_i有重叠的svg的index。e.g., svg_overlaps[0] = [1, 2, 7, 9]
    # 初始化访问标记
    svg_visited = [False for _ in range(len(svgs))]
    # 初始化超出页面边界的标记
    svg_exceedPage = [0 for _ in range(len(svgs))]       # 是否超越边界（artbox），很大，但一般是一个svg的底。  
    # 遍历所有的 SVG 对象
        for i in range(len(svgs)):
            # 获取当前 SVG 的矩形边界 L、U、R、D
            L, U, R, D = svgs[i]['rect'].irect
            # 计算当前 SVG 和页面矩形的重叠比例
            ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L, U, R, D, pageL, pageU, pageR, pageD)
            # 检查当前 SVG 是否在页面范围内
            if (pageL + 20 < L <= R < pageR - 20) and (pageU + 20 < U <= D < pageD - 20):
                # 如果重叠比例大于等于 0.7，增加超出页面的计数
                if ratio_2 >= 0.7:
                    svg_exceedPage[i] += 4
            else:
                # 检查矩形的边界是否超出页面左边
                if L <= pageL:
                    svg_exceedPage[i] += 1
                # 检查矩形的边界是否超出页面右边
                if pageR <= R:
                    svg_exceedPage[i] += 1
                # 检查矩形的边界是否超出页面上边
                if U <= pageU:
                    svg_exceedPage[i] += 1
                # 检查矩形的边界是否超出页面下边
                if pageD <= D:
                    svg_exceedPage[i] += 1
                
        # 如果有两个或以上超出页面的框，则清空 SVG 列表
        if len([x for x in svg_exceedPage if x >= 1]) >= 2:
            svgs = []  # 清空 SVG 列表
            svg_childs = []  # 清空子 SVG 列表
            svg_parents = []  # 清空父 SVG 列表
            svg_overlaps = []  # 清空重叠 SVG 列表
            svg_visited = []  # 清空访问标记列表
            svg_exceedPage = []  # 清空超出页面计数列表
                
        #---------------------------- build graph ----------------------------#
        # 遍历每个 SVG，构建图
        for i, p in enumerate(svgs):
            # 获取当前 SVG 的矩形边界 L1、U1、R1、D1
            L1, U1, R1, D1 = svgs[i]["rect"].irect
            # 遍历所有 SVG 进行比较
            for j in range(len(svgs)):
                if i == j:
                    continue  # 跳过自身比较
                # 获取另一个 SVG 的矩形边界 L2、U2、R2、D2
                L2, U2, R2, D2 = svgs[j]["rect"].irect
                ## 检查当前 SVG 是否包含另一个 SVG
                if check_rect1_contains_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                    svg_childs[i].append(j)  # 将 j 添加为 i 的子 SVG
                    svg_parents[j].append(i)  # 将 i 添加为 j 的父 SVG
                else:
                    ## 检查当前 SVG 是否与另一个 SVG 交叉
                    if check_rect1_overlaps_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                        svg_overlaps[i].append(j)  # 将 j 添加为与 i 重叠的 SVG
    
        #---------------- 确定最终的svg。连通块儿的外围 -------------------#
        eps_ERROR = 5                      # 设置容错值，以防矩形检测不精确
        svg_ID = 0  # 初始化 SVG ID
        svg_final_names = []  # 存储最终识别的 SVG 名称
        svg_final_bboxs = []  # 存储最终识别的 SVG 边界框
        svg_final_visited = []  # 为文本识别准备的访问标记列表
        
        # 获取所有 SVG 的索引并按面积排序
        svg_idxs = [i for i in range(len(svgs))]
        svg_idxs.sort(key = lambda i: -(svgs[i]['rect'].irect[2] - svgs[i]['rect'].irect[0]) * (svgs[i]['rect'].irect[3] - svgs[i]['rect'].irect[1]))   # 按照面积，从大到小排序
         
        # 对识别出的 SVG 进行合并处理
        svg_idxs = [i for i in range(len(svg_final_bboxs))]
        svg_idxs.sort(key = lambda i: (svg_final_bboxs[i][1], svg_final_bboxs[i][0]))   # 按照 (U, L) 排序
        svg_final_names_2 = []  # 存储合并后 SVG 名称
        svg_final_bboxs_2 = []  # 存储合并后 SVG 边界框
        svg_final_visited_2 = []  # 为文本识别准备的访问标记列表
        svg_ID_2 = 0  # 初始化合并后 SVG ID
    # 遍历 svg_final_bboxs 列表的每个元素
    for i in range(len(svg_final_bboxs)):
        # 解包当前元素的边界框坐标
        L1, U1, R1, D1 = svg_final_bboxs[i]
        # 遍历当前元素后面的每个元素
        for j in range(i + 1, len(svg_final_bboxs)):
            # 解包下一个元素的边界框坐标
            L2, U2, R2, D2 = svg_final_bboxs[j]
            # 检查 rect1 是否包含 rect2
            if check_rect1_contains_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                # 标记该元素为已访问
                svg_final_visited[j] = True
                continue
            # 计算两个矩形的重叠比例，判断水平并列
            ratio_1, ratio_2 = calculate_overlapRatio_between_line1_and_line2(U1, D1, U2, D2)
            # 如果重叠比例都大于等于 0.7
            if ratio_1 >= 0.7 and ratio_2 >= 0.7:
                # 如果两个矩形的左边界差距大于等于 20，继续
                if abs(L2 - R1) >= 20:
                    continue
                # 计算合并后的边界框坐标
                LL = min(L1, L2)
                UU = min(U1, U2)
                RR = max(R1, R2)
                DD = max(D1, D2)
                # 更新第一个矩形的边界框为合并后的值
                svg_final_bboxs[i] = (LL, UU, RR, DD)
                # 标记第二个矩形为已访问
                svg_final_visited[j] = True
                continue
            # 计算两个矩形的重叠比例，判断竖直并列
            ratio_1, ratio_2 = calculate_overlapRatio_between_line1_and_line2(L1, R2, L2, R2)
            # 如果重叠比例都大于等于 0.7
            if ratio_1 >= 0.7 and ratio_2 >= 0.7:
                # 如果两个矩形的上下边界差距大于等于 20，继续
                if abs(U2 - D1) >= 20:
                    continue
                # 计算合并后的边界框坐标
                LL = min(L1, L2)
                UU = min(U1, U2)
                RR = max(R1, R2)
                DD = max(D1, D2)
                # 更新第一个矩形的边界框为合并后的值
                svg_final_bboxs[i] = (LL, UU, RR, DD)
                # 标记第二个矩形为已访问
                svg_final_visited[j] = True
    
    # 遍历 svg_final_bboxs 列表，寻找未访问的元素
    for i in range(len(svg_final_bboxs)):
        # 如果当前元素未被访问
        if svg_final_visited[i] == False:
            # 解包边界框坐标
            L, U, R, D = svg_final_bboxs[i]
            # 将当前边界框添加到新列表中
            svg_final_bboxs_2.append((L, U, R, D))
            
            # 调整边界框，考虑误差
            L -= eps_ERROR * 2
            U -= eps_ERROR
            R += eps_ERROR * 2
            D += eps_ERROR
            # cur_svg = page.get_pixmap(clip=(L,U,R,D))  # 生成当前边界框的图片
            # 创建新的图片名称，包含页面和 svg ID
            new_svg_name = "svg_{}_{}.png".format(page_ID, svg_ID_2)      # 图片name
            # cur_svg.save(res_dir_path + '/' + new_svg_name)        # 把图片存出在新建的文件夹，并命名
            # 将新生成的图片名称添加到列表中，方便后续引用
            svg_final_names_2.append(new_svg_name)                      # 把图片的名字存在list中，方便在md中插入引用
            # 将调整后的边界框再次添加到新列表中
            svg_final_bboxs_2.append((L, U, R, D))
            # 添加未访问标记
            svg_final_visited_2.append(False)
            # 增加 svg ID 计数器
            svg_ID_2 += 1
       
    ## svg收尾。识别为drawing，但是在上面没有拼成一张图的。
    # 有收尾才comprehensive
    # xxxx
    # xxxx
    # xxxx
    # xxxx
    
    #--------- 通过json_from_DocXchain来获取，figure, table, equation的bbox ---------#
    # 初始化存储 figure 的边界框列表
    figure_bbox_from_DocXChain = []
    
    # 初始化已访问的图形记录列表
    figure_from_DocXChain_visited = []          # 记忆化
    # 存储重叠比例的列表
    figure_bbox_from_DocXChain_overlappedRatio = []
    
    # 存储单独的图形边界框和名称的列表
    figure_only_from_DocXChain_bboxs = []     # 存储
    figure_only_from_DocXChain_names = []
    figure_only_from_DocXChain_visited = []
    # 初始化图形 ID 计数器
    figure_only_ID = 0
    
    # 获取从 DocXChain 中解析的 JSON 对象
    xf_json = json_from_DocXchain_obj
    # 获取页面宽度
    width_from_json = xf_json['page_info']['width']
    # 获取页面高度
    height_from_json = xf_json['page_info']['height']
    # 计算左右缩放比例
    LR_scaleRatio = width_from_json / (pageR - pageL)
    # 计算上下缩放比例
    UD_scaleRatio = height_from_json / (pageD - pageU)
    
    # 遍历 JSON 中的布局信息
    for xf in xf_json['layout_dets']:
    # {0: 'title', 1: 'figure', 2: 'plain text', 3: 'header', 4: 'page number', 5: 'footnote', 6: 'footer', 7: 'table', 8: 'table caption', 9: 'figure caption', 10: 'equation', 11: 'full column', 12: 'sub column'}
        # 从 xf 字典中获取左边界并按比例缩放
        L = xf['poly'][0] / LR_scaleRatio
        # 从 xf 字典中获取上边界并按比例缩放
        U = xf['poly'][1] / UD_scaleRatio
        # 从 xf 字典中获取右边界并按比例缩放
        R = xf['poly'][2] / LR_scaleRatio
        # 从 xf 字典中获取下边界并按比例缩放
        D = xf['poly'][5] / UD_scaleRatio
        # L += pageL          # 有的页面，artBox偏移了。不在（0,0）
        # R += pageL
        # U += pageU
        # D += pageU
        # 确保 L 和 R 为最小和最大值
        L, R = min(L, R), max(L, R)
        # 确保 U 和 D 为最小和最大值
        U, D = min(U, D), max(U, D)
        # 判断是否为图形类别
        if xf["category_id"] == 1 and xf['score'] >= 0.3:
            # 将有效图形的边界框添加到列表中
            figure_bbox_from_DocXChain.append((L, U, R, D))
            # 标记该图形为未访问
            figure_from_DocXChain_visited.append(False)
            # 初始化重叠率为 0
            figure_bbox_from_DocXChain_overlappedRatio.append(0.0)

    #---------------------- 比对上面识别出来的img,svg 与DocXChain给的figure -----------------------#
    
    ## 比对imgs
    for i, b1 in enumerate(figure_bbox_from_DocXChain):
        # print('--------- DocXChain的图片', b1)
        # 解包图形边界框
        L1, U1, R1, D1 = b1
        # 遍历所有图片边界框进行比较
        for b2 in img_bboxs:
            # print('-------- igms得到的图', b2)
            # 解包图片边界框
            L2, U2, R2, D2 = b2
            # 计算两个边界框的面积
            s1 = abs(R1 - L1) * abs(D1 - U1)
            s2 = abs(R2 - L2) * abs(D2 - U2)
            # 相同
            if check_rect1_sameWith_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                # 标记为已访问
                figure_from_DocXChain_visited[i] = True
            # 包含
            elif check_rect1_contains_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                # 判断面积比例是否满足条件
                if s2 / s1 > 0.8:
                    figure_from_DocXChain_visited[i] = True
            # 被包含
            elif check_rect1_contains_rect2(L2, U2, R2, D2, L1, U1, R1, D1) == True:
                # 判断面积比例是否满足条件
                if s1 / s2 > 0.8:
                    figure_from_DocXChain_visited[i] = True 
            else:
                # 重叠了相当一部分
                # print('进入第3部分')
                # 计算两个边界框的重叠率
                ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
                # 判断重叠率是否满足条件
                if (ratio_1 >= 0.6 and ratio_2 >= 0.6) or (ratio_1 >= 0.8 and s1/s2>0.8) or (ratio_2 >= 0.8 and s2/s1>0.8):
                    figure_from_DocXChain_visited[i] = True
                else:
                    # 更新重叠率
                    figure_bbox_from_DocXChain_overlappedRatio[i] += ratio_1
                    # print('图片的重叠率是{}'.format(ratio_1))


    ## 比对svgs
    # 初始化最终的 SVG 边界框和坏索引列表
    svg_final_bboxs_2_badIdxs = []
    # 遍历 DocXChain 中的每个图形边界框，i 为索引，b1 为边界框坐标
    for i, b1 in enumerate(figure_bbox_from_DocXChain):
        # 解包边界框的左、上、右、下坐标
        L1, U1, R1, D1 = b1
        # 遍历 svg_final_bboxs_2 中的每个图形边界框，j 为索引，b2 为边界框坐标
        for j, b2 in enumerate(svg_final_bboxs_2):
            # 解包边界框的左、上、右、下坐标
            L2, U2, R2, D2 = b2
            # 计算第一个边界框的面积
            s1 = abs(R1 - L1) * abs(D1 - U1)
            # 计算第二个边界框的面积
            s2 = abs(R2 - L2) * abs(D2 - U2)
            # 检查两个边界框是否相同
            if check_rect1_sameWith_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                # 标记第一个边界框为已访问
                figure_from_DocXChain_visited[i] = True
            # 检查第一个边界框是否包含第二个边界框
            elif check_rect1_contains_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                # 标记第一个边界框为已访问
                figure_from_DocXChain_visited[i] = True
            # 检查第二个边界框是否包含第一个边界框
            elif check_rect1_contains_rect2(L2, U2, R2, D2, L1, U1, R1, D1) == True:
                # 如果第一个边界框的面积与第二个边界框的面积比大于 0.7
                if s1 / s2 > 0.7:
                    # 标记第一个边界框为已访问
                    figure_from_DocXChain_visited[i] = True
                else:
                    # 将第二个边界框的索引添加到丢弃列表中
                    svg_final_bboxs_2_badIdxs.append(j)     # svg丢弃。用DocXChain的结果。
            else:
                # 计算两个边界框之间的重叠比率
                ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
                # 检查重叠比率是否达到阈值
                if (ratio_1 >= 0.5 and ratio_2 >= 0.5) or (min(ratio_1, ratio_2) >= 0.4 and max(ratio_1, ratio_2) >= 0.6):
                    # 标记第一个边界框为已访问
                    figure_from_DocXChain_visited[i] = True
                else:
                    # 如果没有达到重叠标准，增加重叠比率
                    figure_bbox_from_DocXChain_overlappedRatio[i] += ratio_1
                    
    # 从 svg_final_bboxs_2 中丢掉错误的边界框
    svg_final_bboxs_2 = [svg_final_bboxs_2[i] for i in range(len(svg_final_bboxs_2)) if i not in set(svg_final_bboxs_2_badIdxs)]
    
    # 遍历已访问状态的边界框，更新访问状态
    for i in range(len(figure_from_DocXChain_visited)):
        # 如果重叠比率达到 0.7，则标记为已访问
        if figure_bbox_from_DocXChain_overlappedRatio[i] >= 0.7:
            figure_from_DocXChain_visited[i] = True
    
    # 遍历 DocXChain 中的边界框，处理未保存的边界框
    for i in range(len(figure_from_DocXChain_visited)):
        # 如果未被访问
        if figure_from_DocXChain_visited[i] == False:
            # 标记为已访问
            figure_from_DocXChain_visited[i] = True
            # 获取当前边界框
            cur_bbox = figure_bbox_from_DocXChain[i]
            # cur_figure = page.get_pixmap(clip=cur_bbox)  # 生成当前边界框的图像
            # 生成新图像的文件名
            new_figure_name = "figure_only_{}_{}.png".format(page_ID, figure_only_ID)      # 图片name
            # cur_figure.save(res_dir_path + '/' + new_figure_name)        # 将图像保存到指定目录
            # 将新图像的文件名添加到列表中
            figure_only_from_DocXChain_names.append(new_figure_name)                      # 把图片的名字存在list中，方便在md中插入引用
            # 将当前边界框添加到边界框列表中
            figure_only_from_DocXChain_bboxs.append(cur_bbox)
            # 将访问状态标记为未访问
            figure_only_from_DocXChain_visited.append(False)
            # 增加图像 ID
            figure_only_ID += 1
    
    # 根据左、上坐标排序图像边界框
    img_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    # 根据左、上坐标排序 SVG 边界框
    svg_final_bboxs_2.sort(key = lambda LURD: (LURD[1], LURD[0]))
    # 根据左、上坐标排序 DocXChain 图像边界框
    figure_only_from_DocXChain_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    # 合并所有边界框
    curPage_all_fig_bboxs = img_bboxs + svg_final_bboxs + figure_only_from_DocXChain_bboxs
    
    #--------------------------- 最后统一去重 -----------------------------------#
    # 根据边界框的面积和左、上坐标进行排序
    curPage_all_fig_bboxs.sort(key = lambda LURD: ( (LURD[2]-LURD[0])*(LURD[3]-LURD[1]) , LURD[0], LURD[1]) )
    
    # 初始化一个集合，用于存储最终的重复边界框
    final_duplicate = set()
    # 遍历所有图像边界框
        for i in range(len(curPage_all_fig_bboxs)):
            # 解构当前边界框的四个坐标
            L1, U1, R1, D1 = curPage_all_fig_bboxs[i]
            # 对每个边界框进行比较
            for j in range(len(curPage_all_fig_bboxs)):
                # 如果是同一个边界框，则跳过
                if i == j:
                    continue
                # 解构另一个边界框的四个坐标
                L2, U2, R2, D2 = curPage_all_fig_bboxs[j]
                # 计算当前边界框的面积
                s1 = abs(R1 - L1) * abs(D1 - U1)
                # 计算另一个边界框的面积
                s2 = abs(R2 - L2) * abs(D2 - U2)
                # 检查第一个边界框是否包含第二个
                if check_rect1_contains_rect2(L2, U2, R2, D2, L1, U1, R1, D1) == True:
                    # 如果包含，则添加到重复集合
                    final_duplicate.add((L1, U1, R1, D1))
                else:
                    # 计算两个边界框的重叠比率
                    ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
                    # 如果满足特定重叠条件，则添加到重复集合
                    if ratio_1 >= 0.8 and ratio_2 <= 0.6:
                        final_duplicate.add((L1, U1, R1, D1))
    
        # 过滤掉重复的边界框
        curPage_all_fig_bboxs = [LURD for LURD in curPage_all_fig_bboxs if LURD not in final_duplicate]
        
        #### 再考虑重叠关系的块
        # 初始化重复集合和合成边界框列表
        final_duplicate = set()
        final_synthetic_bboxs = []
        # 遍历边界框
        for i in range(len(curPage_all_fig_bboxs)):
            # 解构当前边界框的四个坐标
            L1, U1, R1, D1 = curPage_all_fig_bboxs[i]
            # 对每个边界框进行比较
            for j in range(len(curPage_all_fig_bboxs)):
                # 如果是同一个边界框，则跳过
                if i == j:
                    continue
                # 解构另一个边界框的四个坐标
                L2, U2, R2, D2 = curPage_all_fig_bboxs[j]
                # 计算当前边界框的面积
                s1 = abs(R1 - L1) * abs(D1 - U1)
                # 计算另一个边界框的面积
                s2 = abs(R2 - L2) * abs(D2 - U2)
                # 计算两个边界框的重叠比率
                ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
                union_ok = False
                # 检查重叠条件，确定是否合并
                if (ratio_1 >= 0.8 and ratio_2 <= 0.6) or (ratio_1 > 0.6 and ratio_2 > 0.6): 
                    union_ok = True
                # 检查面积比率条件
                if (ratio_1 > 0.2 and s2 / s1 > 5):
                    union_ok = True
                # 检查中心点是否在范围内
                if (L1 <= (L2+R2)/2 <= R1) and (U1 <= (U2+D2)/2 <= D1):
                    union_ok = True
                if (L2 <= (L1+R1)/2 <= R2) and (U2 <= (U1+D1)/2 <= D2):
                    union_ok = True
                # 如果满足合并条件，则添加到重复集合和合成列表
                if union_ok == True:
                    final_duplicate.add((L1, U1, R1, D1))
                    final_duplicate.add((L2, U2, R2, D2))
                    # 计算合成边界框的坐标
                    L3, U3, R3, D3 = min(L1, L2), min(U1, U2), max(R1, R2), max(D1, D2)
                    # 将合成边界框添加到列表
                    final_synthetic_bboxs.append((L3, U3, R3, D3))
    
        # print('---------- curPage_all_fig_bboxs ---------')
        # print(curPage_all_fig_bboxs)
        # 过滤掉重复的边界框
        curPage_all_fig_bboxs = [b for b in curPage_all_fig_bboxs if b not in final_duplicate]    
        # 移除重复的合成边界框
        final_synthetic_bboxs = list(set(final_synthetic_bboxs))
    
    
        ## 再再考虑重叠关系。极端情况下会迭代式地2进1
        # 初始化新的图像列表和已删除图像索引列表
        new_images = []
        droped_img_idx = []
        # 创建包含合成边界框坐标的列表
        image_bboxes = [[b[0], b[1], b[2], b[3]] for b in final_synthetic_bboxs]        
    # 遍历所有图像边界框的索引 i
    for i in range(0, len(image_bboxes)):
        # 遍历所有图像边界框的索引 j，j 从 i+1 开始，避免重复比较
        for j in range(i+1, len(image_bboxes)):
            # 检查 j 是否不在被删除的图像索引中
            if j not in droped_img_idx:
                # 解构 image_bboxes[j] 中的坐标
                L2, U2, R2, D2 = image_bboxes[j]
                # 计算图像边界框 i 的面积
                s1 = abs(R1 - L1) * abs(D1 - U1)
                # 计算图像边界框 j 的面积
                s2 = abs(R2 - L2) * abs(D2 - U2)
                # 计算两个矩形的重叠比例
                ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
                # 初始化合并标志为 False
                union_ok = False
                # 判断重叠比例是否满足条件，若满足则标记为可以合并
                if (ratio_1 >= 0.8 and ratio_2 <= 0.6) or (ratio_1 > 0.6 and ratio_2 > 0.6): 
                    union_ok = True
                # 判断面积比例是否满足条件，若满足则标记为可以合并
                if (ratio_1 > 0.2 and s2 / s1 > 5):
                    union_ok = True
                # 判断图像边界框 j 的中心点是否在图像边界框 i 内
                if (L1 <= (L2+R2)/2 <= R1) and (U1 <= (U2+D2)/2 <= D1):
                    union_ok = True
                # 判断图像边界框 i 的中心点是否在图像边界框 j 内
                if (L2 <= (L1+R1)/2 <= R2) and (U2 <= (U1+D1)/2 <= D2):
                    union_ok = True
                # 如果可以合并，则执行合并操作
                if union_ok == True:
                    # 更新图像边界框 i 为合并后的边界框
                    image_bboxes[i][0], image_bboxes[i][1],image_bboxes[i][2],image_bboxes[i][3] = min(image_bboxes[i][0], image_bboxes[j][0]), min(image_bboxes[i][1], image_bboxes[j][1]), max(image_bboxes[i][2], image_bboxes[j][2]), max(image_bboxes[i][3], image_bboxes[j][3])
                    # 将 j 添加到被删除的图像索引中
                    droped_img_idx.append(j)
            
    # 遍历所有图像边界框的索引 i
    for i in range(0, len(image_bboxes)):
        # 检查 i 是否不在被删除的图像索引中
        if i not in droped_img_idx:
            # 将不被删除的图像边界框添加到新图像列表中
            new_images.append(image_bboxes[i])
    
    
    # find_union_FLAG = True
    # while find_union_FLAG == True:
    #     find_union_FLAG = False
    #     final_duplicate = set()
    #     tmp = []
    #     # 遍历所有合成边界框的索引 i
    #     for i in range(len(final_synthetic_bboxs)):
    #         # 解构合成边界框 i 的坐标
    #         L1, U1, R1, D1 = final_synthetic_bboxs[i]
    #         # 遍历所有合成边界框的索引 j
    #         for j in range(len(final_synthetic_bboxs)):
    #             # 跳过自身比较
    #             if i == j:
    #                 continue
    #             # 解构合成边界框 j 的坐标
    #             L2, U2, R2, D2 = final_synthetic_bboxs[j]
    #             # 计算合成边界框 i 的面积
    #             s1 = abs(R1 - L1) * abs(D1 - U1)
    #             # 计算合成边界框 j 的面积
    #             s2 = abs(R2 - L2) * abs(D2 - U2)
    #             # 计算两个矩形的重叠比例
    #             ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
    #             # 初始化合并标志为 False
    #             union_ok = False
    #             # 判断重叠比例是否满足条件，若满足则标记为可以合并
    #             if (ratio_1 >= 0.8 and ratio_2 <= 0.6) or (ratio_1 > 0.6 and ratio_2 > 0.6): 
    #                 union_ok = True
    #             # 判断面积比例是否满足条件，若满足则标记为可以合并
    #             if (ratio_1 > 0.2 and s2 / s1 > 5):
    #                 union_ok = True
    #             # 判断合成边界框 j 的中心点是否在合成边界框 i 内
    #             if (L1 <= (L2+R2)/2 <= R1) and (U1 <= (U2+D2)/2 <= D1):
    #                 union_ok = True
    #             # 判断合成边界框 i 的中心点是否在合成边界框 j 内
    #             if (L2 <= (L1+R1)/2 <= R2) and (U2 <= (U1+D1)/2 <= D2):
    #                 union_ok = True
    #             # 如果可以合并，则执行合并操作
    #             if union_ok == True:
    #                 find_union_FLAG = True
    #                 # 将重叠的边界框添加到重复集合中
    #                 final_duplicate.add((L1, U1, R1, D1))
    #                 final_duplicate.add((L2, U2, R2, D2))
    #                 # 计算合并后的边界框
    #                 L3, U3, R3, D3 = min(L1, L2), min(U1, U2), max(R1, R2), max(D1, D2)
    #                 # 将合并后的边界框添加到临时列表中
    #                 tmp.append((L3, U3, R3, D3)) 
    #     # 如果发生了合并，将临时列表去重
    #     if find_union_FLAG == True:
    #         tmp = list(set(tmp))
    # 创建一个最终的合成边界框列表，暂时将其复制到 final_synthetic_bboxs 中（注释掉）
    # final_synthetic_bboxs = tmp[:]
    
    # 将当前页面的所有图形边界框与最终合成边界框进行合并
    # curPage_all_fig_bboxs += final_synthetic_bboxs
    # 输出合成边界框的分隔符
    # print('--------- final synthetic')
    # 输出最终合成边界框的内容（注释掉）
    # print(final_synthetic_bboxs)
    #**************************************************************************#
    # 生成一个包含当前页面所有图形边界框的图像信息列表
    images1 = [[img[0], img[1], img[2], img[3]] for img in curPage_all_fig_bboxs]
    # 将当前页面的图像信息与新的图像信息合并
    images = images1 + new_images
    # 返回合并后的图像信息列表
    return images
```