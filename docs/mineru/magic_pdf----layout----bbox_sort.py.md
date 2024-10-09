# `.\MinerU\magic_pdf\layout\bbox_sort.py`

```
# 定义bbox的结构，包括坐标、内容、索引和类型等信息，初始idx_x和idx_y为None
# 其中x0, y0代表左上角坐标，x1, y1代表右下角坐标，坐标原点在左上角。

# 导入处理页面分割的函数
from magic_pdf.layout.layout_spiler_recog import get_spilter_of_page
# 导入用于判断矩形重叠关系的函数
from magic_pdf.libs.boxbase import _is_in, _is_in_or_part_overlap, _is_vertical_full_overlap
# 导入求最大值的函数
from magic_pdf.libs.commons import mymax

# 定义常量，表示bbox的各个索引
X0_IDX = 0
Y0_IDX = 1
X1_IDX = 2
Y1_IDX = 3
CONTENT_IDX = 4
IDX_X = 5
IDX_Y = 6
CONTENT_TYPE_IDX = 7

# 定义扩展bbox的各个索引
X0_EXT_IDX = 8
Y0_EXT_IDX = 9
X1_EXT_IDX = 10
Y1_EXT_IDX = 11

# 准备布局分割所需的bbox
def prepare_bboxes_for_layout_split(image_info, image_backup_info, table_info, inline_eq_info, interline_eq_info, text_raw_blocks: dict, page_boundry, page):
    """
    text_raw_blocks:结构参考test/assets/papre/pymu_textblocks.json
    重新组装bbox成列表，每个元素包含位置信息、内容及类型等，初始idx_x和idx_y为None
    """
    all_bboxes = []  # 初始化所有bbox列表
    
    # 遍历每个图像信息
    for image in image_info:
        box = image['bbox']  # 获取当前图像的bbox
        # 过滤掉长宽均小于50的图片，以避免影响布局
        if abs(box[0]-box[2]) < 50 and abs(box[1]-box[3]) < 50:
            continue  # 小图片直接跳过
        # 将有效的图像bbox添加到all_bboxes中
        all_bboxes.append([box[0], box[1], box[2], box[3], None, None, None, 'image', None, None, None, None])
        
    # 遍历每个表格信息
    for table in table_info:
        box = table['bbox']  # 获取当前表格的bbox
        # 将表格bbox添加到all_bboxes中
        all_bboxes.append([box[0], box[1], box[2], box[3], None, None, None, 'table', None, None, None, None])
    
    """由于公式与段落混合，因此公式不再参与layout划分，无需加入all_bboxes"""
    # 加入文本块
    text_block_temp = []  # 初始化文本块临时列表
    for block in text_raw_blocks:
        bbox = block['bbox']  # 获取当前文本块的bbox
        # 将文本块的bbox添加到临时列表中
        text_block_temp.append([bbox[0], bbox[1], bbox[2], bbox[3], None, None, None, 'text', None, None, None, None])
        
    # 解决文本块之间的重叠问题
    text_block_new = resolve_bbox_overlap_for_layout_det(text_block_temp)   
    # 过滤掉线条bbox，避免无限循环
    text_block_new = filter_lines_bbox(text_block_new) 
    
    """找出会影响layout的色块、横向分割线"""
    # 获取页面中的分割bbox
    spilter_bboxes = get_spilter_of_page(page, [b['bbox'] for b in image_info]+[b['bbox'] for b in image_backup_info], [b['bbox'] for b in table_info])
    # 去掉在spilter_bboxes中存在的文本块
    if len(spilter_bboxes) > 0:
        text_block_new = [box for box in text_block_new if not any([_is_in_or_part_overlap(box[:4], spilter_bbox) for spilter_bbox in spilter_bboxes])]
        
    # 将有效的文本块添加到all_bboxes中
    for bbox in text_block_new:
        all_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], None, None, None, 'text', None, None, None, None]) 
        
    # 将分割bbox添加到all_bboxes中
    for bbox in spilter_bboxes:
        all_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], None, None, None, 'spilter', None, None, None, None])
    
    return all_bboxes  # 返回所有的bbox列表

# 处理bbox重叠的函数
def resolve_bbox_overlap_for_layout_det(bboxes:list):
    """
    1. 去掉bbox互相包含的，去掉被包含的
    2. 上下方向上如果有重叠，就扩大大box范围，直到覆盖小box
    """
    # 定义一个内部函数，用于判断第i个框是否被其他框包含
        def _is_in_other_bbox(i:int):
            """
            判断i个box是否被其他box有所包含
            """
            # 遍历所有边界框
            for j in range(0, len(bboxes)):
                # 检查j不是i，并判断i框是否被j框包含
                if j!=i and _is_in(bboxes[i][:4], bboxes[j][:4]):
                    return True  # 如果包含，返回True
                
                # 该条件被注释掉，表示考虑底部完全重叠的情况
                # elif j!=i and _is_bottom_full_overlap(bboxes[i][:4], bboxes[j][:4]):
                #     return True
                
            return False  # 如果没有被其他框包含，返回False
        
        # 首先去掉被包含的bbox，创建新边界框列表
        new_bbox_1 = []
        # 遍历所有边界框
        for i in range(0, len(bboxes)):
            # 如果当前框不被其他框包含，则添加到新列表中
            if not _is_in_other_bbox(i):
                new_bbox_1.append(bboxes[i])
                
        # 其次扩展大的box，创建新边界框列表
        new_box = []
        new_bbox_2 = []
        len_1 = len(new_bbox_2)  # 记录new_bbox_2的长度
        while True:
            merged_idx = []  # 存储已合并框的索引
            # 遍历新边界框列表
            for i in range(0, len(new_bbox_1)):
                if i in merged_idx:  # 如果已合并，跳过
                    continue
                # 嵌套遍历，寻找可以合并的框
                for j in range(i+1, len(new_bbox_1)):
                    if j in merged_idx:  # 如果已合并，跳过
                        continue
                    bx1 = new_bbox_1[i]  # 当前框
                    bx2 = new_bbox_1[j]  # 待比较框
                    # 检查是否是不同的框，并判断是否垂直完全重叠
                    if i!=j and _is_vertical_full_overlap(bx1[:4], bx2[:4]):
                        # 合并两个框，获取新的边界框
                        merged_box = min([bx1[0], bx2[0]]), min([bx1[1], bx2[1]]), max([bx1[2], bx2[2]]), max([bx1[3], bx2[3]])
                        new_bbox_2.append(merged_box)  # 添加合并后的框
                        merged_idx.append(i)  # 记录已合并框索引
                        merged_idx.append(j)
                        
            # 遍历新边界框，添加未合并的框
            for i in range(0, len(new_bbox_1)): # 没有合并的加入进来
                if i not in merged_idx:
                    new_bbox_2.append(new_bbox_1[i])        
    
            # 如果没有合并框或合并前后长度相同，则结束循环
            if len(new_bbox_2)==0 or len_1==len(new_bbox_2):
                break
            else:
                len_1 = len(new_bbox_2)  # 更新新边界框的长度
                new_box = new_bbox_2  # 更新合并框列表
                new_bbox_1, new_bbox_2 = new_bbox_2, []  # 交换新边界框列表
        
        return new_box  # 返回合并后的边界框列表
# 定义一个函数，过滤掉bbox为空的行
def filter_lines_bbox(bboxes: list):
    # 创建一个新列表，用于存放非空的bbox
    new_box = []
    # 遍历所有的bbox
    for box in bboxes:
        # 提取bbox的四个坐标值
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        # 如果bbox的宽度或高度小于等于1，则跳过该bbox
        if abs(x0-x1)<=1 or abs(y0-y1)<=1:
            continue
        else:
            # 将有效的bbox添加到新列表中
            new_box.append(box)
    # 返回过滤后的bbox列表
    return new_box


################################################################################
# 第一种排序算法
# 以下是基于延长线遮挡做的一个算法
#
################################################################################
# 定义一个函数，寻找this_bbox左边的所有bbox
def find_all_left_bbox(this_bbox, all_bboxes) -> list:
    # 从all_bboxes中筛选出所有左侧bbox
    left_boxes = [box for box in all_bboxes if box[X1_IDX] <= this_bbox[X0_IDX]]
    # 返回左侧bbox列表
    return left_boxes


# 定义一个函数，寻找this_bbox上面的所有bbox
def find_all_top_bbox(this_bbox, all_bboxes) -> list:
    # 从all_bboxes中筛选出所有上方bbox
    top_boxes = [box for box in all_bboxes if box[Y1_IDX] <= this_bbox[Y0_IDX]]
    # 返回上方bbox列表
    return top_boxes


# 定义一个函数，寻找this_bbox在all_bboxes中的遮挡深度 idx_x
def get_and_set_idx_x(this_bbox, all_bboxes) -> int:
    # 如果this_bbox的idx_x已定义，直接返回
    if this_bbox[IDX_X] is not None:
        return this_bbox[IDX_X]
    else:
        # 查找this_bbox左侧的所有bbox
        all_left_bboxes = find_all_left_bbox(this_bbox, all_bboxes)
        # 如果没有左侧bbox，设置idx_x为0
        if len(all_left_bboxes) == 0:
            this_bbox[IDX_X] = 0
        else:
            # 递归获取左侧bbox的idx_x值
            all_left_bboxes_idx = [get_and_set_idx_x(bbox, all_bboxes) for bbox in all_left_bboxes]
            # 取左侧bbox idx_x的最大值
            max_idx_x = mymax(all_left_bboxes_idx)
            # 设置当前bbox的idx_x为最大值加1
            this_bbox[IDX_X] = max_idx_x + 1
        # 返回当前bbox的idx_x
        return this_bbox[IDX_X]


# 定义一个函数，寻找this_bbox在all_bboxes中y方向的遮挡深度 idx_y
def get_and_set_idx_y(this_bbox, all_bboxes) -> int:
    # 如果this_bbox的idx_y已定义，直接返回
    if this_bbox[IDX_Y] is not None:
        return this_bbox[IDX_Y]
    else:
        # 查找this_bbox上方的所有bbox
        all_top_bboxes = find_all_top_bbox(this_bbox, all_bboxes)
        # 如果没有上方bbox，设置idx_y为0
        if len(all_top_bboxes) == 0:
            this_bbox[IDX_Y] = 0
        else:
            # 递归获取上方bbox的idx_y值
            all_top_bboxes_idx = [get_and_set_idx_y(bbox, all_bboxes) for bbox in all_top_bboxes]
            # 取上方bbox idx_y的最大值
            max_idx_y = mymax(all_top_bboxes_idx)
            # 设置当前bbox的idx_y为最大值加1
            this_bbox[IDX_Y] = max_idx_y + 1
        # 返回当前bbox的idx_y
        return this_bbox[IDX_Y]


# 定义一个函数，对所有bbox进行排序
def bbox_sort(all_bboxes: list):
    # 获取每个bbox的idx_x值
    all_bboxes_idx_x = [get_and_set_idx_x(bbox, all_bboxes) for bbox in all_bboxes]
    # 获取每个bbox的idx_y值
    all_bboxes_idx_y = [get_and_set_idx_y(bbox, all_bboxes) for bbox in all_bboxes]
    # 将idx_x和idx_y配对成元组列表
    all_bboxes_idx = [(idx_x, idx_y) for idx_x, idx_y in zip(all_bboxes_idx_x, all_bboxes_idx_y)]

    # 计算排序键，将idx_x和idx_y合并为一个数以保证先按X排序，X相同时按Y排序
    all_bboxes_idx = [idx_x_y[0] * 100000 + idx_x_y[1] for idx_x_y in all_bboxes_idx]
    # 将索引和对应的bbox配对
    all_bboxes_idx = list(zip(all_bboxes_idx, all_bboxes))
    # 根据排序键进行排序
    all_bboxes_idx.sort(key=lambda x: x[0])
    # 提取排序后的bbox列表
    sorted_bboxes = [bbox for idx, bbox in all_bboxes_idx]
    # 返回排序后的bbox列表
    return sorted_bboxes


################################################################################
# 第二种排序算法
# 下面的算法在计算idx_x和idx_y的时候不考虑延长线，而只考虑实际的长或者宽被遮挡的情况
#
################################################################################
# 定义一个函数，在all_bboxes里找到所有右侧高度和this_bbox有重叠的bbox
def find_left_nearest_bbox(this_bbox, all_bboxes) -> list:
    # 函数实现省略
    # 从所有边界框中筛选出左侧的框
    left_boxes = [box for box in all_bboxes if box[X1_IDX] <= this_bbox[X0_IDX] and any([
        # 检查当前框是否与目标框在Y轴上重叠
        box[Y0_IDX] < this_bbox[Y0_IDX] < box[Y1_IDX], box[Y0_IDX] < this_bbox[Y1_IDX] < box[Y1_IDX],
        this_bbox[Y0_IDX] < box[Y0_IDX] < this_bbox[Y1_IDX], this_bbox[Y0_IDX] < box[Y1_IDX] < this_bbox[Y1_IDX],
        # 检查边界框完全重合的情况
        box[Y0_IDX] == this_bbox[Y0_IDX] and box[Y1_IDX] == this_bbox[Y1_IDX]])]
        
    # 如果找到左侧的框，则进一步处理
    if len(left_boxes) > 0:
        # 根据X轴坐标排序，寻找最靠近目标框的边界框
        left_boxes.sort(key=lambda x: x[X1_IDX], reverse=True)
        # 只保留距离最近的那个框
        left_boxes = [left_boxes[0]]
    else:
        # 如果没有找到左侧的框，返回空列表
        left_boxes = []
    # 返回找到的左侧框
    return left_boxes
# 寻找this_bbox在all_bboxes中的被直接遮挡的深度 idx_x
def get_and_set_idx_x_2(this_bbox, all_bboxes):
    # 如果this_bbox的idx_x已经被设置，直接返回其值
    if this_bbox[IDX_X] is not None:
        return this_bbox[IDX_X]
    else:
        # 找到与this_bbox左侧最近的bbox
        left_nearest_bbox = find_left_nearest_bbox(this_bbox, all_bboxes)
        # 如果没有找到左侧的bbox，设置idx_x为0
        if len(left_nearest_bbox) == 0:
            this_bbox[IDX_X] = 0
        else:
            # 递归获取左侧最近bbox的idx_x
            left_idx_x = get_and_set_idx_x_2(left_nearest_bbox[0], all_bboxes)
            # 将当前bbox的idx_x设置为左侧idx_x加1
            this_bbox[IDX_X] = left_idx_x + 1
        # 返回当前bbox的idx_x
        return this_bbox[IDX_X]


# 在all_bboxes里找到所有下侧宽度和this_bbox有重叠的bbox
def find_top_nearest_bbox(this_bbox, all_bboxes) -> list:
    # 遍历all_bboxes，找到下侧与this_bbox重叠的bbox
    top_boxes = [box for box in all_bboxes if box[Y1_IDX] <= this_bbox[Y0_IDX] and any([
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], 
        box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
        this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], 
        this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX] == this_bbox[X0_IDX] and box[X1_IDX] == this_bbox[X1_IDX]])]
    # 如果找到重叠的bbox，按Y坐标降序排序，取最高的那个
    if len(top_boxes) > 0:
        top_boxes.sort(key=lambda x: x[Y1_IDX], reverse=True)
        top_boxes = [top_boxes[0]]  # 只保留距离this_bbox最近的一个
    else:
        top_boxes = []  # 如果没有找到，返回空列表
    return top_boxes


# 寻找this_bbox在all_bboxes中的被直接遮挡的深度 idx_y
def get_and_set_idx_y_2(this_bbox, all_bboxes):
    # 如果this_bbox的idx_y已经被设置，直接返回其值
    if this_bbox[IDX_Y] is not None:
        return this_bbox[IDX_Y]
    else:
        # 找到与this_bbox顶部最近的bbox
        top_nearest_bbox = find_top_nearest_bbox(this_bbox, all_bboxes)
        # 如果没有找到顶部的bbox，设置idx_y为0
        if len(top_nearest_bbox) == 0:
            this_bbox[IDX_Y] = 0
        else:
            # 递归获取顶部最近bbox的idx_y
            top_idx_y = get_and_set_idx_y_2(top_nearest_bbox[0], all_bboxes)
            # 将当前bbox的idx_y设置为顶部idx_y加1
            this_bbox[IDX_Y] = top_idx_y + 1
        # 返回当前bbox的idx_y
        return this_bbox[IDX_Y]


# 对所有bbox进行排序，并返回排序后的结果
def paper_bbox_sort(all_bboxes: list, page_width, page_height):
    # 获取所有bbox的idx_x
    all_bboxes_idx_x = [get_and_set_idx_x_2(bbox, all_bboxes) for bbox in all_bboxes]
    # 获取所有bbox的idx_y
    all_bboxes_idx_y = [get_and_set_idx_y_2(bbox, all_bboxes) for bbox in all_bboxes]
    # 将idx_x和idx_y组合成元组列表
    all_bboxes_idx = [(idx_x, idx_y) for idx_x, idx_y in zip(all_bboxes_idx_x, all_bboxes_idx_y)]

    # 将idx_x和idx_y转换成一个单一的点，用于排序
    all_bboxes_idx = [idx_x_y[0] * 100000 + idx_x_y[1] for idx_x_y in all_bboxes_idx]  
    # 将组合后的idx和bbox结合在一起
    all_bboxes_idx = list(zip(all_bboxes_idx, all_bboxes))
    # 按照idx进行排序
    all_bboxes_idx.sort(key=lambda x: x[0])
    # 取出排序后的bbox
    sorted_bboxes = [bbox for idx, bbox in all_bboxes_idx]
    return sorted_bboxes

################################################################################
# 排序算法说明，假设page的最左侧为X0，最右侧为X1，最上侧为Y0，最下侧为Y1
# 此算法在第二种算法基础上增加对bbox的预处理步骤
    # 对 bbox 列表按 y 方向进行排序，确保从上到下排列
    - 对bbox进行y方向排序，然后从上到下遍历所有bbox，如果当前bbox和下一个bbox的x0, x1等于X0, X1，那么就合并这两个bbox。
# 在垂直方向上扩展bbox，主要步骤包括切割和合并
3. 然后在垂直方向上对bbox进行扩展。扩展方法是：
    # 从页面中切割掉合并后的水平bbox，得到多个新的block
    - 首先从page上切割掉合并后的水平bbox, 得到几个新的block
    针对每个block
    # 更新x0为左侧邻近bbox的最大x1值或保持不变
    - x0: 扎到位于左侧x=x0延长线的左侧所有的bboxes, 找到最大的x1,让x0=x1+1。如果没有，则x0=X0
    # 更新x1为右侧邻近bbox的最小x0值或保持不变
    - x1: 找到位于右侧x=x1延长线右侧所有的bboxes， 找到最小的x0, 让x1=x0-1。如果没有，则x1=X1
    # 在垂直方向上合并所有连续的block
    随后在垂直方向上合并所有的连续的block，方法如下：
    # 对block进行x方向排序并遍历，合并相邻的block
    - 对block进行x方向排序，然后从左到右遍历所有block，如果当前block和下一个block的x0, x1相等，那么就合并这两个block。
    # 判断分割完成状态并标记布局好坏
    如果垂直切分后所有小bbox都被分配到了一个block, 那么分割就完成了。这些合并后的block打上标签'GOOD_LAYOUT’
    # 如果有block未能完全分割，则标记为'坏布局'
    如果在某个垂直方向上无法被完全分割到一个block，那么就将这个block打上标签'BAD_LAYOUT'。
    # 完成页面预处理，标记每个block的布局状态
    至此完成，一个页面的预处理，天然的block要么属于'GOOD_LAYOUT'，要么属于'BAD_LAYOUT'。针对含有'BAD_LAYOUT'的页面，可以先按照自上而下，自左到右进行天然排序，也可以先过滤掉这种书籍。
    # 后续条件加强说明
    (完成条件下次加强：进行水平方向切分，把混乱的layout部分尽可能切割出去)
################################################################################
def find_left_neighbor_bboxes(this_bbox, all_bboxes) -> list:
    # 寻找all_bboxes中所有与this_bbox重叠的左侧bbox
    """
    在all_bboxes里找到所有右侧高度和this_bbox有重叠的bbox
    这里使用扩展之后的bbox
    """
    # 通过条件过滤获取左侧重叠的bbox
    left_boxes = [box for box in all_bboxes if box[X1_EXT_IDX] <= this_bbox[X0_EXT_IDX] and any([
         # 检查重叠条件
         box[Y0_EXT_IDX] < this_bbox[Y0_EXT_IDX] < box[Y1_EXT_IDX], box[Y0_EXT_IDX] < this_bbox[Y1_EXT_IDX] < box[Y1_EXT_IDX],
         this_bbox[Y0_EXT_IDX] < box[Y0_EXT_IDX] < this_bbox[Y1_EXT_IDX], this_bbox[Y0_EXT_IDX] < box[Y1_EXT_IDX] < this_bbox[Y1_EXT_IDX],
         box[Y0_EXT_IDX]==this_bbox[Y0_EXT_IDX] and box[Y1_EXT_IDX]==this_bbox[Y1_EXT_IDX]])]
        
    # 如果找到左侧bbox，按x1降序排序
    # 然后再过滤一下，找到水平上距离this_bbox最近的那个
    if len(left_boxes) > 0:
        left_boxes.sort(key=lambda x: x[X1_EXT_IDX], reverse=True)
        left_boxes = left_boxes
    # 如果没有找到，设置为空列表
    else:
        left_boxes = []
    # 返回找到的左侧bbox列表
    return left_boxes

def find_top_neighbor_bboxes(this_bbox, all_bboxes) -> list:
    # 寻找all_bboxes中所有与this_bbox重叠的上侧bbox
    """
    在all_bboxes里找到所有下侧宽度和this_bbox有重叠的bbox
    这里使用扩展之后的bbox
    """
    # 通过条件过滤获取上侧重叠的bbox
    top_boxes = [box for box in all_bboxes if box[Y1_EXT_IDX] <= this_bbox[Y0_EXT_IDX] and any([
        # 检查重叠条件
        box[X0_EXT_IDX] < this_bbox[X0_EXT_IDX] < box[X1_EXT_IDX], box[X0_EXT_IDX] < this_bbox[X1_EXT_IDX] < box[X1_EXT_IDX],
         this_bbox[X0_EXT_IDX] < box[X0_EXT_IDX] < this_bbox[X1_EXT_IDX], this_bbox[X0_EXT_IDX] < box[X1_EXT_IDX] < this_bbox[X1_EXT_IDX],
        box[X0_EXT_IDX]==this_bbox[X0_EXT_IDX] and box[X1_EXT_IDX]==this_bbox[X1_EXT_IDX]])]
    # 如果找到上侧bbox，按y1降序排序
    # 然后再过滤一下，找到水平上距离this_bbox最近的那个
    if len(top_boxes) > 0:
        top_boxes.sort(key=lambda x: x[Y1_EXT_IDX], reverse=True)
        top_boxes = top_boxes
    # 如果没有找到，设置为空列表
    else:
        top_boxes = []
    # 返回找到的上侧bbox列表
    return top_boxes

def get_and_set_idx_x_2_ext(this_bbox, all_bboxes):
    # 寻找this_bbox在all_bboxes中的被直接遮挡的深度 idx_x
    """
    寻找this_bbox在all_bboxes中的被直接遮挡的深度 idx_x
    这个遮挡深度不考虑延长线，而是被实际的长或者宽遮挡的情况
    """
    # 如果已有idx_x则直接返回
    if this_bbox[IDX_X] is not None:
        return this_bbox[IDX_X]
    else:
        # 找到左侧最近的bbox
        left_nearest_bbox = find_left_neighbor_bboxes(this_bbox, all_bboxes)
        # 如果没有找到左侧bbox，idx_x设为0
        if len(left_nearest_bbox) == 0:
            this_bbox[IDX_X] = 0
        else:
            # 获取所有左侧bbox的idx_x并更新
            left_idx_x = [get_and_set_idx_x_2(b, all_bboxes) for b in left_nearest_bbox]
            this_bbox[IDX_X] = mymax(left_idx_x) + 1
        # 返回更新后的idx_x
        return this_bbox[IDX_X]
   
def get_and_set_idx_y_2_ext(this_bbox, all_bboxes):
    # 代码注释继续...
    # 寻找this_bbox在all_bboxes中的被直接遮挡的深度 idx_y
    # 这个遮挡深度不考虑延长线，而是被实际的长或者宽遮挡的情况
    if this_bbox[IDX_Y] is not None:
        # 如果this_bbox的遮挡深度已定义，直接返回该值
        return this_bbox[IDX_Y]
    else:
        # 找到this_bbox的上方邻近边界框
        top_nearest_bbox = find_top_neighbor_bboxes(this_bbox, all_bboxes)
        if len(top_nearest_bbox) == 0:
            # 如果没有邻近边界框，设置遮挡深度为0
            this_bbox[IDX_Y] = 0
        else:
            # 获取邻近边界框的遮挡深度并设置到this_bbox
            top_idx_y = [get_and_set_idx_y_2_ext(b, all_bboxes) for b in top_nearest_bbox]
            this_bbox[IDX_Y] = mymax(top_idx_y) + 1
        # 返回计算后的遮挡深度
        return this_bbox[IDX_Y]
# 对所有边界框进行排序处理，返回排序后的边界框列表
def _paper_bbox_sort_ext(all_bboxes: list):
    # 获取每个边界框在 X 轴上的索引
    all_bboxes_idx_x = [get_and_set_idx_x_2_ext(bbox, all_bboxes) for bbox in all_bboxes]
    # 获取每个边界框在 Y 轴上的索引
    all_bboxes_idx_y = [get_and_set_idx_y_2_ext(bbox, all_bboxes) for bbox in all_bboxes]
    # 将 X 和 Y 的索引组合成元组
    all_bboxes_idx = [(idx_x, idx_y) for idx_x, idx_y in zip(all_bboxes_idx_x, all_bboxes_idx_y)]
    
    # 将索引元组转换为单个数值，以确保按 X 排序，若 X 相同则按 Y 排序
    all_bboxes_idx = [idx_x_y[0] * 100000 + idx_x_y[1] for idx_x_y in all_bboxes_idx]
    # 将索引与原始边界框进行配对
    all_bboxes_idx = list(zip(all_bboxes_idx, all_bboxes))
    # 根据索引排序边界框
    all_bboxes_idx.sort(key=lambda x: x[0])
    # 提取排序后的边界框
    sorted_bboxes = [bbox for idx, bbox in all_bboxes_idx]
    # 返回排序后的边界框列表
    return sorted_bboxes

# ===============================================================================================
# 寻找 this_bbox 左边的所有边界框，使用延长线
def find_left_bbox_ext_line(this_bbox, all_bboxes) -> list:
    """
    寻找this_bbox左边的所有bbox, 使用延长线
    """
    # 获取所有在 this_bbox 左侧的边界框
    left_boxes = [box for box in all_bboxes if box[X1_IDX] <= this_bbox[X0_IDX]]
    # 如果找到左侧的边界框
    if len(left_boxes):
        # 根据 X1 值降序排序
        left_boxes.sort(key=lambda x: x[X1_IDX], reverse=True)
        # 选择排序后的第一个边界框
        left_boxes = left_boxes[0]
    else:
        # 未找到左侧边界框
        left_boxes = None
    
    # 返回找到的边界框或 None
    return left_boxes

# 寻找 this_bbox 右边的所有边界框，使用延长线
def find_right_bbox_ext_line(this_bbox, all_bboxes) -> list:
    """
    寻找this_bbox右边的所有bbox, 使用延长线
    """
    # 获取所有在 this_bbox 右侧的边界框
    right_boxes = [box for box in all_bboxes if box[X0_IDX] >= this_bbox[X1_IDX]]
    # 如果找到右侧的边界框
    if len(right_boxes):
        # 根据 X0 值升序排序
        right_boxes.sort(key=lambda x: x[X0_IDX])
        # 选择排序后的第一个边界框
        right_boxes = right_boxes[0]
    else:
        # 未找到右侧边界框
        right_boxes = None
    # 返回找到的边界框或 None
    return right_boxes

# =============================================================================================

# 寻找在 all_bboxes 中与 this_bbox 左侧重叠的边界框
def find_left_nearest_bbox_direct(this_bbox, all_bboxes) -> list:
    """
    在all_bboxes里找到所有右侧高度和this_bbox有重叠的bbox， 不用延长线并且不能像
    """
    # 获取所有在 this_bbox 左侧且有 Y 轴重叠的边界框
    left_boxes = [box for box in all_bboxes if box[X1_IDX] <= this_bbox[X0_IDX] and any([
         box[Y0_IDX] < this_bbox[Y0_IDX] < box[Y1_IDX], box[Y0_IDX] < this_bbox[Y1_IDX] < box[Y1_IDX],
         this_bbox[Y0_IDX] < box[Y0_IDX] < this_bbox[Y1_IDX], this_bbox[Y0_IDX] < box[Y1_IDX] < this_bbox[Y1_IDX],
         box[Y0_IDX]==this_bbox[Y0_IDX] and box[Y1_IDX]==this_bbox[Y1_IDX]])]
        
    # 如果找到左侧边界框，按 X1 值降序排序
    if len(left_boxes) > 0:
        left_boxes.sort(key=lambda x: x[X1_EXT_IDX] if x[X1_EXT_IDX] else x[X1_IDX], reverse=True)
        # 选择排序后的第一个边界框
        left_boxes = left_boxes[0]
    else:
        # 未找到左侧边界框
        left_boxes = None
    # 返回找到的边界框或 None
    return left_boxes

# 寻找在 this_bbox 右侧且直接遮挡的边界框
def find_right_nearst_bbox_direct(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox右侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    # 获取所有在 this_bbox 右侧且有 Y 轴重叠的边界框
    right_bboxes = [box for box in all_bboxes if box[X0_IDX] >= this_bbox[X1_IDX] and any([
        this_bbox[Y0_IDX] < box[Y0_IDX] < this_bbox[Y1_IDX], this_bbox[Y0_IDX] < box[Y1_IDX] < this_bbox[Y1_IDX],
        box[Y0_IDX] < this_bbox[Y0_IDX] < box[Y1_IDX], box[Y0_IDX] < this_bbox[Y1_IDX] < box[Y1_IDX],
        box[Y0_IDX]==this_bbox[Y0_IDX] and box[Y1_IDX]==this_bbox[Y1_IDX]])]
    
    # 检查右侧边界框是否有元素
        if len(right_bboxes)>0:
            # 对右侧边界框进行排序，依据第一个有效的坐标值
            right_bboxes.sort(key=lambda x: x[X0_EXT_IDX] if x[X0_EXT_IDX] else x[X0_IDX])
            # 取排序后的第一个边界框
            right_bboxes = right_bboxes[0]
        else:
            # 如果没有右侧边界框，赋值为 None
            right_bboxes = None
        # 返回最终的右侧边界框
        return right_bboxes
# 重置所有边界框的 X 和 Y 索引为 None
def reset_idx_x_y(all_boxes:list)->list:
    # 遍历每个边界框
    for box in all_boxes:
        # 将边界框的 X 索引设为 None
        box[IDX_X] = None
        # 将边界框的 Y 索引设为 None
        box[IDX_Y] = None
        
    # 返回修改后的边界框列表
    return all_boxes

# ===================================================================================================
# 找到在给定边界框上方且距离最近的直接遮挡的边界框
def find_top_nearest_bbox_direct(this_bbox, bboxes_collection) -> list:
    """
    找到在this_bbox上方且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    # 筛选出在 this_bbox 上方的边界框，并满足遮挡条件
    top_bboxes = [box for box in bboxes_collection if box[Y1_IDX] <= this_bbox[Y0_IDX] and any([
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
         this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    # 如果找到的上方边界框不为空
    if len(top_bboxes) > 0:
        # 按 Y1 索引降序排序，寻找最近的边界框
        top_bboxes.sort(key=lambda x: x[Y1_IDX], reverse=True)
        # 取出距离最近的边界框
        top_bboxes = top_bboxes[0]
    else:
        # 如果没有找到，设为 None
        top_bboxes = None
    # 返回找到的边界框
    return top_bboxes

# 找到在给定边界框下方且距离最近的直接遮挡的边界框
def find_bottom_nearest_bbox_direct(this_bbox, bboxes_collection) -> list:
    """
    找到在this_bbox下方且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    # 筛选出在 this_bbox 下方的边界框，并满足遮挡条件
    bottom_bboxes = [box for box in bboxes_collection if box[Y0_IDX] >= this_bbox[Y1_IDX] and any([
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
         this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    # 如果找到的下方边界框不为空
    if len(bottom_bboxes) > 0:
        # 按 Y0 索引升序排序，寻找最近的边界框
        bottom_bboxes.sort(key=lambda x: x[Y0_IDX])
        # 取出距离最近的边界框
        bottom_bboxes = bottom_bboxes[0]
    else:
        # 如果没有找到，设为 None
        bottom_bboxes = None
    # 返回找到的边界框
    return bottom_bboxes

# 找到边界框集合的边界，返回最小和最大坐标
def find_boundry_bboxes(bboxes:list) -> tuple:
    """
    找到bboxes的边界——找到所有bbox里最小的(x0, y0), 最大的(x1, y1)
    """
    # 初始化边界坐标为第一个边界框的坐标
    x0, y0, x1, y1 = bboxes[0][X0_IDX], bboxes[0][Y0_IDX], bboxes[0][X1_IDX], bboxes[0][Y1_IDX]
    # 遍历所有边界框，更新边界坐标
    for box in bboxes:
        # 更新最小的 X0
        x0 = min(box[X0_IDX], x0)
        # 更新最小的 Y0
        y0 = min(box[Y0_IDX], y0)
        # 更新最大的 X1
        x1 = max(box[X1_IDX], x1)
        # 更新最大的 Y1
        y1 = max(box[Y1_IDX], y1)
        
    # 返回边界坐标
    return x0, y0, x1, y1
    

# 在垂直方向上扩展能够直接垂直打通的边界框
def extend_bbox_vertical(bboxes:list, boundry_x0, boundry_y0, boundry_x1, boundry_y1) -> list:
    """
    在垂直方向上扩展能够直接垂直打通的bbox,也就是那些上下都没有其他box的bbox
    """
    # 遍历所有的边界框
    for box in bboxes:
        # 找到与当前框顶部最近的边界框
        top_nearest_bbox = find_top_nearest_bbox_direct(box, bboxes)
        # 找到与当前框底部最近的边界框
        bottom_nearest_bbox = find_bottom_nearest_bbox_direct(box, bboxes)
        # 如果当前框上方和下方都没有最近的边界框，说明它独占一列
        if top_nearest_bbox is None and bottom_nearest_bbox is None: # 独占一列
            # 设置扩展边界框的左上角坐标为当前框的左上角坐标
            box[X0_EXT_IDX] = box[X0_IDX]
            # 设置扩展边界框的左下角坐标为边界的下限
            box[Y0_EXT_IDX] = boundry_y0
            # 设置扩展边界框的右上角坐标为当前框的右上角坐标
            box[X1_EXT_IDX] = box[X1_IDX]
            # 设置扩展边界框的右下角坐标为边界的上限
            box[Y1_EXT_IDX] = boundry_y1
        # else:  # 下面的代码被注释掉了
        #     if top_nearest_bbox is None:  # 如果没有顶部最近框
        #         box[Y0_EXT_IDX] = boundry_y0  # 设置扩展框的上边界为边界下限
        #     else:  # 否则
        #         box[Y0_EXT_IDX] = top_nearest_bbox[Y1_IDX] + 1  # 设置扩展框的上边界为顶部最近框的下边界 + 1
        #     if bottom_nearest_bbox is None:  # 如果没有底部最近框
        #         box[Y1_EXT_IDX] = boundry_y1  # 设置扩展框的下边界为边界上限
        #     else:  # 否则
        #         box[Y1_EXT_IDX] = bottom_nearest_bbox[Y0_IDX] - 1  # 设置扩展框的下边界为底部最近框的上边界 - 1
        #     box[X0_EXT_IDX] = box[X0_IDX]  # 设置扩展框的左边界为当前框的左边界
        #     box[X1_EXT_IDX] = box[X1_IDX]  # 设置扩展框的右边界为当前框的右边界
    # 返回处理后的所有边界框
    return bboxes
# ===================================================================================================

def paper_bbox_sort_v2(all_bboxes: list, page_width:int, page_height:int):
    """
    增加预处理行为的排序:
    return:
    [
        {
            "layout_bbox": [x0, y0, x1, y1],
            "layout_label":"GOOD_LAYOUT/BAD_LAYOUT",
            "content_bboxes": [] #每个元素都是[x0, y0, x1, y1, block_content, idx_x, idx_y, content_type, ext_x0, ext_y0, ext_x1, ext_y1], 并且顺序就是阅读顺序
        }
    ]
    """
    # 初始化返回结果列表
    sorted_layouts = [] # 最后的返回结果
    # 定义页面边界坐标
    page_x0, page_y0, page_x1, page_y1 = 1, 1, page_width-1, page_height-1
    
    # 对所有的bbox进行初步排序
    all_bboxes = paper_bbox_sort(all_bboxes) # 大致拍下序
    # 首先在水平方向上扩展独占一行的bbox
    for bbox in all_bboxes:
        # 找到当前bbox左边最近的bbox
        left_nearest_bbox = find_left_nearest_bbox_direct(bbox, all_bboxes) # 非扩展线
        # 找到当前bbox右边最近的bbox
        right_nearest_bbox = find_right_nearst_bbox_direct(bbox, all_bboxes)
        # 如果左右都没有邻接的bbox，则标记为独占一行
        if left_nearest_bbox is None and right_nearest_bbox is None: # 独占一行
            # 扩展bbox到页面左边界
            bbox[X0_EXT_IDX] = page_x0
            # y0坐标保持不变
            bbox[Y0_EXT_IDX] = bbox[Y0_IDX]
            # 扩展bbox到页面右边界
            bbox[X1_EXT_IDX] = page_x1
            # y1坐标保持不变
            bbox[Y1_EXT_IDX] = bbox[Y1_IDX]
            
    # 如果独占一行的bbox已扩展到页面边界，合并连续的bbox形成一个group
    if len(all_bboxes)==1:
        # 返回包含单一独占行的layout信息
        return [{"layout_bbox": [page_x0, page_y0, page_x1, page_y1], "layout_label":"GOOD_LAYOUT", "content_bboxes": all_bboxes}]
    if len(all_bboxes)==0:
        # 如果没有bbox，返回空列表
        return []
    
    """
    然后合并所有连续水平方向的bbox.
    
    """
    # 按照y0坐标排序所有bbox
    all_bboxes.sort(key=lambda x: x[Y0_IDX])
    # 用于存储水平bbox
    h_bboxes = []
    # 当前水平bbox组
    h_bbox_group = []
    # 垂直框（未使用）
    v_boxes = []

    for bbox in all_bboxes:
        # 检查当前bbox是否位于页面的最左侧和最右侧
        if bbox[X0_IDX] == page_x0 and bbox[X1_IDX] == page_x1:
            # 将符合条件的bbox加入当前组
            h_bbox_group.append(bbox)
        else:
            # 如果当前组非空，则将其加入h_bboxes
            if len(h_bbox_group)>0:
                h_bboxes.append(h_bbox_group) 
                h_bbox_group = []
    # 最后一个group
    if len(h_bbox_group)>0:
        # 将最后的组添加到h_bboxes
        h_bboxes.append(h_bbox_group)

    """
    现在h_bboxes里面是所有的group了，每个group都是一个list
    对h_bboxes里的每个group进行计算放回到sorted_layouts里
    """
    for gp in h_bboxes:
        # 对每个组按照y0坐标排序
        gp.sort(key=lambda x: x[Y0_IDX])
        # 创建布局信息
        block_info = {"layout_label":"GOOD_LAYOUT", "content_bboxes": gp}
        # 计算当前组的布局边界，找到最小的x0,y0和最大的x1,y1
        x0, y0, x1, y1 = gp[0][X0_EXT_IDX], gp[0][Y0_EXT_IDX], gp[-1][X1_EXT_IDX], gp[-1][Y1_EXT_IDX]
        # 更新布局信息的边界
        block_info["layout_bbox"] = [x0, y0, x1, y1]
        # 将布局信息添加到结果列表
        sorted_layouts.append(block_info)
        
    # 接下来利用这些连续的水平bbox的layout_bbox的y0, y1，从水平上切分开其余的为几个部分
    # 初始化水平切分线列表
    h_split_lines = [page_y0]
    for gp in h_bboxes:
        # 提取当前组的布局边界
        layout_bbox = gp['layout_bbox']
        y0, y1 = layout_bbox[1], layout_bbox[3]
        # 将y0和y1添加到切分线列表
        h_split_lines.append(y0)
        h_split_lines.append(y1)
    # 添加页面底部边界到切分线
    h_split_lines.append(page_y1)
    
    # 未切分的bbox列表（后续处理）
    unsplited_bboxes = []
    # 遍历 h_split_lines 列表中的每两个元素
    for i in range(0, len(h_split_lines), 2):
        # 将当前两个元素分别赋值给 start_y0 和 start_y1
        start_y0, start_y1 = h_split_lines[i:i+2]
        # 找出在 [start_y0, start_y1] 区间内的所有 bbox，形成一个未分割的板块
        bboxes_in_block = [bbox for bbox in all_bboxes if bbox[Y0_IDX]>=start_y0 and bbox[Y1_IDX]<=start_y1]
        # 将找到的未分割 bbox 列表添加到 unsplited_bboxes 中
        unsplited_bboxes.append(bboxes_in_block)
    # ================== 至此，水平方向的 已经切分排序完毕====================================
    """
    接下来针对每个非水平的部分切分垂直方向的
    此时，只剩下了无法被完全水平打通的 bbox 了。对这些 bbox，优先进行垂直扩展，然后进行垂直切分。
    分 3 步：
    1. 先把能完全垂直打通的隔离出去当做一个 layout
    2. 其余的先垂直切分
    3. 垂直切分之后的部分再尝试水平切分
    4. 剩下的不能被切分的各个部分当成一个 layout
    """
    # 对未分割的每部分进行垂直切分
    for bboxes_in_block in unsplited_bboxes:
        # 计算当前 block 的边界框
        boundry_x0, boundry_y0, boundry_x1, boundry_y1 = find_boundry_bboxes(bboxes_in_block) 
        # 进行垂直方向上的扩展，得到扩展后的 bbox 列表
        extended_vertical_bboxes = extend_bbox_vertical(bboxes_in_block, boundry_x0, boundry_y0, boundry_x1, boundry_y1)
        # 对扩展后的 bbox 进行垂直方向上的切分
        extended_vertical_bboxes.sort(key=lambda x: x[X0_IDX]) # x方向上从小到大，代表了从左到右读取
        v_boxes_group = [] # 用于存储当前分组的 bbox
        for bbox in extended_vertical_bboxes:
            # 如果当前 bbox 的上下边界与边界框一致，则加入当前组
            if bbox[Y0_IDX]==boundry_y0 and bbox[Y1_IDX]==boundry_y1:
                v_boxes_group.append(bbox)
            else:
                # 如果当前组不为空，将其添加到 v_boxes 中，并重置当前组
                if len(v_boxes_group)>0:
                    v_boxes.append(v_boxes_group)
                    v_boxes_group = []
                    
        # 如果当前组仍然有未添加的 bbox，添加到 v_boxes
        if len(v_boxes_group)>0:
            v_boxes.append(v_boxes_group)
            
        # 把连续的垂直部分加入到 sorted_layouts 里
        for gp in v_boxes:
            # 按 x0 排序
            gp.sort(key=lambda x: x[X0_IDX])
            # 创建包含布局信息的字典
            block_info = {"layout_label":"GOOD_LAYOUT", "content_bboxes": gp}
            # 计算当前组的布局 bbox，即最小的 x0, y0 和最大的 x1, y1
            x0, y0, x1, y1 = gp[0][X0_EXT_IDX], gp[0][Y0_EXT_IDX], gp[-1][X1_EXT_IDX], gp[-1][Y1_EXT_IDX]
            block_info["layout_bbox"] = [x0, y0, x1, y1]
            # 将布局信息添加到 sorted_layouts
            sorted_layouts.append(block_info)
            
        # 在垂直方向上，划分子块，用贯通的垂直线进行切分
        v_split_lines = [boundry_x0] # 初始化分割线列表
        for gp in v_boxes:
            # 获取当前组的布局 bbox
            layout_bbox = gp['layout_bbox']
            x0, x1 = layout_bbox[0], layout_bbox[2]
            v_split_lines.append(x0) # 添加左边界
            v_split_lines.append(x1) # 添加右边界
        v_split_lines.append(boundry_x1) # 添加右侧边界
        
    # 重置 bbox 的索引
    reset_idx_x_y(all_bboxes)
    # 对所有 bbox 进行排序并返回结果
    all_boxes = _paper_bbox_sort_ext(all_bboxes)
    return all_boxes
```