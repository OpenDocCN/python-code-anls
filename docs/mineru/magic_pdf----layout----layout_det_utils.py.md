# `.\MinerU\magic_pdf\layout\layout_det_utils.py`

```
# 从magic_pdf.layout.bbox_sort导入边界框索引常量
from magic_pdf.layout.bbox_sort import X0_EXT_IDX, X0_IDX, X1_EXT_IDX, X1_IDX, Y0_IDX, Y1_EXT_IDX, Y1_IDX
# 从magic_pdf.libs.boxbase导入边界框重叠检测函数
from magic_pdf.libs.boxbase import _is_bottom_full_overlap, _left_intersect, _right_intersect


# 查找与给定边界框this_bbox在垂直方向上重叠的所有左侧边界框
def find_all_left_bbox_direct(this_bbox, all_bboxes) -> list:
    """
    在all_bboxes里找到所有右侧垂直方向上和this_bbox有重叠的bbox， 不用延长线
    并且要考虑两个box左右相交的情况，如果相交了，那么右侧的box就不算最左侧。
    """
    # 遍历所有边界框，筛选出与this_bbox重叠且在其左侧的边界框
    left_boxes = [box for box in all_bboxes if box[X1_IDX] <= this_bbox[X0_IDX] 
         and any([
         # 检查边界框在y轴上的重叠情况
         box[Y0_IDX] < this_bbox[Y0_IDX] < box[Y1_IDX], box[Y0_IDX] < this_bbox[Y1_IDX] < box[Y1_IDX],
         this_bbox[Y0_IDX] < box[Y0_IDX] < this_bbox[Y1_IDX], this_bbox[Y0_IDX] < box[Y1_IDX] < this_bbox[Y1_IDX],
         # 检查两个边界框是否完全重合
         box[Y0_IDX]==this_bbox[Y0_IDX] and box[Y1_IDX]==this_bbox[Y1_IDX]]) or _left_intersect(box[:4], this_bbox[:4])]
        
    # 如果找到重叠的左侧边界框，则按照x1坐标排序，取距离this_bbox最近的一个
    if len(left_boxes) > 0:
        left_boxes.sort(key=lambda x: x[X1_EXT_IDX] if x[X1_EXT_IDX] else x[X1_IDX], reverse=True)
        left_boxes = left_boxes[0]
    else:
        # 如果没有找到，则设置为None
        left_boxes = None
    # 返回找到的左侧边界框
    return left_boxes

# 查找与给定边界框this_bbox在右侧且最近的边界框
def find_all_right_bbox_direct(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox右侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    # 遍历所有边界框，筛选出与this_bbox右侧且在y轴上有重叠的边界框
    right_bboxes = [box for box in all_bboxes if box[X0_IDX] >= this_bbox[X1_IDX] 
        and any([
        # 检查边界框在y轴上的重叠情况
        this_bbox[Y0_IDX] < box[Y0_IDX] < this_bbox[Y1_IDX], this_bbox[Y0_IDX] < box[Y1_IDX] < this_bbox[Y1_IDX],
        box[Y0_IDX] < this_bbox[Y0_IDX] < box[Y1_IDX], box[Y0_IDX] < this_bbox[Y1_IDX] < box[Y1_IDX],
        # 检查两个边界框是否完全重合
        box[Y0_IDX]==this_bbox[Y0_IDX] and box[Y1_IDX]==this_bbox[Y1_IDX]]) or _right_intersect(this_bbox[:4], box[:4])]
    
    # 如果找到重叠的右侧边界框，则按照x0坐标排序，取距离this_bbox最近的一个
    if len(right_bboxes)>0:
        right_bboxes.sort(key=lambda x: x[X0_EXT_IDX] if x[X0_EXT_IDX] else x[X0_IDX])
        right_bboxes = right_bboxes[0]
    else:
        # 如果没有找到，则设置为None
        right_bboxes = None
    # 返回找到的右侧边界框
    return right_bboxes

# 查找与给定边界框this_bbox在上侧且最近的边界框
def find_all_top_bbox_direct(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox上侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    # 遍历所有边界框，筛选出与this_bbox上侧且在x轴上有重叠的边界框
    top_bboxes = [box for box in all_bboxes if box[Y1_IDX] <= this_bbox[Y0_IDX] and any([
        # 检查边界框在x轴上的重叠情况
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
        this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        # 检查两个边界框是否完全重合
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    
    # 如果找到重叠的上侧边界框，则按照y1坐标排序，取距离this_bbox最近的一个
    if len(top_bboxes)>0:
        top_bboxes.sort(key=lambda x: x[Y1_EXT_IDX] if x[Y1_EXT_IDX] else x[Y1_IDX], reverse=True)
        top_bboxes = top_bboxes[0]
    else:
        # 如果没有找到，则设置为None
        top_bboxes = None
    # 返回找到的上侧边界框
    return top_bboxes

# 查找与给定边界框this_bbox在下侧且最近的边界框
def find_all_bottom_bbox_direct(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox下侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    # 筛选出所有在当前边界框下方的边界框
        bottom_bboxes = [box for box in all_bboxes if box[Y0_IDX] >= this_bbox[Y1_IDX] and any([
            # 检查当前边界框左边界与其他框的关系
            this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], 
            # 检查当前边界框右边界与其他框的关系
            this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
            # 检查其他框左边界与当前边界框的关系
            box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], 
            # 检查其他框右边界与当前边界框的关系
            box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
            # 检查框是否完全重合
            box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
        
        # 如果存在下方框，进行排序
        if len(bottom_bboxes)>0:
            # 按照 Y0_IDX 进行排序
            bottom_bboxes.sort(key=lambda x:  x[Y0_IDX])
            # 取出第一个边界框
            bottom_bboxes = bottom_bboxes[0]
        else:
            # 如果没有下方框，则返回 None
            bottom_bboxes = None
        # 返回下方框
        return bottom_bboxes
# ===================================================================================================================
# 定义一个函数，找到在this_bbox下侧且最近的直接遮挡的bbox
def find_bottom_bbox_direct_from_right_edge(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox下侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    # 筛选出在this_bbox下方且可能遮挡的所有bbox
    bottom_bboxes = [box for box in all_bboxes if box[Y0_IDX] >= this_bbox[Y1_IDX] and any([
        this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    
    if len(bottom_bboxes)>0:
        # 按照y0升序排序，找到最靠近this_bbox的bbox
        bottom_bboxes.sort(key=lambda x: x[Y0_IDX])
        # 过滤出y0最小的bbox
        bottom_bboxes = [box for box in bottom_bboxes if box[Y0_IDX]==bottom_bboxes[0][Y0_IDX]]
        # 在y1相同的情况下，找到x1最大的bbox
        bottom_bboxes.sort(key=lambda x: x[X1_IDX], reverse=True)
        bottom_bboxes = bottom_bboxes[0]
    else:
        # 如果没有找到符合条件的bbox，返回None
        bottom_bboxes = None
    # 返回找到的bbox或None
    return bottom_bboxes

# 定义一个函数，找到在this_bbox下侧且最近的直接遮挡的bbox
def find_bottom_bbox_direct_from_left_edge(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox下侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    # 筛选出在this_bbox下方且可能遮挡的所有bbox
    bottom_bboxes = [box for box in all_bboxes if box[Y0_IDX] >= this_bbox[Y1_IDX] and any([
        this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    
    if len(bottom_bboxes)>0:
        # 按照y0升序排序，找到最靠近this_bbox的bbox
        bottom_bboxes.sort(key=lambda x: x[Y0_IDX])
        # 过滤出y0最小的bbox
        bottom_bboxes = [box for box in bottom_bboxes if box[Y0_IDX]==bottom_bboxes[0][Y0_IDX]]
        # 在y0相同的情况下，找到x0最小的bbox
        bottom_bboxes.sort(key=lambda x: x[X0_IDX])
        bottom_bboxes = bottom_bboxes[0]
    else:
        # 如果没有找到符合条件的bbox，返回None
        bottom_bboxes = None
    # 返回找到的bbox或None
    return bottom_bboxes

# 定义一个函数，找到在this_bbox上侧且最近的直接遮挡的bbox
def find_top_bbox_direct_from_left_edge(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox上侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    # 筛选出在this_bbox上方且可能遮挡的所有bbox
    top_bboxes = [box for box in all_bboxes if box[Y1_IDX] <= this_bbox[Y0_IDX] and any([
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
        this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    
    if len(top_bboxes)>0:
        # 按照y1降序排序，找到最靠近this_bbox的bbox
        top_bboxes.sort(key=lambda x: x[Y1_IDX], reverse=True)
        # 过滤出y1最大的bbox
        top_bboxes = [box for box in top_bboxes if box[Y1_IDX]==top_bboxes[0][Y1_IDX]]
        # 在y1相同的情况下，找到x0最小的bbox
        top_bboxes.sort(key=lambda x: x[X0_IDX])
        top_bboxes = top_bboxes[0]
    else:
        # 如果没有找到符合条件的bbox，返回None
        top_bboxes = None
    # 返回找到的bbox或None
    return top_bboxes

# 定义一个函数，找到在this_bbox上侧且最近的直接遮挡的bbox
def find_top_bbox_direct_from_right_edge(this_bbox, all_bboxes) -> list:
    #
    # 找到在this_bbox上侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    top_bboxes = [box for box in all_bboxes if box[Y1_IDX] <= this_bbox[Y0_IDX] and any([
        # 检查box的下边界是否在this_bbox的上边界之上，并且box与this_bbox有重叠
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], 
        box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
        this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], 
        this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[Y1_IDX],
        # 检查box与this_bbox完全重叠的情况
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    
    # 如果找到了符合条件的bbox
    if len(top_bboxes)>0:
        # 根据Y1_IDX降序排列，获取上边界最大值的bbox
        top_bboxes.sort(key=lambda x: x[Y1_IDX], reverse=True)
        # 保留y1最大值相同的bbox
        top_bboxes = [box for box in top_bboxes if box[Y1_IDX]==top_bboxes[0][Y1_IDX]]
        # 再根据X1_IDX降序排列，获取x1最大值的bbox
        top_bboxes.sort(key=lambda x: x[X1_IDX], reverse=True)
        # 取出第一个bbox作为结果
        top_bboxes = top_bboxes[0]
    else:
        # 如果没有找到符合条件的bbox，返回None
        top_bboxes = None
    # 返回找到的bbox或None
    return top_bboxes
# ===================================================================================================================

# 定义函数，获取最左边的bbox
def get_left_edge_bboxes(all_bboxes) -> list:
    """
    返回最左边的bbox
    """
    # 过滤出所有没有左边相邻bbox的bbox
    left_bboxes = [box for box in all_bboxes if find_all_left_bbox_direct(box, all_bboxes) is None]
    # 返回最左边的bbox列表
    return left_bboxes
    
# 定义函数，获取最右边的bbox
def get_right_edge_bboxes(all_bboxes) -> list:
    """
    返回最右边的bbox
    """
    # 过滤出所有没有右边相邻bbox的bbox
    right_bboxes = [box for box in all_bboxes if find_all_right_bbox_direct(box, all_bboxes) is None]
    # 返回最右边的bbox列表
    return right_bboxes

# 定义函数，修正垂直方向上的bbox位置
def fix_vertical_bbox_pos(bboxes:list):
    """
    检查这批bbox在垂直方向是否有轻微的重叠，如果重叠了，就把重叠的bbox往下移动一点
    在x方向上必须一个包含或者被包含，或者完全重叠，不能只有部分重叠
    """
    # 根据bbox的上边界进行排序
    bboxes.sort(key=lambda x: x[Y0_IDX]) # 从上向下排列
    # 遍历每个bbox
    for i in range(0, len(bboxes)):
        # 遍历当前bbox之后的所有bbox
        for j in range(i+1, len(bboxes)):
            # 检查两个bbox是否有完全重叠的情况
            if _is_bottom_full_overlap(bboxes[i][:4], bboxes[j][:4]):
                # 如果有部分重叠，将下方的bbox向下移动一点
                bboxes[j][Y0_IDX] = bboxes[i][Y1_IDX] + 2 # 2是个经验值
                break
    # 返回修正后的bbox列表
    return bboxes
```