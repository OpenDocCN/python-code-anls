# `.\MinerU\magic_pdf\pre_proc\remove_bbox_overlap.py`

```
# 从 magic_pdf.libs.boxbase 导入处理边界框重叠的函数
from magic_pdf.libs.boxbase import _is_in_or_part_overlap, _is_in, _is_part_overlap
# 从 magic_pdf.libs.drop_reason 导入表示丢弃原因的类
from magic_pdf.libs.drop_reason import DropReason

# 定义一个函数，用于移除两个边界框之间的重叠部分
def _remove_overlap_between_bbox(bbox1, bbox2):
   # 检查两个边界框是否部分重叠
   if _is_part_overlap(bbox1, bbox2):
        # 解构边界框 bbox1 的坐标
        ix0, iy0, ix1, iy1 = bbox1
        # 解构边界框 bbox2 的坐标
        x0, y0, x1, y1 = bbox2

        # 计算两个边界框重叠区域的宽度和高度
        diff_x = min(x1, ix1) - max(x0, ix0)
        diff_y = min(y1, iy1) - max(y0, iy0)

        # 根据重叠区域的高度和宽度，决定如何调整边界框
        if diff_y > diff_x:
            # 如果 bbox2 的右边界在 bbox1 的右边界右侧
            if x1 >= ix1:
                mid = (x0 + ix1) // 2  # 计算中间点
                ix1 = min(mid - 0.25, ix1)  # 调整 bbox1 的右边界
                x0 = max(mid + 0.25, x0)  # 调整 bbox2 的左边界
            else:
                mid = (ix0 + x1) // 2  # 计算中间点
                ix0 = max(mid + 0.25, ix0)  # 调整 bbox1 的左边界
                x1 = min(mid - 0.25, x1)  # 调整 bbox2 的右边界
        else:
            # 如果 bbox2 的上边界在 bbox1 的上边界上方
            if y1 >= iy1:
                mid = (y0 + iy1) // 2  # 计算中间点
                y0 = max(mid + 0.25, y0)  # 调整 bbox2 的下边界
                iy1 = min(iy1, mid-0.25)  # 调整 bbox1 的上边界
            else:
                mid = (iy0 + y1) // 2  # 计算中间点
                y1 = min(y1, mid-0.25)  # 调整 bbox2 的上边界
                iy0 = max(mid + 0.25, iy0)  # 调整 bbox1 的下边界

        # 如果调整后的边界框有效，则返回它们
        if ix1 > ix0 and iy1 > iy0 and y1 > y0 and x1 > x0:
            bbox1 = [ix0, iy0, ix1, iy1]  # 更新 bbox1
            bbox2 = [x0, y0, x1, y1]  # 更新 bbox2
            return bbox1, bbox2, None  # 返回调整后的边界框和 None（表示没有丢弃原因）
        else:
            # 如果调整后无效，则返回丢弃原因
            return bbox1, bbox2, DropReason.NEGATIVE_BBOX_AREA
   else:
       # 如果没有重叠，则返回原始边界框和 None
       return bbox1, bbox2, None


# 定义一个函数，移除一组边界框之间的重叠
def _remove_overlap_between_bboxes(arr):
    drop_reasons = []  # 存储丢弃原因的列表
    N = len(arr)  # 获取边界框数量
    keeps = [True] * N  # 初始化保留标志，默认都保留
    res = [None] * N  # 初始化结果列表，默认都为 None
    # 遍历所有边界框，检查是否被包含在其他边界框内
    for i in range(N):
        for j in range(N):
            if i == j:  # 跳过自己
                continue
            if _is_in(arr[i]["bbox"], arr[j]["bbox"]):  # 检查 arr[i] 是否在 arr[j] 内
                keeps[i] = False  # 如果是，则标记为不保留

    # 遍历所有边界框，根据保留标志和重叠情况处理
    for idx, v in enumerate(arr):
        if not keeps[idx]:  # 如果不保留，则跳过
            continue
        for i in range(N):
            if res[i] is None:  # 如果结果列表的当前项为 None，则跳过
                continue
        
            # 移除重叠并获取可能的丢弃原因
            bbox1, bbox2, drop_reason = _remove_overlap_between_bbox(v["bbox"], res[i]["bbox"])
            if drop_reason is None:  # 如果没有丢弃原因
                v["bbox"] = bbox1  # 更新边界框
                res[i]["bbox"] = bbox2  # 更新结果中的边界框
            else:
                # 如果有丢弃原因，根据分数决定保留哪一个
                if v["score"] > res[i]["score"]:
                    keeps[i] = False  # 标记较低分数的框为不保留
                    res[i] = None  # 将结果中的对应框设为 None
                else:
                    keeps[idx] = False  # 否则标记当前框为不保留
                drop_reasons.append(drop_reasons)  # 记录丢弃原因
        if keeps[idx]:  # 如果当前框被保留
            res[idx] = v  # 将其加入结果列表
    return res, drop_reasons  # 返回结果和丢弃原因


# 定义一个函数，用于处理一组 span 的边界框重叠
def remove_overlap_between_bbox_for_span(spans):
    # 创建一个包含边界框和分数的数组
    arr = [{"bbox": span["bbox"], "score": span.get("score", 0.1)} for span in spans ]
    res, drop_reasons = _remove_overlap_between_bboxes(arr)  # 调用移除重叠的函数
    ret = []  # 初始化结果列表
    # 遍历结果，构建最终返回的列表
    for i in range(len(res)):
        if res[i] is None:  # 如果结果为 None，跳过
            continue
        spans[i]["bbox"] = res[i]["bbox"]  # 更新原始 spans 的边界框
        ret.append(spans[i])  # 添加到结果列表
    return ret, drop_reasons  # 返回结果和丢弃原因


# 定义一个函数，用于处理一组边界框的重叠
def remove_overlap_between_bbox_for_block(all_bboxes):
    # 创建一个包含边界框和分数的数组
    arr = [{"bbox": bbox[:4], "score": bbox[-1]} for bbox in all_bboxes ]
    res, drop_reasons = _remove_overlap_between_bboxes(arr)  # 调用移除重叠的函数
    ret = []  # 初始化结果列表
    # 遍历结果列表的索引
        for i in range(len(res)):
            # 如果当前结果为空，则跳过该次循环
            if res[i] is None:
                continue
            # 将结果中当前索引的边界框信息更新到 all_bboxes 的对应位置
            all_bboxes[i][:4] = res[i]["bbox"]
            # 将更新后的边界框添加到返回列表中
            ret.append(all_bboxes[i])
        # 返回边界框列表和丢弃原因
        return ret, drop_reasons
```