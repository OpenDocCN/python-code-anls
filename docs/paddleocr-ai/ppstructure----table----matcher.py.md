# `.\PaddleOCR\ppstructure\table\matcher.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

import numpy as np
from ppstructure.table.table_master_match import deal_eb_token, deal_bb

# 计算两个矩形框之间的距离
def distance(box_1, box_2):
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2
    dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4 - x2) + abs(y4 - y2)
    dis_2 = abs(x3 - x1) + abs(y3 - y1)
    dis_3 = abs(x4 - x2) + abs(y4 - y2)
    return dis + min(dis_2, dis_3)

# 计算两个矩形框之间的 IoU（Intersection over Union）
def compute_iou(rec1, rec2):
    """
    计算 IoU
    :param rec1: (y0, x0, y1, x1)，表示
            (顶部，左侧，底部，右侧)
    :param rec2: (y0, x0, y1, x1)
    :return: IoU 的标量值
    """
    # 计算每个矩形的面积
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # 计算两个矩形的总面积
    sum_area = S_rec1 + S_rec2

    # 找到相交矩形的每条边
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # 判断是否有相交
    if left_line >= right_line or top_line >= bottom_line:
        return 0.0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

# 表格匹配类
class TableMatch:
    # 初始化函数，设置是否过滤 OCR 结果和是否使用主模型
    def __init__(self, filter_ocr_result=False, use_master=False):
        # 设置是否过滤 OCR 结果
        self.filter_ocr_result = filter_ocr_result
        # 设置是否使用主模型
        self.use_master = use_master

    # 调用函数，处理结构化结果、检测框和识别结果
    def __call__(self, structure_res, dt_boxes, rec_res):
        # 获取结构化结果和预测框
        pred_structures, pred_bboxes = structure_res
        # 如果需要过滤 OCR 结果
        if self.filter_ocr_result:
            # 过滤 OCR 结果
            dt_boxes, rec_res = self._filter_ocr_result(pred_bboxes, dt_boxes,
                                                        rec_res)
        # 匹配结果框和预测框
        matched_index = self.match_result(dt_boxes, pred_bboxes)
        # 如果使用主模型
        if self.use_master:
            # 获取主模型的预测 HTML 和预测结果
            pred_html, pred = self.get_pred_html_master(pred_structures,
                                                        matched_index, rec_res)
        else:
            # 获取预测 HTML 和预测结果
            pred_html, pred = self.get_pred_html(pred_structures, matched_index,
                                                 rec_res)
        # 返回预测 HTML
        return pred_html

    # 匹配结果框和预测框
    def match_result(self, dt_boxes, pred_bboxes):
        # 初始化匹配字典
        matched = {}
        # 遍历检测框
        for i, gt_box in enumerate(dt_boxes):
            distances = []
            # 遍历预测框
            for j, pred_box in enumerate(pred_bboxes):
                # 如果预测框为 8 个点的形式，转换为左上角和右下角坐标形式
                if len(pred_box) == 8:
                    pred_box = [
                        np.min(pred_box[0::2]), np.min(pred_box[1::2]),
                        np.max(pred_box[0::2]), np.max(pred_box[1::2])
                    ]
                # 计算距离和 IOU
                distances.append((distance(gt_box, pred_box),
                                  1. - compute_iou(gt_box, pred_box)
                                  ))  # compute iou and l1 distance
            sorted_distances = distances.copy()
            # 根据 IOU 和距离选择检测框
            sorted_distances = sorted(
                sorted_distances, key=lambda item: (item[1], item[0]))
            # 如果最匹配的预测框索引不在匹配字典中，添加到匹配字典
            if distances.index(sorted_distances[0]) not in matched.keys():
                matched[distances.index(sorted_distances[0])] = [i]
            else:
                matched[distances.index(sorted_distances[0])].append(i)
        # 返回匹配结果
        return matched
    # 获取预测的 HTML 结构
    def get_pred_html(self, pred_structures, matched_index, ocr_contents):
        # 存储最终的 HTML 结果
        end_html = []
        # 初始化 td 索引
        td_index = 0
        # 遍历预测的结构标签
        for tag in pred_structures:
            # 如果当前标签是 '</td>'
            if '</td>' in tag:
                # 如果当前标签是 '<td></td>'
                if '<td></td>' == tag:
                    end_html.extend('<td>')
                # 如果当前 td 索引在匹配索引中
                if td_index in matched_index.keys():
                    # 初始化是否包含 <b> 标签的标志
                    b_with = False
                    # 如果 OCR 内容中包含 '<b>'，并且匹配索引中有多个元素
                    if '<b>' in ocr_contents[matched_index[td_index][0]] and len(matched_index[td_index]) > 1:
                        b_with = True
                        end_html.extend('<b>')
                    # 遍历匹配索引中的内容
                    for i, td_index_index in enumerate(matched_index[td_index]):
                        # 获取 OCR 内容
                        content = ocr_contents[td_index_index][0]
                        # 如果匹配索引中有多个元素
                        if len(matched_index[td_index]) > 1:
                            # 如果内容为空，则跳过
                            if len(content) == 0:
                                continue
                            # 如果内容以空格开头，则去除空格
                            if content[0] == ' ':
                                content = content[1:]
                            # 如果内容包含 '<b>'，则去除前三个字符
                            if '<b>' in content:
                                content = content[3:]
                            # 如果内容包含 '</b>'，则去除后四个字符
                            if '</b>' in content:
                                content = content[:-4]
                            # 如果内容为空，则跳过
                            if len(content) == 0:
                                continue
                            # 如果不是最后一个元素，并且内容最后一个字符不是空格，则添加空格
                            if i != len(matched_index[td_index]) - 1 and ' ' != content[-1]:
                                content += ' '
                        # 添加内容到最终 HTML 结果
                        end_html.extend(content)
                    # 如果包含 <b> 标签，则添加 '</b>'
                    if b_with:
                        end_html.extend('</b>')
                # 如果当前标签是 '<td></td>'
                if '<td></td>' == tag:
                    end_html.append('</td>')
                else:
                    end_html.append(tag)
                # 更新 td 索引
                td_index += 1
            else:
                end_html.append(tag)
        # 返回最终的 HTML 结果
        return ''.join(end_html), end_html
    # 获取预测结构、匹配索引和 OCR 内容，返回 HTML 结果和处理后的 HTML 结构列表
    def get_pred_html_master(self, pred_structures, matched_index, ocr_contents):
        # 初始化空的 HTML 结构列表
        end_html = []
        # 初始化 td 索引
        td_index = 0
        # 遍历预测结构中的每个 token
        for token in pred_structures:
            # 如果 token 中包含 '</td>'
            if '</td>' in token:
                # 初始化文本内容为空
                txt = ''
                # 初始化是否包含 <b> 标签为 False
                b_with = False
                # 如果当前 td 索引在匹配索引中
                if td_index in matched_index.keys():
                    # 如果 OCR 内容中匹配索引对应的内容包含 '<b>' 并且匹配索引长度大于 1
                    if '<b>' in ocr_contents[matched_index[td_index][0]] and len(matched_index[td_index]) > 1:
                        b_with = True
                    # 遍历匹配索引对应的内容
                    for i, td_index_index in enumerate(matched_index[td_index]):
                        # 获取 OCR 内容
                        content = ocr_contents[td_index_index][0]
                        # 如果匹配索引长度大于 1
                        if len(matched_index[td_index]) > 1:
                            # 如果内容为空，则跳过
                            if len(content) == 0:
                                continue
                            # 如果内容以空格开头，则去除空格
                            if content[0] == ' ':
                                content = content[1:]
                            # 如果内容包含 '<b>'，则去除前三个字符
                            if '<b>' in content:
                                content = content[3:]
                            # 如果内容包含 '</b>'，则去除后四个字符
                            if '</b>' in content:
                                content = content[:-4]
                            # 如果内容为空，则跳过
                            if len(content) == 0:
                                continue
                            # 如果不是最后一个内容并且内容最后一个字符不是空格，则添加空格
                            if i != len(matched_index[td_index]) - 1 and ' ' != content[-1]:
                                content += ' '
                        # 拼接文本内容
                        txt += content
                # 如果包含 <b> 标签
                if b_with:
                    txt = '<b>{}</b>'.format(txt)
                # 如果 token 为 '<td></td>'
                if '<td></td>' == token:
                    token = '<td>{}</td>'.format(txt)
                else:
                    token = '{}</td>'.format(txt)
                # td 索引加一
                td_index += 1
            # 处理特殊 token
            token = deal_eb_token(token)
            # 将处理后的 token 添加到 HTML 结构列表中
            end_html.append(token)
        # 将 HTML 结构列表转换为字符串
        html = ''.join(end_html)
        # 处理 HTML 结构中的特殊标签
        html = deal_bb(html)
        # 返回处理后的 HTML 和 HTML 结构列表
        return html, end_html
    # 过滤OCR结果，根据预测边界框的最小y坐标确定过滤条件
    y1 = pred_bboxes[:, 1::2].min()
    # 初始化新的边界框列表和识别结果列表
    new_dt_boxes = []
    new_rec_res = []

    # 遍历检测到的边界框和识别结果，根据条件进行过滤
    for box, rec in zip(dt_boxes, rec_res):
        # 如果边界框的最大y坐标小于y1，则跳过该边界框
        if np.max(box[1::2]) < y1:
            continue
        # 将符合条件的边界框和识别结果添加到新的列表中
        new_dt_boxes.append(box)
        new_rec_res.append(rec)
    
    # 返回过滤后的边界框列表和识别结果列表
    return new_dt_boxes, new_rec_res
```