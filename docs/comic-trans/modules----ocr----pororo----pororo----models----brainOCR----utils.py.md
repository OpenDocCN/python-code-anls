# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\utils.py`

```py
"""
This code is adapted from https://github.com/JaidedAI/EasyOCR/blob/8af936ba1b2f3c230968dc1022d0cd3e9ca1efbb/easyocr/utils.py
"""

# 导入必要的库和模块
import math  # 导入数学函数库
import os  # 导入操作系统接口模块
from urllib.request import urlretrieve  # 从 urllib 库中导入 urlretrieve 函数

import cv2  # 导入 OpenCV 库
import numpy as np  # 导入 NumPy 库
import torch  # 导入 PyTorch 深度学习库
from PIL import Image  # 从 PIL 库中导入 Image 模块
from torch import Tensor  # 从 PyTorch 库中导入 Tensor 类

from .imgproc import load_image  # 从当前包中导入 imgproc 模块中的 load_image 函数


def consecutive(data, mode: str = "first", stepsize: int = 1):
    # 将数据按照连续性分组
    group = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    group = [item for item in group if len(item) > 0]

    if mode == "first":
        result = [l[0] for l in group]  # 返回每组的第一个元素
    elif mode == "last":
        result = [l[-1] for l in group]  # 返回每组的最后一个元素
    return result


def word_segmentation(
    mat,
    separator_idx={
        "th": [1, 2],
        "en": [3, 4]
    },
    separator_idx_list=[1, 2, 3, 4],
):
    result = []  # 初始化结果列表
    sep_list = []  # 初始化分隔符列表
    start_idx = 0  # 初始化起始索引
    sep_lang = ""  # 初始化分隔符语言

    # 遍历分隔符索引列表
    for sep_idx in separator_idx_list:
        if sep_idx % 2 == 0:
            mode = "first"
        else:
            mode = "last"
        # 找出 mat 中值为 sep_idx 的连续块
        a = consecutive(np.argwhere(mat == sep_idx).flatten(), mode)
        new_sep = [[item, sep_idx] for item in a]
        sep_list += new_sep
    sep_list = sorted(sep_list, key=lambda x: x[0])  # 按照位置排序

    # 根据分隔符列表分割 mat
    for sep in sep_list:
        for lang in separator_idx.keys():
            if sep[1] == separator_idx[lang][0]:  # 如果是起始语言分隔符
                sep_lang = lang
                sep_start_idx = sep[0]
            elif sep[1] == separator_idx[lang][1]:  # 如果是结束语言分隔符
                if sep_lang == lang:  # 检查是否与之前的起始语言匹配
                    new_sep_pair = [lang, [sep_start_idx + 1, sep[0] - 1]]
                    if sep_start_idx > start_idx:
                        result.append(["", [start_idx, sep_start_idx - 1]])
                    start_idx = sep[0] + 1
                    result.append(new_sep_pair)
                sep_lang = ""  # 重置分隔符语言

    # 将剩余部分添加到结果中
    if start_idx <= len(mat) - 1:
        result.append(["", [start_idx, len(mat) - 1]])
    return result


# code is based from https://github.com/githubharald/CTCDecoder/blob/master/src/BeamSearch.py
class BeamEntry:
    "information about one single beam at specific time-step"

    def __init__(self):
        self.prTotal = 0  # 总概率（包括空白和非空白）
        self.prNonBlank = 0  # 非空白概率
        self.prBlank = 0  # 空白概率
        self.prText = 1  # 语言模型分数
        self.lmApplied = False  # 是否已应用语言模型标志
        self.labeling = ()  # beam 标签


class BeamState:
    "information about the beams at specific time-step"

    def __init__(self):
        self.entries = {}  # 初始化 beam 列表

    def norm(self):
        "length-normalise LM score"
        # 对所有 beam 的语言模型分数进行长度归一化
        for (k, _) in self.entries.items():
            labelingLen = len(self.entries[k].labeling)
            self.entries[k].prText = self.entries[k].prText**(
                1.0 / (labelingLen if labelingLen else 1.0))
    def sort(self):
        "return beam-labelings, sorted by probability"
        # 从 self.entries 中提取所有的 beam 对象
        beams = [v for (_, v) in self.entries.items()]
        # 根据每个 beam 的 prTotal * prText 的乘积进行排序，降序排列
        sortedBeams = sorted(
            beams,
            reverse=True,
            key=lambda x: x.prTotal * x.prText,
        )
        # 返回排序后的每个 beam 对象的 labeling 列表
        return [x.labeling for x in sortedBeams]

    def wordsearch(self, classes, ignore_idx, maxCandidate, dict_list):
        # 从 self.entries 中提取所有的 beam 对象
        beams = [v for (_, v) in self.entries.items()]
        # 根据每个 beam 的 prTotal * prText 的乘积进行排序，降序排列
        sortedBeams = sorted(
            beams,
            reverse=True,
            key=lambda x: x.prTotal * x.prText,
        )
        # 如果排序后的 beam 数量超过 maxCandidate，则仅保留前 maxCandidate 个
        if len(sortedBeams) > maxCandidate:
            sortedBeams = sortedBeams[:maxCandidate]

        # 遍历排序后的 beam 对象列表
        for j, candidate in enumerate(sortedBeams):
            idx_list = candidate.labeling
            text = ""
            # 根据 labeling 中的索引生成对应的文本
            for i, l in enumerate(idx_list):
                # 如果索引不在 ignore_idx 中，并且不是连续相同的索引，则将对应的类别加入文本中
                if l not in ignore_idx and (
                        not (i > 0 and idx_list[i - 1] == idx_list[i])):
                    text += classes[l]

            # 如果是第一个候选项，则将其设为最佳文本
            if j == 0:
                best_text = text
            # 如果生成的文本在 dict_list 中，则将其设为最佳文本，并结束搜索
            if text in dict_list:
                # print('found text: ', text)
                best_text = text
                break
            else:
                pass
                # print('not in dict: ', text)
        # 返回找到的最佳文本
        return best_text
# 将语句 labeling 转换为 numpy 数组，方便后续操作
labeling = np.array(labeling)

# 压缩连续的空白标签（blankIdx），只保留第一个
idx = np.where(~((np.roll(labeling, 1) == labeling) & (labeling == blankIdx)))[0]
labeling = labeling[idx]

# 去除不同字符之间的空白标签（blankIdx）
idx = np.where(~((np.roll(labeling, 1) != np.roll(labeling, -1)) & (labeling == blankIdx)))[0]

# 如果 labeling 非空，则确保最后一个标签也被保留在 idx 中
if len(labeling) > 0:
    last_idx = len(labeling) - 1
    if last_idx not in idx:
        idx = np.append(idx, [last_idx])

# 使用 idx 索引更新 labeling，转换为元组返回
labeling = labeling[idx]

return tuple(labeling)
    # 遍历时间步长范围内的每个时间步 t
    for t in range(maxT):
        # 创建一个新的 BeamState 对象作为当前时间步的状态容器
        curr = BeamState()
        
        # 获取上一时间步中得分最高的 beam-labelings（束搜索结果）并按得分排序，选取前 beam_width 个
        bestLabelings = last.sort()[0:beam_width]
        
        # 遍历得分最高的 beam-labelings
        for labeling in bestLabelings:
            # 计算以非空结束的路径的概率
            prNonBlank = 0
            
            # 如果 labeling 不为空，则计算以最后一个字符为结尾的路径重复概率
            if labeling:
                prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]
            
            # 计算以空白符结尾的路径的概率
            prBlank = last.entries[labeling].prTotal * mat[t, blankIdx]
            
            # 简化当前的 labeling（去除连续的重复字符）
            labeling = simplify_label(labeling, blankIdx)
            
            # 将当前 labeling 添加到当前时间步的 beam 中
            addBeam(curr, labeling)
            
            # 填充数据到当前 labeling 的条目中
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].prNonBlank += prNonBlank
            curr.entries[labeling].prBlank += prBlank
            curr.entries[labeling].prTotal += prBlank + prNonBlank
            curr.entries[labeling].prText = last.entries[labeling].prText
            
            # 标记当前的 beam-labeling 已经应用了语言模型，这个信息在前一个时间步已经应用过
            curr.entries[labeling].lmApplied = True
            
            # 扩展当前的 beam-labeling
            # 找出当前时间步中概率高的字符索引
            char_highscore = np.where(mat[t, :] >= 0.5 / maxC)[0]
            
            # 遍历高概率字符索引
            for c in char_highscore:
                # 添加新字符 c 到当前的 beam-labeling 中
                newLabeling = labeling + (c,)
                newLabeling = simplify_label(newLabeling, blankIdx)
                
                # 如果新的 labeling 包含末尾重复的字符，则只考虑以空白符结尾的路径
                if labeling and labeling[-1] == c:
                    prNonBlank = mat[t, c] * last.entries[labeling].prBlank
                else:
                    prNonBlank = mat[t, c] * last.entries[labeling].prTotal
                
                # 将新的 labeling 添加到当前时间步的 beam 中
                addBeam(curr, newLabeling)
                
                # 填充数据到新的 labeling 的条目中
                curr.entries[newLabeling].labeling = newLabeling
                curr.entries[newLabeling].prNonBlank += prNonBlank
                curr.entries[newLabeling].prTotal += prNonBlank
                
                # 应用语言模型到新的 labeling
                applyLM(
                    curr.entries[labeling],
                    curr.entries[newLabeling],
                    classes,
                    lm_model,
                    lm_factor,
                )
        
        # 将当前时间步的 beam 状态更新为最新的 curr
        last = curr
    # 根据 beam-labeling-length 规范化 LM 分数
    last.norm()

    # 获取最可能的标签序列
    bestLabeling = last.sort()[0]

    # 初始化结果字符串
    res = ""

    # 遍历最佳标签序列
    for i, l in enumerate(bestLabeling):
        # 移除重复字符和空白
        if l != ignore_idx and (not (i > 0 and bestLabeling[i - 1] == bestLabeling[i])):
            res += classes[l]

    # 返回处理后的结果字符串
    return res
class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, vocab: list):
        # 创建字符到索引的映射字典和索引到字符的映射字典
        self.char2idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(vocab)}
        # 忽略索引为0的字符
        self.ignored_index = 0
        self.vocab = vocab

    def encode(self, texts: list):
        """
        Convert input texts into indices
        texts (list): text labels of each image. [batch_size]

        Returns
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        # 计算每个文本的长度
        lengths = [len(text) for text in texts]
        # 将所有文本拼接成一个字符串
        concatenated_text = "".join(texts)
        # 将字符转换成对应的索引
        indices = [self.char2idx[char] for char in concatenated_text]

        return torch.IntTensor(indices), torch.IntTensor(lengths)

    def decode_greedy(self, indices: Tensor, lengths: Tensor):
        """convert text-index into text-label.

        :param indices (1D int32 Tensor): [N*length,]
        :param lengths (1D int32 Tensor): [N,]
        :return:
        """
        texts = []
        index = 0
        for length in lengths:
            # 根据长度从索引中提取对应的文本部分
            text = indices[index:index + length]

            chars = []
            for i in range(length):
                # 跳过被忽略的索引和重复的字符
                if (text[i] != self.ignored_index) and (
                        not (i > 0 and text[i - 1] == text[i])
                ):  # removing repeated characters and blank (and separator).
                    chars.append(self.idx2char[text[i].item()])
            texts.append("".join(chars))
            index += length
        return texts

    def decode_beamsearch(self, mat, lm_model, lm_factor, beam_width: int = 5):
        # 使用集束搜索解码CTC输出
        texts = []
        for i in range(mat.shape[0]):
            text = ctcBeamSearch(
                mat[i],
                self.vocab,
                self.ignored_index,
                lm_model,
                lm_factor,
                beam_width,
            )
            texts.append(text)
        return texts


def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图像的高度，即右上角和右下角y坐标的最大距离或左上角和左下角y坐标的最大距离
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],
         [0, maxHeight - 1]],
        dtype="float32",
    )

    # 计算透视变换矩阵并应用它
    M = cv2.getPerspectiveTransform(rect, dst)
    # 使用cv2.warpPerspective函数对输入的图像image进行透视变换，变换矩阵为M，目标图像大小为(maxWidth, maxHeight)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # 返回经过透视变换后的图像warped
    return warped
def group_text_box(
    polys,
    slope_ths: float = 0.1,
    ycenter_ths: float = 0.5,
    height_ths: float = 0.5,
    width_ths: float = 1.0,
    add_margin: float = 0.05,
):
    # poly top-left, top-right, low-right, low-left
    # 初始化四个空列表用于存储不同类型的多边形框
    horizontal_list, free_list, combined_list, merged_list = [], [], [], []

    # 遍历输入的多边形列表
    for poly in polys:
        # 计算多边形的上下两条边的斜率
        slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
        slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))

        # 判断多边形是否为水平方向的框
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            # 计算多边形的边界坐标
            x_max = max([poly[0], poly[2], poly[4], poly[6]])
            x_min = min([poly[0], poly[2], poly[4], poly[6]])
            y_max = max([poly[1], poly[3], poly[5], poly[7]])
            y_min = min([poly[1], poly[3], poly[5], poly[7]])
            # 将水平方向的框加入到列表中，包括左右边界、上下边界、中心位置及高度
            horizontal_list.append([
                x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min
            ])
        else:
            # 计算多边形的高度并增加边界
            height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
            margin = int(1.44 * add_margin * height)

            # 计算多边形各个顶点的扩展位置
            theta13 = abs(
                np.arctan(
                    (poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4]))))
            theta24 = abs(
                np.arctan(
                    (poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6]))))
            x1 = poly[0] - np.cos(theta13) * margin
            y1 = poly[1] - np.sin(theta13) * margin
            x2 = poly[2] + np.cos(theta24) * margin
            y2 = poly[3] - np.sin(theta24) * margin
            x3 = poly[4] + np.cos(theta13) * margin
            y3 = poly[5] + np.sin(theta13) * margin
            x4 = poly[6] - np.cos(theta24) * margin
            y4 = poly[7] + np.sin(theta24) * margin

            # 将扩展后的多边形顶点坐标加入自由列表
            free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # 对水平方向的框按中心位置排序
    horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # 合并框
    new_box = []
    for poly in horizontal_list:
        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            # 判断是否与已有框可合并
            if (abs(np.mean(b_height) - poly[5]) < height_ths *
                    np.mean(b_height)) and (abs(np.mean(b_ycenter) - poly[4]) <
                                            ycenter_ths * np.mean(b_height)):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    # 再次排序合并的框列表
    for boxes in combined_list:
        if len(boxes) == 1:  # 如果每行只有一个框
            box = boxes[0]
            margin = int(add_margin * box[5])  # 计算边距
            merged_list.append([
                box[0] - margin, box[1] + margin, box[2] - margin,
                box[3] + margin
            ])
        else:  # 如果每行有多个框
            boxes = sorted(boxes, key=lambda item: item[0])  # 根据框的左上角 x 坐标排序

            merged_box, new_box = [], []
            for box in boxes:
                if len(new_box) == 0:
                    x_max = box[1]
                    new_box.append(box)
                else:
                    if abs(box[0] - x_max) < width_ths * (
                            box[3] - box[2]):  # 合并框
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        x_max = box[1]
                        merged_box.append(new_box)
                        new_box = [box]
            if len(new_box) > 0:
                merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1:  # 同一行内相邻的框
                    x_min = min(mbox, key=lambda x: x[0])[0]  # 最左侧框的左上角 x 坐标
                    x_max = max(mbox, key=lambda x: x[1])[1]  # 最右侧框的右下角 x 坐标
                    y_min = min(mbox, key=lambda x: x[2])[2]  # 最上侧框的左上角 y 坐标
                    y_max = max(mbox, key=lambda x: x[3])[3]  # 最下侧框的右下角 y 坐标

                    margin = int(add_margin * (y_max - y_min))  # 计算边距

                    merged_list.append([
                        x_min - margin, x_max + margin, y_min - margin,
                        y_max + margin
                    ])
                else:  # 同一行内非相邻的框
                    box = mbox[0]

                    margin = int(add_margin * (box[3] - box[2]))  # 计算边距
                    merged_list.append([
                        box[0] - margin,
                        box[1] + margin,
                        box[2] - margin,
                        box[3] + margin,
                    ])
    # 可能需要检查框是否真的在图像内
    return merged_list, free_list  # 返回合并后的框列表和未合并的框列表
# 根据水平文本框列表、自由形文本框列表、图像数据和模型高度获取图像列表
def get_image_list(horizontal_list: list,
                   free_list: list,
                   img: np.ndarray,
                   model_height: int = 64):
    image_list = []  # 初始化空的图像列表
    maximum_y, maximum_x = img.shape  # 获取图像的最大高度和宽度

    max_ratio_hori, max_ratio_free = 1, 1  # 初始化水平和自由形文本框的最大宽高比

    # 遍历自由形文本框列表
    for box in free_list:
        rect = np.array(box, dtype="float32")  # 将文本框坐标转换为浮点型数组
        transformed_img = four_point_transform(img, rect)  # 对图像进行透视变换，以匹配文本框
        ratio = transformed_img.shape[1] / transformed_img.shape[0]  # 计算变换后图像的宽高比
        crop_img = cv2.resize(
            transformed_img,
            (int(model_height * ratio), model_height),  # 调整图像尺寸至模型指定高度
            interpolation=Image.LANCZOS,
        )
        image_list.append((box, crop_img))  # 将文本框坐标及调整后的图像加入图像列表
        max_ratio_free = max(ratio, max_ratio_free)  # 更新自由形文本框的最大宽高比

    max_ratio_free = math.ceil(max_ratio_free)  # 对最大宽高比向上取整

    # 遍历水平文本框列表
    for box in horizontal_list:
        x_min = max(0, box[0])  # 水平文本框左上角 x 坐标
        x_max = min(box[1], maximum_x)  # 水平文本框右下角 x 坐标
        y_min = max(0, box[2])  # 水平文本框左上角 y 坐标
        y_max = min(box[3], maximum_y)  # 水平文本框右下角 y 坐标
        crop_img = img[y_min:y_max, x_min:x_max]  # 裁剪出水平文本框区域的图像
        width = x_max - x_min  # 水平文本框宽度
        height = y_max - y_min  # 水平文本框高度
        ratio = width / height  # 计算宽高比
        crop_img = cv2.resize(
            crop_img,
            (int(model_height * ratio), model_height),  # 调整图像尺寸至模型指定高度
            interpolation=Image.LANCZOS,
        )
        image_list.append((
            [
                [x_min, y_min],  # 左上角坐标
                [x_max, y_min],  # 右上角坐标
                [x_max, y_max],  # 右下角坐标
                [x_min, y_max],  # 左下角坐标
            ],
            crop_img,
        ))
        max_ratio_hori = max(ratio, max_ratio_hori)  # 更新水平文本框的最大宽高比

    max_ratio_hori = math.ceil(max_ratio_hori)  # 对最大宽高比向上取整
    max_ratio = max(max_ratio_hori, max_ratio_free)  # 获取最大宽高比
    max_width = math.ceil(max_ratio) * model_height  # 计算最大宽度

    image_list = sorted(
        image_list, key=lambda item: item[0][0][1])  # 按文本框左上角纵坐标排序，以便排列成段
    return image_list, max_width  # 返回图像列表和最大宽度


def diff(input_list):
    return max(input_list) - min(input_list)  # 计算列表中最大值与最小值的差


def get_paragraph(raw_result,
                  x_ths: int = 1,
                  y_ths: float = 0.5,
                  mode: str = "ltr"):
    # 创建基本属性列表
    box_group = []
    for box in raw_result:
        all_x = [int(coord[0]) for coord in box[0]]  # 提取文本框所有 x 坐标
        all_y = [int(coord[1]) for coord in box[0]]  # 提取文本框所有 y 坐标
        min_x = min(all_x)  # 获取最小 x 坐标
        max_x = max(all_x)  # 获取最大 x 坐标
        min_y = min(all_y)  # 获取最小 y 坐标
        max_y = max(all_y)  # 获取最大 y 坐标
        height = max_y - min_y  # 计算文本框高度
        box_group.append([
            box[1], min_x, max_x, min_y, max_y, height, 0.5 * (min_y + max_y), 0
        ])  # 添加文本框信息到组列表，最后一个元素表示分组

    current_group = 1  # 设置当前分组编号为 1
    # 只要还有未分组的盒子，继续循环
    while len([box for box in box_group if box[7] == 0]) > 0:
        # group0 = non-group，筛选出尚未分组的盒子
        box_group0 = [box for box in box_group if box[7] == 0]
        
        # 如果当前组中没有任何盒子
        if len([box for box in box_group if box[7] == current_group]) == 0:
            # 将第一个盒子分配给新组
            box_group0[0][7] = current_group
        
        # 否则，尝试添加到当前组
        else:
            # 获取当前组中的所有盒子
            current_box_group = [box for box in box_group if box[7] == current_group]
            
            # 计算当前组盒子高度的平均值
            mean_height = np.mean([box[5] for box in current_box_group])
            
            # 计算水平方向的最小和最大范围
            min_gx = min([box[1] for box in current_box_group]) - x_ths * mean_height
            max_gx = max([box[2] for box in current_box_group]) + x_ths * mean_height
            
            # 计算垂直方向的最小和最大范围
            min_gy = min([box[3] for box in current_box_group]) - y_ths * mean_height
            max_gy = max([box[4] for box in current_box_group]) + y_ths * mean_height
            
            add_box = False
            
            # 遍历未分组的盒子，尝试将其加入到当前组中
            for box in box_group0:
                same_horizontal_level = (min_gx <= box[1] <= max_gx) or (min_gx <= box[2] <= max_gx)
                same_vertical_level = (min_gy <= box[3] <= max_gy) or (min_gy <= box[4] <= max_gy)
                
                # 如果盒子与当前组水平和垂直方向符合条件，则将其加入当前组
                if same_horizontal_level and same_vertical_level:
                    box[7] = current_group
                    add_box = True
                    break
            
            # 如果无法将更多盒子加入当前组，则转移到下一个组
            if not add_box:
                current_group += 1
        
    # 段落中的盒子重新排序的结果列表
    result = []
    # 对 box_group 中第 7 个元素去重，生成一个集合
    for i in set(box[7] for box in box_group):
        # 从 box_group 中筛选出第 7 个元素等于当前 i 值的所有盒子
        current_box_group = [box for box in box_group if box[7] == i]
        # 计算当前盒子组中盒子高度的平均值
        mean_height = np.mean([box[5] for box in current_box_group])
        # 获取当前盒子组中最左侧的 x 坐标
        min_gx = min([box[1] for box in current_box_group])
        # 获取当前盒子组中最右侧的 x 坐标
        max_gx = max([box[2] for box in current_box_group])
        # 获取当前盒子组中最上方的 y 坐标
        min_gy = min([box[3] for box in current_box_group])
        # 获取当前盒子组中最下方的 y 坐标
        max_gy = max([box[4] for box in current_box_group])

        # 初始化一个空字符串，用于存储最终的文本
        text = ""
        # 只要当前盒子组还有元素，就进行循环
        while len(current_box_group) > 0:
            # 找出当前盒子组中高度最小的盒子
            highest = min([box[6] for box in current_box_group])
            # 筛选出高度小于最小高度加上平均高度四分之一的候选盒子
            candidates = [
                box for box in current_box_group
                if box[6] < highest + 0.4 * mean_height
            ]
            # 如果模式是从左到右
            if mode == "ltr":
                # 找出候选盒子中最左侧的 x 坐标
                most_left = min([box[1] for box in candidates])
                # 遍历候选盒子，找到最左侧的盒子
                for box in candidates:
                    if box[1] == most_left:
                        best_box = box
            # 如果模式是从右到左
            elif mode == "rtl":
                # 找出候选盒子中最右侧的 x 坐标
                most_right = max([box[2] for box in candidates])
                # 遍历候选盒子，找到最右侧的盒子
                for box in candidates:
                    if box[2] == most_right:
                        best_box = box
            # 将最佳盒子中的文本内容添加到 text 中
            text += " " + best_box[0]
            # 从当前盒子组中移除已处理的最佳盒子
            current_box_group.remove(best_box)

        # 将结果添加到 result 列表中，包括盒子坐标和组合后的文本（去除开头空格）
        result.append([
            [
                [min_gx, min_gy],
                [max_gx, min_gy],
                [max_gx, max_gy],
                [min_gx, max_gy],
            ],
            text[1:],
        ])

    # 返回最终结果列表
    return result
# 定义一个打印进度条的函数，用于在命令行中显示进度条
def printProgressBar(
    prefix="",
    suffix="",
    decimals: int = 1,
    length: int = 100,
    fill: str = "█",
    printEnd: str = "\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        prefix      - Optional  : 前缀字符串 (Str)
        suffix      - Optional  : 后缀字符串 (Str)
        decimals    - Optional  : 百分比显示精度 (Int)
        length      - Optional  : 进
```