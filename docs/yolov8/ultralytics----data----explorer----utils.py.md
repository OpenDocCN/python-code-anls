# `.\yolov8\ultralytics\data\explorer\utils.py`

```py
# 导入必要的模块和库
import getpass  # 导入获取用户信息的模块
from typing import List  # 引入列表类型提示

import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数值计算

# 导入Ultralytics项目中的数据增强、日志等工具
from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER as logger
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.plotting import plot_images


def get_table_schema(vector_size):
    """提取并返回数据库表的模式。"""
    # 导入LanceModel和Vector类
    from lancedb.pydantic import LanceModel, Vector

    # 定义表模式Schema类
    class Schema(LanceModel):
        im_file: str  # 图像文件名
        labels: List[str]  # 标签列表
        cls: List[int]  # 类别列表
        bboxes: List[List[float]]  # 边界框列表
        masks: List[List[List[int]]]  # 掩模列表
        keypoints: List[List[List[float]]]  # 关键点列表
        vector: Vector(vector_size)  # 特征向量

    return Schema


def get_sim_index_schema():
    """返回具有指定向量大小的数据库表的LanceModel模式。"""
    # 导入LanceModel类
    from lancedb.pydantic import LanceModel

    # 定义模式Schema类
    class Schema(LanceModel):
        idx: int  # 索引
        im_file: str  # 图像文件名
        count: int  # 计数
        sim_im_files: List[str]  # 相似图像文件列表

    return Schema


def sanitize_batch(batch, dataset_info):
    """清理推断的输入批次，确保格式和维度正确。"""
    # 将类别转换为扁平整数列表
    batch["cls"] = batch["cls"].flatten().int().tolist()
    # 按类别对边界框和类别进行排序
    box_cls_pair = sorted(zip(batch["bboxes"].tolist(), batch["cls"]), key=lambda x: x[1])
    batch["bboxes"] = [box for box, _ in box_cls_pair]  # 更新边界框列表
    batch["cls"] = [cls for _, cls in box_cls_pair]  # 更新类别列表
    # 根据类别索引获取标签名称
    batch["labels"] = [dataset_info["names"][i] for i in batch["cls"]]
    # 将掩模和关键点转换为列表形式
    batch["masks"] = batch["masks"].tolist() if "masks" in batch else [[[]]]
    batch["keypoints"] = batch["keypoints"].tolist() if "keypoints" in batch else [[[]]]
    return batch


def plot_query_result(similar_set, plot_labels=True):
    """
    绘制相似集合中的图像。

    Args:
        similar_set (list): 包含相似数据点的Pyarrow或pandas对象
        plot_labels (bool): 是否绘制标签
    """
    import pandas  # 为更快的'import ultralytics'而导入

    # 如果similar_set是DataFrame，则转换为字典
    similar_set = (
        similar_set.to_dict(orient="list") if isinstance(similar_set, pandas.DataFrame) else similar_set.to_pydict()
    )
    empty_masks = [[[]]]
    empty_boxes = [[]]
    
    # 获取相似集合中的图像、边界框、掩模、关键点和类别
    images = similar_set.get("im_file", [])
    bboxes = similar_set.get("bboxes", []) if similar_set.get("bboxes") is not empty_boxes else []
    masks = similar_set.get("masks") if similar_set.get("masks")[0] != empty_masks else []
    kpts = similar_set.get("keypoints") if similar_set.get("keypoints")[0] != empty_masks else []
    cls = similar_set.get("cls", [])

    plot_size = 640  # 绘图尺寸
    imgs, batch_idx, plot_boxes, plot_masks, plot_kpts = [], [], [], [], []
    # 遍历图像列表，并用索引 i 和图像文件路径 imf 迭代
    for i, imf in enumerate(images):
        # 使用 OpenCV 读取图像文件
        im = cv2.imread(imf)
        # 将图像从 BGR 格式转换为 RGB 格式
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # 获取图像的高度 h 和宽度 w
        h, w = im.shape[:2]
        # 计算缩放比例 r，使得图像可以适应预定的绘图大小 plot_size
        r = min(plot_size / h, plot_size / w)
        # 使用 LetterBox 函数对图像进行处理，并将通道顺序从 HWC 转换为 CHW
        imgs.append(LetterBox(plot_size, center=False)(image=im).transpose(2, 0, 1))
        
        # 如果需要绘制标签
        if plot_labels:
            # 如果当前图像存在边界框信息
            if len(bboxes) > i and len(bboxes[i]) > 0:
                # 将边界框坐标根据缩放比例 r 进行调整
                box = np.array(bboxes[i], dtype=np.float32)
                box[:, [0, 2]] *= r
                box[:, [1, 3]] *= r
                plot_boxes.append(box)
            
            # 如果当前图像存在掩码信息
            if len(masks) > i and len(masks[i]) > 0:
                # 取出掩码并使用 LetterBox 处理
                mask = np.array(masks[i], dtype=np.uint8)[0]
                plot_masks.append(LetterBox(plot_size, center=False)(image=mask))
            
            # 如果当前图像存在关键点信息
            if len(kpts) > i and kpts[i] is not None:
                # 取出关键点坐标并根据缩放比例 r 进行调整
                kpt = np.array(kpts[i], dtype=np.float32)
                kpt[:, :, :2] *= r
                plot_kpts.append(kpt)
        
        # 将当前图像索引 i 添加到 batch_idx 列表中，其长度与当前图像的边界框数量相同
        batch_idx.append(np.ones(len(np.array(bboxes[i], dtype=np.float32))) * i)
    
    # 将所有处理后的图像堆叠成一个批次 imgs
    imgs = np.stack(imgs, axis=0)
    # 将所有处理后的掩码堆叠成一个批次 masks，如果没有掩码则创建空数组
    masks = np.stack(plot_masks, axis=0) if plot_masks else np.zeros(0, dtype=np.uint8)
    # 将所有处理后的关键点堆叠成一个数组 kpts，如果没有关键点则创建空数组
    kpts = np.concatenate(plot_kpts, axis=0) if plot_kpts else np.zeros((0, 51), dtype=np.float32)
    # 将所有处理后的边界框坐标从 xyxy 格式转换为 xywh 格式，如果没有边界框则创建空数组
    boxes = xyxy2xywh(np.concatenate(plot_boxes, axis=0)) if plot_boxes else np.zeros(0, dtype=np.float32)
    # 将 batch_idx 数组连接起来，形成一个批次索引
    batch_idx = np.concatenate(batch_idx, axis=0)
    # 将类别列表 cls 中所有元素连接成一个数组 cls
    cls = np.concatenate([np.array(c, dtype=np.int32) for c in cls], axis=0)
    
    # 调用 plot_images 函数，绘制所有处理后的图像及其相关信息，并返回结果
    return plot_images(
        imgs, batch_idx, cls, bboxes=boxes, masks=masks, kpts=kpts, max_subplots=len(images), save=False, threaded=False
    )
def prompt_sql_query(query):
    """提示用户输入 SQL 查询，然后使用 OpenAI 模型生成完整的 SQL 查询语句"""

    # 检查是否符合 openai 要求
    check_requirements("openai>=1.6.1")
    # 导入 OpenAI 模块
    from openai import OpenAI

    # 如果 SETTINGS 中未设置 openai_api_key，则提示用户输入
    if not SETTINGS["openai_api_key"]:
        logger.warning("OpenAI API key not found in settings. Please enter your API key below.")
        openai_api_key = getpass.getpass("OpenAI API key: ")
        SETTINGS.update({"openai_api_key": openai_api_key})
    # 创建 OpenAI 对象并使用设置的 API key
    openai = OpenAI(api_key=SETTINGS["openai_api_key"])

    # 准备对话消息列表
    messages = [
        {
            "role": "system",
            "content": """
                You are a helpful data scientist proficient in SQL. You need to output exactly one SQL query based on
                the following schema and a user request. You only need to output the format with fixed selection
                statement that selects everything from "'table'", like `SELECT * from 'table'`

                Schema:
                im_file: string not null
                labels: list<item: string> not null
                child 0, item: string
                cls: list<item: int64> not null
                child 0, item: int64
                bboxes: list<item: list<item: double>> not null
                child 0, item: list<item: double>
                    child 0, item: double
                masks: list<item: list<item: list<item: int64>> not null
                child 0, item: list<item: list<item: int64>>
                    child 0, item: list<item: int64>
                        child 0, item: int64
                keypoints: list<item: list<item: list<item: double>> not null
                child 0, item: list<item: list<item: double>>
                    child 0, item: list<item: double>
                        child 0, item: double
                vector: fixed_size_list<item: float>[256] not null
                child 0, item: float

                Some details about the schema:
                - the "labels" column contains the string values like 'person' and 'dog' for the respective objects
                    in each image
                - the "cls" column contains the integer values on these classes that map them the labels

                Example of a correct query:
                request - Get all data points that contain 2 or more people and at least one dog
                correct query-
                SELECT * FROM 'table' WHERE  ARRAY_LENGTH(cls) >= 2  AND ARRAY_LENGTH(FILTER(labels, x -> x = 'person')) >= 2  AND ARRAY_LENGTH(FILTER(labels, x -> x = 'dog')) >= 1;
             """,
        },
        {"role": "user", "content": f"{query}"},  # 用户输入的查询消息
    ]

    # 调用 OpenAI 模型生成回应
    response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content  # 返回生成的完整 SQL 查询语句
```