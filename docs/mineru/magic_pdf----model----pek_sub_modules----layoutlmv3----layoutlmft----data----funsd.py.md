# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\layoutlmft\data\funsd.py`

```
# coding=utf-8
''' 
# 定义编码格式为 UTF-8 
'''
# 引用 JSON 模块用于处理 JSON 数据
import json
# 引用 OS 模块用于与操作系统交互
import os

# 引用 datasets 库用于数据集的处理
import datasets

# 从本地模块中引入图像加载和边界框归一化的工具
from .image_utils import load_image, normalize_bbox

# 获取日志记录器，用于输出日志信息
logger = datasets.logging.get_logger(__name__)

# 引用数据集的文献
_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

# 数据集的描述信息
_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""

# 定义 FUNSD 的配置类，继承自 datasets.BuilderConfig
class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    # 初始化配置类
    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        # 调用父类构造函数
        super(FunsdConfig, self).__init__(**kwargs)

# 定义 FUNSD 数据集类，继承自 datasets.GeneratorBasedBuilder
class Funsd(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    # 定义可用的配置
    BUILDER_CONFIGS = [
        # 创建 FUNSD 的配置对象
        FunsdConfig(name="funsd", version=datasets.Version("1.0.0"), description="FUNSD dataset"),
    ]

    # 定义数据集的信息
    def _info(self):
        # 返回数据集的信息，包括描述和特征
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    # 定义数据集中的 ID 字段
                    "id": datasets.Value("string"),
                    # 定义 token 字段，包含字符串序列
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    # 定义边界框字段，包含整数序列的序列
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    # 定义命名实体标签字段
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                        )
                    ),
                    # 定义图像字段，包含形状为 (3, 224, 224) 的数组
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    # 定义图像路径字段
                    "image_path": datasets.Value("string"),
                }
            ),
            # 不需要监督键
            supervised_keys=None,
            # 设置数据集主页链接
            homepage="https://guillaumejaume.github.io/FUNSD/",
            # 设置引用信息
            citation=_CITATION,
        )

    # 定义数据集的分割生成器
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # 下载并解压数据集文件
        downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        # 返回训练和测试数据的分割生成器
        return [
            datasets.SplitGenerator(
                # 定义训练集生成器
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/training_data/"}
            ),
            datasets.SplitGenerator(
                # 定义测试集生成器
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/"}
            ),
        ]
    # 获取每一行的边界框
    def get_line_bbox(self, bboxs):
        # 提取所有边界框的 x 坐标（偶数索引）
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        # 提取所有边界框的 y 坐标（奇数索引）
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        # 计算 x 和 y 坐标的最小值和最大值，形成整体边界框
        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        # 确保计算出的边界框有效
        assert x1 >= x0 and y1 >= y0
        # 为每个边界框创建一个统一的边界框列表
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        # 返回所有行的边界框
        return bbox

    # 生成示例数据
    def _generate_examples(self, filepath):
        # 记录生成示例数据的日志信息
        logger.info("⏳ Generating examples from = %s", filepath)
        # 定义注释文件夹的路径
        ann_dir = os.path.join(filepath, "annotations")
        # 定义图像文件夹的路径
        img_dir = os.path.join(filepath, "images")
        # 遍历注释文件夹中的文件
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            # 初始化令牌、边界框和命名实体标签的列表
            tokens = []
            bboxes = []
            ner_tags = []

            # 获取当前文件的完整路径
            file_path = os.path.join(ann_dir, file)
            # 打开文件并加载 JSON 数据
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            # 构建对应的图像路径，并将扩展名从 json 更改为 png
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            # 加载图像并获取其尺寸
            image, size = load_image(image_path)
            # 遍历数据中的每个表单项
            for item in data["form"]:
                # 初始化当前行的边界框列表
                cur_line_bboxes = []
                # 获取当前项的单词和标签
                words, label = item["words"], item["label"]
                # 过滤掉文本为空的单词
                words = [w for w in words if w["text"].strip() != ""]
                # 如果当前行没有单词，跳过
                if len(words) == 0:
                    continue
                # 如果标签为 'other'
                if label == "other":
                    # 将每个单词添加到令牌和命名实体标签中
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append("O")  # 代表其他
                        # 归一化当前单词的边界框
                        cur_line_bboxes.append(normalize_bbox(w["box"], size))
                else:
                    # 对第一个单词进行特别处理
                    tokens.append(words[0]["text"])
                    ner_tags.append("B-" + label.upper())  # 代表开始
                    cur_line_bboxes.append(normalize_bbox(words[0]["box"], size))
                    # 对后续单词进行处理
                    for w in words[1:]:
                        tokens.append(w["text"])
                        ner_tags.append("I-" + label.upper())  # 代表内部
                        cur_line_bboxes.append(normalize_bbox(w["box"], size))
                # 默认使用分段级别布局
                # 如果不想使用分段级别布局，请注释以下行
                cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                # box = normalize_bbox(item["box"], size)
                # cur_line_bboxes = [box for _ in range(len(words))]
                # 扩展边界框列表，加入当前行的边界框
                bboxes.extend(cur_line_bboxes)
            # 生成的示例数据以元组形式返回
            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags,
                         "image": image, "image_path": image_path}
```