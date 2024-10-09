# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\layoutlmft\data\cord.py`

```
# 引用地址，说明此代码的来源
'''
Reference: https://huggingface.co/datasets/pierresi/cord/blob/main/cord.py
'''

# 导入所需的库
import json
import os
from pathlib import Path
import datasets
from .image_utils import load_image, normalize_bbox
# 获取日志记录器以记录信息
logger = datasets.logging.get_logger(__name__)
# CORD 数据集的引用信息
_CITATION = """\
@article{park2019cord,
  title={CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author={Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk}
  booktitle={Document Intelligence Workshop at Neural Information Processing Systems}
  year={2019}
}
"""
# CORD 数据集的描述链接
_DESCRIPTION = """\
https://github.com/clovaai/cord/
"""

# 将四个坐标点转换为边界框
def quad_to_box(quad):
    # test 87 的注释标注错误
    box = (
        max(0, quad["x1"]),  # 获取左上角x坐标，确保不小于0
        max(0, quad["y1"]),  # 获取左上角y坐标，确保不小于0
        quad["x3"],          # 获取右下角x坐标
        quad["x3"]          # 获取右下角y坐标
    )
    # 如果右下角y坐标小于左上角y坐标，交换这两个值
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    # 如果右下角x坐标小于左上角x坐标，交换这两个值
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    # 返回调整后的边界框
    return box

# 从给定 URL 获取 Google Drive 的直接下载链接
def _get_drive_url(url):
    base_url = 'https://drive.google.com/uc?id='  # Google Drive 的基础下载 URL
    split_url = url.split('/')  # 将 URL 按斜杠分割
    return base_url + split_url[5]  # 返回直接下载链接

# 存储数据集下载链接的列表
_URLS = [
    _get_drive_url("https://drive.google.com/file/d/1MqhTbcj-AHXOqYoeoh12aRUwIprzTJYI/"),  # 第一个文件的下载链接
    _get_drive_url("https://drive.google.com/file/d/1wYdp5nC9LnHQZ2FcmOoC0eClyWvcuARU/")   # 第二个文件的下载链接
    # 如果自动下载数据集失败，可以手动下载并修改代码获取本地数据集。
    # 或者可以使用以下链接，请遵循 CORD 的原始许可协议进行使用。
    # "https://layoutlm.blob.core.windows.net/cord/CORD-1k-001.zip",
    # "https://layoutlm.blob.core.windows.net/cord/CORD-1k-002.zip"
]

# CORD 数据集的构建配置类
class CordConfig(datasets.BuilderConfig):
    """BuilderConfig for CORD"""
    def __init__(self, **kwargs):
        """BuilderConfig for CORD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CordConfig, self).__init__(**kwargs)  # 调用父类的构造函数

# CORD 数据集的生成器基础类
class Cord(datasets.GeneratorBasedBuilder):
    # 定义可用的构建配置
    BUILDER_CONFIGS = [
        CordConfig(name="cord", version=datasets.Version("1.0.0"), description="CORD dataset"),  # 配置名称、版本和描述
    ]
    # 定义一个名为 _info 的方法
        def _info(self):
            # 返回一个包含数据集信息的对象
            return datasets.DatasetInfo(
                # 设置数据集的描述信息
                description=_DESCRIPTION,
                # 定义数据集的特征
                features=datasets.Features(
                    {
                        # 定义特征 "id"，类型为字符串
                        "id": datasets.Value("string"),
                        # 定义特征 "words"，类型为字符串的序列
                        "words": datasets.Sequence(datasets.Value("string")),
                        # 定义特征 "bboxes"，类型为整型的序列的序列
                        "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                        # 定义特征 "ner_tags"，为一系列类别标签
                        "ner_tags": datasets.Sequence(
                            datasets.features.ClassLabel(
                                # 指定类别标签的名称
                                names=["O","B-MENU.NM","B-MENU.NUM","B-MENU.UNITPRICE","B-MENU.CNT","B-MENU.DISCOUNTPRICE","B-MENU.PRICE","B-MENU.ITEMSUBTOTAL","B-MENU.VATYN","B-MENU.ETC","B-MENU.SUB_NM","B-MENU.SUB_UNITPRICE","B-MENU.SUB_CNT","B-MENU.SUB_PRICE","B-MENU.SUB_ETC","B-VOID_MENU.NM","B-VOID_MENU.PRICE","B-SUB_TOTAL.SUBTOTAL_PRICE","B-SUB_TOTAL.DISCOUNT_PRICE","B-SUB_TOTAL.SERVICE_PRICE","B-SUB_TOTAL.OTHERSVC_PRICE","B-SUB_TOTAL.TAX_PRICE","B-SUB_TOTAL.ETC","B-TOTAL.TOTAL_PRICE","B-TOTAL.TOTAL_ETC","B-TOTAL.CASHPRICE","B-TOTAL.CHANGEPRICE","B-TOTAL.CREDITCARDPRICE","B-TOTAL.EMONEYPRICE","B-TOTAL.MENUTYPE_CNT","B-TOTAL.MENUQTY_CNT","I-MENU.NM","I-MENU.NUM","I-MENU.UNITPRICE","I-MENU.CNT","I-MENU.DISCOUNTPRICE","I-MENU.PRICE","I-MENU.ITEMSUBTOTAL","I-MENU.VATYN","I-MENU.ETC","I-MENU.SUB_NM","I-MENU.SUB_UNITPRICE","I-MENU.SUB_CNT","I-MENU.SUB_PRICE","I-MENU.SUB_ETC","I-VOID_MENU.NM","I-VOID_MENU.PRICE","I-SUB_TOTAL.SUBTOTAL_PRICE","I-SUB_TOTAL.DISCOUNT_PRICE","I-SUB_TOTAL.SERVICE_PRICE","I-SUB_TOTAL.OTHERSVC_PRICE","I-SUB_TOTAL.TAX_PRICE","I-SUB_TOTAL.ETC","I-TOTAL.TOTAL_PRICE","I-TOTAL.TOTAL_ETC","I-TOTAL.CASHPRICE","I-TOTAL.CHANGEPRICE","I-TOTAL.CREDITCARDPRICE","I-TOTAL.EMONEYPRICE","I-TOTAL.MENUTYPE_CNT","I-TOTAL.MENUQTY_CNT"]
                            )
                        ),
                        # 定义特征 "image"，形状为 (3, 224, 224) 的 3D 数组，数据类型为无符号整型
                        "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                        # 定义特征 "image_path"，类型为字符串
                        "image_path": datasets.Value("string"),
                    }
                ),
                # 不使用监督学习的键
                supervised_keys=None,
                # 设置引用文献
                citation=_CITATION,
                # 设置数据集的主页链接
                homepage="https://github.com/clovaai/cord/",
            )
    # 定义一个私有方法，用于分割生成器
    def _split_generators(self, dl_manager):
        # 返回 SplitGenerators 的方法
        """Returns SplitGenerators."""
        # 使用本地文件，位于 data_dir
        """Uses local files located with data_dir"""
        # 下载并解压指定的文件
        downloaded_file = dl_manager.download_and_extract(_URLS)
        # 将第二个 URL 的文件移动到第一个 URL 的目录中
        dest = Path(downloaded_file[0])/"CORD"
        # 遍历训练、验证和测试数据集
        for split in ["train", "dev", "test"]:
            # 遍历文件类型，包括图像和 JSON
            for file_type in ["image", "json"]:
                # 跳过测试集的 JSON 文件
                if split == "test" and file_type == "json":
                    continue
                # 获取当前分割和文件类型下的文件
                files = (Path(downloaded_file[1])/"CORD"/split/file_type).iterdir()
                # 遍历所有文件
                for f in files:
                    # 重命名并移动文件到目标目录
                    os.rename(f, dest/split/file_type/f.name)
        # 返回训练、验证和测试的分割生成器
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": dest/"train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dest/"dev"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": dest/"test"}
            ),
        ]
    
    # 定义一个获取行边界框的函数
    def get_line_bbox(self, bboxs):
        # 提取所有 x 坐标
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        # 提取所有 y 坐标
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]
    
        # 计算 x 和 y 坐标的最小和最大值
        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
    
        # 确保 x1 和 y1 不小于 x0 和 y0
        assert x1 >= x0 and y1 >= y0
        # 为每个边界框创建一个统一的边界框列表
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        # 返回边界框列表
        return bbox
    # 定义生成示例的私有方法，接受文件路径作为参数
    def _generate_examples(self, filepath):
        # 记录生成示例的开始信息，包含文件路径
        logger.info("⏳ Generating examples from = %s", filepath)
        # 构建注释文件所在目录的路径
        ann_dir = os.path.join(filepath, "json")
        # 构建图像文件所在目录的路径
        img_dir = os.path.join(filepath, "image")
        # 遍历注释目录中的每个文件，使用文件的索引和文件名
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            # 初始化存储单词的列表
            words = []
            # 初始化存储边界框的列表
            bboxes = []
            # 初始化存储命名实体标签的列表
            ner_tags = []
            # 构建当前文件的完整路径
            file_path = os.path.join(ann_dir, file)
            # 以读取模式打开注释文件，并指定编码为 UTF-8
            with open(file_path, "r", encoding="utf8") as f:
                # 解析 JSON 格式的数据
                data = json.load(f)
            # 构建当前图像的路径
            image_path = os.path.join(img_dir, file)
            # 将路径中的“json”替换为“png”，得到图像文件的路径
            image_path = image_path.replace("json", "png")
            # 加载图像并获取其大小
            image, size = load_image(image_path)
            # 遍历有效行中的每一项
            for item in data["valid_line"]:
                # 初始化当前行的边界框列表
                cur_line_bboxes = []
                # 提取当前行的单词和标签
                line_words, label = item["words"], item["category"]
                # 过滤掉文本为空的单词
                line_words = [w for w in line_words if w["text"].strip() != ""]
                # 如果当前行没有有效单词，则跳过
                if len(line_words) == 0:
                    continue
                # 如果标签为“其他”，则处理所有单词
                if label == "other":
                    for w in line_words:
                        # 添加单词文本到列表
                        words.append(w["text"])
                        # 为单词标记为“其他”
                        ner_tags.append("O")
                        # 归一化当前单词的边界框并添加到列表
                        cur_line_bboxes.append(normalize_bbox(quad_to_box(w["quad"]), size))
                else:
                    # 添加第一个单词文本到列表
                    words.append(line_words[0]["text"])
                    # 为第一个单词标记为该类别的开始
                    ner_tags.append("B-" + label.upper())
                    # 归一化第一个单词的边界框并添加到列表
                    cur_line_bboxes.append(normalize_bbox(quad_to_box(line_words[0]["quad"]), size))
                    # 处理后续单词
                    for w in line_words[1:]:
                        # 添加单词文本到列表
                        words.append(w["text"])
                        # 为后续单词标记为该类别的内部
                        ner_tags.append("I-" + label.upper())
                        # 归一化当前单词的边界框并添加到列表
                        cur_line_bboxes.append(normalize_bbox(quad_to_box(w["quad"]), size))
                # 默认情况下，使用段落级布局
                # 如果不想使用段落级布局，可以注释掉以下行
                # 计算当前行的边界框
                cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                # 扩展总边界框列表
                bboxes.extend(cur_line_bboxes)
            # 生成当前示例的 GUID 和相关信息
            yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "ner_tags": ner_tags,
                         "image": image, "image_path": image_path}
```