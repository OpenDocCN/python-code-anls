# `.\pytorch\scripts\release_notes\classifier.py`

```py
import argparse
import math
import pickle
import random
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, List

import common
import pandas as pd
import torchtext
from torchtext.functional import to_tensor
from tqdm import tqdm

import torch
import torch.nn as nn


XLMR_BASE = torchtext.models.XLMR_BASE_ENCODER
# 设置默认设备为 CUDA 如果可用，否则为 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

HAS_IMBLEARN = False
try:
    import imblearn
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

# 文件名最大长度用于文件路径截断
MAX_LEN_FILE = 6

UNKNOWN_TOKEN = "<Unknown>"

# Utilities for working with a truncated file graph

# 截断文件路径至指定长度
def truncate_file(file: Path, max_len: int = 5):
    return ("/").join(file.parts[:max_len])

# 构建截断后文件集合
def build_file_set(all_files: List[Path], max_len: int):
    truncated_files = [truncate_file(file, max_len) for file in all_files]
    return set(truncated_files)

@dataclass
class CommitClassifierInputs:
    title: List[str]
    files: List[str]
    author: List[str]

@dataclass
class CategoryConfig:
    categories: List[str]
    input_dim: int = 768
    inner_dim: int = 128
    dropout: float = 0.1
    activation = nn.ReLU
    embedding_dim: int = 8
    file_embedding_dim: int = 32

class CommitClassifier(nn.Module):
    def __init__(
        self,
        encoder_base: torchtext.models.XLMR_BASE_ENCODER,
        author_map: Dict[str, int],
        file_map: Dict[str, int],
        config: CategoryConfig,
    ):
        super().__init__()
        # 设置模型编码器和转换器
        self.encoder = encoder_base.get_model().requires_grad_(False)
        self.transform = encoder_base.transform()
        self.author_map = author_map
        self.file_map = file_map
        self.categories = config.categories
        self.num_authors = len(author_map)
        self.num_files = len(file_map)
        # 设置作者的嵌入表
        self.embedding_table = nn.Embedding(self.num_authors, config.embedding_dim)
        # 设置文件的嵌入袋
        self.file_embedding_bag = nn.EmbeddingBag(
            self.num_files, config.file_embedding_dim, mode="sum"
        )
        # 设置标题的密集层
        self.dense_title = nn.Linear(config.input_dim, config.inner_dim)
        # 设置文件的密集层
        self.dense_files = nn.Linear(config.file_embedding_dim, config.inner_dim)
        # 设置作者的密集层
        self.dense_author = nn.Linear(config.embedding_dim, config.inner_dim)
        self.dropout = nn.Dropout(config.dropout)
        # 设置标题、文件、作者的输出层
        self.out_proj_title = nn.Linear(config.inner_dim, len(self.categories))
        self.out_proj_files = nn.Linear(config.inner_dim, len(self.categories))
        self.out_proj_author = nn.Linear(config.inner_dim, len(self.categories))
        self.activation_fn = config.activation()
    def forward(self, input_batch: CommitClassifierInputs):
        # Encode input title
        # 从输入中获取标题列表
        title: List[str] = input_batch.title
        # 将标题转换为张量，并进行填充，转移到指定设备
        model_input = to_tensor(self.transform(title), padding_value=1).to(device)
        # 使用编码器对标题进行编码
        title_features = self.encoder(model_input)
        # 提取标题的嵌入表示中的第一个位置的特征
        title_embed = title_features[:, 0, :]
        # 对标题嵌入进行丢弃操作
        title_embed = self.dropout(title_embed)
        # 通过全连接层处理标题嵌入
        title_embed = self.dense_title(title_embed)
        # 应用激活函数到标题嵌入
        title_embed = self.activation_fn(title_embed)
        # 再次对标题嵌入进行丢弃操作
        title_embed = self.dropout(title_embed)
        # 通过最终的全连接层处理标题嵌入，生成最终的标题表示
        title_embed = self.out_proj_title(title_embed)

        # 处理文件列表
        files: list[str] = input_batch.files
        batch_file_indexes = []
        for file in files:
            # 对文件路径进行截断处理，确保不超过最大长度
            paths = [
                truncate_file(Path(file_part), MAX_LEN_FILE)
                for file_part in file.split(" ")
            ]
            # 将截断后的文件路径映射为对应的索引，并添加到批次的文件索引列表中
            batch_file_indexes.append(
                [
                    self.file_map.get(file, self.file_map[UNKNOWN_TOKEN])
                    for file in paths
                ]
            )

        # 将批次中的文件索引展开成一维张量
        flat_indexes = torch.tensor(
            list(chain.from_iterable(batch_file_indexes)),
            dtype=torch.long,
            device=device,
        )
        # 计算文件索引在批次中的偏移量
        offsets = [0]
        offsets.extend(len(files) for files in batch_file_indexes[:-1])
        offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        offsets = offsets.cumsum(dim=0)

        # 使用文件索引和偏移量生成文件嵌入
        files_embed = self.file_embedding_bag(flat_indexes, offsets)
        # 通过全连接层处理文件嵌入
        files_embed = self.dense_files(files_embed)
        # 应用激活函数到文件嵌入
        files_embed = self.activation_fn(files_embed)
        # 对文件嵌入进行丢弃操作
        files_embed = self.dropout(files_embed)
        # 通过最终的全连接层处理文件嵌入，生成最终的文件表示
        files_embed = self.out_proj_files(files_embed)

        # 添加作者嵌入
        authors: List[str] = input_batch.author
        # 将作者映射为对应的 ID
        author_ids = [
            self.author_map.get(author, self.author_map[UNKNOWN_TOKEN])
            for author in authors
        ]
        # 转换为张量并移动到指定设备
        author_ids = torch.tensor(author_ids).to(device)
        # 获取作者嵌入
        author_embed = self.embedding_table(author_ids)
        # 通过全连接层处理作者嵌入
        author_embed = self.dense_author(author_embed)
        # 应用激活函数到作者嵌入
        author_embed = self.activation_fn(author_embed)
        # 对作者嵌入进行丢弃操作
        author_embed = self.dropout(author_embed)
        # 通过最终的全连接层处理作者嵌入，生成最终的作者表示
        author_embed = self.out_proj_author(author_embed)

        # 返回标题嵌入、文件嵌入和作者嵌入的加和作为最终的输出
        return title_embed + files_embed + author_embed

    def convert_index_to_category_name(self, most_likely_index):
        # 根据最可能的索引返回对应的类别名称
        if isinstance(most_likely_index, int):
            return self.categories[most_likely_index]
        elif isinstance(most_likely_index, torch.Tensor):
            return [self.categories[i] for i in most_likely_index]

    def get_most_likely_category_name(self, inpt):
        # 输入是包含标题和作者键的字典
        # 获取前向传播的结果（logits）
        logits = self.forward(inpt)
        # 找到 logits 中最大值的索引
        most_likely_index = torch.argmax(logits, dim=1)
        # 将最可能的索引转换为类别名称并返回
        return self.convert_index_to_category_name(most_likely_index)
# 根据数据文件夹路径和重新生成标志，获取训练和验证数据集
def get_train_val_data(data_folder: Path, regen_data: bool, train_percentage=0.95):
    # 如果不需要重新生成数据，并且已经存在训练集和验证集的 CSV 文件
    if (
        not regen_data
        and Path(data_folder / "train_df.csv").exists()
        and Path(data_folder / "val_df.csv").exists()
    ):
        # 从已有的 CSV 文件中读取训练集和验证集数据
        train_data = pd.read_csv(data_folder / "train_df.csv")
        val_data = pd.read_csv(data_folder / "val_df.csv")
        return train_data, val_data
    else:
        # 如果需要重新生成数据，或者训练集和验证集的 CSV 文件不存在
        print("Train, Val, Test Split not found generating from scratch.")
        # 从 commitlist.csv 文件中读取全部数据
        commit_list_df = pd.read_csv(data_folder / "commitlist.csv")
        # 从 commitlist.csv 中选择 category 为 "Uncategorized" 的数据作为测试集
        test_df = commit_list_df[commit_list_df["category"] == "Uncategorized"]
        # 从 commitlist.csv 中选择 category 不为 "Uncategorized" 的数据作为全部的训练集
        all_train_df = commit_list_df[commit_list_df["category"] != "Uncategorized"]
        # 从训练集中去除 category 为 "skip" 的数据，因为其类别严重不平衡
        print(
            "We are removing skip categories, YOU MIGHT WANT TO CHANGE THIS, BUT THIS IS A MORE HELPFUL CLASSIFIER FOR LABELING."
        )
        all_train_df = all_train_df[all_train_df["category"] != "skip"]
        # 将训练集数据随机打乱顺序
        all_train_df = all_train_df.sample(frac=1).reset_index(drop=True)
        # 根据给定的训练集百分比划分训练集和验证集
        split_index = math.floor(train_percentage * len(all_train_df))
        train_df = all_train_df[:split_index]
        val_df = all_train_df[split_index:]
        print("Train data size: ", len(train_df))
        print("Val data size: ", len(val_df))

        # 将测试集、训练集和验证集数据保存到 CSV 文件中
        test_df.to_csv(data_folder / "test_df.csv", index=False)
        train_df.to_csv(data_folder / "train_df.csv", index=False)
        val_df.to_csv(data_folder / "val_df.csv", index=False)
        return train_df, val_df


# 根据数据文件夹路径和重新生成标志，获取作者映射字典
def get_author_map(data_folder: Path, regen_data, assert_stored=False):
    # 如果不需要重新生成数据，并且已经存在作者映射文件 author_map.pkl
    if not regen_data and Path(data_folder / "author_map.pkl").exists():
        # 直接从 author_map.pkl 文件中加载并返回作者映射字典
        with open(data_folder / "author_map.pkl", "rb") as f:
            return pickle.load(f)
    else:
        # 如果需要重新生成数据，或者作者映射文件不存在
        if assert_stored:
            # 如果设置了 assert_stored 标志，抛出文件未找到异常
            raise FileNotFoundError(
                "Author map not found, you are loading for inference you need to have an author map!"
            )
        print("Regenerating Author Map")
        # 从 commitlist.csv 文件中读取所有数据
        all_data = pd.read_csv(data_folder / "commitlist.csv")
        # 获取所有不重复的作者列表
        authors = all_data.author.unique().tolist()
        # 将 UNKNOWN_TOKEN 添加到作者列表末尾
        authors.append(UNKNOWN_TOKEN)
        # 创建作者到索引的映射字典
        author_map = {author: i for i, author in enumerate(authors)}
        # 将作者映射字典保存到 author_map.pkl 文件中
        with open(data_folder / "author_map.pkl", "wb") as f:
            pickle.dump(author_map, f)
        return author_map


# 根据数据文件夹路径和重新生成标志，获取文件映射字典
def get_file_map(data_folder: Path, regen_data, assert_stored=False):
    # 如果不需要重新生成数据，并且已经存在文件映射文件 file_map.pkl
    if not regen_data and Path(data_folder / "file_map.pkl").exists():
        # 直接从 file_map.pkl 文件中加载并返回文件映射字典
        with open(data_folder / "file_map.pkl", "rb") as f:
            return pickle.load(f)
    else:
        # 如果条件不满足，则执行以下代码块
        if assert_stored:
            # 如果需要确保已经存在文件映射，但未找到时，抛出文件未找到异常
            raise FileNotFoundError(
                "File map not found, you are loading for inference you need to have a file map!"
            )
        # 输出信息，表示正在重新生成文件映射
        print("Regenerating File Map")
        # 从 CSV 文件中读取所有数据
        all_data = pd.read_csv(data_folder / "commitlist.csv")
        # 获取所有文件更改列表
        files = all_data.files_changed.to_list()

        # 初始化空列表以存储所有文件路径
        all_files = []
        # 遍历文件列表中的每个文件路径
        for file in files:
            # 将文件路径按空格拆分为部分路径，并转换为 Path 对象后存入 paths 列表
            paths = [Path(file_part) for file_part in file.split(" ")]
            # 将所有路径扩展到 all_files 列表中
            all_files.extend(paths)
        # 将一个特定的未知令牌路径添加到 all_files 列表末尾
        all_files.append(Path(UNKNOWN_TOKEN))
        # 构建文件集合，限制文件名长度为 MAX_LEN_FILE
        file_set = build_file_set(all_files, MAX_LEN_FILE)
        # 创建文件名到索引的映射字典
        file_map = {file: i for i, file in enumerate(file_set)}
        # 将文件名映射字典保存到指定路径的 pickle 文件中
        with open(data_folder / "file_map.pkl", "wb") as f:
            pickle.dump(file_map, f)
        # 返回生成的文件名映射字典
        return file_map
# 为训练生成数据集

def get_title_files_author_categories_zip_list(dataframe: pd.DataFrame):
    # 从 DataFrame 中获取标题列表
    title = dataframe.title.to_list()
    # 从 DataFrame 中获取文件列表的字符串表示
    files_str = dataframe.files_changed.to_list()
    # 从 DataFrame 中获取作者列表，并用未知标记填充缺失值
    author = dataframe.author.fillna(UNKNOWN_TOKEN).to_list()
    # 从 DataFrame 中获取类别列表
    category = dataframe.category.to_list()
    # 将标题、文件、作者和类别列表组合成一个元组的列表
    return list(zip(title, files_str, author, category))


def generate_batch(batch):
    # 将批次解压缩为各自的列表
    title, files, author, category = zip(*batch)
    # 将元组转换为列表
    title = list(title)
    files = list(files)
    author = list(author)
    category = list(category)
    # 创建目标张量，其中每个类别通过其索引映射到张量，然后移动到指定设备上
    targets = torch.tensor([common.categories.index(cat) for cat in category]).to(
        device
    )
    # 返回 CommitClassifierInputs 类型的输入和目标张量
    return CommitClassifierInputs(title, files, author), targets


def train_step(batch, model, optimizer, loss):
    # 从批次中获取输入和目标
    inpt, targets = batch
    # 清空优化器的梯度
    optimizer.zero_grad()
    # 使用模型计算输出
    output = model(inpt)
    # 计算损失
    l = loss(output, targets)
    # 反向传播损失
    l.backward()
    # 优化模型参数
    optimizer.step()
    # 返回损失值
    return l


@torch.no_grad()
def eval_step(batch, model, loss):
    # 从批次中获取输入和目标
    inpt, targets = batch
    # 使用模型计算输出
    output = model(inpt)
    # 计算损失
    l = loss(output, targets)
    # 返回损失值
    return l


def balance_dataset(dataset: List):
    # 如果没有安装 imbalanced-learn 库，直接返回数据集
    if not HAS_IMBLEARN:
        return dataset
    # 解压缩数据集为各自的列表
    title, files, author, category = zip(*dataset)
    # 将类别转换为其在公共类别列表中的索引
    category = [common.categories.index(cat) for cat in category]
    # 将标题、文件和作者组合成输入数据列表
    inpt_data = list(zip(title, files, author))
    from imblearn.over_sampling import RandomOverSampler

    # 使用 RandomOverSampler 进行过采样
    rus = RandomOverSampler(random_state=42)
    X, y = rus.fit_resample(inpt_data, category)
    # 将过采样后的数据进行合并并随机采样以保持数据平衡
    merged = list(zip(X, y))
    merged = random.sample(merged, k=2 * len(dataset))
    X, y = zip(*merged)
    # 重新构建数据集，将重新平衡的类别映射回原始的类别标签
    rebuilt_dataset = []
    for i in range(len(X)):
        rebuilt_dataset.append((*X[i], common.categories[y[i]]))
    # 返回重新平衡后的数据集
    return rebuilt_dataset


def gen_class_weights(dataset: List):
    # 导入 Counter 类用于计数
    from collections import Counter

    # 设置平滑值
    epsilon = 1e-1
    # 解压缩数据集为各自的列表
    title, files, author, category = zip(*dataset)
    # 将类别转换为其在公共类别列表中的索引
    category = [common.categories.index(cat) for cat in category]
    # 使用 Counter 统计每个类别的数量
    counter = Counter(category)
    # 计算用于计算权重的类别的三分之一分位数
    percentile_33 = len(category) // 3
    # 获取最常见和最不常见的类别
    most_common = counter.most_common(percentile_33)
    least_common = counter.most_common()[-percentile_33:]
    # 平滑处理顶部和底部类别的数量
    smoothed_top = sum(i[1] + epsilon for i in most_common) / len(most_common)
    smoothed_bottom = sum(i[1] + epsilon for i in least_common) / len(least_common) // 3
    # 计算类别权重张量
    class_weights = torch.tensor(
        [
            1.0 / (min(max(counter[i], smoothed_bottom), smoothed_top) + epsilon)
            for i in range(len(common.categories))
        ],
        device=device,
    )
    # 返回计算得到的类别权重张量
    return class_weights


def train(save_path: Path, data_folder: Path, regen_data: bool, resample: bool):
    # 获取训练集和验证集数据
    train_data, val_data = get_train_val_data(data_folder, regen_data)
    # 获取训练集和验证集的标题、文件、作者和类别的列表
    train_zip_list = get_title_files_author_categories_zip_list(train_data)
    val_zip_list = get_title_files_author_categories_zip_list(val_data)

    # 创建分类器配置对象，指定公共类别列表
    classifier_config = CategoryConfig(common.categories)
    # 获取作者映射数据和文件映射数据
    author_map = get_author_map(data_folder, regen_data)
    file_map = get_file_map(data_folder, regen_data)
    
    # 使用 XLM-R 模型和获取的映射数据初始化 CommitClassifier，并将其移动到指定设备（如 GPU）
    commit_classifier = CommitClassifier(
        XLMR_BASE, author_map, file_map, classifier_config
    ).to(device)

    # 根据训练数据生成类别权重
    class_weights = gen_class_weights(train_zip_list)
    
    # 使用交叉熵损失函数初始化损失函数，并传入类别权重
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # 使用 Adam 优化器来优化 CommitClassifier 的参数，设置学习率为 3e-3
    optimizer = torch.optim.Adam(commit_classifier.parameters(), lr=3e-3)

    # 定义训练周期数和批量大小
    num_epochs = 25
    batch_size = 256

    # 如果需要重新采样数据
    if resample:
        # 不使用重新平衡数据的功能
        train_zip_list = balance_dataset(train_zip_list)
        
    # 计算训练数据集的大小
    data_size = len(train_zip_list)

    # 打印训练样本数量信息
    print(f"Training on {data_size} examples.")
    
    # 生成验证集的批次数据，将所有验证数据装载为一个批次
    val_batch = generate_batch(val_zip_list)

    # 开始训练周期循环，使用 tqdm 进行进度条显示
    for i in tqdm(range(num_epochs), desc="Epochs"):
        start = 0
        # 随机打乱训练数据集
        random.shuffle(train_zip_list)
        while start < data_size:
            end = start + batch_size
            # 如果最后一个批次需要更大则调整大小
            if end > data_size:
                end = data_size
            # 获取当前批次的训练数据并生成对应的批次数据
            train_batch = train_zip_list[start:end]
            train_batch = generate_batch(train_batch)
            # 执行训练步骤，更新模型参数
            l = train_step(train_batch, commit_classifier, optimizer, loss)
            start = end

        # 执行验证步骤，计算验证集的损失
        val_l = eval_step(val_batch, commit_classifier, loss)
        
        # 打印当前训练周期的训练损失和验证损失
        tqdm.write(
            f"Finished epoch {i} with a train loss of: {l.item()} and a val_loss of: {val_l.item()}"
        )

    # 在不进行梯度计算的情况下，设置 CommitClassifier 为评估模式
    with torch.no_grad():
        commit_classifier.eval()
        # 获取验证集输入数据和目标数据
        val_inpts, val_targets = val_batch
        # 使用模型进行验证集预测
        val_output = commit_classifier(val_inpts)
        # 计算验证集的预测类别，并计算准确率
        val_preds = torch.argmax(val_output, dim=1)
        val_acc = torch.sum(val_preds == val_targets).item() / len(val_preds)
        # 打印最终的验证准确率
        print(f"Final Validation accuracy is {val_acc}")

    # 打印保存路径信息，并将模型参数保存到指定路径
    print(f"Jobs done! Saving to {save_path}")
    torch.save(commit_classifier.state_dict(), save_path)
# 主函数，程序的入口点
def main():
    # 创建命令行参数解析器，设置工具的描述信息
    parser = argparse.ArgumentParser(
        description="Tool to create a classifier for helping to categorize commits"
    )

    # 添加命令行参数选项：--train，用于指示是否训练一个新的分类器
    parser.add_argument("--train", action="store_true", help="Train a new classifier")

    # 添加命令行参数选项：--commit_data_folder，用于指定提交数据的文件夹路径
    parser.add_argument("--commit_data_folder", default="results/classifier/")

    # 添加命令行参数选项：--save_path，用于指定分类器模型的保存路径
    parser.add_argument(
        "--save_path", default="results/classifier/commit_classifier.pt"
    )

    # 添加命令行参数选项：--regen_data，如果设置则重新生成训练数据，有助于在标记更多示例并希望重新训练时使用
    parser.add_argument(
        "--regen_data",
        action="store_true",
        help="Regenerate the training data, helps if labeled more examples and want to re-train.",
    )

    # 添加命令行参数选项：--resample，如果设置则重新采样训练数据以达到平衡（仅在安装了imblearn时有效）
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Resample the training data to be balanced. (Only works if imblearn is installed.)",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 如果传入了 --train 参数，则执行训练函数并提前返回
    if args.train:
        train(
            Path(args.save_path),
            Path(args.commit_data_folder),
            args.regen_data,
            args.resample,
        )
        return

    # 否则打印提示信息，说明当前文件仅用于训练新的分类器，需要传入 --train 参数来执行训练
    print(
        "Currently this file only trains a new classifier please pass in --train to train a new classifier"
    )


# 如果当前脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```