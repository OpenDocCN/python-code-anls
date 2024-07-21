# `.\pytorch\benchmarks\dynamo\training_loss.py`

```
# 导入命令行参数解析模块
import argparse
# 导入检查模块，用于检查源码
import inspect
# 导入操作系统功能模块
import os
# 导入系统特定的功能和参数的模块
import sys
# 导入时间模块
import time
# 导入时间增量模块
from datetime import timedelta

# 从 datasets 库中导入加载数据集的函数
from datasets import load_dataset, load_metric
# 从 transformers 库中导入自动模型和分词器的类
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 导入 PyTorch 深度学习框架
import torch

# 导入 Torch 内部特定的动态库
import torch._dynamo
# 从 Torch 工具模块中导入数据加载器类
from torch.utils.data import DataLoader

# 设置 Torch 后端允许使用 TF32 模式进行 CUDA 矩阵乘法计算
torch.backends.cuda.matmul.allow_tf32 = True

# 提示用户如果运行完整的训练/评估示例可能会下载约 84G 的数据集
print("You will download around 84G dataset if you run this end to end training/evaluation example.")

# 设置环境变量，禁用 Tokenizers 的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 根据 CUDA 是否可用选择设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def data_processing(num_samples, batch_size):
    # 加载 Yelp 全评论数据集
    dataset = load_dataset("yelp_review_full")
    # 根据 BERT-base-cased 模型加载对应的分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # 定义分词函数
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # 对数据集应用分词函数，以批处理方式处理
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 移除数据集中的文本列，并将标签列重命名为 labels
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # 将数据集格式设置为 Torch 格式
    tokenized_datasets.set_format("torch")

    # 选择部分训练集和评估集
    small_train_dataset = tokenized_datasets["train"].select(range(num_samples))
    small_eval_dataset = tokenized_datasets["test"].select(range(num_samples))

    # 创建训练数据加载器和评估数据加载器
    train_dataloader = DataLoader(small_train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader


def training_iter_fn(batch, model, optimizer):
    # 使用模型进行前向传播和计算损失
    outputs = model(**batch)
    loss = outputs.loss
    # 反向传播和优化器更新
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def model_training_evaluation(
    backend, train_dataloader, eval_dataloader, model, optimizer, num_epochs, evaluation
):
    # 将模型移动到指定设备（CUDA 或 CPU）
    model.to(device)
    # 设置模型为训练模式
    model.train()
    # 初始化损失记录列表
    loss_history = []
    
    # 根据后端类型选择优化训练迭代函数
    if not backend:
        # 使用原生 PyTorch 运行
        opt_training_iter_fn = training_iter_fn
    else:
        # 支持的后端类型包括: eager, aot_eager, aot_nvfuser 和 inductor
        opt_training_iter_fn = torch._dynamo.optimize(backend)(training_iter_fn)
    
    # 进行指定轮数的训练循环
    for epoch in range(num_epochs):
        running_loss = 0.0
        # 遍历训练数据加载器中的每个批次
        for i, batch in enumerate(train_dataloader, 0):
            # 将批次数据移动到指定设备
            batch = {k: v.to(device) for k, v in batch.items()}
            # 使用优化后的训练迭代函数计算损失
            loss = opt_training_iter_fn(batch, model, optimizer)
            # 记录损失值
            running_loss += loss.item()
            # 每处理100个批次记录平均损失到历史记录
            if i % 100 == 99:
                loss_history.append(running_loss / 100)
                running_loss = 0.0
    # 如果进行评估
    if evaluation:
        # 加载精度度量指标模块，这里使用"accuracy"指标
        metric = load_metric("accuracy")
        # 将模型设置为评估模式，这通常会关闭 dropout 等训练时特有的行为
        model.eval()
        
        # 如果没有指定后端优化，则直接使用原始模型
        if not backend:
            opt_model = model
        else:
            # 使用指定的后端优化模型
            opt_model = torch._dynamo.optimize(backend)(model)
        
        # 遍历评估数据加载器的每个批次
        for batch in eval_dataloader:
            # 将批次数据中的每个张量移到指定的设备上（通常是 GPU）
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 在评估模式下，不计算梯度，以提高性能
            with torch.no_grad():
                # 使用优化后的模型计算输出结果
                outputs = opt_model(**batch)

            # 从输出结果中获取逻辑回归值
            logits = outputs.logits
            # 计算预测值，取最大概率对应的类别
            predictions = torch.argmax(logits, dim=-1)
            
            # 将预测值和真实标签添加到精度度量器中，用于后续计算精度
            metric.add_batch(predictions=predictions, references=batch["labels"])

        # 返回损失历史和通过度量器计算的评估指标值
        return loss_history, metric.compute()
    
    else:
        # 如果不进行评估，直接返回损失历史和空的评估指标值
        return loss_history, None
# 检查参考损失和结果损失的长度是否相等
def check_loss(ref_loss, res_loss):
    assert len(ref_loss) == len(res_loss)
    # 获取损失列表的长度
    length = len(ref_loss)
    # 取长度和 10 的最小值，作为下文要使用的数值 x
    x = min(length, 10)
    # 检查最近 x 个结果损失的平均值是否小于等于最近 x 个参考损失的平均值加上 1e-1
    if sum(res_loss[-x:]) / 10 <= sum(ref_loss[-x:]) / 10 + 1e-1:
        return True
    else:
        return False


# 解析命令行参数
def parse_args():
    # 创建参数解析器对象，并设置描述信息
    parser = argparse.ArgumentParser(
        description="TorchDynamo end to end training/evaluation benchmark"
    )
    # 添加命令行参数 --epochs，指定训练的轮数，默认为 10
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    # 添加命令行参数 --num-samples，指定训练或评估的样本数，默认为 1000
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="number of samples to train/eval (default: 1000)",
    )
    # 添加命令行参数 --batch-size，指定训练的批量大小，默认为 8
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )
    # 添加命令行参数 --lr，指定学习率，默认为 5e-5
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="learning rate (default: 5e-5)"
    )
    # 添加命令行参数 --backend，选择训练/评估模型所用的后端，默认为 "inductor"
    parser.add_argument(
        "--backend",
        choices=torch._dynamo.list_backends(exclude_tags=None),
        default="inductor",
        help="train/evaluate model with a given backend (default: inductor)",
    )
    # 添加命令行参数 --optimizer，选择训练模型所用的优化器，默认为 "Adam"
    parser.add_argument(
        "--optimizer",
        default="Adam",
        help="train model using a given optimizer (default: Adam)",
    )
    # 添加命令行参数 --evaluation，是否在模型训练后进行评估，默认为 False
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="running evaluation after model training",
    )
    # 解析命令行参数并返回结果
    args = parser.parse_args()
    return args


# 主函数，程序的入口点
def main():
    # 解析命令行参数
    args = parse_args()
    # 数据处理，获取训练集和评估集的数据加载器
    train_dataloader, eval_dataloader = data_processing(
        args.num_samples, args.batch_size
    )
    # 使用预训练的 "bert-base-cased" 模型创建分类模型
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    # 根据命令行参数选择优化器类，并根据是否支持 capturable 参数进行实例化
    optimizer_cls = getattr(sys.modules["torch.optim"], args.optimizer)
    if "capturable" in inspect.signature(optimizer_cls).parameters.keys():
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, capturable=True)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    # 记录本地时间，用于计算本地训练时间
    native_start = time.time()
    # 使用原生 PyTorch 进行模型训练和评估，获取参考损失和准确率
    ref_loss, accuracy = model_training_evaluation(
        None,
        train_dataloader,
        eval_dataloader,
        model,
        optimizer,
        args.epochs,
        args.evaluation,
    )
    # 记录本地训练结束时间
    native_end = time.time()
    # 使用 TorchDynamo 进行模型训练和评估，获取结果损失和准确率
    res_loss, accuracy = model_training_evaluation(
        args.backend,
        train_dataloader,
        eval_dataloader,
        model,
        optimizer,
        args.epochs,
        args.evaluation,
    )
    # 记录 TorchDynamo 训练结束时间
    dynamo_end = time.time()
    # 检查结果损失是否小于等于参考损失，若是则打印通过信息，否则打印失败信息
    if check_loss(ref_loss, res_loss):
        print(
            "[PASSED] TorchDynamo end to end training loss is less than or equal to native PyTorch"
        )
    else:
        print(
            "[FAILED] TorchDynamo end to end training loss is greater than native Pytorch"
        )
    # 若需要评估模型，则打印模型准确率信息
    if args.evaluation:
        print(f"Model accuracy: {accuracy}")
    # 计算本地训练时间和 TorchDynamo 训练时间
    native_elapsed = native_end - native_start
    dynamo_elapsed = dynamo_end - native_end
    # 打印使用的参数信息，包括训练周期数、后端和优化器
    print(
        f"Train model on {args.epochs} epochs with backend {args.backend} and optimizer {args.optimizer}:"
    )
    # 打印每个 epoch 在 PyTorch 上的平均耗时
    print(f"PyTorch spent {timedelta(seconds=native_elapsed/args.epochs)} per epoch")
    # 打印每个 epoch 在 TorchDynamo 上的平均耗时
    print(
        f"TorchDynamo spent {timedelta(seconds=dynamo_elapsed/args.epochs)} per epoch"
    )
# 如果当前脚本作为主程序执行（而不是作为模块被导入），则执行 main() 函数
if __name__ == "__main__":
    main()
```