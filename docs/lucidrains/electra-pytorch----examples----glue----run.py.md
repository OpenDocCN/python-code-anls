# `.\lucidrains\electra-pytorch\examples\glue\run.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 Google AI Language Team Authors 和 HuggingFace Inc. 团队所有，以及 NVIDIA 公司所有
# 根据 Apache 许可证 2.0 版本使用此文件，详细信息请访问 http://www.apache.org/licenses/LICENSE-2.0
# 除非符合许可证规定或书面同意，否则不得使用此文件
# 根据许可证规定，软件按"原样"分发，不提供任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

""" 在 GLUE 上对库模型进行序列分类微调（Bert、XLM、XLNet、RoBERTa、Albert、XLM-RoBERTa）。"""

# 导入所需的库
import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# 导入自定义的计算指标函数
from metrics import glue_compute_metrics as compute_metrics
# 导入数据处理函数
from processors import glue_convert_examples_to_features as convert_examples_to_features
# 导入输出模式
from processors import glue_output_modes as output_modes
# 导入处理器
from processors import glue_processors as processors
# 导入任务标签数量
from processors import glue_tasks_num_labels as task_num_labels

# 设置日志记录器
logger = logging.getLogger(__name__)

##################################################
# 适配 Google 风格的 GLUE 代码

# Tokenizer 适配器类
class TokenizerAdapter:
    def __init__(self, tokenizer, pad_token, cls_token="[CLS]", sep_token="[SEP]"):
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token

    # 将 tokens 转换为 ids
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    # 截断序列
    def truncate_sequences(
        self,
        ids,
        pair_ids,
        num_tokens_to_remove,
        truncation_strategy,
        stride,
    ):
        # 确保 ids 的长度大于要移除的 tokens 数量
        assert len(ids) > num_tokens_to_remove
        # 计算窗口长度
        window_len = min(len(ids), stride + num_tokens_to_remove)
        # 获取溢出的 tokens
        overflowing_tokens = ids[-window_len:]
        # 截断 ids
        ids = ids[:-num_tokens_to_remove]

        return (ids, pair_ids, overflowing_tokens)
    # 对输入文本进行编码，生成输入的 token ids 和 token type ids
    def encode_plus(self, text, text_pair, add_special_tokens, max_length, return_token_type_ids):

        # 对第一个文本进行 tokenization，转换成 token ids
        token_ids_0 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        len_ids = len(token_ids_0)
        # 如果有第二个文本，则对其进行 tokenization，转换成 token ids
        if text_pair:
            token_ids_1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text_pair))
            len_pair_ids = len(token_ids_1)
        else:
            token_ids_1 = None
            len_pair_ids = 0

 
        # 截断文本
        assert add_special_tokens
        num_special_tokens_to_add = (2 if not text_pair else 3)
        total_len = len_ids + len_pair_ids + num_special_tokens_to_add
        # 如果总长度超过最大长度，则进行截断
        if max_length and total_len > max_length:
            token_ids_0, token_ids_1, overflowing_tokens = self.truncate_sequences(
                token_ids_0,
                pair_ids=token_ids_1,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy='only_first', # TODO(nijkamp): is this the correct truncation strategy for all GLUE tasks?
                stride=0,
            )


        # 添加特殊 token
        cls = [self.tokenizer.vocab[self.cls_token]]
        sep = [self.tokenizer.vocab[self.sep_token]]

        if not text_pair:

            input_ids = cls + token_ids_0 + sep
            token_type_ids = len(cls + token_ids_0 + sep) * [0]

        else:

            input_ids = cls + token_ids_0 + sep + token_ids_1 + sep
            token_type_ids = len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

        assert len(input_ids) <= max_length

        # 返回编码结果
        return {"input_ids": input_ids, "token_type_ids": token_type_ids}

    # 返回 tokenizer 的词汇表长度
    def __len__(self):
        return len(self.tokenizer.vocab)

    # 保存预训练模型到指定目录
    def save_pretrained(self, outputdir):
        pass
# 将给定的 tokenizer 和 pad_token 封装成 TokenizerAdapter 对象并返回
def wrap_tokenizer(tokenizer, pad_token):
    return TokenizerAdapter(tokenizer, pad_token)


##################################################
# distilled Google-like/HF glue code

# 设置随机种子，确保实验的可重复性
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# 创建一个学习率调度器，包括线性增加和线性减少学习率
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

# 训练模型
def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    # 设置训练批次大小
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # 创建训练数据采样器
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # 创建训练数据加载器
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # 计算总的训练步数
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # 准备优化器和调度器（线性增加和减少学习率）
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # 检查是否存在保存的优化器或调度器状态
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # 加载优化器和调度器状态
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))

    # 如果启用混合精度训练
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # 多 GPU 训练（应在 apex fp16 初始化之后）
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 分布式训练（应在 apex fp16 初始化之后）
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # 开始训练
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # 打印训练批次的总大小（包括并行、分布式和累积），根据参数计算得出
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    # 打印梯度累积步数
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # 打印总优化步数
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # 检查是否从检查点继续训练
    if os.path.exists(args.model_name_or_path):
        # 将 global_step 设置为模型路径中最后一个保存检查点的 global_step
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    # 将模型梯度置零
    model.zero_grad()
    # 创建训练迭代器
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # 为了可重现性而添加在这里
    # 返回 global_step 和 tr_loss/global_step
    return global_step, tr_loss / global_step
def evaluate(args, model, tokenizer, prefix=""):
    # 循环处理 MNLI 双重评估（匹配，不匹配）
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # 注意 DistributedSampler 会随机采样
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # 多 GPU 评估
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # 评估！
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
            print(preds)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key]))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # 确保在分布式训练中只有第一个进程处理数据集，其他进程将使用缓存

    processor = processors[task]()
    output_mode = output_modes[task]
    # 从缓存或数据集文件加载数据特征
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    # 检查缓存文件是否存在且不覆盖缓存时
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        # 输出日志信息，加载缓存文件中的特征
        logger.info("Loading features from cached file %s", cached_features_file)
        # 从缓存文件中加载特征数据
        features = torch.load(cached_features_file)
    else:
        # 输出日志信息，从数据集文件中创建特征
        logger.info("Creating features from dataset file at %s", args.data_dir)
        # 获取标签列表
        label_list = processor.get_labels()
        # 如果任务是 mnli 或 mnli-mm 且模型类型是 roberta 或 xlmroberta
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(在 RoBERTa 预训练模型中交换标签索引)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        # 获取示例数据
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        # 将示例转换为特征
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=False,  # 在 xlnet 中左侧填充
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        # 如果本地进程的索引为 -1 或 0
        if args.local_rank in [-1, 0]:
            # 输出日志信息，将特征保存到缓存文件中
            logger.info("Saving features into cached file %s", cached_features_file)
            # 将特征保存到缓存文件中
            torch.save(features, cached_features_file)

    # 如果本地进程的索引为 0 且不是评估模式
    if args.local_rank == 0 and not evaluate:
        # 确保只有分布式训练中的第一个进程处理数据集，其他进程将使用缓存
        torch.distributed.barrier()

    # 转换为张量并构建数据集
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    # 如果输出模式是分类
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    # 如果输出模��是回归
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    # 构建张量数据集
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
# 定义主函数，设置默认参数task='MRPC', seed=42, ckpt='output/pretrain/2020-08-28-02-41-37/ckpt/60000'
def main(task='MRPC', seed=42, ckpt='output/pretrain/2020-08-28-02-41-37/ckpt/60000'):
    # 创建参数解析器
    parser = argparse.ArgumentParser()

    # 必需参数
    # 指定输入数据目录，应包含任务的.tsv文件（或其他数据文件）
    parser.add_argument(
        "--data_dir",
        default=f'data/glue_data/{task}',
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    # 模型类型，默认为"bert"
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
    )
    # 模型名称或路径，默认为ckpt
    parser.add_argument(
        "--model_name_or_path",
        default=ckpt,
        type=str,
    )
    # 词汇表路径，默认为'data/vocab.txt'
    parser.add_argument(
        "--vocab_path",
        default='data/vocab.txt',
        type=str,
    )
    # 任务名称，默认为task
    parser.add_argument(
        "--task_name",
        default=task,
        type=str,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    # 输出目录，默认为'output/glue'
    parser.add_argument(
        "--output_dir",
        default='output/glue',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # 其他参数
    # 缓存目录，默认为空字符串
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    # 最大序列长度，默认为128
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    # 是否进行训练，默认为True
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    # 是否在开发集上进行评估，默认为True
    parser.add_argument("--do_eval", default=True, help="Whether to run eval on the dev set.")
    # 训练期间是否进行评估，默认为True
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    # 是否使用小写模型，默认为True
    parser.add_argument(
        "--do_lower_case", default=True, help="Set this flag if you are using an uncased model.",
    )

    # 每个GPU/CPU的训练批次大小，默认为32
    parser.add_argument(
        "--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.",
    )
    # 每个GPU/CPU的评估批次大小，默认为8
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    # 累积梯度更新的步数，默认为1
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # Adam优化器的初始学习率，默认为2e-5
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    # 权重衰减，默认为0.0
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    # Adam优化器��epsilon值，默认为1e-8
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    # 最大梯度范数，默认为1.0
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # 总训练周期数，默认为3.0
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    # 最大步数，默认为-1
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    # 线性预热步数，默认为0
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    # 每X次更新步骤记录一次日志，默认为500
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    # 每X次更新步骤保存一次检查点，默认为500
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    # 添加一个参数，用于评估所有具有与 model_name 相同前缀和以步数结尾的检查点
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    # 添加一个参数，用于避免在可用时使用 CUDA
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    # 添加一个参数，用于覆盖输出目录的内容
    parser.add_argument(
        "--overwrite_output_dir", default=True, help="Overwrite the content of the output directory",
    )
    # 添加一个参数，用于覆盖缓存的训练和评估集
    parser.add_argument(
        "--overwrite_cache", default=True, help="Overwrite the cached training and evaluation sets",
    )
    # 添加一个参数，用于初始化随机种子
    parser.add_argument("--seed", type=int, default=seed, help="random seed for initialization")

    # 添加一个参数，用于指定是否使用 16 位（混合）精度
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    # 添加一个参数，用于指定 fp16 的优化级别
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    # 添加一个参数，用于分布式训练中的本地排名
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # 添加一个参数，用于远程调试的服务器 IP 地址
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    # 添加一个参数，用于远程调试的服务器端口
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    # 解析参数
    args = parser.parse_args()

    # 如果输出目录已存在且不为空，并且需要训练且不覆盖输出目录，则引发 ValueError
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # 如果需要远程调试，则设置远程调试
    if args.server_ip and args.server_port:
        # 远程调试 - 参考 https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # 设置 CUDA、GPU 和分布式训练
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    args.device = device

    # 设置日志记录
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # 设置随机种子
    set_seed(args)

    # 准备 GLUE 任务
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # 加载预训练模型和分词器
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # 确保只有分布式训练中的第一个进程会下载模型和词汇表

    from transformers import AutoConfig, AutoModelForSequenceClassification
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # 从预训练模型中加载自动序列分类模型
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # 导入自定义的新标记器
    from pretraining.openwebtext.dataset import new_tokenizer
    # 使用新标记器包装标记器，并设置填充标记
    tokenizer = wrap_tokenizer(new_tokenizer(args.vocab_path), pad_token='[PAD]')

    # 如果本地进程的排名为0，则执行分布式训练中的同步操作
    if args.local_rank == 0:
        torch.distributed.barrier()  # 确保只有分布式训练中的第一个进程会下载模型和词汇表

    # 将模型移动到指定设备
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # 训练
    if args.do_train:
        # 加载并缓存训练数据集示例
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        # 训练模型并获取全局步数和训练损失
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # 保存最佳实践：如果使用默认名称为模型，则可以使用from_pretrained()重新加载它
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # 如果需要，创建输出目录
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # 保存训练后的模型、配置和标记器使用`save_pretrained()`方法
        # 可以使用`from_pretrained()`重新加载它们
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # 处理分布式/并行训练
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # 良好的实践：将训练参数与训练后的模型一起保存
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # 加载已经微调的训练模型和词汇表
        model = model_to_save
        # TODO(nijkamp): 我们忽略模型序列化
        # model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        # tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # 评估
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # TODO(nijkamp): 我们忽略模型序列化
        # tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # 减少日志记录
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            # TODO(nijkamp): 我们忽略模型序列化
            # model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results
# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```