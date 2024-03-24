# `.\lucidrains\reformer-pytorch\pretraining\self-supervised.py`

```
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm

from reformer_pytorch import Reformer, ReformerLM
from transformers import BertTokenizer, PreTrainedTokenizer
from fairseq.optim.adafactor import Adafactor
import os
import json
import logging
from datetime import datetime

# 定义一个自定义数据集类，用于处理Wiki数据集
class WikiDataset(Dataset):

    def __init__(self, path="", prefix="train"):
        # 确保给定的路径是一个目录
        assert os.path.isdir(path)

        self.documents = []
        filename_list = os.listdir(path)
        # 遍历目录下的文件
        for file in filename_list:
            path_to_file = os.path.join(path, file)
            # 如果不是文件则跳过
            if not os.path.isfile(path_to_file):
                continue
            self.documents.append(path_to_file)

    def __len__(self):
        """ Returns the number of documents. """
        return len(self.documents)

    def __getitem__(self, idx):
        document_path = self.documents[idx]
        document_name = document_path.split("/")[-1]

        items = []

        with open(document_path, encoding="utf-8") as source:
            raw_text = source.readlines()
            # 读取每个文档中的文本内容
            for obj in raw_text:
                text = json.loads(obj)['text']
                # 替换文本中的换行符和多余空格
                text = re.sub('\\n', ' ', text)
                text = re.sub('\\s+', ' ', text)
                items.append(text)

        return items

# 定义一个Reformer模型训练器类
class ReformerTrainer(object):

    def __init__(self,
                 dataset,
                 model,
                 tokenizer,
                 device=None,
                 train_batch_size=8,
                 eval_batch_size=None,
                 tb_writer=True,
                 tb_dir='./tb_logs',
                 log_dir='./logs'):
        """
        Provides an easy to use class for pretraining and evaluating a Reformer Model.

        :param dataset: (torch.utils.data.Dataset) containing all of the data you wish to utilize during training.
        :param model: (reformer_pytorch.Reformer)
        :param tokenizer: (transformers.PreTrainedTokenizer) defaults to BertTokenizer ('bert-base-case')
        :param device: provide manual device placement. If None, will default to cuda:0 if available.
        :param tb_writer: (bool) Whether to write to tensorboard or not.
        :param tb_dir: (str) Where to write TB logs to.
        :param log_dir: (str) Where to write generic logs to.
        """

        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tb_writer = tb_writer
        self.log_dir = log_dir

        # 如果未提供tokenizer，则使用默认的BertTokenizer
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        # 如果未提供device，则根据是否有cuda选择设备
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # 如果未提供eval_batch_size，则使用train_batch_size
        if eval_batch_size is None:
            self.eval_batch_size = train_batch_size

        # 如果需要写入tensorboard，则初始化SummaryWriter
        if tb_writer:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=tb_dir)

        # 配置日志记录
        logging.basicConfig(filename=f'{log_dir}/{datetime.now().date()}.log', level=logging.INFO)
    def build_dataloaders(self, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
        """
        Builds the Training and Eval DataLoaders

        :param train_test_split: The ratio split of test to train data.
        :param train_shuffle: (bool) True if you wish to shuffle the train_dataset.
        :param eval_shuffle: (bool) True if you wish to shuffle the eval_dataset.
        :return: train dataloader and evaluation dataloader.
        """
        # 获取数据集的长度
        dataset_len = len(self.dataset)
        # 计算用于评估的数据集长度
        eval_len = int(dataset_len * train_test_split)
        # 计算用于训练的数据集长度
        train_len = dataset_len - eval_len
        # 随机划分数据集为训练集和评估集
        train_dataset, eval_dataset = random_split(self.dataset, (train_len, eval_len))
        # 创建训练数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=train_shuffle)
        # 创建评估数据加载器
        eval_loader = DataLoader(eval_dataset, batch_size=self.eval_batch_size, shuffle=eval_shuffle)
        # 记录日志信息
        logging.info(f'''train_dataloader size: {len(train_loader.dataset)} | shuffle: {train_shuffle}
                         eval_dataloader size: {len(eval_loader.dataset)} | shuffle: {eval_shuffle}''')
        # 返回训练数据加载器和评估数据加载器
        return train_loader, eval_loader

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability=0.15, pad=True):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        # 复制输入作为标签
        labels = inputs.clone()
        # 创建概率矩阵，用于控制MASK的概率
        probability_matrix = torch.full(labels.shape, mlm_probability)
        # 获取特殊标记的掩码
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        # 根据特殊标记掩码更新概率矩阵
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        # 如果存在填充标记，将填充标记的位置概率设为0
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        # 生成MASK的索引
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # 将非MASK的标记设为-100，用于计算损失
        labels[~masked_indices] = -100

        # 80%的情况下，用[MASK]替换MASK的输入标记
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10%的情况下，用随机词替换MASK的输入标记
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 如果需要填充，将输入和标签填充到最大长度
        if pad:
            input_pads = self.tokenizer.max_len - inputs.shape[-1]
            label_pads = self.tokenizer.max_len - labels.shape[-1]

            inputs = F.pad(inputs, pad=(0, input_pads), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, pad=(0, label_pads), value=self.tokenizer.pad_token_id)

        # 剩余的10%的情况下，保持MASK的输入标记不变
        return inputs, labels

    def _tokenize_input_ids(self, input_ids: list, pad_to_max_length: bool = True):
        """
        Helper function to clean up the train and eval functions
        :param input_ids: inputs to tokenize.
        :param pad_to_max_length: Whether you want to pad the inputs to the tokenizer.max_len
        :return: Tensor containing training data.
        """
        # 将输入ID列表转换为张量
        inputs = torch.cat(
            [
                self.tokenizer.encode(
                    input_ids[i],
                    add_special_tokens=True,
                    max_length=self.tokenizer.max_len,
                    pad_to_max_length=pad_to_max_length,
                    return_tensors='pt'
                ) \
                for i in range(len(input_ids))
            ]
        )
        return inputs
    def evaluate(self, dataloader):
        """
        Runs through the provided dataloader with torch.no_grad()
        :param dataloader: (torch.utils.data.DataLoader) Evaluation DataLoader
        :return: None
        """
        # 定义交叉熵损失函数
        loss_fn = nn.CrossEntropyLoss()

        # 如果有多个 GPU 并且模型不是 nn.DataParallel 类型，则使用 nn.DataParallel 包装模型
        if self.n_gpu > 1 and not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        # 将模型设置为评估模式
        self.model.eval()
        eval_loss = 0.0
        perplexity = 0.0
        eval_steps = 0

        # 记录当前时间并输出评估信息
        logging.info(f'{datetime.now()} | Evaluating...')
        # 遍历数据加载器中的每个批次数据
        for step, batch in tqdm(enumerate(dataloader), desc='Evaluating', leave=True, total=len(dataloader)):
            # 遍历批次中的每个数据
            for data in batch:
                # 对输入数据进行标记化处理，并填充到最大长度
                inputs = self._tokenize_input_ids(data, pad_to_max_length=True)
                # 对输入数据进行掩码处理
                inputs, labels = self.mask_tokens(inputs)
                # 将输入数据和标签移动到设备上
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 使用 torch.no_grad() 禁用梯度计算
                with torch.no_grad():
                    # 获取模型的输出
                    output = self.model(inputs)

                # 计算损失的掩码
                loss_mx = labels != -100
                output_ids = output[loss_mx].view(-1, self.tokenizer.vocab_size)
                labels = labels[loss_mx].view(-1)
                # 计算临时评估损失和困惑度
                tmp_eval_loss = loss_fn(output_ids, labels)
                tmp_perplexity = torch.exp(tmp_eval_loss)

                # 如果有多个 GPU，则计算平均损失
                if self.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()

                # 累加评估损失和困惑度
                eval_loss += tmp_eval_loss.item()
                perplexity += tmp_perplexity.item()
                eval_steps += 1

            # 计算平均评估损失和困惑度
            eval_loss /= eval_steps
            perplexity /= eval_steps

            # 如果有 TensorBoard 写入器，则记录评估损失和困惑度
            if self.tb_writer:
                self.writer.add_scalar('Eval/Loss', eval_loss, eval_steps)
                self.writer.close()
                self.writer.add_scalar('Perplexity', perplexity, eval_steps)
                self.writer.close()
            # 输出评估信息
            logging.info(f'{datetime.now()} | Step: {step} | Eval Loss: {eval_loss} | Perplexity: {perplexity}')

        return None
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 创建一个WikiDataset对象，指定数据集路径
    dataset = WikiDataset(path='D:/data/enwiki')
    # 从预训练的bert-base-cased模型中加载分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # 设置分词器的最大长度为128
    tokenizer.max_len = 128
    # 创建一个ReformerLM模型对象，设置相关参数
    model = ReformerLM(
        num_tokens=tokenizer.vocab_size,
        dim=512,
        depth=6,
        heads=8,
        max_seq_len=tokenizer.max_len,
        causal=True
    )
    # 创建一个ReformerTrainer对象，传入数据集、模型、分词器等参数
    trainer = ReformerTrainer(dataset, model, tokenizer, train_batch_size=32, eval_batch_size=32)
    # 构建训练集和验证集的数据加载器
    train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.90)
    # 训练模型，返回训练后的模型
    model = trainer.train(epochs=3,
                          train_dataloader=train_dataloader,
                          eval_dataloader=eval_dataloader,
                          log_steps=10,
                          ckpt_steps=100,
                          ckpt_dir='./ckpts',
                          gradient_accumulation_steps=1)
    # 保存训练后的模型到指定路径
    torch.save(model, './ckpts/model.bin')
```