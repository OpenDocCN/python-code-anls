# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\modules\prediction.py`

```py
# 导入PyTorch库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入PyTorch中的函数库
import torch.nn.functional as F

# 如果CUDA可用，则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义注意力机制的类
class
    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        # 获取批量大小
        batch_size = batch_H.size(0)
        # 确定序列长度（加1是为了考虑句子末尾的[s]标记）
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        # 初始化输出隐藏状态为零张量
        output_hiddens = (torch.FloatTensor(
            batch_size, num_steps, self.hidden_size).fill_(0).to(device))
        
        # 初始化解码器的隐藏状态为零张量
        hidden = (
            torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
            torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
        )

        # 如果是训练阶段
        if is_train:
            # 逐步处理序列中的每个字符
            for i in range(num_steps):
                # 将文本字符索引转换为独热向量
                char_onehots = self._char_to_onehot(text[:, i],
                                                    onehot_dim=self.num_classes)
                # 使用注意力机制处理解码器的隐藏状态和编码器的上下文特征
                hidden, alpha = self.attention_cell(hidden, batch_H,
                                                    char_onehots)
                # 将当前时间步的隐藏状态存储到输出中
                output_hiddens[:, i, :] = hidden[
                    0]  # LSTM hidden index (0: hidden, 1: Cell)
            # 通过生成器生成输出概率分布
            probs = self.generator(output_hiddens)

        # 如果是推理阶段
        else:
            # 初始化目标序列，以[GO] token开始
            targets = torch.LongTensor(batch_size).fill_(0).to(
                device)  # [GO] token
            # 初始化概率张量为零张量
            probs = (torch.FloatTensor(batch_size, num_steps,
                                       self.num_classes).fill_(0).to(device))

            # 逐步生成输出概率分布
            for i in range(num_steps):
                # 将目标字符索引转换为独热向量
                char_onehots = self._char_to_onehot(targets,
                                                    onehot_dim=self.num_classes)
                # 使用注意力机制处理解码器的隐藏状态和编码器的上下文特征
                hidden, alpha = self.attention_cell(hidden, batch_H,
                                                    char_onehots)
                # 通过生成器生成当前时间步的输出概率分布
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                # 选择下一个输入字符作为下一个时间步的目标
                _, next_input = probs_step.max(1)
                targets = next_input

        # 返回最终的输出概率分布
        return probs  # batch_size x num_steps x num_classes
class AttentionCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # 将 batch_H 投影到隐藏空间，形状为 [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        # 将上一个隐藏状态投影到隐藏空间，并增加维度以匹配 batch_H_proj 的形状
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        # 计算注意力分数 e
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # 形状为 batch_size x num_encoder_step x 1

        # 对注意力分数 e 进行 softmax 操作，沿着 num_encoder_step 维度
        alpha = F.softmax(e, dim=1)
        # 计算加权后的上下文向量 context
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # 形状为 batch_size x hidden_size
        # 将上下文向量和字符的 one-hot 编码拼接起来
        concat_context = torch.cat([context, char_onehots], 1)  # 形状为 batch_size x (hidden_size + num_embeddings)
        # 使用当前的上下文和隐藏状态更新当前隐藏状态 cur_hidden
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
```