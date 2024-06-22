# `.\SentEval\examples\models.py`

```py
# 导入必要的库和模块：numpy、time、torch 和 torch 的神经网络模块 nn
import numpy as np
import time
import torch
import torch.nn as nn

# 定义一个名为 InferSent 的神经网络模型，继承自 nn.Module
class InferSent(nn.Module):

    def __init__(self, config):
        super(InferSent, self).__init__()
        # 初始化模型的配置参数
        self.bsize = config['bsize']  # 批量大小
        self.word_emb_dim = config['word_emb_dim']  # 词嵌入维度
        self.enc_lstm_dim = config['enc_lstm_dim']  # LSTM 编码器维度
        self.pool_type = config['pool_type']  # 池化类型（mean 或 max）
        self.dpout_model = config['dpout_model']  # 模型的 dropout 比率
        self.version = 1 if 'version' not in config else config['version']  # 模型版本，如果未指定则默认为 1

        # 创建一个双向 LSTM 编码器
        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=self.dpout_model)

        # 断言检查模型版本是否为 1 或 2
        assert self.version in [1, 2]
        # 根据模型版本初始化特定参数
        if self.version == 1:
            self.bos = '<s>'  # 开始符号
            self.eos = '</s>'  # 结束符号
            self.max_pad = True  # 是否使用最大填充
            self.moses_tok = False  # 是否使用 Moses 分词器
        elif self.version == 2:
            self.bos = '<p>'  # 开始符号
            self.eos = '</p>'  # 结束符号
            self.max_pad = False  # 是否使用最大填充
            self.moses_tok = True  # 是否使用 Moses 分词器

    def is_cuda(self):
        # 判断模型参数是否在 GPU 上
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple):
        # sent_tuple 包含句子和句子长度信息
        sent, sent_len = sent_tuple

        # 根据句子长度降序排序，并记录索引
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        # 根据排序后的索引重新排列输入的句子张量
        idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, idx_sort)

        # 在循环网络中处理填充部分的输入数据
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # 获取 LSTM 的输出
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # 根据原始顺序重新排列输出张量
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, idx_unsort)

        # 池化操作，根据池化类型不同进行不同处理
        if self.pool_type == "mean":
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb
    # 设置词向量路径
    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    # 获取句子列表中的词汇表
    def get_word_dict(self, sentences, tokenize=True):
        # 创建空的词典
        word_dict = {}
        # 对每个句子进行分词处理（如果需要），然后遍历
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                # 如果词不在词典中，加入词典
                if word not in word_dict:
                    word_dict[word] = ''
        # 添加起始和结束符到词典中
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    # 获取词向量
    def get_w2v(self, word_dict):
        # 断言确保已设置词向量路径
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # 创建空的词向量字典
        word_vec = {}
        # 打开词向量文件，读取每一行
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                # 如果词在词汇表中，将其词向量添加到词向量字典中
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        # 打印找到的带有词向量的词数量
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    # 获取前 K 个词向量
    def get_w2v_k(self, K):
        # 断言确保已设置词向量路径
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # 初始化计数器和词向量字典
        k = 0
        word_vec = {}
        # 打开词向量文件，读取每一行
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                # 如果计数器小于等于 K，将词向量添加到词向量字典中
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                # 如果计数器大于 K，检查是否是起始或结束符，如果是，添加其词向量
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')
                # 如果计数器大于 K，并且起始和结束符都在词向量字典中，结束循环
                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    # 构建词汇表
    def build_vocab(self, sentences, tokenize=True):
        # 断言确保已设置词向量路径
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # 获取词汇表
        word_dict = self.get_word_dict(sentences, tokenize)
        # 获取词向量并存储在对象属性中
        self.word_vec = self.get_w2v(word_dict)
        # 打印词汇表大小
        print('Vocab size : %s' % (len(self.word_vec)))

    # 使用前 K 个最频繁出现的词建立词汇表
    def build_vocab_k_words(self, K):
        # 断言确保已设置词向量路径
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # 获取前 K 个词向量并存储在对象属性中
        self.word_vec = self.get_w2v_k(K)
        # 打印词汇表大小
        print('Vocab size : %s' % (K))

    # 更新词汇表
    def update_vocab(self, sentences, tokenize=True):
        # 断言确保已设置词向量路径和已构建的词汇表
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        # 获取词汇表
        word_dict = self.get_word_dict(sentences, tokenize)

        # 仅保留新词汇表中的新词
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # 更新词汇表
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        # 打印新词汇表大小及新增词汇数
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    def get_batch(self, batch):
        # 按照长度降序排列的批量输入
        # batch: (bsize, max_len, word_dim)
        # 创建一个全零的数组，用于存储批量的词嵌入表示
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        # 遍历批量数据
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                # 使用预训练的词向量模型获取当前词的词嵌入表示，并放入对应位置
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def tokenize(self, s):
        # 导入NLTK的分词模块
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            # 使用MOSES分词工具进行分词，处理连字符
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # 修复MOSES分词的问题
            return s.split()
        else:
            # 使用NLTK默认分词器进行分词
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        # 对每个句子进行预处理，添加起始和结束标记
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        # 统计所有句子中的词数
        n_w = np.sum([len(x) for x in sentences])

        # 过滤掉没有词向量的词语
        for i in range(len(sentences)):
            # 只保留存在词向量的词语
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                # 如果整个句子没有任何词有对应的词向量，则发出警告并用结束标记替换
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        # 计算处理后句子的长度数组
        lengths = np.array([len(s) for s in sentences])
        # 统计处理后所有句子的词数
        n_wk = np.sum(lengths)
        if verbose:
            # 输出保留的词数百分比信息
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # 按句子长度降序排列
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        # 根据排列顺序重新排列句子数组
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        # 准备样本数据，包括预处理句子和计算句子长度等信息
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        # 按照指定批量大小遍历句子集合
        for stidx in range(0, len(sentences), bsize):
            # 获取当前批量的词嵌入表示
            batch = self.get_batch(sentences[stidx:stidx + bsize])
            # 如果使用GPU，将数据移到GPU上
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                # 使用模型进行前向推断得到输出，将结果转换为NumPy数组
                batch = self.forward((batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        # 将所有批量的嵌入表示垂直堆叠成一个大的数组
        embeddings = np.vstack(embeddings)

        # 恢复到原始顺序
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            # 输出编码速度信息
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings
    # 定义一个方法 visualize，用于可视化输入句子的词重要性
    def visualize(self, sent, tokenize=True):
        # 如果 tokenize 参数为 True，则调用 self.tokenize 方法对句子进行分词处理
        sent = sent.split() if not tokenize else self.tokenize(sent)
        # 在分词后的句子首尾添加起始符号 self.bos 和结束符号 self.eos
        sent = [[self.bos] + [word for word in sent if word in self.word_vec] + [self.eos]]

        # 如果句子只包含起始和结束符号，则发出警告并替换句子内容
        if ' '.join(sent[0]) == '%s %s' % (self.bos, self.eos):
            import warnings
            warnings.warn('No words in "%s" have w2v vectors. Replacing \
                           by "%s %s"..' % (sent, self.bos, self.eos))
        
        # 调用 get_batch 方法，生成句子的批次数据
        batch = self.get_batch(sent)

        # 如果当前环境支持 CUDA 加速，并且模型已经加载到 CUDA 上，则将 batch 数据转移到 CUDA 上
        if self.is_cuda():
            batch = batch.cuda()

        # 对输入的句子进行编码，得到输出 output
        output = self.enc_lstm(batch)[0]
        
        # 在 output 中找到每个批次中的最大值及其索引
        output, idxs = torch.max(output, 0)
        
        # 将 idxs 转换为 numpy 数组
        idxs = idxs.data.cpu().numpy()
        
        # 计算每个词的重要性
        argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

        # 可视化模型结果
        import matplotlib.pyplot as plt
        x = range(len(sent[0]))
        y = [100.0 * n / np.sum(argmaxs) for n in argmaxs]
        plt.xticks(x, sent[0], rotation=45)
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()

        # 返回 output 和 idxs
        return output, idxs
```