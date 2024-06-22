# `.\SentEval\examples\bow.py`

```py
# 导入必要的模块和库
from __future__ import absolute_import, division, unicode_literals  # 兼容Python2和Python3的未来特性导入
import sys  # 系统相关的操作
import io  # 用于处理IO操作的库
import numpy as np  # 处理数值计算的库
import logging  # 日志记录模块

# 设置路径常量
PATH_TO_SENTEVAL = '../'  # SentEval库的路径
PATH_TO_DATA = '../data'  # 数据集的路径
PATH_TO_VEC = 'fasttext/crawl-300d-2M.vec'  # 词向量文件的路径

# 导入SentEval库
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval  # SentEval库的引入

# 创建词典函数
def create_dictionary(sentences, threshold=0):
    words = {}
    # 遍历每个句子
    for s in sentences:
        # 遍历句子中的每个词
        for word in s:
            words[word] = words.get(word, 0) + 1

    # 如果指定了阈值，筛选出现频率高于阈值的词
    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords

    # 添加特殊标记到词典中
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    # 按词频倒序排序词典
    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # 逆序排序
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id  # 返回词语到ID的映射和ID到词语的映射

# 从词汇表中获取词向量（如GloVe、Word2Vec、FastText等）
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # 如果是Word2Vec或FastText文件，跳过第一行（通常是文件的元数据）
        # next(f)

        # 遍历词向量文件中的每一行
        for line in f:
            word, vec = line.split(' ', 1)
            # 如果词在词汇表中，将其词向量转换为numpy数组保存
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    # 记录找到的词向量数量
    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    
    return word_vec  # 返回词向量字典

# SentEval的准备和批处理函数
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)  # 创建词典并存储在params中
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)  # 获取词向量并存储在params中
    params.wvec_dim = 300  # 设置词向量维度为300
    return

# SentEval的批处理函数
def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]  # 处理空句子，用点号代替
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])  # 获取每个词的词向量
        if not sentvec:
            vec = np.zeros(params.wvec_dim)  # 如果句子中没有词向量，则使用全零向量
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)  # 计算句子的平均向量作为句子向量
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)  # 将所有句子向量堆叠成矩阵
    return embeddings  # 返回批量句子的词向量表示

# 设置SentEval的参数
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# 设置日志记录器的格式和级别
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # 初始化SentEval引擎
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # 定义要评估的任务列表，包括一系列语义相关任务和句子相似性任务
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    # 使用评估器对给定的任务列表进行评估，得到评估结果
    results = se.eval(transfer_tasks)
    # 打印评估结果
    print(results)
```