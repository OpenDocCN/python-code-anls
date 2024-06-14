# `.\SentEval\examples\infersent.py`

```py
# 版权声明和许可证信息
# 版权所有 © 2017 至今，Facebook, Inc.
#
# 此源代码根据位于根目录下 LICENSE 文件中的许可证授权。

"""
InferSent 模型。详见 https://github.com/facebookresearch/InferSent.
"""

# 导入必要的库
from __future__ import absolute_import, division, unicode_literals  # 导入 Python 未来支持的特性
import sys  # 导入 sys 模块，用于操作系统相关功能
import os   # 导入 os 模块，用于操作系统相关功能
import torch    # 导入 PyTorch 深度学习库
import logging  # 导入 logging 模块，用于日志记录

# 从 InferSent 仓库中导入 models.py
from models import InferSent

# 设置路径常量
PATH_SENTEVAL = '../'   # SentEval 的根路径
PATH_TO_DATA = '../data'    # 数据集的路径
PATH_TO_W2V = 'PATH/TO/glove.840B.300d.txt'  # 或者 'crawl-300d-2M.vec'（适用于 V2 版本）的路径
MODEL_PATH = 'infersent1.pkl'   # InferSent 模型的路径
V = 1   # InferSent 的版本号

# 断言确保 MODEL_PATH 和 PATH_TO_W2V 文件存在
assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
    '请设置正确的 MODEL 和 GloVe 的路径'

# 导入 senteval 库
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    # 构建词汇表，将样本转换为字符串列表
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch):
    # 将批次中的句子连接成字符串列表
    sentences = [' '.join(s) for s in batch]
    # 使用 InferSent 模型编码句子，生成嵌入向量
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings


"""
在 Transfer Tasks（SentEval）上评估训练好的模型
"""

# 定义 SentEval 的参数
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# 设置日志记录格式
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # 加载 InferSent 模型
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))   # 加载模型的预训练参数
    model.set_w2v_path(PATH_TO_W2V)   # 设置词向量文件的路径

    params_senteval['infersent'] = model.cuda()   # 将模型加载到 CUDA 上进行加速

    se = senteval.engine.SE(params_senteval, batcher, prepare)   # 初始化 SentEval 引擎
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)   # 在所有 Transfer Tasks 上进行评估
    print(results)   # 打印评估结果
```