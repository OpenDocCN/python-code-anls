# `.\SentEval\examples\gensen.py`

```py
"""
Clone GenSen repo here: https://github.com/Maluuba/gensen.git
And follow instructions for loading the model used in batcher
"""

# 从未来版本导入必需的绝对导入、除法和Unicode字面量支持
from __future__ import absolute_import, division, unicode_literals

# 导入系统相关的库
import sys
import logging
# 导入 GenSen 包
from gensen import GenSen, GenSenSingle

# 设置路径常量
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# 导入 SentEval 包
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# SentEval 的准备函数，此处暂无需执行任何操作
def prepare(params, samples):
    return

# SentEval 的批处理函数
def batcher(params, batch):
    # 将批次中的每个句子转换为字符串形式，如果句子为空则用句点表示
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    # 使用 GenSen 获取句子表示，采用最后一个池化策略，并返回 NumPy 数组形式的嵌入
    _, reps_h_t = gensen.get_representation(
        sentences, pool='last', return_numpy=True, tokenize=True
    )
    embeddings = reps_h_t
    return embeddings

# 加载 GenSen 单模型
gensen_1 = GenSenSingle(
    model_folder='../data/models',
    filename_prefix='nli_large_bothskip',
    pretrained_emb='../data/embedding/glove.840B.300d.h5'
)
gensen_2 = GenSenSingle(
    model_folder='../data/models',
    filename_prefix='nli_large_bothskip_parse',
    pretrained_emb='../data/embedding/glove.840B.300d.h5'
)
# 创建 GenSen 对象，用于生成句子表示
gensen_encoder = GenSen(gensen_1, gensen_2)
# 获取句子表示，使用最后一个池化策略，返回 NumPy 数组形式的表示结果
reps_h, reps_h_t = gensen.get_representation(
    sentences, pool='last', return_numpy=True, tokenize=True
)

# 设置 SentEval 参数
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
params_senteval['gensen'] = gensen_encoder

# 配置日志记录格式和级别
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # 创建 SentEval 引擎对象
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # 定义要评估的传递任务列表
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    # 执行评估并打印结果
    results = se.eval(transfer_tasks)
    print(results)
```