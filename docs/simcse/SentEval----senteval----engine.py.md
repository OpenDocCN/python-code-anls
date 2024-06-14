# `.\SentEval\senteval\engine.py`

```
'''
Generic sentence evaluation scripts wrapper
'''

# 导入必要的模块和函数
from __future__ import absolute_import, division, unicode_literals
from senteval import utils  # 导入自定义的工具模块
from senteval.binary import CREval, MREval, MPQAEval, SUBJEval  # 导入不同任务的评估类
from senteval.snli import SNLIEval  # 导入 SNLI 任务的评估类
from senteval.trec import TRECEval  # 导入 TREC 任务的评估类
from senteval.sick import SICKEntailmentEval, SICKEval  # 导入 SICK 相关任务的评估类
from senteval.mrpc import MRPCEval  # 导入 MRPC 任务的评估类
from senteval.sts import (  # 导入 STS 相关任务的评估类
    STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval,
    STSBenchmarkEval, SICKRelatednessEval, STSBenchmarkFinetune
)
from senteval.sst import SSTEval  # 导入 SST 任务的评估类
from senteval.rank import ImageCaptionRetrievalEval  # 导入图像标题检索任务的评估类
from senteval.probing import *  # 导入探测任务相关模块

# 定义主类 SE
class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # 初始化函数，接收参数 params, batcher, prepare
        # 将 params 转换为点符号字典对象
        params = utils.dotdict(params)
        # 如果 params 中未设置 usepytorch 参数，则默认为 True
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        # 如果 params 中未设置 seed 参数，则默认为 1111
        params.seed = 1111 if 'seed' not in params else params.seed

        # 如果 params 中未设置 batch_size 参数，则默认为 128
        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        # 如果 params 中未设置 nhid 参数，则默认为 0
        params.nhid = 0 if 'nhid' not in params else params.nhid
        # 如果 params 中未设置 kfold 参数，则默认为 5
        params.kfold = 5 if 'kfold' not in params else params.kfold

        # 如果 params 中未设置 classifier 参数或者为空，则设置一个空的字典作为 classifier 参数
        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        # 断言检查，确保 params.classifier 中包含 nhid 参数
        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        # 将初始化后的 params 赋值给 self.params
        self.params = params

        # 将 batcher 赋值给 self.batcher
        self.batcher = batcher
        # 如果 prepare 不为 None，则将其赋值给 self.prepare，否则设置一个空函数作为 prepare
        self.prepare = prepare if prepare else lambda x, y: None

        # 设置支持的任务列表
        self.list_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                           'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
                           'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13',
                           'STS14', 'STS15', 'STS16',
                           'Length', 'WordContent', 'Depth', 'TopConstituents',
                           'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                           'OddManOut', 'CoordinationInversion', 'SICKRelatedness-finetune', 'STSBenchmark-finetune', 'STSBenchmark-fix']
```