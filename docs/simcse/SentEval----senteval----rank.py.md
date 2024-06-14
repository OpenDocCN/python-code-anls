# `.\SentEval\senteval\rank.py`

```
'''
Image-Caption Retrieval with COCO dataset
'''
# 引入必要的库和模块
from __future__ import absolute_import, division, unicode_literals

import os
import sys
import logging
import numpy as np

try:
    import cPickle as pickle  # 尝试导入 cPickle，Python 2.x 中的 pickle 实现
except ImportError:
    import pickle  # Python 3.x 中使用标准库 pickle

# 导入自定义模块 ImageSentenceRankingPytorch，用于图像-句子检索的评估
from senteval.tools.ranking import ImageSentenceRankingPytorch

class ImageCaptionRetrievalEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task: Image Caption Retrieval *****\n\n')

        # 设置随机种子
        self.seed = seed
        # 加载数据集文件，并保存在 coco_data 字典中
        train, dev, test = self.loadFile(task_path)
        self.coco_data = {'train': train, 'dev': dev, 'test': test}

    def do_prepare(self, params, prepare):
        # 获取训练、开发和测试集中的所有句子样本
        samples = self.coco_data['train']['sent'] + \
                  self.coco_data['dev']['sent'] + \
                  self.coco_data['test']['sent']
        # 调用 prepare 函数准备数据集样本
        prepare(params, samples)

    def loadFile(self, fpath):
        # 初始化 coco 字典用于存储数据集
        coco = {}

        for split in ['train', 'valid', 'test']:
            list_sent = []
            list_img_feat = []
            if sys.version_info < (3, 0):
                # 使用 pickle 加载数据文件（Python 2.x）
                with open(os.path.join(fpath, split + '.pkl')) as f:
                    cocodata = pickle.load(f)
            else:
                # 使用 pickle 加载数据文件（Python 3.x），指定 Latin-1 编码
                with open(os.path.join(fpath, split + '.pkl'), 'rb') as f:
                    cocodata = pickle.load(f, encoding='latin1')

            # 遍历每个图像键
            for imgkey in range(len(cocodata['features'])):
                # 检查每个图像至少有 5 个相关的描述
                assert len(cocodata['image_to_caption_ids'][imgkey]) >= 5, \
                       cocodata['image_to_caption_ids'][imgkey]
                # 对每个图像的前 5 个描述进行处理
                for captkey in cocodata['image_to_caption_ids'][imgkey][0:5]:
                    # 获取清理后的描述文本，并在末尾添加标点符号
                    sent = cocodata['captions'][captkey]['cleaned_caption']
                    sent += ' .'  # 在 COCO 数据集中，句子末尾添加标点
                    # 将处理后的句子编码为 UTF-8 字节并分割存入列表
                    list_sent.append(sent.encode('utf-8').split())
                    # 将图像特征添加到列表中
                    list_img_feat.append(cocodata['features'][imgkey])
            
            # 确保句子列表和图像特征列表长度相等，且每个图像有 5 个描述
            assert len(list_sent) == len(list_img_feat) and \
                len(list_sent) % 5 == 0
            # 将图像特征列表转换为 numpy 数组，数据类型为 float32
            list_img_feat = np.array(list_img_feat).astype('float32')
            # 将处理后的数据存入 coco 字典中的相应分割集合
            coco[split] = {'sent': list_sent, 'imgfeat': list_img_feat}
        
        return coco['train'], coco['valid'], coco['test']
    def run(self, params, batcher):
        # 初始化嵌入字典，包含训练、开发和测试集的句子和图像特征
        coco_embed = {'train': {'sentfeat': [], 'imgfeat': []},
                      'dev': {'sentfeat': [], 'imgfeat': []},
                      'test': {'sentfeat': [], 'imgfeat': []}}

        # 遍历 self.coco_data 中的每个键（可能是'train', 'dev', 'test'）
        for key in self.coco_data:
            logging.info('Computing embedding for {0}'.format(key))
            # 将每个键对应的'sent'值转换为 NumPy 数组，并按内容排序
            self.coco_data[key]['sent'] = np.array(self.coco_data[key]['sent'])
            self.coco_data[key]['sent'], idx_sort = np.sort(self.coco_data[key]['sent']), np.argsort(self.coco_data[key]['sent'])
            idx_unsort = np.argsort(idx_sort)

            # 初始化当前键对应的句子特征列表
            coco_embed[key]['X'] = []
            nsent = len(self.coco_data[key]['sent'])
            # 按照批量大小 params.batch_size，对当前键的句子进行批处理
            for ii in range(0, nsent, params.batch_size):
                batch = self.coco_data[key]['sent'][ii:ii + params.batch_size]
                # 使用 batcher 函数计算批量句子的嵌入
                embeddings = batcher(params, batch)
                coco_embed[key]['sentfeat'].append(embeddings)
            # 根据 idx_unsort 重新排列嵌入特征
            coco_embed[key]['sentfeat'] = np.vstack(coco_embed[key]['sentfeat'])[idx_unsort]
            # 将当前键对应的图像特征转换为 NumPy 数组
            coco_embed[key]['imgfeat'] = np.array(self.coco_data[key]['imgfeat'])
            logging.info('Computed {0} embeddings'.format(key))

        # 设置模型的配置参数
        config = {'seed': self.seed, 'projdim': 1000, 'margin': 0.2}
        # 初始化图像-句子排名模型，使用训练、开发和测试集的嵌入数据和配置
        clf = ImageSentenceRankingPytorch(train=coco_embed['train'],
                                          valid=coco_embed['dev'],
                                          test=coco_embed['test'],
                                          config=config)

        # 运行模型并获取最佳开发集分数及测试集评估指标
        bestdevscore, r1_i2t, r5_i2t, r10_i2t, medr_i2t, \
            r1_t2i, r5_t2i, r10_t2i, medr_t2i = clf.run()

        # 记录图像到文本和文本到图像的测试集评估指标
        logging.debug("\nTest scores | Image to text: \
            {0}, {1}, {2}, {3}".format(r1_i2t, r5_i2t, r10_i2t, medr_i2t))
        logging.debug("Test scores | Text to image: \
            {0}, {1}, {2}, {3}\n".format(r1_t2i, r5_t2i, r10_t2i, medr_t2i))

        # 返回结果字典，包括开发集准确度、测试集评估指标和数据集大小信息
        return {'devacc': bestdevscore,
                'acc': [(r1_i2t, r5_i2t, r10_i2t, medr_i2t),
                        (r1_t2i, r5_t2i, r10_t2i, medr_t2i)],
                'ndev': len(coco_embed['dev']['sentfeat']),
                'ntest': len(coco_embed['test']['sentfeat'])}
```