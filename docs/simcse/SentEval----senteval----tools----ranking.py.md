# `.\SentEval\senteval\tools\ranking.py`

```
"""
Image Annotation/Search for COCO with Pytorch
"""
# 导入必要的库和模块
from __future__ import absolute_import, division, unicode_literals

import logging  # 导入日志记录模块
import copy  # 导入复制模块
import numpy as np  # 导入NumPy库

import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块
from torch.autograd import Variable  # 导入自动求导变量
import torch.optim as optim  # 导入优化器模块


class COCOProjNet(nn.Module):
    """
    COCOProjNet类，用于定义COCO投影网络模型
    """
    def __init__(self, config):
        """
        初始化函数，定义网络结构

        Args:
        - config (dict): 包含模型配置参数的字典，包括imgdim、sentdim、projdim等
        """
        super(COCOProjNet, self).__init__()
        self.imgdim = config['imgdim']  # 图像维度
        self.sentdim = config['sentdim']  # 句子维度
        self.projdim = config['projdim']  # 投影维度
        # 图像投影层
        self.imgproj = nn.Sequential(
                        nn.Linear(self.imgdim, self.projdim),
                        )
        # 句子投影层
        self.sentproj = nn.Sequential(
                        nn.Linear(self.sentdim, self.projdim),
                        )

    def forward(self, img, sent, imgc, sentc):
        """
        前向传播函数，执行数据处理和计算

        Args:
        - img (torch.Tensor): 主图像数据张量，大小为(bsize, imgdim)
        - sent (torch.Tensor): 主句子数据张量，大小为(bsize, sentdim)
        - imgc (torch.Tensor): 对比图像数据张量，大小为(bsize, ncontrast, imgdim)
        - sentc (torch.Tensor): 对比句子数据张量，大小为(bsize, ncontrast, sentdim)

        Returns:
        - anchor1 (torch.Tensor): 锚点1的投影结果
        - anchor2 (torch.Tensor): 锚点2的投影结果
        - img_sentc (torch.Tensor): 图像与对比句子的投影结果
        - sent_imgc (torch.Tensor): 句子与对比图像的投影结果
        """
        # 对主图像进行扩展和变形
        img = img.unsqueeze(1).expand_as(imgc).contiguous()
        img = img.view(-1, self.imgdim)
        imgc = imgc.view(-1, self.imgdim)
        # 对主句子进行扩展和变形
        sent = sent.unsqueeze(1).expand_as(sentc).contiguous()
        sent = sent.view(-1, self.sentdim)
        sentc = sentc.view(-1, self.sentdim)

        # 计算主图像和主句子的投影
        imgproj = self.imgproj(img)
        imgproj = imgproj / torch.sqrt(torch.pow(imgproj, 2).sum(1, keepdim=True)).expand_as(imgproj)
        imgcproj = self.imgproj(imgc)
        imgcproj = imgcproj / torch.sqrt(torch.pow(imgcproj, 2).sum(1, keepdim=True)).expand_as(imgcproj)
        sentproj = self.sentproj(sent)
        sentproj = sentproj / torch.sqrt(torch.pow(sentproj, 2).sum(1, keepdim=True)).expand_as(sentproj)
        sentcproj = self.sentproj(sentc)
        sentcproj = sentcproj / torch.sqrt(torch.pow(sentcproj, 2).sum(1, keepdim=True)).expand_as(sentcproj)

        # 计算锚点投影结果
        anchor1 = torch.sum((imgproj*sentproj), 1)
        anchor2 = torch.sum((sentproj*imgproj), 1)
        img_sentc = torch.sum((imgproj*sentcproj), 1)
        sent_imgc = torch.sum((sentproj*imgcproj), 1)

        # 返回投影结果
        return anchor1, anchor2, img_sentc, sent_imgc

    def proj_sentence(self, sent):
        """
        对句子进行投影

        Args:
        - sent (torch.Tensor): 输入的句子张量，大小为(bsize, sentdim)

        Returns:
        - output (torch.Tensor): 投影后的结果，大小为(bsize, projdim)
        """
        output = self.sentproj(sent)
        output = output / torch.sqrt(torch.pow(output, 2).sum(1, keepdim=True)).expand_as(output)
        return output  # 返回投影后的句子结果，大小为(bsize, projdim)

    def proj_image(self, img):
        """
        对图像进行投影

        Args:
        - img (torch.Tensor): 输入的图像张量，大小为(bsize, imgdim)

        Returns:
        - output (torch.Tensor): 投影后的结果，大小为(bsize, projdim)
        """
        output = self.imgproj(img)
        output = output / torch.sqrt(torch.pow(output, 2).sum(1, keepdim=True)).expand_as(output)
        return output  # 返回投影后的图像结果，大小为(bsize, projdim)


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss，用于计算损失函数
    """
    def __init__(self, margin):
        """
        初始化函数

        Args:
        - margin (float): 损失函数的边界值
        """
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
    # 定义前向传播函数，计算损失函数
    def forward(self, anchor1, anchor2, img_sentc, sent_imgc):
        # 计算句子到图像的损失，使用 margin ranking loss，并进行逐元素求和
        cost_sent = torch.clamp(self.margin - anchor1 + img_sentc,
                                min=0.0).sum()
        # 计算图像到句子的损失，同样使用 margin ranking loss，并进行逐元素求和
        cost_img = torch.clamp(self.margin - anchor2 + sent_imgc,
                               min=0.0).sum()
        # 总损失为句子损失和图像损失的和
        loss = cost_sent + cost_img
        # 返回计算得到的总损失
        return loss
# 定义一个 ImageSentenceRankingPytorch 类，用于在 COCO 数据集上使用 Pytorch 实现图像与句子的排序任务
class ImageSentenceRankingPytorch(object):
    # 初始化方法，接受训练、验证和测试数据集以及配置参数
    def __init__(self, train, valid, test, config):
        # 设置随机种子
        self.seed = config['seed']
        np.random.seed(self.seed)  # 设置 NumPy 随机种子
        torch.manual_seed(self.seed)  # 设置 Pytorch 随机种子
        torch.cuda.manual_seed(self.seed)  # 设置 CUDA 随机种子

        # 初始化训练、验证和测试数据集
        self.train = train
        self.valid = valid
        self.test = test

        # 获取图像特征维度和句子特征维度
        self.imgdim = len(train['imgfeat'][0])  # 图像特征的维度
        self.sentdim = len(train['sentfeat'][0])  # 句子特征的维度

        # 从配置中获取投影维度和损失函数的边界值
        self.projdim = config['projdim']  # 投影维度
        self.margin = config['margin']  # 损失函数的边界值

        # 设置批处理大小、负样本数量和最大迭代次数以及是否启用早停
        self.batch_size = 128  # 批处理大小
        self.ncontrast = 30  # 负样本数量
        self.maxepoch = 20  # 最大迭代次数
        self.early_stop = True  # 是否启用早停

        # 配置模型参数
        config_model = {'imgdim': self.imgdim, 'sentdim': self.sentdim,
                        'projdim': self.projdim}
        self.model = COCOProjNet(config_model).cuda()  # 创建并移动模型到 GPU

        # 定义损失函数为 PairwiseRankingLoss，并将其移动到 GPU
        self.loss_fn = PairwiseRankingLoss(margin=self.margin).cuda()

        # 使用 Adam 优化器优化模型参数
        self.optimizer = optim.Adam(self.model.parameters())

    # 数据准备方法，接受训练集和验证集的文本和图像特征，并将它们转换为 Pytorch 的 Tensor 类型
    def prepare_data(self, trainTxt, trainImg, devTxt, devImg,
                     testTxt, testImg):
        trainTxt = torch.FloatTensor(trainTxt)  # 转换训练文本特征为 FloatTensor
        trainImg = torch.FloatTensor(trainImg)  # 转换训练图像特征为 FloatTensor
        devTxt = torch.FloatTensor(devTxt).cuda()  # 转换验证文本特征为 FloatTensor 并移动到 GPU
        devImg = torch.FloatTensor(devImg).cuda()  # 转换验证图像特征为 FloatTensor 并移动到 GPU
        testTxt = torch.FloatTensor(testTxt).cuda()  # 转换测试文本特征为 FloatTensor 并移动到 GPU
        testImg = torch.FloatTensor(testImg).cuda()  # 转换测试图像特征为 FloatTensor 并移动到 GPU

        return trainTxt, trainImg, devTxt, devImg, testTxt, testImg
    # 定义一个方法来训练模型，使用给定的训练和开发数据集，执行指定数量的 epochs
    def trainepoch(self, trainTxt, trainImg, devTxt, devImg, nepoches=1):
        # 将模型设置为训练模式
        self.model.train()
        # 对每个 epoch 进行迭代
        for _ in range(self.nepoch, self.nepoch + nepoches):
            # 随机排列训练文本的索引
            permutation = list(np.random.permutation(len(trainTxt)))
            # 用于记录所有批次的损失
            all_costs = []
            # 对训练数据进行批处理
            for i in range(0, len(trainTxt), self.batch_size):
                # 每处理 self.batch_size*500 个数据点输出一次日志信息
                if i % (self.batch_size*500) == 0 and i > 0:
                    logging.info('samples : {0}'.format(i))
                    # 计算并记录开发集上的 Image to Text 评估指标
                    r1_i2t, r5_i2t, r10_i2t, medr_i2t = self.i2t(devImg, devTxt)
                    logging.info("Image to text: {0}, {1}, {2}, {3}".format(
                        r1_i2t, r5_i2t, r10_i2t, medr_i2t))
                    # 计算并记录开发集上的 Text to Image 评估指标
                    r1_t2i, r5_t2i, r10_t2i, medr_t2i = self.t2i(devImg, devTxt)
                    logging.info("Text to Image: {0}, {1}, {2}, {3}".format(
                        r1_t2i, r5_t2i, r10_t2i, medr_t2i))
                # 从排列好的索引中取出当前批次的索引
                idx = torch.LongTensor(permutation[i:i + self.batch_size])
                # 将训练图像数据和文本数据封装成变量，并放到 GPU 上
                imgbatch = Variable(trainImg.index_select(0, idx)).cuda()
                sentbatch = Variable(trainTxt.index_select(0, idx)).cuda()

                # 随机选择用于对比的图像和文本的索引
                idximgc = np.random.choice(permutation[:i] +
                                           permutation[i + self.batch_size:],
                                           self.ncontrast*idx.size(0))
                idxsentc = np.random.choice(permutation[:i] +
                                            permutation[i + self.batch_size:],
                                            self.ncontrast*idx.size(0))
                idximgc = torch.LongTensor(idximgc)
                idxsentc = torch.LongTensor(idxsentc)
                # 将对比图像和文本的索引封装成变量，并重新组织成指定维度的张量，并放到 GPU 上
                imgcbatch = Variable(trainImg.index_select(0, idximgc)).view(
                    -1, self.ncontrast, self.imgdim).cuda()
                sentcbatch = Variable(trainTxt.index_select(0, idxsentc)).view(
                    -1, self.ncontrast, self.sentdim).cuda()

                # 调用模型进行前向传播
                anchor1, anchor2, img_sentc, sent_imgc = self.model(
                    imgbatch, sentbatch, imgcbatch, sentcbatch)
                # 计算损失
                loss = self.loss_fn(anchor1, anchor2, img_sentc, sent_imgc)
                # 记录当前批次的损失值
                all_costs.append(loss.data.item())
                # 清除优化器的梯度信息
                self.optimizer.zero_grad()
                # 反向传播，计算梯度
                loss.backward()
                # 更新模型参数
                self.optimizer.step()
        # 更新当前 epoch 数量
        self.nepoch += nepoches
    def t2i(self, images, captions):
        """
        Images: (5N, imgdim) matrix of images
        Captions: (5N, sentdim) matrix of captions
        """
        # 使用 torch.no_grad() 上下文管理器，确保在评估模式下执行，不计算梯度
        with torch.no_grad():
            # 初始化空列表用于存储图像和句子的嵌入向量
            img_embed, sent_embed = [], []

            # 按照批次大小迭代处理图像数据
            for i in range(0, len(images), self.batch_size):
                # 对图像进行投影，并将结果添加到 img_embed 列表中
                img_embed.append(self.model.proj_image(
                    Variable(images[i:i + self.batch_size])))
                
                # 对句子进行投影，并将结果添加到 sent_embed 列表中
                sent_embed.append(self.model.proj_sentence(
                    Variable(captions[i:i + self.batch_size])))
            
            # 将列表中的张量连接成一个大的张量，并提取其中的数据部分
            img_embed = torch.cat(img_embed, 0).data
            sent_embed = torch.cat(sent_embed, 0).data

            # 计算图像嵌入向量的数量
            npts = int(img_embed.size(0) / 5)
            
            # 创建一个 LongTensor，包含从 0 到 len(img_embed) 的索引，步长为 5
            idxs = torch.cuda.LongTensor(range(0, len(img_embed), 5))
            
            # 使用索引从 img_embed 中选择特定的图像嵌入向量
            ims = img_embed.index_select(0, idxs)

            # 初始化一个全零数组用于存储排名结果
            ranks = np.zeros(5 * npts)

            # 对每个样本进行处理，计算其与所有图像的相似度得分
            for index in range(npts):
                # 获取当前样本对应的查询句子嵌入向量
                queries = sent_embed[5*index: 5*index + 5]

                # 计算查询句子嵌入向量与所有图像嵌入向量之间的相似度得分
                scores = torch.mm(queries, ims.transpose(0, 1)).cpu().numpy()

                # 初始化一个全零数组用于存储排序后的索引
                inds = np.zeros(scores.shape)

                # 对每个相似度得分向量进行处理，获取其排序后的索引
                for i in range(len(inds)):
                    inds[i] = np.argsort(scores[i])[::-1]

                    # 找到查询结果中与当前索引匹配的位置，并存储在 ranks 数组中
                    ranks[5 * index + i] = np.where(inds[i] == index)[0][0]

            # 计算检索指标：r1、r5、r10、medr
            r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
            r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
            r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
            medr = np.floor(np.median(ranks)) + 1
            
            # 返回评估指标元组
            return (r1, r5, r10, medr)
    def i2t(self, images, captions):
        """
        Images: (5N, imgdim) matrix of images
        Captions: (5N, sentdim) matrix of captions
        """
        with torch.no_grad():
            # Project images and captions
            img_embed, sent_embed = [], []
            # 分批次处理图片和标题向量
            for i in range(0, len(images), self.batch_size):
                # 将批次的图片投影到模型空间
                img_embed.append(self.model.proj_image(
                    Variable(images[i:i + self.batch_size])))
                # 将批次的标题投影到模型空间
                sent_embed.append(self.model.proj_sentence(
                    Variable(captions[i:i + self.batch_size])))
            # 将列表转换为张量，并提取数据部分
            img_embed = torch.cat(img_embed, 0).data
            sent_embed = torch.cat(sent_embed, 0).data

            # 计算图片数量
            npts = int(img_embed.size(0) / 5)
            index_list = []

            # 初始化排名数组
            ranks = np.zeros(npts)
            for index in range(npts):

                # 获取查询图片
                query_img = img_embed[5 * index]

                # 计算得分
                scores = torch.mm(query_img.view(1, -1),
                                  sent_embed.transpose(0, 1)).view(-1)
                scores = scores.cpu().numpy()
                inds = np.argsort(scores)[::-1]
                index_list.append(inds[0])

                # 计算排名
                rank = 1e20
                for i in range(5*index, 5*index + 5, 1):
                    tmp = np.where(inds == i)[0][0]
                    if tmp < rank:
                        rank = tmp
                ranks[index] = rank

            # 计算评价指标
            r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
            r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
            r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
            medr = np.floor(np.median(ranks)) + 1
            return (r1, r5, r10, medr)
```