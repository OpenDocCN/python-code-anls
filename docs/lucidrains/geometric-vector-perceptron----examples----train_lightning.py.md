# `.\lucidrains\geometric-vector-perceptron\examples\train_lightning.py`

```
import gc
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from loguru import logger
from pytorch_lightning.callbacks import (
    GPUStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from examples.data_handler import kabsch_torch, scn_cloud_mask
from examples.data_utils import (
    encode_whole_bonds,
    encode_whole_protein,
    from_encode_to_pred,
    prot_covalent_bond,
)
from examples.scn_data_module import ScnDataModule
from geometric_vector_perceptron.geometric_vector_perceptron import GVP_Network

# 定义一个继承自 LightningModule 的结构模型类
class StructureModel(pl.LightningModule):
    # 静态方法，用于添加模型特定参数
    @staticmethod
    def add_model_specific_args(parent_parser):
        # 创建参数解析器
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # 添加模型参数
        parser.add_argument("--depth", type=int, default=4)
        parser.add_argument("--cutoffs", type=float, default=1.0)
        parser.add_argument("--noise", type=float, default=1.0)
        # 添加优化器和调度器参数
        parser.add_argument("--init_lr", type=float, default=1e-3)

        return parser

    # 初始化方法，接受模型参数
    def __init__(
        self,
        depth: int = 1,
        cutoffs: float = 1.0,
        noise: float = 1.0,
        init_lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__()

        # 保存超参数
        self.save_hyperparameters()
        
        # 定义需要的信息字典
        self.needed_info = {
            "cutoffs": [cutoffs], # -1e-3 for just covalent, "30_closest", 5. for under 5A, etc
            "bond_scales": [1, 2, 4],
            "aa_pos_scales": [1, 2, 4, 8, 16, 32, 64, 128],
            "atom_pos_scales": [1, 2, 4, 8, 16, 32],
            "dist2ca_norm_scales": [1, 2, 4],
            "bb_norms_atoms": [0.5],  # will encode 3 vectors with this
        }

        # 创建 GVP_Network 模型
        self.model = GVP_Network(
            n_layers=depth,
            feats_x_in=48,
            vectors_x_in=7,
            feats_x_out=48,
            vectors_x_out=7,
            feats_edge_in=8,
            vectors_edge_in=1,
            feats_edge_out=8,
            vectors_edge_out=1,
            embedding_nums=[36, 20],
            embedding_dims=[16, 16],
            edge_embedding_nums=[2],
            edge_embedding_dims=[2],
            residual=True,
            recalc=1
        )

        self.noise = noise
        self.init_lr = init_lr

        self.baseline_losses = [] # 存储基准损失
        self.epoch_losses = [] # 存储每个 epoch 的损失
    # 定义前向传播函数，接受序列、真实坐标、角度、填充序列、掩码作为输入
    def forward(self, seq, true_coords, angles, padding_seq, mask):
        # 获取需要的信息
        needed_info = self.needed_info
        # 获取设备信息
        device = true_coords.device

        # 将序列截取到填充序列之前的部分
        needed_info["seq"] = seq[: (-padding_seq) or None]
        # 计算蛋白质的共价键
        needed_info["covalent_bond"] = prot_covalent_bond(needed_info["seq"])

        # 对整个蛋白质进行编码
        pre_target = encode_whole_protein(
            seq,
            true_coords,
            angles,
            padding_seq,
            needed_info=needed_info,
            free_mem=True,
        )
        pre_target_x, _, _, embedd_info = pre_target

        # 对蛋白质进行编码并加入噪声
        encoded = encode_whole_protein(
            seq,
            true_coords + self.noise * torch.randn_like(true_coords),
            angles,
            padding_seq,
            needed_info=needed_info,
            free_mem=True,
        )

        x, edge_index, edge_attrs, embedd_info = encoded

        # 创建批次信息
        batch = torch.tensor([0 for i in range(x.shape[0])], device=x.device).long()

        # 添加位置坐标
        cloud_mask = scn_cloud_mask(seq[: (-padding_seq) or None]).to(device)
        chain_mask = mask[: (-padding_seq) or None].unsqueeze(-1) * cloud_mask
        flat_chain_mask = rearrange(chain_mask.bool(), "l c -> (l c)")
        cloud_mask = cloud_mask.bool()
        flat_cloud_mask = rearrange(cloud_mask, "l c -> (l c)")

        # 部分重新计算边
        recalc_edge = partial(
            encode_whole_bonds,
            x_format="encode",
            embedd_info=embedd_info,
            needed_info=needed_info,
            free_mem=True,
        )

        # 预测
        scores = self.model.forward(
            x,
            edge_index,
            batch=batch,
            edge_attr=edge_attrs,
            recalc_edge=recalc_edge,
            verbose=False,
        )

        # 格式化预测、基线和目标
        target = from_encode_to_pred(
            pre_target_x, embedd_info=embedd_info, needed_info=needed_info
        )
        pred = from_encode_to_pred(
            scores, embedd_info=embedd_info, needed_info=needed_info
        )
        base = from_encode_to_pred(x, embedd_info=embedd_info, needed_info=needed_info)

        # 计算误差

        # 选项1：损失是输出令牌的均方误差
        # loss_ = (target-pred)**2
        # loss  = loss_.mean()

        # 选项2：损失是重构坐标的RMSD
        target_coords = target[:, 3:4] * target[:, :3]
        pred_coords = pred[:, 3:4] * pred[:, :3]
        base_coords = base[:, 3:4] * base[:, :3]

        ## 对齐 - 有时svc失败 - 不知道为什么
        try:
            pred_aligned, target_aligned = kabsch_torch(pred_coords.t(), target_coords.t()) # (3, N)
            base_aligned, _ = kabsch_torch(base_coords.t(), target_coords.t())
            loss = ( (pred_aligned.t() - target_aligned.t())[flat_chain_mask[flat_cloud_mask]]**2 ).mean()**0.5 
            loss_base = ( (base_aligned.t() - target_aligned.t())[flat_chain_mask[flat_cloud_mask]]**2 ).mean()**0.5 
        except:
            pred_aligned, target_aligned = None, None
            print("svd failed convergence, ep:", ep)
            loss = ( (pred_coords.t() - target_coords.t())[flat_chain_mask[flat_cloud_mask]]**2 ).mean()**0.5
            loss_base = ( (base_coords - target_coords)[flat_chain_mask[flat_cloud_mask]]**2 ).mean()**0.5 

        # 释放GPU内存
        del true_coords, angles, pre_target_x, edge_index, edge_attrs
        del scores, target_coords, pred_coords, base_coords
        del encoded, pre_target, target_aligned, pred_aligned
        gc.collect()

        # 返回损失
        return {"loss": loss, "loss_base": loss_base}

    # 配置优化器
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        return optimizer

    # 训练开始时的操作
    def on_train_start(self) -> None:
        self.baseline_losses = []
        self.epoch_losses = []
    # 训练步骤，接收一个批次数据和批次索引
    def training_step(self, batch, batch_idx):
        # 调用前向传播函数得到输出
        output = self.forward(**batch)
        # 获取损失值和基准损失值
        loss = output["loss"]
        loss_base = output["loss_base"]

        # 如果损失值为空或为 NaN，则返回 None
        if loss is None or torch.isnan(loss):
            return None

        # 将损失值和基准损失值添加到对应的列表中
        self.epoch_losses.append(loss.item())
        self.baseline_losses.append(loss_base.item())

        # 记录训练损失值到日志中，显示在进度条中
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_loss_base", output["loss_base"], on_epoch=True, prog_bar=False)

        # 返回损失值
        return loss

    # 训练结束时的操作
    def on_train_end(self) -> None:
        # 创建一个图形窗口
        plt.figure(figsize=(15, 6))
        # 设置图形标题
        plt.title(
            f"Loss Evolution - Denoising of Gaussian-masked Coordinates (mu=0, sigma={self.noise})"
        )
        # 绘制训练损失值随时间的变化曲线

        # 绘制滑动窗口平均值曲线
        for window in [8, 16, 32]:
            # 计算滑动窗口平均值
            plt.plot(
                [
                    np.mean(self.epoch_losses[:window][0 : i + 1])
                    for i in range(min(window, len(self.epoch_losses))
                ]
                + [
                    np.mean(self.epoch_losses[i : i + window + 1])
                    for i in range(len(self.epoch_losses) - window)
                ],
                label="Window mean n={0}".format(window),
            )

        # 绘制基准损失值的水平虚线
        plt.plot(
            np.ones(len(self.epoch_losses)) * np.mean(self.baseline_losses),
            "k--",
            label="Baseline",
        )

        # 设置 x 轴范围
        plt.xlim(-0.01 * len(self.epoch_losses), 1.01 * len(self.epoch_losses))
        # 设置 y 轴标签
        plt.ylabel("RMSD")
        # 设置 x 轴标签
        plt.xlabel("Batch number")
        # 添加图例
        plt.legend()
        # 保存图形为 PDF 文件
        plt.savefig("loss.pdf")

    # 验证步骤，接收一个批次数据和批次索引
    def validation_step(self, batch, batch_idx):
        # 调用前向传播函数得到输出，并记录验证损失值到日志中
        output = self.forward(**batch)
        self.log("val_loss", output["loss"], on_epoch=True, sync_dist=True)
        self.log("val_loss_base", output["loss_base"], on_epoch=True, sync_dist=True)

    # 测试步骤，接收一个批次数据和批次索引
    def test_step(self, batch, batch_idx):
        # 调用前向传播函数得到输出，并记录测试损失值到日志中
        output = self.forward(**batch)
        self.log("test_loss", output["loss"], on_epoch=True, sync_dist=True)
        self.log("test_loss_base", output["loss_base"], on_epoch=True, sync_dist=True)
# 根据参数获取训练器对象
def get_trainer(args):
    # 设置随机种子
    pl.seed_everything(args.seed)

    # 创建日志记录器
    root_dir = Path(args.default_root_dir).expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    tb_save_dir = root_dir / "tb"
    tb_logger = TensorBoardLogger(save_dir=tb_save_dir)
    loggers = [tb_logger]
    logger.info(f"Run tensorboard --logdir {tb_save_dir}")

    # 创建回调函数
    ckpt_cb = ModelCheckpoint(verbose=True)
    lr_cb = LearningRateMonitor(logging_interval="step")
    pb_cb = ProgressBar(refresh_rate=args.progress_bar_refresh_rate)
    callbacks = [lr_cb, pb_cb]

    callbacks.append(ckpt_cb)

    gpu_cb = GPUStatsMonitor()
    callbacks.append(gpu_cb)

    plugins = []
    # 根据参数创建训练器对象
    trainer = pl.Trainer.from_argparse_args(
        args, logger=loggers, callbacks=callbacks, plugins=plugins
    )

    return trainer


def main(args):
    # 创建数据模块对象
    dm = ScnDataModule(**vars(args))
    # 创建模型对象
    model = StructureModel(**vars(args))
    # 获取训练器对象
    trainer = get_trainer(args)
    # 训练模型
    trainer.fit(model, datamodule=dm)
    # 测试模型并获取指标
    metrics = trainer.test(model, datamodule=dm)
    print("test", metrics)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=23333, help="Seed everything.")

    # 添加模型特定参数
    parser = StructureModel.add_model_specific_args(parser)

    # 添加数据特定参数
    parser = ScnDataModule.add_data_specific_args(parser)

    # 添加训练器参数
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # 打印参数
    pprint(vars(args))
    # 执行主函数
    main(args)
```