# `.\lucidrains\byol-pytorch\examples\lightning\train.py`

```
# 导入所需的库
import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

# 导入自定义的 BYOL 模块和 pytorch lightning 模块
from byol_pytorch import BYOL
import pytorch_lightning as pl

# 加载预训练的 resnet 50 模型
resnet = models.resnet50(pretrained=True)

# 解析命令行参数
parser = argparse.ArgumentParser(description='byol-lightning-test')
parser.add_argument('--image_folder', type=str, required=True,
                    help='path to your folder of images for self-supervised learning')
args = parser.parse_args()

# 定义常量
BATCH_SIZE = 32
EPOCHS = 1000
LR = 3e-4
NUM_GPUS = 2
IMAGE_SIZE = 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()

# 定义 pytorch lightning 模块
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

# 定义处理灰度图像的函数
def expand_greyscale(t):
    return t.expand(3, -1, -1)

# 定义图像数据集类
class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

# 主程序入口
if __name__ == '__main__':
    # 创建图像数据集对象
    ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    # 创建数据加载器
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    # 创建自监督学习模型
    model = SelfSupervisedLearner(
        resnet,
        image_size=IMAGE_SIZE,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    )

    # 创建训练器
    trainer = pl.Trainer(
        gpus=NUM_GPUS,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True
    )

    # 训练模型
    trainer.fit(model, train_loader)
```