# `.\lucidrains\med-seg-diff-pytorch\driver.py`

```py
import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.optim import AdamW
from lion_pytorch import Lion
from med_seg_diff_pytorch import Unet, MedSegDiff
from med_seg_diff_pytorch.dataset import ISICDataset, GenericNpyDataset
from accelerate import Accelerator
import wandb

## Parse CLI arguments ##
def parse_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数选项
    parser.add_argument('-slr', '--scale_lr', action='store_true', help="Whether to scale lr.")
    parser.add_argument('-rt', '--report_to', type=str, default="wandb", choices=["wandb"],
                        help="Where to log to. Currently only supports wandb")
    parser.add_argument('-ld', '--logging_dir', type=str, default="logs", help="Logging dir.")
    parser.add_argument('-od', '--output_dir', type=str, default="output", help="Output dir.")
    parser.add_argument('-mp', '--mixed_precision', type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Whether to do mixed precision")
    parser.add_argument('-ga', '--gradient_accumulation_steps', type=int, default=4,
                        help="The number of gradient accumulation steps.")
    parser.add_argument('-img', '--img_folder', type=str, default='ISBI2016_ISIC_Part3B_Training_Data',
                        help='The image file path from data_path')
    parser.add_argument('-csv', '--csv_file', type=str, default='ISBI2016_ISIC_Part3B_Training_GroundTruth.csv',
                        help='The csv file to load in from data_path')
    parser.add_argument('-sc', '--self_condition', action='store_true', help='Whether to do self condition')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.95,
                        help='The beta1 parameter for the Adam optimizer.')
    parser.add_argument('-ab2', '--adam_beta2', type=float, default=0.999,
                        help='The beta2 parameter for the Adam optimizer.')
    parser.add_argument('-aw', '--adam_weight_decay', type=float, default=1e-6,
                        help='Weight decay magnitude for the Adam optimizer.')
    parser.add_argument('-ae', '--adam_epsilon', type=float, default=1e-08,
                        help='Epsilon value for the Adam optimizer.')
    parser.add_argument('-ul', '--use_lion', type=bool, default=False, help='use Lion optimizer')
    parser.add_argument('-ic', '--mask_channels', type=int, default=1, help='input channels for training (default: 3)')
    parser.add_argument('-c', '--input_img_channels', type=int, default=3,
                        help='output channels for training (default: 3)')
    parser.add_argument('-is', '--image_size', type=int, default=128, help='input image size (default: 128)')
    parser.add_argument('-dd', '--data_path', default='./data', help='directory of input image')
    parser.add_argument('-d', '--dim', type=int, default=64, help='dim (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs (default: 10000)')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='batch size to train on (default: 8)')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of timesteps (default: 1000)')
    parser.add_argument('-ds', '--dataset', default='generic', help='Dataset to use')
    parser.add_argument('--save_every', type=int, default=100, help='save_every n epochs (default: 100)')
    parser.add_argument('--load_model_from', default=None, help='path to pt file to load from')
    # 解析参数并返回
    return parser.parse_args()


def load_data(args):
    # 加载数据集
    # 如果数据集为ISIC，则定义ISIC数据集的转换操作列表
    if args.dataset == 'ISIC':
        transform_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), ]
        transform_train = transforms.Compose(transform_list)
        # 创建ISIC数据集对象
        dataset = ISICDataset(args.data_path, args.csv_file, args.img_folder, transform=transform_train, training=True,
                              flip_p=0.5)
    # 如果数据集为generic，则定义generic数据集的转换操作列表
    elif args.dataset == 'generic':
        transform_list = [transforms.ToPILImage(), transforms.Resize(args.image_size), transforms.ToTensor()]
        transform_train = transforms.Compose(transform_list)
        # 创建generic数据集对象
        dataset = GenericNpyDataset(args.data_path, transform=transform_train, test_flag=False)
    # 如果数据集不是ISIC或generic，则抛出未实现错误
    else:
        raise NotImplementedError(f"Your dataset {args.dataset} hasn't been implemented yet.")

    ## 定义PyTorch数据生成器
    training_generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True)

    # 返回训练数据生成器
    return training_generator
def main():
    # 解析命令行参数
    args = parse_args()
    # 创建检查点目录
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    # 创建日志目录
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    # 如果目录不存在则创建
    os.makedirs(checkpoint_dir, exist_ok=True)
    # 初始化加速器
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )
    # 如果是主进程则初始化跟踪器
    if accelerator.is_main_process:
        accelerator.init_trackers("med-seg-diff", config=vars(args))

    ## DEFINE MODEL ##
    # 定义模型
    model = Unet(
        dim=args.dim,
        image_size=args.image_size,
        dim_mults=(1, 2, 4, 8),
        mask_channels=args.mask_channels,
        input_img_channels=args.input_img_channels,
        self_condition=args.self_condition
    )

    ## LOAD DATA ##
    # 加载数据
    data_loader = load_data(args)
    # 如果需要缩放学习率，则重新计算学习率
    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )

    ## Initialize optimizer
    # 初始化优化器
    if not args.use_lion:
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = Lion(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay
        )

    ## TRAIN MODEL ##
    counter = 0
    # 准备模型、优化器和数据加载器
    model, optimizer, data_loader = accelerator.prepare(
        model, optimizer, data_loader
    )
    # 创建 MedSegDiff 对象
    diffusion = MedSegDiff(
        model,
        timesteps=args.timesteps
    ).to(accelerator.device)

    # 如果指定了加载模型的路径，则加载模型
    if args.load_model_from is not None:
        save_dict = torch.load(args.load_model_from)
        diffusion.model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        accelerator.print(f'Loaded from {args.load_model_from}')

    ## Iterate across training loop
    # 遍历训练循环
    for epoch in range(args.epochs):
        running_loss = 0.0
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        for (img, mask) in tqdm(data_loader):
            with accelerator.accumulate(model):
                loss = diffusion(mask, img)
                running_loss += loss.item() * img.size(0)
                accelerator.log({'loss': loss})  # Log loss to wandb
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        counter += 1
        epoch_loss = running_loss / len(data_loader)
        print('Training Loss : {:.4f}'.format(epoch_loss))
        ## INFERENCE ##

        # 如果满足保存间隔条件，则保存模型
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(checkpoint_dir, f'state_dict_epoch_{epoch}_loss_{epoch_loss}.pt'))

            # 生成预测结果
            pred = diffusion.sample(img).cpu().detach().numpy()

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    # 保存每个批次的一张图像
                    tracker.log(
                        {'pred-img-mask': [wandb.Image(pred[0, 0, :, :]), wandb.Image(img[0, 0, :, :]),
                                           wandb.Image(mask[0, 0, :, :])]}
                    )


if __name__ == '__main__':
    main()
```