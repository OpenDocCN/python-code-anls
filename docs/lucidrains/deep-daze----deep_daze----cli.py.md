# `.\lucidrains\deep-daze\deep_daze\cli.py`

```py
import sys
# 导入系统模块

import fire
# 导入fire模块，用于命令行接口

from deep_daze import Imagine
# 从deep_daze模块中导入Imagine类

def train(
        text=None,
        img=None,
        learning_rate=1e-5,
        num_layers=16,
        hidden_size=256,
        batch_size=4,
        gradient_accumulate_every=4,
        epochs=20,
        iterations=1050,
        save_every=100,
        image_width=512,
        deeper=False,
        overwrite=False,
        save_progress=True,
        seed=None,
        open_folder=True,
        save_date_time=False,
        start_image_path=None,
        start_image_train_iters=50,
        theta_initial=None,
        theta_hidden=None,
        start_image_lr=3e-4,
        lower_bound_cutout=0.1,
        upper_bound_cutout=1.0,
        saturate_bound=False,
        create_story=False,
        story_start_words=5,
        story_words_per_epoch=5,
        story_separator=None,
        averaging_weight=0.3,
        gauss_sampling=False,
        gauss_mean=0.6,
        gauss_std=0.2,
        do_cutout=True,
        center_bias=False,
        center_focus=2,
        jit=True,
        save_gif=False,
        save_video=False,
        model_name="ViT-B/32",
        optimizer="AdamP"
):
    """
    :param text: (required) A phrase less than 77 tokens which you would like to visualize.
    # 要可视化的短语，不超过77个标记
    :param img: The path to a jpg or png image which you would like to imagine. Can be combined with text.
    # 要想象的jpg或png图像的路径，可以与文本结合使用
    :param learning_rate: The learning rate of the neural net.
    # 神经网络的学习率
    :param hidden_size: The hidden layer size of the Siren net.
    # Siren网络的隐藏层大小
    :param num_layers: The number of hidden layers to use in the Siren neural net.
    # 在Siren神经网络中使用的隐藏层数量
    :param batch_size: The number of generated images to pass into Siren before calculating loss. Decreasing this can lower memory and accuracy.
    # 在计算损失之前传递给Siren的生成图像数量。减少此值可以降低内存和准确性
    :param gradient_accumulate_every: Calculate a weighted loss of n samples for each iteration. Increasing this can help increase accuracy with lower batch sizes.
    # 每次迭代计算n个样本的加权损失。增加此值可以帮助提高较小批次大小的准确性
    :param epochs: The number of epochs to run.
    # 要运行的周期数
    :param iterations: The number of times to calculate and backpropagate loss in a given epoch.
    # 在给定周期内计算和反向传播损失的次数
    :param save_progress: Whether or not to save images generated before training Siren is complete.
    # 是否保存在训练Siren完成之前生成的图像
    :param save_every: Generate an image every time iterations is a multiple of this number.
    # 每次迭代是此数字的倍数时生成一幅图像
    :param open_folder:  Whether or not to open a folder showing your generated images.
    # 是否打开显示生成图像的文件夹
    :param overwrite: Whether or not to overwrite existing generated images of the same name.
    # 是否覆盖同名的现有生成图像
    :param deeper: Uses a Siren neural net with 32 hidden layers.
    # 使用具有32个隐藏层的Siren神经网络
    :param image_width: The desired resolution of the image.
    # 图像的期望分辨率
    :param seed: A seed to be used for deterministic runs.
    # 用于确定性运行的种子
    :param save_date_time: Save files with a timestamp prepended e.g. `%y%m%d-%H%M%S-my_phrase_here.png`
    # 保存带有时间戳前缀的文件，��如`%y%m%d-%H%M%S-my_phrase_here.png`
    :param start_image_path: Path to the image you would like to prime the generator with initially
    # 要首先使用的图像的路径
    :param start_image_train_iters: Number of iterations for priming, defaults to 50
    # 用于初始化的迭代次数，默认为50
    :param theta_initial: Hyperparameter describing the frequency of the color space. Only applies to the first layer of the network.
    # 描述颜色空间频率的超参数。仅适用于网络的第一层
    :param theta_hidden: Hyperparameter describing the frequency of the color space. Only applies to the hidden layers of the network.
    # 描述颜色空间频率的超参数。仅适用于网络的隐藏层
    :param start_image_lr: Learning rate for the start image training.
    # 用于初始图像训练的学习率
    :param upper_bound_cutout: The upper bound for the cutouts used in generation.
    # 用于生成的切割的上限
    :param lower_bound_cutout: The lower bound for the cutouts used in generation.
    # 用于生成的切割的下限
    :param saturate_bound: If True, the LOWER_BOUND_CUTOUT is linearly increased to 0.75 during training.
    # 如果为True，则在训练过程中将LOWER_BOUND_CUTOUT线性增加到0.75
    :param create_story: Creates a story by optimizing each epoch on a new sliding-window of the input words. If this is enabled, much longer texts than 77 tokens can be used. Requires save_progress to visualize the transitions of the story.
    # 通过在输入单词的新滑动窗口上优化每个时期来创建故事。如果启用此功能，则可以使用比77个标记更长的文本。需要save_progress来可视化故事的转换
    :param story_start_words: Only used if create_story is True. How many words to optimize on for the first epoch.
    # 仅在create_story为True时使用。第一个时期要优化的单词数
    :param story_words_per_epoch: Only used if create_story is True. How many words to add to the optimization goal per epoch after the first one.
    # 仅在create_story为True时使用。在第一个时期之后每个时期要添加到优化目标中的单词数
    :param story_separator: Only used if create_story is True. Defines a separator like '.' that splits the text into groups for each epoch. Separator needs to be in the text otherwise it will be ignored!
    :param averaging_weight: How much to weigh the averaged features of the random cutouts over the individual random cutouts. Increasing this value leads to more details being represented at the cost of some global coherence and a parcellation into smaller scenes.
    :param gauss_sampling: Whether to use sampling from a Gaussian distribution instead of a uniform distribution.
    :param gauss_mean: The mean of the Gaussian sampling distribution.
    :param gauss_std: The standard deviation of the Gaussian sampling distribution.
    :param do_cutouts: Whether to use random cutouts as an augmentation. This basically needs to be turned on unless some new augmentations are added in code eventually.
    :param center_bias: Whether to use a Gaussian distribution centered around the center of the image to sample the locations of random cutouts instead of a uniform distribution. Leads to the main generated objects to be more focused in the center.
    :param center_focus: How much to focus on the center if using center_bias. std = sampling_range / center_focus. High values lead to a very correct representation in the center but washed out colors and details towards the edges,
    :param jit: Whether to use the jit-compiled CLIP model. The jit model is faster, but only compatible with torch version 1.7.1.
    :param save_gif: Only used if save_progress is True. Saves a GIF animation of the generation procedure using the saved frames.
    :param save_video: Only used if save_progress is True. Saves a MP4 animation of the generation procedure using the saved frames.
    """
    # Don't instantiate imagine if the user just wants help.
    # 检查命令行参数中是否包含"--help"，如果有则打印使用信息并退出程序
    if any("--help" in arg for arg in sys.argv):
        print("Type `imagine --help` for usage info.")
        sys.exit()

    # 根据参数设置生成的图像的层数
    num_layers = 32 if deeper else num_layers

    # 创建 Imagine 对象，传入各种参数
    imagine = Imagine(
        text=text,
        img=img,
        lr=learning_rate,
        num_layers=num_layers,
        batch_size=batch_size,
        gradient_accumulate_every=gradient_accumulate_every,
        epochs=epochs,
        iterations=iterations,
        image_width=image_width,
        save_every=save_every,
        save_progress=save_progress,
        seed=seed,
        open_folder=open_folder,
        save_date_time=save_date_time,
        start_image_path=start_image_path,
        start_image_train_iters=start_image_train_iters,
        theta_initial=theta_initial,
        theta_hidden=theta_hidden,
        start_image_lr=start_image_lr,
        lower_bound_cutout=lower_bound_cutout,
        upper_bound_cutout=upper_bound_cutout,
        saturate_bound=saturate_bound,
        create_story=create_story,
        story_start_words=story_start_words,
        story_words_per_epoch=story_words_per_epoch,
        story_separator=story_separator,
        averaging_weight=averaging_weight,
        gauss_sampling=gauss_sampling,
        gauss_mean=gauss_mean,
        gauss_std=gauss_std,
        do_cutout=do_cutout,
        center_bias=center_bias,
        center_focus=center_focus,
        jit=jit,
        hidden_size=hidden_size,
        model_name=model_name,
        optimizer=optimizer,
        save_gif=save_gif,
        save_video=save_video,
    )

    # 打印提示信息
    print('Starting up...')
    # 如果不覆盖已存在的生成图像文件，并且文件已存在，则询问用户是否覆盖
    if not overwrite and imagine.filename.exists():
        answer = input('Imagined image already exists, do you want to overwrite? (y/n) ').lower()
        if answer not in ('yes', 'y'):
            sys.exit()

    # 调用 Imagine 对象，开始生成图像
    imagine()
# 定义主函数入口
def main():
    # 使用 Fire 库将 train 函数转换为命令行接口
    fire.Fire(train)
```