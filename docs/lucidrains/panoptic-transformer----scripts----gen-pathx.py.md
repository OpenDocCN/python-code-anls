# `.\lucidrains\panoptic-transformer\scripts\gen-pathx.py`

```py
# 导入所需的库
import time
import sys
import numpy as np
import os

# 导入自定义的 snakes2 模块
import snakes2

# 定义一个参数类，用于设置各种参数
class Args:
    def __init__(self,
                 contour_path = './contour', batch_id=0, n_images = 200000,
                 window_size=[256,256], padding=22, antialias_scale = 4,
                 LABEL =1, seed_distance= 27, marker_radius = 3,
                 contour_length=15, distractor_length=5, num_distractor_snakes=6, snake_contrast_list=[1.], use_single_paddles=True,
                 max_target_contour_retrial = 4, max_distractor_contour_retrial = 4, max_paddle_retrial=2,
                 continuity = 1.4, paddle_length=5, paddle_thickness=1.5, paddle_margin_list=[4], paddle_contrast_list=[1.],
                 pause_display=False, save_images=True, save_metadata=True):

        # 初始化参数
        self.contour_path = contour_path
        self.batch_id = batch_id
        self.n_images = n_images

        self.window_size = window_size
        self.padding = padding
        self.antialias_scale = antialias_scale

        self.LABEL = LABEL
        self.seed_distance = seed_distance
        self.marker_radius = marker_radius
        self.contour_length = contour_length
        self.distractor_length = distractor_length
        self.num_distractor_snakes = num_distractor_snakes
        self.snake_contrast_list = snake_contrast_list
        self.use_single_paddles = use_single_paddles

        self.max_target_contour_retrial = max_target_contour_retrial
        self.max_distractor_contour_retrial = max_distractor_contour_retrial
        self.max_paddle_retrial = max_paddle_retrial

        self.continuity = continuity
        self.paddle_length = paddle_length
        self.paddle_thickness = paddle_thickness
        self.paddle_margin_list = paddle_margin_list # 如果列表中有多个元素，每个图像将采样一个数字
        self.paddle_contrast_list = paddle_contrast_list # 如果列表中有多个元素，每个 paddle 将采样一个数字

        self.pause_display = pause_display
        self.save_images = save_images
        self.save_metadata = save_metadata

# 记录开始时间
t = time.time()
# 创建参数对象
args = Args()

# 从命令行参数中获取机器数量、当前 ID 和总图像数量
num_machines = int(sys.argv[1])
current_id = int(sys.argv[2])
args.batch_id = current_id
total_images = int(sys.argv[3])
args.n_images = total_images/num_machines
dataset_root = './pathx-data' #'/media/data_cifs/pathfinder_seg/'

# 根据命令行参数设置数据集根目录
if len(sys.argv)==4:
    print('Using default path...')
elif len(sys.argv)==5:
    print('Using custom save path...')
    dataset_root = str(sys.argv[4])

# 设置一些参数的值
args.padding = 1
args.antialias_scale = 4
args.paddle_margin_list = [2,3]
args.seed_distance = 20
args.window_size = [128,128]
args.marker_radius = 3
args.contour_length = 14
args.paddle_thickness = 1.5
args.antialias_scale = 2
args.continuity = 1.8  # 从 1.8 到 0.8，步长为 66%
args.distractor_length = args.contour_length // 3
args.num_distractor_snakes = 35 / args.distractor_length
args.snake_contrast_list = [0.9]

args.use_single_paddles = False
args.segmentation_task = False # False
args.segmentation_task_double_circle = False

# 设置轮廓路径
dataset_subpath = 'curv_baseline'
args.contour_path = os.path.join(dataset_root, dataset_subpath)

# 调用 snakes2 模块中的 from_wrapper 函数，传入参数对象
snakes2.from_wrapper(args)
```