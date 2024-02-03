# `.\PaddleOCR\StyleText\utils\sys_funcs.py`

```
# 版权声明和许可信息
#
# 本代码版权归 PaddlePaddle 作者所有。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本许可，除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
import sys
import os
import errno
import paddle

# 获取全局参数检查列表
def get_check_global_params(mode):
    check_params = [
        'use_gpu', 'max_text_length', 'image_shape', 'image_shape',
        'character_type', 'loss_type'
    ]
    if mode == "train_eval":
        check_params = check_params + [
            'train_batch_size_per_card', 'test_batch_size_per_card'
        ]
    elif mode == "test":
        check_params = check_params + ['test_batch_size_per_card']
    return check_params

# 检查 GPU 使用情况
def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as true while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"
    if use_gpu:
        try:
            if not paddle.is_compiled_with_cuda():
                print(err)
                sys.exit(1)
        except:
            print("Fail to check gpu state.")
            sys.exit(1)

# 如果路径不存在，则创建目录，忽略多进程同时创建目录时的异常
def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    # 检查路径是否存在，如果不存在则创建
    if not os.path.exists(path):
        # 尝试创建路径
        try:
            os.makedirs(path)
        # 捕获可能出现的异常
        except OSError as e:
            # 如果路径已存在且是一个目录，则记录警告信息
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            # 如果出现其他异常，则抛出错误
            else:
                raise OSError('Failed to mkdir {}'.format(path))
```