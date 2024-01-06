# `transformer_vq\src\transformer_vq\utils\io.py`

```
# 导入 TensorFlow 库
import tensorflow as tf
# 从 flax.training 模块中导入 checkpoints
from flax.training import checkpoints

# 检查参数是否为 None，如果是则抛出数值错误
def check_not_none(s):
    if s is None:
        raise ValueError("argument cannot be None.")

# 保存检查点文件
def save_checkpoint(target, save_dir, prefix, step, keep):
    # 检查保存目录是否为 None
    check_not_none(save_dir)
    # 使用多进程保存检查点
    checkpoints.save_checkpoint_multiprocess(
        save_dir,  # 保存目录
        target=target,  # 目标对象
        step=step,  # 步数
        prefix=f"{prefix}_",  # 文件名前缀
        keep=keep,  # 保留的检查点数量
        overwrite=False,  # 是否覆盖已存在的检查点
        keep_every_n_steps=None,  # 每隔多少步保存一个检查点
        async_manager=None,  # 异步管理器
# 定义一个默认值为None的orbax_checkpointer变量
orbax_checkpointer=None,

# 加载检查点，恢复训练状态
def load_checkpoint(train_state, load_dir, prefix):
    # 检查load_dir是否为None
    check_not_none(load_dir)
    # 从检查点目录load_dir中恢复训练状态
    train_state = checkpoints.restore_checkpoint(
        ckpt_dir=load_dir,
        target=train_state,
        prefix=prefix,
        step=None,
    )
    return train_state

# 保存文本内容到指定文件
def save_text(target, dirname, fname, mode="w"):
    # 拼接目录和文件名，得到文件路径
    fp = tf.io.gfile.join(dirname, fname)
    # 使用tf.io.gfile.GFile打开文件
    with tf.io.gfile.GFile(fp, mode=mode) as f:
        # 写入目标文本内容
        f.write(target)
        # 刷新缓冲区
        f.flush()
# 将目标数据编码为 PNG 格式的字节流，并转换为 numpy 数组
target = tf.io.encode_png(target).numpy()
# 调用 save_text 函数保存目标数据到指定目录下的文件中，以二进制写入模式
save_text(target=target, dirname=dirname, fname=fname, mode="wb")
```