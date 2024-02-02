# `transformer_vq\src\transformer_vq\utils\io.py`

```py
# 导入 TensorFlow 库
import tensorflow as tf
# 从 flax.training 模块中导入 checkpoints 函数
from flax.training import checkpoints


# 定义函数，检查参数是否为 None
def check_not_none(s):
    # 如果参数为 None，则抛出数值错误异常
    if s is None:
        raise ValueError("argument cannot be None.")


# 定义函数，保存检查点
def save_checkpoint(target, save_dir, prefix, step, keep):
    # 检查保存目录是否为 None
    check_not_none(save_dir)
    # 调用 checkpoints 模块的 save_checkpoint_multiprocess 函数保存检查点
    checkpoints.save_checkpoint_multiprocess(
        save_dir,
        target=target,
        step=step,
        prefix=f"{prefix}_",
        keep=keep,
        overwrite=False,
        keep_every_n_steps=None,
        async_manager=None,
        orbax_checkpointer=None,
    )


# 定义函数，加载检查点
def load_checkpoint(train_state, load_dir, prefix):
    # 检查加载目录是否为 None
    check_not_none(load_dir)
    # 调用 checkpoints 模块的 restore_checkpoint 函数加载检查点
    train_state = checkpoints.restore_checkpoint(
        ckpt_dir=load_dir,
        target=train_state,
        prefix=prefix,
        step=None,
    )
    # 返回加载后的训练状态
    return train_state


# 定义函数，保存文本数据
def save_text(target, dirname, fname, mode="w"):
    # 拼接文件路径
    fp = tf.io.gfile.join(dirname, fname)
    # 使用 tf.io.gfile.GFile 打开文件，并写入数据
    with tf.io.gfile.GFile(fp, mode=mode) as f:
        f.write(target)
        f.flush()


# 定义函数，保存像素数据
def save_pixels(target, dirname, fname):
    # 将目标数据编码为 PNG 格式的字节流
    target = tf.io.encode_png(target).numpy()
    # 调用 save_text 函数保存字节流数据
    save_text(target=target, dirname=dirname, fname=fname, mode="wb")
```