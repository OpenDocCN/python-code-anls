# `.\lucidrains\memformer\memformer\mrbp.py`

```py
# 导入 torch 库
import torch
# 从 operator 库中导入 itemgetter 函数

# 定义内存回放反向传播函数，接受模型、源数据、目标数据、源数据掩码和目标数据掩码作为参数
def memory_replay_backprop(
    model,
    src,
    tgt,
    src_mask = None,
    tgt_mask = None
):
    # 获取源数据的 batch 大小
    b, *_ = src.shape

    # 从编码器获取初始内存和最大序列长度
    mem_init = model.get_initial_mem(b)
    max_seq_len = model.encoder.max_seq_len

    # 实例化内存回放缓冲区
    replay_buffer = [mem_init]

    # 拆分序列和掩码
    src_segs = src.split(max_seq_len, dim = 1)
    num_segs = len(src_segs)
    src_mask_segs = src_mask.split(max_seq_len, dim = 1) if src_mask is not None else ((None,) * num_segs)

    # 目前假设目标序列和掩码在最后一个段中传递
    # 待办事项 - 允许在任何段中连接目标序列
    # 并将自定义损失附加到编码器输出
    tgt_segs = ((None,) * (num_segs - 1)) + (tgt,)
    tgt_mask_segs = ((None,) * (num_segs - 1)) + (tgt_mask,)

    # 运行前向传播并收集所有内存
    prev_mem = mem_init
    with torch.no_grad():
        for i in range(num_segs - 1):
            src, src_mask = map(itemgetter(i), (src_segs, src_mask_segs))
            _, mem, _ = model(src, src_mask = src_mask, mems = prev_mem)
            replay_buffer.append(mem)
            prev_mem = mem

    # 逐个段进行反向传播，从最后一步到第一步
    mem_grad = torch.zeros_like(prev_mem)
    for i in reversed(range(num_segs)):
        src, src_mask, tgt, tgt_mask, mems = map(itemgetter(i), (src_segs, src_mask_segs, tgt_segs, tgt_mask_segs, replay_buffer))
        mems = mems.requires_grad_()

        _, mems_next, tgt_loss = model(src = src, tgt = tgt, src_mask = src_mask, tgt_mask = tgt_mask, mems = mems)
        tgt_loss.backward(retain_graph = True)
        mems_next.backward(mem_grad, retain_graph = True)

        # 如果不是最后一步，则将下一个内存的梯度传递回一步
        if i != 0:
            mem_grad.copy_(mems.grad.data)
```