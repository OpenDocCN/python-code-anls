# `.\PaddleOCR\ppocr\utils\e2e_utils\extract_batchsize.py`

```
import paddle
import numpy as np
import copy

# 定义函数 org_tcl_rois，用于处理一些数据
def org_tcl_rois(batch_size, pos_lists, pos_masks, label_lists, tcl_bs):
    # 初始化空列表
    pos_lists_, pos_masks_, label_lists_ = [], [], []
    # 计算每个 GPU 处理的图片数量
    img_bs = batch_size
    ngpu = int(batch_size / img_bs)
    # 获取图片 ID 数组
    img_ids = np.array(pos_lists, dtype=np.int32)[:, 0, 0].copy()
    # 初始化分割后的列表
    pos_lists_split, pos_masks_split, label_lists_split = [], [], []
    for i in range(ngpu):
        pos_lists_split.append([])
        pos_masks_split.append([])
        label_lists_split.append([])

    # 根据图片 ID 进行分割
    for i in range(img_ids.shape[0]):
        img_id = img_ids[i]
        gpu_id = int(img_id / img_bs)
        img_id = img_id % img_bs
        pos_list = pos_lists[i].copy()
        pos_list[:, 0] = img_id
        pos_lists_split[gpu_id].append(pos_list)
        pos_masks_split[gpu_id].append(pos_masks[i].copy())
        label_lists_split[gpu_id].append(copy.deepcopy(label_lists[i]))
    
    # 根据条件重复或删除数据
    for i in range(ngpu):
        vp_len = len(pos_lists_split[i])
        if vp_len <= tcl_bs:
            for j in range(0, tcl_bs - vp_len):
                pos_list = pos_lists_split[i][j].copy()
                pos_lists_split[i].append(pos_list)
                pos_mask = pos_masks_split[i][j].copy()
                pos_masks_split[i].append(pos_mask)
                label_list = copy.deepcopy(label_lists_split[i][j])
                label_lists_split[i].append(label_list)
        else:
            for j in range(0, vp_len - tcl_bs):
                c_len = len(pos_lists_split[i])
                pop_id = np.random.permutation(c_len)[0]
                pos_lists_split[i].pop(pop_id)
                pos_masks_split[i].pop(pop_id)
                label_lists_split[i].pop(pop_id)
    
    # 合并数据
    for i in range(ngpu):
        pos_lists_.extend(pos_lists_split[i])
        pos_masks_.extend(pos_masks_split[i])
        label_lists_.extend(label_lists_split[i])
    
    # 返回处理后的数据
    return pos_lists_, pos_masks_, label_lists_
# 对输入数据进行预处理，返回处理后的数据
def pre_process(label_list, pos_list, pos_mask, max_text_length, max_text_nums,
                pad_num, tcl_bs):
    # 将标签列表转换为 numpy 数组
    label_list = label_list.numpy()
    # 获取标签列表的形状信息
    batch, _, _, _ = label_list.shape
    # 将位置列表转换为 numpy 数组
    pos_list = pos_list.numpy()
    # 将位置掩码转换为 numpy 数组
    pos_mask = pos_mask.numpy()
    # 初始化临时位置列表、位置掩码列表和标签列表
    pos_list_t = []
    pos_mask_t = []
    label_list_t = []
    # 遍历每个批次
    for i in range(batch):
        # 遍历每个文本序列
        for j in range(max_text_nums):
            # 如果位置掩码中存在非零值
            if pos_mask[i, j].any():
                # 将位置列表、位置掩码和标签添加到临时列表中
                pos_list_t.append(pos_list[i][j])
                pos_mask_t.append(pos_mask[i][j])
                label_list_t.append(label_list[i][j])
    # 调用 org_tcl_rois 函数对位置列表、位置掩码和标签列表进行处理
    pos_list, pos_mask, label_list = org_tcl_rois(batch, pos_list_t, pos_mask_t,
                                                  label_list_t, tcl_bs)
    # 初始化标签列表
    label = []
    # 将标签列表转换为列表形式
    tt = [l.tolist() for l in label_list]
    # 遍历每个批次
    for i in range(tcl_bs):
        k = 0
        # 遍历每个文本序列的最大长度
        for j in range(max_text_length):
            # 如果标签不是填充值
            if tt[i][j][0] != pad_num:
                k += 1
            else:
                break
        # 将计算得到的标签长度添加到标签列表中
        label.append(k)
    # 将标签列表转换为 PaddlePaddle 张量
    label = paddle.to_tensor(label)
    # 将标签列表转换为 int64 类型
    label = paddle.cast(label, dtype='int64')
    # 将位置列表转换为 PaddlePaddle 张量
    pos_list = paddle.to_tensor(pos_list)
    # 将位置掩码转换为 PaddlePaddle 张量
    pos_mask = paddle.to_tensor(pos_mask)
    # 将标签列表转换为 PaddlePaddle 张量并去除多余的维度
    label_list = paddle.squeeze(paddle.to_tensor(label_list), axis=2)
    # 将标签列表转换为 int32 类型
    label_list = paddle.cast(label_list, dtype='int32')
    # 返回处理后的位置列表、位置掩码、标签列表和标签
    return pos_list, pos_mask, label_list, label
```