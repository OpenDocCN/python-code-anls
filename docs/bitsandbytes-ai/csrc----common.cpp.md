# `bitsandbytes\csrc\common.cpp`

```py
void *quantize_block(void *arguments) {
    // 1. find absmax in block
    // 2. divide input value by absmax to normalize into [-1.0, 1.0]
    // 3. do binary search to find the closest value
    // 4. check minimal distance
    // 5. store index

    // 将传入的参数转换为结构体指针
    struct quantize_block_args *args = (quantize_block_args *) arguments;

    // 1. find absmax in block
    // 初始化块内的最大绝对值为负无穷
    float absmax_block = -FLT_MAX;
    // 遍历块内的元素，找到最大的绝对值
    for (long long i = args->block_idx; i < args->block_end; i++)
        absmax_block = fmax(absmax_block, fabs(args->A[i]));

    // 将块内的最大绝对值存储到数组中
    args->absmax[args->block_idx / args->blocksize] = absmax_block;

    // 遍历块内的元素
    for (long long i = args->block_idx; i < args->block_end; i++) {
        // 2. divide input value by absmax to normalize into [-1.0, 1.0]
        // 3. do binary search to find the closest value
        // 将输入值除以 absmax 以将其归一化到 [-1.0, 1.0] 范围内
        float normed_value = args->A[i] / absmax_block;
        // 使用二分查找找到最接近的值的索引
        long long idx = args->bin_searcher->scalar(normed_value);

        // 4. check minimal distance
        // 二分查找总是返回左侧的值，这可能不是最接近的值
        if (idx < 255) {
            // 计算当前值与左右两个值的距离
            float dist_left = fabs(normed_value - (args->code[idx]));
            float dist_right = fabs(normed_value - (args->code[idx + 1]));
            // 如果右侧的距离更小，则选择右侧的值
            if (dist_right < dist_left) { idx += 1; }
        }

        // 5. store index
        // 将索引存储到输出数组中
        args->out[i] = (unsigned char) idx;
    }

    return NULL;
}
```