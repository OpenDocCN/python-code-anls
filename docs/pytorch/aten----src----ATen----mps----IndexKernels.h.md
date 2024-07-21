# `.\pytorch\aten\src\ATen\mps\IndexKernels.h`

```py
#pragma once

namespace at::mps {

static const char * indexing_metal_shaders = R"INDEX_METAL(
#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

#if __METAL_VERSION__ < 300
struct IndexAB {
    // Allow up to 16 indices
    metal::array<constant void *, 16>  indexArray [[ id(0) ]];
};
#else
struct IndexAB {
    constant int64_t* indexArray;
};

#endif

template<typename T, typename OffsetsT>
kernel void index_select(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB           [[buffer(0)]],   // 索引数组结构体数组的常量指针
#else
    constant IndexAB  & indexAB           [[buffer(0)]],   // 索引数组结构体的常量引用
#endif
    constant void     * indexSizes        [[buffer(1)]],   // 索引大小的常量指针
    constant void     * indexStrides      [[buffer(2)]],   // 索引步长的常量指针
    constant OffsetsT * offsets           [[buffer(3)]],   // 偏移量的常量指针
    constant void     * inputData         [[buffer(4)]],   // 输入数据的常量指针
    device   void     * outputData        [[buffer(5)]],   // 输出数据的设备指针
    constant uint32_t & num_indices       [[buffer(6)]],   // 索引数量的常量引用
    uint thread_index [[thread_position_in_grid]]) {       // 线程索引
    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;  // 将索引大小转换为常量 int64_t 指针
    constant int64_t * index_strides = (constant int64_t *)indexStrides;  // 将索引步长转换为常量 int64_t 指针
    int64_t offset = 0;  // 初始化偏移量
    for (uint32_t i = 0; i < num_indices; i++) {  // 遍历索引数量
#if __METAL_VERSION__ >= 300
        constant int64_t* indexArray = indexAB[i].indexArray;  // 使用索引数组结构体中的索引数组
#else
        constant int64_t* indexArray = (constant int64_t*)indexAB.indexArray[i];  // 从索引数组结构体获取索引数组
#endif
        int64_t index = indexArray[offsets[thread_index].z / sizeof(int64_t)];  // 计算索引位置

        if (index < 0) {  // 处理负索引的情况
            index += index_sizes[i];
        }
        offset += index * index_strides[i];  // 计算偏移量
     }
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x);  // 计算输出数据位置
    constant T * in  = (constant T*)((constant char*)inputData  + offsets[thread_index].y + offset);  // 计算输入数据位置
    *out = *in;  // 执行数据复制
}

template<typename T, typename OffsetsT>
void index_put_impl(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB,  // 索引数组结构体的常量指针
#else
    constant IndexAB  & indexAB,  // 索引数组结构体的常量引用
#endif
    constant int64_t  * index_sizes,  // 索引大小的常量指针
    constant int64_t  * index_strides,  // 索引步长的常量指针
    constant OffsetsT * offsets,  // 偏移量的常量指针
    constant void     * inputData,  // 输入数据的常量指针
    device   void     * outputData,  // 输出数据的设备指针
    constant uint32_t & num_indices,  // 索引数量的常量引用
    uint thread_index) {  // 线程索引
    int64_t offset = 0;  // 初始化偏移量
    for (uint32_t i = 0; i < num_indices; i++) {  // 遍历索引数量
#if __METAL_VERSION__ >= 300
        constant int64_t* indexArray = indexAB[i].indexArray;  // 使用索引数组结构体中的索引数组
#else
        constant int64_t* indexArray = (constant int64_t*)indexAB.indexArray[i];  // 从索引数组结构体获取索引数组
#endif
        int64_t index = indexArray[offsets[thread_index].z / sizeof(int64_t)];  // 计算索引位置

        if (index < 0) {  // 处理负索引的情况
            index += index_sizes[i];
        }
        offset += index * index_strides[i];  // 计算偏移量
    }
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x + offset);  // 计算输出数据位置
    constant T * in  = (constant T*)((constant char*)inputData  + offsets[thread_index].y);  // 计算输入数据位置
    *out = *in;  // 执行数据复制
}

template<typename T, typename OffsetsT>
kernel void index_put_serial(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB           [[buffer(0)]],


    # 定义一个常量指针 indexAB，指向类型为 IndexAB 的对象，使用在缓冲区0中
#else
    constant IndexAB  & indexAB           [[buffer(0)]],   // 定义一个常量引用 IndexAB 对象，从第一个缓冲区获取
#endif
    constant void     * indexSizes        [[buffer(1)]],   // 定义一个指向常量数据的指针，从第二个缓冲区获取
    constant void     * indexStrides      [[buffer(2)]],   // 定义一个指向常量数据的指针，从第三个缓冲区获取
    constant OffsetsT * offsets           [[buffer(3)]],   // 定义一个指向 OffsetsT 类型数据的指针，从第四个缓冲区获取
    constant void     * inputData         [[buffer(4)]],   // 定义一个指向常量数据的指针，从第五个缓冲区获取
    device   void     * outputData        [[buffer(5)]],   // 定义一个指向设备数据的指针，从第六个缓冲区获取
    constant uint32_t & num_indices       [[buffer(6)]],   // 定义一个常量引用 uint32_t 类型的数据，从第七个缓冲区获取
    constant uint     * numIters          [[buffer(7)]],   // 定义一个指向常量 uint 类型数据的指针，从第八个缓冲区获取
    uint thread_index [[thread_position_in_grid]]) {        // 定义一个无符号整数 thread_index，表示线程在网格中的位置索引

    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;     // 将 indexSizes 转换为常量 int64_t 类型指针，并赋给 index_sizes
    constant int64_t * index_strides = (constant int64_t *)indexStrides;   // 将 indexStrides 转换为常量 int64_t 类型指针，并赋给 index_strides

    // 循环迭代，从 0 到 *numIters - 1
    for (uint iter_i = 0; iter_i < *numIters; iter_i++) {
        // 调用模板函数 index_put_impl<T>，传递相应参数进行索引操作
        index_put_impl<T>(indexAB, index_sizes, index_strides, offsets, inputData, outputData, num_indices, iter_i);
    }
}

template<typename T, typename OffsetsT>
kernel void index_put(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB           [[buffer(0)]],   // 如果 Metal 版本 >= 300，定义一个常量指针 IndexAB 对象，从第一个缓冲区获取
#else
    constant IndexAB  & indexAB           [[buffer(0)]],   // 如果 Metal 版本 < 300，定义一个常量引用 IndexAB 对象，从第一个缓冲区获取
#endif
    constant void     * indexSizes        [[buffer(1)]],   // 定义一个指向常量数据的指针，从第二个缓冲区获取
    constant void     * indexStrides      [[buffer(2)]],   // 定义一个指向常量数据的指针，从第三个缓冲区获取
    constant OffsetsT * offsets           [[buffer(3)]],   // 定义一个指向 OffsetsT 类型数据的指针，从第四个缓冲区获取
    constant void     * inputData         [[buffer(4)]],   // 定义一个指向常量数据的指针，从第五个缓冲区获取
    device   void     * outputData        [[buffer(5)]],   // 定义一个指向设备数据的指针，从第六个缓冲区获取
    constant uint32_t & num_indices       [[buffer(6)]],   // 定义一个常量引用 uint32_t 类型的数据，从第七个缓冲区获取
    uint thread_index [[thread_position_in_grid]]) {        // 定义一个无符号整数 thread_index，表示线程在网格中的位置索引

    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;     // 将 indexSizes 转换为常量 int64_t 类型指针，并赋给 index_sizes
    constant int64_t * index_strides = (constant int64_t *)indexStrides;   // 将 indexStrides 转换为常量 int64_t 类型指针，并赋给 index_strides
    // 调用模板函数 index_put_impl<T>，传递相应参数进行索引操作
    index_put_impl<T>(indexAB, index_sizes, index_strides, offsets, inputData, outputData, num_indices, thread_index);
}

#if __METAL_VERSION__ < 300
#define REGISTER_INDEX_OP(DTYPE_SIZE, IDX_SIZE, DTYPE, INDEX_OP_TYPE, IDX_DTYPE)   \
template                                                                           \
[[host_name("index_" #INDEX_OP_TYPE "_" #DTYPE_SIZE "_" #IDX_SIZE)]]               \
kernel void index_ ## INDEX_OP_TYPE<DTYPE, IDX_DTYPE>(                             \
    constant IndexAB & indexAB           [[buffer(0)]],                            \
    constant void    * indexSizes        [[buffer(1)]],                            \
    constant void    * indexStrides      [[buffer(2)]],                            \
    constant IDX_DTYPE   * offsets           [[buffer(3)]],                        \
    constant void    * inputData         [[buffer(4)]],                            \
    device   void    * outputData        [[buffer(5)]],                            \
    constant uint32_t & num_indices      [[buffer(6)]],                            \
    uint thread_index [[thread_position_in_grid]]);
#else
#define REGISTER_INDEX_OP(DTYPE_SIZE, IDX_SIZE, DTYPE, INDEX_OP_TYPE, IDX_DTYPE)   \
template                                                                           \
#if __METAL_VERSION__ < 300
// 定义单线程索引操作的模板，使用给定的数据类型、索引类型和操作类型
template
// 在 Metal 中定义主机函数名称，用于生成唯一的函数名
[[host_name("index_" #INDEX_OP_TYPE "_" #DTYPE_SIZE "_" #IDX_SIZE)]]
// Metal 内核函数，实现索引操作
kernel void index_ ## INDEX_OP_TYPE<DTYPE, IDX_DTYPE>(
    // Metal 缓冲区：存储 IndexAB 结构体，用于索引操作
    constant IndexAB   & indexAB           [[buffer(0)]],
    // Metal 缓冲区：存储索引大小信息
    constant void      * indexSizes        [[buffer(1)]],
    // Metal 缓冲区：存储索引步长信息
    constant void      * indexStrides      [[buffer(2)]],
    // Metal 缓冲区：存储偏移量信息
    constant IDX_DTYPE * offsets           [[buffer(3)]],
    // Metal 缓冲区：存储输入数据
    constant void      * inputData         [[buffer(4)]],
    // Metal 缓冲区：存储输出数据
    device   void      * outputData        [[buffer(5)]],
    // Metal 缓冲区：存储索引的数量
    constant uint32_t  & num_indices       [[buffer(6)]],
    // Metal 缓冲区：存储迭代次数
    constant uint      * numIters          [[buffer(7)]],
    // Metal 线程索引，表示当前线程在整个线程网格中的位置
    uint thread_index [[thread_position_in_grid]]);
#else
// 如果 Metal 版本高于或等于 300，则使用条件编译的 else 分支
#define REGISTER_SINGLE_THREADED_INDEX_OP(DTYPE_SIZE, IDX_SIZE, DTYPE, INDEX_OP_TYPE, IDX_DTYPE)   \
// 此处留空，没有额外的代码需要添加
template                                                                                           \
[[host_name("index_" #INDEX_OP_TYPE "_" #DTYPE_SIZE "_" #IDX_SIZE)]]                               \
kernel void index_ ## INDEX_OP_TYPE<DTYPE, IDX_DTYPE>(                                             \
    constant IndexAB   * indexAB           [[buffer(0)]],                                          \
    constant void      * indexSizes        [[buffer(1)]],                                          \
    constant void      * indexStrides      [[buffer(2)]],                                          \
    constant IDX_DTYPE * offsets           [[buffer(3)]],                                          \
    constant void      * inputData         [[buffer(4)]],                                          \
    device   void      * outputData        [[buffer(5)]],                                          \
    constant uint32_t  & num_indices       [[buffer(6)]],                                          \
    constant uint      * numIters          [[buffer(7)]],                                          \
    uint thread_index [[thread_position_in_grid]]);
#endif

#define REGISTER_SINGLE_THREADED_INDEX_OP_ALL_DTYPES(INDEX_OP_TYPE)                   \
    REGISTER_SINGLE_THREADED_INDEX_OP(8bit,  idx32, char,  INDEX_OP_TYPE, uint3);     \
    REGISTER_SINGLE_THREADED_INDEX_OP(8bit,  idx64, char,  INDEX_OP_TYPE, ulong3);    \
    REGISTER_SINGLE_THREADED_INDEX_OP(16bit, idx32, short, INDEX_OP_TYPE, uint3);     \
    REGISTER_SINGLE_THREADED_INDEX_OP(16bit, idx64, short, INDEX_OP_TYPE, ulong3);    \
    REGISTER_SINGLE_THREADED_INDEX_OP(32bit, idx32, int,   INDEX_OP_TYPE, uint3);     \
    REGISTER_SINGLE_THREADED_INDEX_OP(32bit, idx64, int,   INDEX_OP_TYPE, ulong3);    \
    REGISTER_SINGLE_THREADED_INDEX_OP(64bit, idx32, long,  INDEX_OP_TYPE, uint3);     \
    REGISTER_SINGLE_THREADED_INDEX_OP(64bit, idx64, long,  INDEX_OP_TYPE, ulong3);

REGISTER_SINGLE_THREADED_INDEX_OP_ALL_DTYPES(put_serial);

// 定义模板函数 kernel_index_offsets，计算索引偏移量
template<typename StridesT, typename DataT>
kernel void kernel_index_offsets(constant StridesT * strides         [[buffer(0)]],
                                device DataT      * data_offsets    [[buffer(1)]],
                                constant uint     * iter_shape      [[buffer(2)]],
                                constant uint     & num_dimensions  [[buffer(3)]],
                                uint thread_index [[thread_position_in_grid]]) {
    // 将 data_offsets 的当前索引位置初始化为 0
    data_offsets[thread_index] = 0;
    // 初始化索引为当前线程的索引
    uint32_t idx = thread_index;
    // 遍历每一个维度
    for (uint32_t dim = 0; dim < num_dimensions; dim++) {
        // 计算当前维度上的余数
        uint32_t remainder = idx % iter_shape[dim];
        // 更新 idx，准备计算下一个维度的索引
        idx /= iter_shape[dim];

        // 根据当前维度的余数和对应的步长，更新 data_offsets
        data_offsets[thread_index] += remainder * DataT(strides[dim]);
    }
}

template
[[host_name("kernel_index_offsets_32")]]
kernel void kernel_index_offsets<packed_uint3, uint3>(
                constant packed_uint3 * strides         [[buffer(0)]],
                device uint3          * data_offsets    [[buffer(1)]],
                constant uint         * iter_shape      [[buffer(2)]],
                constant uint         & num_dimensions  [[buffer(3)]],
                uint thread_index [[thread_position_in_grid]]);


// 定义 Metal GPU 计算核函数，计算索引偏移量
kernel void kernel_index_offsets<packed_uint3, uint3>(
    // 输入参数：步长数组，用于计算偏移量
    constant packed_uint3 * strides         [[buffer(0)]],
    // 输出参数：数据偏移量数组，保存每个线程的结果
    device uint3          * data_offsets    [[buffer(1)]],
    // 输入参数：迭代形状数组，指定每个维度的迭代次数
    constant uint         * iter_shape      [[buffer(2)]],
    // 输入参数：维度数量，定义迭代的维度数目
    constant uint         & num_dimensions  [[buffer(3)]],
    // 线程索引，标识当前线程在网格中的位置
    uint thread_index [[thread_position_in_grid]]);



template
[[host_name("kernel_index_offsets_64")]]
kernel void kernel_index_offsets<packed_uint3, ulong3>(
                constant packed_uint3 * strides         [[buffer(0)]],
                device ulong3          * data_offsets    [[buffer(1)]],
                constant uint         * iter_shape      [[buffer(2)]],
                constant uint         & num_dimensions  [[buffer(3)]],
                uint thread_index [[thread_position_in_grid]]);


// 定义 Metal GPU 计算核函数，计算索引偏移量（64位版本）
template
[[host_name("kernel_index_offsets_64")]]
kernel void kernel_index_offsets<packed_uint3, ulong3>(
    // 输入参数：步长数组，用于计算偏移量
    constant packed_uint3 * strides         [[buffer(0)]],
    // 输出参数：数据偏移量数组，保存每个线程的结果（64位无符号整数）
    device ulong3          * data_offsets    [[buffer(1)]],
    // 输入参数：迭代形状数组，指定每个维度的迭代次数
    constant uint         * iter_shape      [[buffer(2)]],
    // 输入参数：维度数量，定义迭代的维度数目
    constant uint         & num_dimensions  [[buffer(3)]],
    // 线程索引，标识当前线程在网格中的位置
    uint thread_index [[thread_position_in_grid]]);



template<typename T, typename E, typename OffsetsT>
kernel void index_put_accumulate_native_dtypes(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB     [[buffer(0)]],
#else
    constant IndexAB  & indexAB     [[buffer(0)]],
#endif
    constant void     * indexSizes   [[buffer(1)]],
    constant void     * indexStrides [[buffer(2)]],
    constant OffsetsT * offsets      [[buffer(3)]],
    constant void     * inputData    [[buffer(4)]],
    device void       * outputData   [[buffer(5)]],
    constant uint32_t & num_indices  [[buffer(6)]],
    uint thread_index [[thread_position_in_grid]]) {
    // 解析索引数组的大小和步长
    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;
    constant int64_t * index_strides = (constant int64_t *)indexStrides;
    int64_t offset = 0;
    // 对每个索引执行累加操作
    for (uint32_t i = 0; i < num_indices; i++) {
#if __METAL_VERSION__ >= 300
        // 获取当前索引数组的指针
        constant int64_t* indexArray = indexAB[i].indexArray;
#else
        // 获取当前索引数组的指针（低版本兼容）
        constant int64_t* indexArray = (constant int64_t*)indexAB.indexArray[i];
#endif
        // 计算当前索引的偏移量
        int64_t index = indexArray[offsets[thread_index].z / sizeof(int64_t)];
        // 处理负数索引，将其映射到有效范围内
        if (index < 0) {
            index += index_sizes[i];
        }
        // 计算偏移量并更新
        offset += index * index_strides[i];
    }
    // 计算输出数据的地址
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x + offset);
    // 计算输入数据的地址
    constant E * in  = (constant E*)((constant char*)inputData  + offsets[thread_index].y);
    // 原子操作：将输入数据累加到输出数据
    atomic_fetch_add_explicit(out, *in, memory_order_relaxed);
}



template<typename T>
__attribute__((__always_inline__)) void atomic_fetch_add_relaxed(device void * addr, T value) {
    // 转换地址类型为原子无符号整数指针
    device atomic_uint* uintAddr = (device atomic_uint*)addr;
    // 使用松散内存顺序加载原子变量值
    uint expected = atomic_load_explicit(uintAddr, memory_order_relaxed);
    // 计算更新后的值
    T updated = as_type<T>(expected) + value;
    // 使用松散内存顺序进行比较和交换操作，直至成功
    while (!atomic_compare_exchange_weak_explicit(uintAddr, &expected, as_type<uint>(updated), memory_order_relaxed, memory_order_relaxed)) {
        updated = as_type<T>(expected) + value;
    }
}



template<typename T, typename OffsetsT>
kernel void atomic_index_put_accumulate(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB           [[buffer(0)]],
#else


// 定义 Metal GPU 计算核函数，支持原子索引累加操作
template<typename T, typename OffsetsT>
kernel void atomic_index_put_accumulate(
#if __METAL_VERSION__ >= 300
    // 输入参数：索引 A/B 结构体数组
    constant IndexAB  * indexAB           [[buffer(0)]],
#else
    // 输入参数：索引 A/B 结构体
    constant IndexAB  & indexAB           [[buffer(0)]],
#endif
    // 输入参数：索引大小数组
    constant void     * indexSizes         [[buffer(1)]],
    // 输入参数：索引步长数组
    constant void     * indexStrides       [[buffer(2)]],
    // 输入参数：偏移量数组
    constant OffsetsT * offsets            [[buffer(3)]],
    // 输入参数：输入数据数组
    constant void     * inputData          [[buffer(4)]],
    // 输出参数：输出数据数组
    device void       * outputData         [[buffer(5)]],
    // 输入参数：索引数量
    constant uint32_t & num_indices        [[buffer(6)]],
    // 线程索引，标识当前线程在网格中的位置
    uint thread_index [[thread_position_in_grid]]) {
    # 声明一个常量变量 IndexAB，并将其与 indexAB 绑定在一起，位于缓冲区的第一个位置
    constant IndexAB & indexAB [[buffer(0)]],
#endif
    // 声明 indexSizes 参数为常量指针，用于存储索引大小信息
    constant void     * indexSizes        [[buffer(1)]],
    // 声明 indexStrides 参数为常量指针，用于存储索引步长信息
    constant void     * indexStrides      [[buffer(2)]],
    // 声明 offsets 参数为 OffsetsT 类型的常量指针，用于存储偏移量信息
    constant OffsetsT * offsets           [[buffer(3)]],
    // 声明 inputData 参数为常量指针，用于存储输入数据
    constant void     * inputData         [[buffer(4)]],
    // 声明 outputData 参数为设备端的指针，用于存储输出数据
    device   void     * outputData        [[buffer(5)]],
    // 声明 num_indices 参数为无符号整数常量引用，用于存储索引数量
    constant uint32_t & num_indices       [[buffer(6)]],
    // 声明 thread_index 参数为无符号整数，表示线程在网格中的位置索引
    uint thread_index [[thread_position_in_grid]]) {
    // 将 indexSizes 转换为 int64_t 类型的常量指针
    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;
    // 将 indexStrides 转换为 int64_t 类型的常量指针
    constant int64_t * index_strides = (constant int64_t *)indexStrides;
    // 初始化偏移量为 0
    int64_t offset = 0;
    // 循环遍历 num_indices 次，处理每个索引
    for (uint32_t i = 0; i < num_indices; i++) {
#if __METAL_VERSION__ >= 300
        // 如果 Metal 版本大于等于 300，则使用 indexAB[i].indexArray 获取索引数组
        constant int64_t* indexArray = indexAB[i].indexArray;
#else
        // 如果 Metal 版本小于 300，则使用 indexAB.indexArray[i] 获取索引数组
        constant int64_t* indexArray = (constant int64_t*)indexAB.indexArray[i];
#endif
        // 根据偏移量获取索引值
        int64_t index = indexArray[offsets[thread_index].z / sizeof(int64_t)];
        // 如果索引值小于 0，则加上对应索引的大小
        if (index < 0) {
            index += index_sizes[i];
        }
        // 更新偏移量
        offset += index * index_strides[i];
    }
    // 计算输出数据的地址
    device void * out = (device void*)((device char*)outputData + offsets[thread_index].x + offset);
    // 获取输入数据的地址
    constant T  * in  = (constant T*)((constant char*)inputData + offsets[thread_index].y);
    // 对输出数据执行原子加操作
    atomic_fetch_add_relaxed<T>(out, *in);
}

template
[[host_name("index_put_accumulate_32bit_float_idx32")]]
kernel void atomic_index_put_accumulate<float, uint3>(
#if __METAL_VERSION__ >= 300
    // 如果 Metal 版本大于等于 300，则使用 indexAB 参数作为常量指针数组
    constant IndexAB  * indexAB     [[buffer(0)]],
#else
    // 如果 Metal 版本小于 300，则使用 indexAB 参数作为常量引用
    constant IndexAB  & indexAB     [[buffer(0)]],
#endif
    // 其余参数与前面的 kernel 类似，具体含义参见上述注释
    constant void     * indexSizes   [[buffer(1)]],
    constant void     * indexStrides [[buffer(2)]],
    constant uint3    * offsets      [[buffer(3)]],
    constant void     * inputData    [[buffer(4)]],
    device   void     * outputData   [[buffer(5)]],
    constant uint32_t & num_indices  [[buffer(6)]],
    uint thread_index [[thread_position_in_grid]]);

template
[[host_name("index_put_accumulate_32bit_float_idx64")]]
kernel void atomic_index_put_accumulate<float, ulong3>(
#if __METAL_VERSION__ >= 300
    // 如果 Metal 版本大于等于 300，则使用 indexAB 参数作为常量指针数组
    constant IndexAB  * indexAB     [[buffer(0)]],
#else
    // 如果 Metal 版本小于 300，则使用 indexAB 参数作为常量引用
    constant IndexAB  & indexAB     [[buffer(0)]],
#endif
    // 其余参数与前面的 kernel 类似，具体含义参见上述注释
    constant void     * indexSizes   [[buffer(1)]],
    constant void     * indexStrides [[buffer(2)]],
    constant ulong3   * offsets      [[buffer(3)]],
    constant void     * inputData    [[buffer(4)]],
    device   void     * outputData   [[buffer(5)]],
    constant uint32_t & num_indices  [[buffer(6)]],
    uint thread_index [[thread_position_in_grid]]);

template
[[host_name("index_put_accumulate_32bit_int_idx32")]]
kernel void index_put_accumulate_native_dtypes<atomic_int, int, uint3>(
#if __METAL_VERSION__ >= 300
    // 如果 Metal 版本大于等于 300，则使用 indexAB 参数作为常量指针数组
    constant IndexAB  * indexAB     [[buffer(0)]],
#else
    // 如果 Metal 版本小于 300，则使用 indexAB 参数作为常量引用
    constant IndexAB  & indexAB     [[buffer(0)]],
#endif
    // 其余参数与前面的 kernel 类似，具体含义参见上述注释
    constant void     * indexSizes   [[buffer(1)]],
    constant void     * indexStrides [[buffer(2)]],
    constant uint3    * offsets      [[buffer(3)]],
    constant void     * inputData    [[buffer(4)]],
    // 定义名为 device 的指针变量，指向 void 类型的数据，用于存储输出数据，绑定到索引为 5 的缓冲区
    device void * outputData [[buffer(5)]],
    // 定义名为 num_indices 的常量引用，指向 uint32_t 类型的数据，用于存储索引数量，绑定到索引为 6 的缓冲区
    constant uint32_t & num_indices [[buffer(6)]],
    // 定义名为 thread_index 的 uint 变量，存储当前线程在网格中的位置索引
    uint thread_index [[thread_position_in_grid]]);
// 定义一个模板字符串，包含 Metal 标注指令，用于 kernel 函数的声明
[[host_name("index_put_accumulate_32bit_int_idx64")]]
kernel void index_put_accumulate_native_dtypes<atomic_int, int, ulong3>(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB     [[buffer(0)]],
#else
    constant IndexAB  & indexAB     [[buffer(0)]],
#endif
    constant void     * indexSizes   [[buffer(1)]],
    constant void     * indexStrides [[buffer(2)]],
    constant ulong3   * offsets      [[buffer(3)]],
    constant void     * inputData    [[buffer(4)]],
    device   void     * outputData   [[buffer(5)]],
    constant uint32_t & num_indices  [[buffer(6)]],
    uint thread_index                [[thread_position_in_grid]]);
// 上述代码定义了一个 Metal kernel 函数，用于执行特定的索引和累加操作，处理原子整型和指定数据类型的输入输出数据。

static const char *SCATTER_OPS_TEMPLATE = R"METAL_SCATTER(
// 定义一个 Metal 字符串模板，用于 scatter_kernel 函数的实现

struct __attribute__ ((packed)) packed_uint5{
  uint32_t x; uint32_t y; uint32_t z; uint32_t w; uint32_t u;
};

// 定义一个通用的类型转换模板函数
template<typename Y, typename X>
Y cast(const X x);

// 特化模板函数，将 {0} 类型转换为 {1} 类型
template<>
{1} cast<{1}, {0}>(const {0} x) {{
 return {2};
}}

// Metal kernel 函数，用于处理五维索引的散射操作
kernel void scatter_kernel_5(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint5 & size   [[buffer(2)]],
                             constant packed_uint5 & stride [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint5 local_index;
    local_index.x = linear_index / (size.u * size.w * size.z * size.y) % size.x;
    local_index.y = linear_index / (size.u * size.w * size.z) % size.y;
    local_index.z = linear_index / (size.u * size.w) % size.z;
    local_index.w = linear_index / size.u % size.w;
    local_index.u = linear_index % size.u;

    packed_uint5 strided_index;
    strided_index.x = local_index.x * stride.x;
    strided_index.y = local_index.y * stride.y;
    strided_index.z = local_index.z * stride.z;
    strided_index.w = local_index.w * stride.w;
    strided_index.u = local_index.u * stride.u;

    dst[strided_index.x + strided_index.y + strided_index.z + strided_index.w + strided_index.u] = cast<{1}>(src[linear_index]);
}

// Metal kernel 函数，用于处理四维索引的散射操作
kernel void scatter_kernel_4(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint4 & size   [[buffer(2)]],
                             constant packed_uint4 & stride [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint4 local_index;
    local_index.x = linear_index / (size[3] * size[2] * size[1]) % size[0];
    # 计算在四维数组中的索引，每个维度的值计算如下：
    # local_index.y: 计算线性索引除以第四维度和第三维度的乘积再取模第二维度的结果
    local_index.y = linear_index / (size[3] * size[2]) % size[1];
    # local_index.z: 计算线性索引除以第四维度再取模第三维度的结果
    local_index.z = linear_index / size[3] % size[2];
    # local_index.w: 计算线性索引对第四维度取模的结果
    local_index.w = linear_index % size[3];

    # 计算目标数组中的跨步索引，将本地索引乘以给定的步幅得到跨步索引
    const packed_uint4 strided_index = local_index * stride;
    # 将源数组中的线性索引处的元素转换为目标类型后存放在目标数组中对应的位置
    dst[strided_index.x + strided_index.y + strided_index.z + strided_index.w] = cast<{1}>(src[linear_index]);
kernel void scatter_kernel_3(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint3 & size   [[buffer(2)]],
                             constant packed_uint3 & stride [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    // 检查当前线性索引是否超出元素数量，如果是则返回
    if (linear_index >= numel) return;

    // 将 src_ 转换为常量指针指向类型为 {0} 的数据
    constant {0} * src = (constant {0} *)src_;
    // 将 dst_ 转换为设备指针指向类型为 {1} 的数据
    device {1} * dst = (device {1} *)dst_;

    // 计算本地索引
    packed_uint3 local_index;
    local_index.x = linear_index / (size[2] * size[1]) % size[0];
    local_index.y = linear_index / size[2] % size[1];
    local_index.z = linear_index % size[2];

    // 计算带步长的索引
    const packed_uint3 strided_index = local_index * stride;
    // 将 src 中的数据强制转换为类型 {1} 并写入 dst 中对应的位置
    dst[strided_index.x + strided_index.y + strided_index.z] = cast<{1}>(src[linear_index]);
}}

kernel void scatter_kernel_2(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint2 & size   [[buffer(2)]],
                             constant packed_uint2 & stride [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    // 检查当前线性索引是否超出元素数量，如果是则返回
    if (linear_index >= numel) return;

    // 将 src_ 转换为常量指针指向类型为 {0} 的数据
    constant {0} * src = (constant {0} *)src_;
    // 将 dst_ 转换为设备指针指向类型为 {1} 的数据
    device {1} * dst = (device {1} *)dst_;

    // 计算本地索引
    packed_uint2 local_index;
    local_index.x = linear_index / size[1] % size[0];
    local_index.y = linear_index % size[1];

    // 计算带步长的索引
    const packed_uint2 strided_index = local_index * stride;
    // 将 src 中的数据强制转换为类型 {1} 并写入 dst 中对应的位置
    dst[strided_index.x + strided_index.y] = cast<{1}>(src[linear_index]);
}}

kernel void scatter_kernel_1(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant int & size            [[buffer(2)]],
                             constant int & stride          [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    // 检查当前线性索引是否超出元素数量，如果是则返回
    if (linear_index >= numel) return;

    // 将 src_ 转换为常量指针指向类型为 {0} 的数据
    constant {0} * src = (constant {0} *)src_;
    // 将 dst_ 转换为设备指针指向类型为 {1} 的数据
    device {1} * dst = (device {1} *)dst_;

    // 计算本地索引
    const int local_index = linear_index % size;
    // 计算带步长的索引
    const int strided_index = local_index * stride;
    // 将 src 中的数据强制转换为类型 {1} 并写入 dst 中对应的位置
    dst[strided_index] = cast<{1}>(src[linear_index]);
}}
)METAL_SCATTER";

static const char *GATHER_OPS_TEMPLATE = R"METAL_GATHER(
struct __attribute__ ((packed)) packed_uint5{{
  uint32_t x; uint32_t y; uint32_t z; uint32_t w; uint32_t u;
}};

// 模板函数，用于将 X 类型的数据转换为 Y 类型
template<typename Y, typename X>
Y cast(const X x);

// 具体化模板函数，将类型为 {0} 的数据 x 转换为类型为 {1} 的数据
template<>
{1} cast<{1}, {0}>(const {0} x) {{
 return {2};
}}
)METAL_GATHER";


注释：
# 根据线性索引确定当前线程在网格中的位置，进行数据收集操作
kernel void gather_kernel_5(uint linear_index               [[thread_position_in_grid]],
                            # 源数据的常量指针，用于读取数据
                            constant void * src_            [[buffer(0)]],
                            # 目标数据的设备指针，用于写入数据
                            device void * dst_              [[buffer(1)]],
                            # 数据尺寸的压缩表示，用于计算索引
                            constant packed_uint5 & size    [[buffer(2)]],
                            # 步长的压缩表示，用于计算步长索引
                            constant packed_uint5 & stride  [[buffer(3)]],
                            # 数据总数
                            constant uint32_t & numel       [[buffer(4)]]) {
    # 如果线性索引超出数据总数，则退出
    if (linear_index >= numel) return;

    # 将源数据指针转换为特定类型的常量指针
    constant {0} * src = (constant {0} *)src_;
    # 将目标数据指针转换为特定类型的设备指针
    device {1} * dst = (device {1} *)dst_;

    # 声明局部索引变量
    packed_uint5 local_index;
    # 计算当前线性索引在各维度上的局部索引
    local_index.x = linear_index / (size.u * size.w * size.z * size.y) % size.x;
    local_index.y = linear_index / (size.u * size.w * size.z) % size.y;
    local_index.z = linear_index / (size.u * size.w) % size.z;
    local_index.w = linear_index / size.u % size.w;
    local_index.u = linear_index % size.u;

    # 声明步长索引变量
    packed_uint5 strided_index;
    # 计算步长索引在各维度上的值
    strided_index.x = local_index.x * stride.x;
    strided_index.y = local_index.y * stride.y;
    strided_index.z = local_index.z * stride.z;
    strided_index.w = local_index.w * stride.w;
    strided_index.u = local_index.u * stride.u;

    # 将源数据根据步长索引读取，并写入目标数据
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y + strided_index.z + strided_index.w + strided_index.u]);
}}

# 根据线性索引确定当前线程在网格中的位置，进行数据收集操作
kernel void gather_kernel_4(uint linear_index               [[thread_position_in_grid]],
                            # 源数据的常量指针，用于读取数据
                            constant void * src_            [[buffer(0)]],
                            # 目标数据的设备指针，用于写入数据
                            device void * dst_              [[buffer(1)]],
                            # 数据尺寸的压缩表示，用于计算索引
                            constant packed_uint4 & size    [[buffer(2)]],
                            # 步长的压缩表示，用于计算步长索引
                            constant packed_uint4 & stride  [[buffer(3)]],
                            # 数据总数
                            constant uint32_t & numel       [[buffer(4)]]) {
    # 如果线性索引超出数据总数，则退出
    if (linear_index >= numel) return;

    # 将源数据指针转换为特定类型的常量指针
    constant {0} * src = (constant {0} *)src_;
    # 将目标数据指针转换为特定类型的设备指针
    device {1} * dst = (device {1} *)dst_;

    # 声明局部索引变量
    packed_uint4 local_index;
    # 计算当前线性索引在各维度上的局部索引
    local_index.x = linear_index / (size[3] * size[2] * size[1]) % size[0];
    local_index.y = linear_index / (size[3] * size[2]) % size[1];
    local_index.z = linear_index / size[3] % size[2];
    local_index.w = linear_index % size[3];

    # 声明步长索引变量
    const packed_uint4 strided_index = local_index * stride;

    # 将源数据根据步长索引读取，并写入目标数据
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y + strided_index.z + strided_index.w]);
}}

# 根据线性索引确定当前线程在网格中的位置，进行数据收集操作
kernel void gather_kernel_3(uint linear_index               [[thread_position_in_grid]],
                            # 源数据的常量指针，用于读取数据
                            constant void * src_            [[buffer(0)]],
                            # 目标数据的设备指针，用于写入数据
                            device void * dst_              [[buffer(1)]],
                            # 数据尺寸的压缩表示，用于计算索引
                            constant packed_uint3 & size    [[buffer(2)]],
                            # 步长的压缩表示，用于计算步长索引
                            constant packed_uint3 & stride  [[buffer(3)]],
                            # 数据总数
                            constant uint32_t & numel       [[buffer(4)]]) {
    # 如果线性索引超出数据总数，则退出
    if (linear_index >= numel) return;

    # 将源数据指针转换为特定类型的常量指针
    constant {0} * src = (constant {0} *)src_;
    # 将目标数据指针转换为特定类型的设备指针
    device {1} * dst = (device {1} *)dst_;
    # 定义一个结构体或类型为 packed_uint3 的变量 local_index，用于存储本地索引信息
    packed_uint3 local_index;
    # 计算在三维数组中的本地索引 x，表示在第 0 维的位置
    local_index.x = linear_index / (size[2] * size[1]) % size[0];
    # 计算在三维数组中的本地索引 y，表示在第 1 维的位置
    local_index.y = linear_index / size[2] % size[1];
    # 计算在三维数组中的本地索引 z，表示在第 2 维的位置
    local_index.z = linear_index % size[2];

    # 定义一个常量结构体或类型为 packed_uint3 的变量 strided_index，表示按步长跨越的索引
    const packed_uint3 strided_index = local_index * stride;
    # 将源数组 src 中按照 strided_index 计算的索引处的数据转换为目标类型 {1}，并存入目标数组 dst 中的 linear_index 位置
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y + strided_index.z]);
// 定义 Metal 程序的第二个采集核函数，用于在 GPU 上执行数据采集操作
kernel void gather_kernel_2(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant packed_uint2 & size    [[buffer(2)]],
                            constant packed_uint2 & stride  [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    // 如果线性索引超出数据总数，则返回不执行
    if (linear_index >= numel) return;

    // 将输入源转换为常量指针
    constant {0} * src = (constant {0} *)src_;
    // 将输出目标转换为设备指针
    device {1} * dst = (device {1} *)dst_;

    // 定义局部索引对象，计算局部索引 x 和 y 分量
    packed_uint2 local_index;
    local_index.x = linear_index / size[1] % size[0];
    local_index.y = linear_index % size[1];

    // 计算采样后的索引位置
    const packed_uint2 strided_index = local_index * stride;
    // 将采集到的数据转换并存储到输出目标中
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y]);
}}

// 定义 Metal 程序的第一个采集核函数，用于在 GPU 上执行数据采集操作
kernel void gather_kernel_1(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant int & size             [[buffer(2)]],
                            constant int & stride           [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    // 如果线性索引超出数据总数，则返回不执行
    if (linear_index >= numel) return;

    // 将输入源转换为常量指针
    constant {0} * src = (constant {0} *)src_;
    // 将输出目标转换为设备指针
    device {1} * dst = (device {1} *)dst_;

    // 计算局部索引
    const int local_index = linear_index % size;
    // 计算采样后的索引位置
    const int strided_index = local_index * stride;
    // 将采集到的数据转换并存储到输出目标中
    dst[linear_index] = cast<{1}>(src[strided_index]);
}}
```