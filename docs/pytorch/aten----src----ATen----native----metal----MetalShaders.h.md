# `.\pytorch\aten\src\ATen\native\metal\MetalShaders.h`

```py
#ifndef MPSCNNShaders_h
#define MPSCNNShaders_h

// 定义 Metal Shader 代码字符串常量
static const char* PT_METAL_SHADERS = R"PT_METAL_SHADERS(
#include <metal_stdlib>
using namespace metal;

// 定义函数常量，包括 ushort 和 float 类型的参数
constant ushort ushort_arg_0[[function_constant(0)]];
constant ushort ushort_arg_1[[function_constant(1)]];
constant ushort ushort_arg_2[[function_constant(2)]];
constant ushort ushort_arg_3[[function_constant(3)]];
constant ushort ushort_arg_4[[function_constant(4)]];
constant ushort ushort_arg_5[[function_constant(5)]];
constant ushort ushort_arg_6[[function_constant(6)]];
constant ushort ushort_arg_7[[function_constant(7)]];
constant ushort ushort_arg_8[[function_constant(8)]];
constant ushort ushort_arg_9[[function_constant(9)]];
constant ushort ushort_arg_10[[function_constant(10)]];
constant ushort ushort_arg_11[[function_constant(11)]];
constant float float_arg_0 [[function_constant(12)]];
constant float float_arg_1 [[function_constant(13)]];

// 定义一个内联函数，用于计算 ushort 类型的除法向上取整
inline constexpr ushort divRoundUp(ushort x, ushort y) { return (x + (y - 1)) / y; }

// 定义一个枚举类型，表示广播操作的种类
enum broadcastOp {
    Add,  // 加法操作
    Sub,  // 减法操作
    Mul,  // 乘法操作
    Div,  // 除法操作
};

// 实现一个元素级广播操作，针对非数组的情况
void elementwise_broadcast_nonarray(texture2d<half, access::read> in0,
                                   texture2d<half, access::read> in1,
                                   texture2d<half, access::write> out,
                                   ushort2 gid,
                                   broadcastOp op) {
    // 如果全局索引超出输出纹理的尺寸，则返回
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }

    // 确定输入纹理的步长
    ushort2 in0_stride = ushort2(in0.get_width() > 1, in0.get_height() > 1);
    ushort2 in1_stride = ushort2(in1.get_width() > 1, in1.get_height() > 1);

    // 计算输入纹理的全局索引
    ushort2 gid0 = gid.xy * in0_stride;
    ushort2 gid1 = gid.xy * in1_stride;

    // 根据操作类型执行相应的元素级广播操作
    if(op == Add) {
        out.write(in0.read(gid0) + in1.read(gid1), gid);
    } else if(op == Sub) {
        out.write(in0.read(gid0) - in1.read(gid1), gid);
    } else if(op == Mul) {
        out.write(in0.read(gid0) * in1.read(gid1), gid);
    } else if(op == Div) {
        out.write(in0.read(gid0) / in1.read(gid1), gid);
    }
}

// 实现一个元素级广播操作，针对数组的情况
void elementwise_broadcast(texture2d_array<half, access::read> in0,
                           texture2d_array<half, access::read> in1,
                           texture2d_array<half, access::write> out,
                           ushort3 gid,
                           broadcastOp op) {
    // 如果全局索引超出输出纹理的尺寸，则返回
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }

    // 确定输入纹理的步长
    ushort2 in0_stride = ushort2(in0.get_width() > 1, in0.get_height() > 1);
    ushort2 in1_stride = ushort2(in1.get_width() > 1, in1.get_height() > 1);

    // 计算输入纹理的全局索引
    ushort2 gid0 = gid.xy * in0_stride;
    ushort2 gid1 = gid.xy * in1_stride;

    // 根据操作类型执行相应的元素级广播操作
    if(op == Add) {
        out.write(in0.read(gid0, gid.z) + in1.read(gid1, gid.z), gid.xy, gid.z);
    } else if(op == Sub) {
        out.write(in0.read(gid0, gid.z) - in1.read(gid1, gid.z), gid.xy, gid.z);
    } else if(op == Mul) {
        out.write(in0.read(gid0, gid.z) * in1.read(gid1, gid.z), gid.xy, gid.z);
    }
}
    # 如果操作符是除法 (Div)，执行以下操作
    else if(op == Div) {
        # 从输入图像in0和in1中读取指定全局ID gid0 和 gid1 处的像素值，并进行除法运算
        # 将结果写入输出图像out，写入位置为全局ID gid 的 xy 坐标，z 坐标不变
        out.write(in0.read(gid0, gid.z) / in1.read(gid1, gid.z), gid.xy, gid.z);
    }
kernel void elementwise_add_nonarray(texture2d<half, access::read> in0[[texture(0)]],  // 从第一个纹理读取输入数据（半精度浮点数），只读访问
                                     texture2d<half, access::read> in1[[texture(1)]],  // 从第二个纹理读取输入数据（半精度浮点数），只读访问
                                     texture2d<half, access::write> out[[texture(2)]],  // 写入输出数据到第三个纹理（半精度浮点数），可写访问
                                     ushort2 gid[[thread_position_in_grid]]) {  // 获取当前线程在网格中的位置信息，二维索引
    elementwise_broadcast_nonarray(in0, in1, out, gid, Add);  // 调用元素级广播函数，对输入进行逐元素操作（加法）
}

kernel void elementwise_add(texture2d_array<half, access::read> in0[[texture(0)]],  // 从第一个纹理数组读取输入数据（半精度浮点数），只读访问
                            texture2d_array<half, access::read> in1[[texture(1)]],  // 从第二个纹理数组读取输入数据（半精度浮点数），只读访问
                            texture2d_array<half, access::write> out[[texture(2)]],  // 写入输出数据到第三个纹理数组（半精度浮点数），可写访问
                            ushort3 gid[[thread_position_in_grid]]) {  // 获取当前线程在网格中的位置信息，三维索引
    elementwise_broadcast(in0, in1, out, gid, Add);  // 调用元素级广播函数，对输入进行逐元素操作（加法）
}

kernel void elementwise_sub_nonarray(texture2d<half, access::read> in0[[texture(0)]],  // 从第一个纹理读取输入数据（半精度浮点数），只读访问
                                     texture2d<half, access::read> in1[[texture(1)]],  // 从第二个纹理读取输入数据（半精度浮点数），只读访问
                                     texture2d<half, access::write> out[[texture(2)]],  // 写入输出数据到第三个纹理（半精度浮点数），可写访问
                                     ushort2 gid[[thread_position_in_grid]]) {  // 获取当前线程在网格中的位置信息，二维索引
    elementwise_broadcast_nonarray(in0, in1, out, gid, Sub);  // 调用元素级广播函数，对输入进行逐元素操作（减法）
}

kernel void elementwise_sub(texture2d_array<half, access::read> in0[[texture(0)]],  // 从第一个纹理数组读取输入数据（半精度浮点数），只读访问
                            texture2d_array<half, access::read> in1[[texture(1)]],  // 从第二个纹理数组读取输入数据（半精度浮点数），只读访问
                            texture2d_array<half, access::write> out[[texture(2)]],  // 写入输出数据到第三个纹理数组（半精度浮点数），可写访问
                            ushort3 gid[[thread_position_in_grid]]) {  // 获取当前线程在网格中的位置信息，三维索引
    elementwise_broadcast(in0, in1, out, gid, Sub);  // 调用元素级广播函数，对输入进行逐元素操作（减法）
}

kernel void elementwise_mul_nonarray(texture2d<half, access::read> in0[[texture(0)]],  // 从第一个纹理读取输入数据（半精度浮点数），只读访问
                                     texture2d<half, access::read> in1[[texture(1)]],  // 从第二个纹理读取输入数据（半精度浮点数），只读访问
                                     texture2d<half, access::write> out[[texture(2)]],  // 写入输出数据到第三个纹理（半精度浮点数），可写访问
                                     ushort2 gid[[thread_position_in_grid]]) {  // 获取当前线程在网格中的位置信息，二维索引
    elementwise_broadcast_nonarray(in0, in1, out, gid, Mul);  // 调用元素级广播函数，对输入进行逐元素操作（乘法）
}

kernel void elementwise_mul(texture2d_array<half, access::read> in0[[texture(0)]],  // 从第一个纹理数组读取输入数据（半精度浮点数），只读访问
                            texture2d_array<half, access::read> in1[[texture(1)]],  // 从第二个纹理数组读取输入数据（半精度浮点数），只读访问
                            texture2d_array<half, access::write> out[[texture(2)]],  // 写入输出数据到第三个纹理数组（半精度浮点数），可写访问
                            ushort3 gid[[thread_position_in_grid]]) {  // 获取当前线程在网格中的位置信息，三维索引
    elementwise_broadcast(in0, in1, out, gid, Mul);  // 调用元素级广播函数，对输入进行逐元素操作（乘法）
}

kernel void elementwise_div_nonarray(texture2d<half, access::read> in0[[texture(0)]],  // 从第一个纹理读取输入数据（半精度浮点数），只读访问
                                     texture2d<half, access::read> in1[[texture(1)]],  // 从第二个纹理读取输入数据（半精度浮点数），只读访问
                                     texture2d<half, access::write> out[[texture(2)]],  // 写入输出数据到第三个纹理（半精度浮点数），可写访问
                                     ushort2 gid[[thread_position_in_grid]]) {  // 获取当前线程在网格中的位置信息，二维索引
    elementwise_broadcast_nonarray(in0, in1, out, gid, Div);  // 调用元素级广播函数，对输入进行逐元素操作（除法）
}
kernel void elementwise_div(texture2d_array<half, access::read> in0[[texture(0)]],
                            texture2d_array<half, access::read> in1[[texture(1)]],
                            texture2d_array<half, access::write> out[[texture(2)]],
                            ushort3 gid[[thread_position_in_grid]]) {
    // 调用 elementwise_broadcast 函数，对输入纹理进行逐元素除法运算
    elementwise_broadcast(in0, in1, out, gid, Div);
}

kernel void copy_nchw_to_metal(constant float* in[[buffer(0)]],
                               texture2d_array<half, access::write> out[[texture(0)]],
                               ushort3 gid[[thread_position_in_grid]]) {
    // 从缓冲区中读取输入数据并将其复制到 Metal 纹理中，转换为 Metal 纹理数组的格式
    const ushort C = ushort_arg_0;
    const ushort H = ushort_arg_1;
    const ushort W = ushort_arg_2;
    // 如果线程超出了输出 Metal 纹理的宽度或高度，则直接返回
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    // 计算当前像素点在数据中的索引
    const ushort n = gid.z / divRoundUp(C, 4);
    const ushort c = gid.z - n * divRoundUp(C, 4);
    // 定义宏，将 NCHW 格式的数据转换为 Metal 纹理数组格式的数据
#define CHW_TO_CHWP4(idx, n, c_, h, w)                                     \
if ((c_) < C) {                                                          \
trns[idx] = in[n * H * W * C + int(c_) * H * W + int(h) * W + int(w)]; \
} else {                                                                 \
trns[idx] = 0.0h;                                                      \
}
    half4 trns;
    // 分别调用宏，转换四个通道的数据
    CHW_TO_CHWP4(0, n, c * 4 + 0, gid.y, gid.x);
    CHW_TO_CHWP4(1, n, c * 4 + 1, gid.y, gid.x);
    CHW_TO_CHWP4(2, n, c * 4 + 2, gid.y, gid.x);
    CHW_TO_CHWP4(3, n, c * 4 + 3, gid.y, gid.x);
#undef CHW_TO_CHWP4
    // 将转换后的数据写入输出 Metal 纹理数组中的指定位置
    out.write(trns, gid.xy, gid.z);
}

kernel void copy_nchw_to_metal_nonarray(constant float* in[[buffer(0)]],
                                        texture2d<half, access::write> out[[texture(0)]],
                                        ushort2 gid[[thread_position_in_grid]]) {
    // 将 NCHW 格式的数据复制到非数组形式的 Metal 纹理中
    const ushort C = ushort_arg_0;
    const ushort H = ushort_arg_1;
    const ushort W = ushort_arg_2;
    // 如果线程超出了输出 Metal 纹理的宽度或高度，则直接返回
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    half4 trns;
    // 定义宏，将 NCHW 格式的数据转换为 Metal 纹理的数据格式
#define CHW_TO_CHWP4(idx, c, h, w)                        \
if ((c) < C) {                                          \
trns[idx] = in[int(c) * H * W + int(h) * W + int(w)]; \
} else {                                                \
trns[idx] = 0.0h;                                     \
}
    // 分别调用宏，转换四个通道的数据
    CHW_TO_CHWP4(0, 0, gid.y, gid.x);
    CHW_TO_CHWP4(1, 1, gid.y, gid.x);
    CHW_TO_CHWP4(2, 2, gid.y, gid.x);
    CHW_TO_CHWP4(3, 3, gid.y, gid.x);
#undef CHW_TO_CHWP4
    // 将转换后的数据写入输出 Metal 纹理中的指定位置
    out.write(trns, gid.xy);
}

kernel void copy_metal_to_nchw(texture2d_array<half, access::read> in[[texture(0)]],
                               device float* out[[buffer(0)]],
                               ushort3 gid[[thread_position_in_grid]]) {
    // 将 Metal 纹理数组中的数据复制到 NCHW 格式的缓冲区中
    const ushort C = ushort_arg_0;
    const ushort H = ushort_arg_1;
    const ushort W = ushort_arg_2;
    // 如果线程超出了输出缓冲区的宽度或高度，则直接返回
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    // 计算当前像素点在 Metal 纹理数组中的索引，读取数据
    const ushort n = gid.z / divRoundUp(C, 4);
    const ushort c = gid.z - n * divRoundUp(C, 4);
    half4 cs = in.read(gid.xy, gid.z);
    // 将读取的数据写入到输出缓冲区中的指定位置
#define CHWP4_TO_CHW(idx, n, c_, h, w)                                    \
if ((c_) < C) {                                                         \
out[n * H * W * C + int(c_) * H * W + int(h) * W + int(w)] = cs[idx]; \
}

// 宏定义：将输入的数据按照索引 idx 重新组织到输出数组 out 中的指定位置，条件是通道数 c_ 必须小于 C。

    CHWP4_TO_CHW(0, n, c * 4 + 0, gid.y, gid.x);
    CHWP4_TO_CHW(1, n, c * 4 + 1, gid.y, gid.x);
    CHWP4_TO_CHW(2, n, c * 4 + 2, gid.y, gid.x);
    CHWP4_TO_CHW(3, n, c * 4 + 3, gid.y, gid.x);
#undef CHWP4_TO_CHW
// 使用上面定义的宏 CHWP4_TO_CHW 分别处理四个不同的通道数据，将每个通道的数据放置到输出数组 out 的相应位置，并在结束后取消宏定义。

kernel void copy_metal_to_nchw_nonarray(texture2d<half, access::read> in[[texture(0)]],
                                        device float* out[[buffer(0)]],
                                        ushort2 gid[[thread_position_in_grid]]) {
    const ushort C = ushort_arg_0;
    const ushort H = ushort_arg_1;
    const ushort W = ushort_arg_2;
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    half4 cs = in.read(gid.xy);
#define CHWP4_TO_CHW(idx, c, h, w)                       \
if ((c) < C) {                                         \
out[int(c) * H * W + int(h) * W + int(w)] = cs[idx]; \
}
    CHWP4_TO_CHW(0, 0, gid.y, gid.x);
    CHWP4_TO_CHW(1, 1, gid.y, gid.x);
    CHWP4_TO_CHW(2, 2, gid.y, gid.x);
    CHWP4_TO_CHW(3, 3, gid.y, gid.x);
#undef CHWP4_TO_CHW
}
// 将 Metal 格式的输入数据复制到 NCHW 非数组格式的输出数据中，处理每个通道的数据并按照 NCHW 格式重新排列。

kernel void copy(texture2d_array<half, access::read> in[[texture(0)]],
                 texture2d_array<half, access::write> out[[texture(1)]],
                 ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = gid.xy;
    out.write(in.read(gid_, gid.z), gid_, gid.z);
}
// 将输入的二维数组数据复制到输出的二维数组中，处理三维索引 gid 并进行读写操作。

kernel void copy_nonarray(texture2d<half, access::read> in[[texture(0)]],
                          texture2d<half, access::write> out[[texture(1)]],
                          ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    out.write(in.read(gid), gid);
}
// 将非数组格式的输入数据复制到非数组格式的输出数据中，处理二维索引 gid 并进行读写操作。

kernel void copy_offset(texture2d_array<half, access::read> in[[texture(0)]],
                        texture2d_array<half, access::write> out[[texture(1)]],
                        constant ushort* offset_buf[[buffer(0)]],
                        ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = gid.xy;
    out.write(in.read(gid_, gid.z), gid_, gid.z + offset_buf[0]);
}
// 将输入的二维数组数据复制到输出的二维数组中，并使用偏移量进行索引处理，处理三维索引 gid 和偏移量 offset_buf。

kernel void copy_offset_nonarray(texture2d<half, access::read> in[[texture(0)]],
                                 texture2d_array<half, access::write> out[[texture(1)]],
                                 constant ushort* offset_buf[[buffer(0)]],
                                 ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = gid.xy;
    out.write(in.read(gid_), gid_, gid.z + offset_buf[0]);
}
// 将非数组格式的输入数据复制到二维数组格式的输出数据中，并使用偏移量进行索引处理，处理三维索引 gid 和偏移量 offset_buf。

constant bool store_features_out_is_arr = (ushort_arg_3 > 1 || ushort_arg_2 > 4);
// 常量声明：根据参数 ushort_arg_3 和 ushort_arg_2 的值判断是否输出的特征是数组格式。
// 定义常量，确定输出是否为纹理（由 store_features_out_is_arr 反转得来）
constant bool store_features_out_is_tex = !store_features_out_is_arr;

// 存储特征的核函数，将输入纹理数组中的数据写入输出纹理或数组中
kernel void store_features(texture2d_array<half, access::read> in[[texture(0)]],
                           texture2d<half, access::write> out_tex[[texture(1), function_constant(store_features_out_is_tex)]],
                           texture2d_array<half, access::write> out_arr[[texture(1), function_constant(store_features_out_is_arr)]],
                           constant ushort* offset_buf[[buffer(0)]],
                           ushort3 gid[[thread_position_in_grid]]) {
    // 提取二维坐标信息
    ushort2 gid_ = gid.xy;

    // 根据输出类型（纹理还是数组），选择写入函数
    if (store_features_out_is_arr)
        // 将输入纹理数组中的数据写入输出数组
        out_arr.write(in.read(gid_, gid.z * offset_buf[1] + offset_buf[0]), gid_, gid.z);
    else
        // 将输入纹理数组中的数据写入输出纹理
        out_tex.write(in.read(gid_, gid.z * offset_buf[1] + offset_buf[0]), gid_);
}

// 定义常量，确定输入是否为数组（由 ushort_arg_7 和 ushort_arg_6 的比较得来）
constant bool append_features_in_is_arr = (ushort_arg_7 > 1 || ushort_arg_6 > 4);
// 定义常量，确定输入是否为纹理（由 append_features_in_is_arr 反转得来）
constant bool append_features_in_is_tex = !append_features_in_is_arr;

// 追加特征的核函数，将输入纹理或数组中的数据追加到输出数组中
kernel void append_features(texture2d<half, access::read> in_tex[[texture(0), function_constant(append_features_in_is_tex)]],
                            texture2d_array<half, access::read> in_arr[[texture(0), function_constant(append_features_in_is_arr)]],
                            texture2d_array<half, access::write> out[[texture(1)]],
                            constant ushort* offset_buf[[buffer(0)]],
                            ushort3 gid[[thread_position_in_grid]]) {
    // 提取二维坐标信息
    ushort2 gid_ = gid.xy;

    // 计算批次和特征索引
    ushort batch = gid.z / offset_buf[0];
    ushort feature = gid.z % offset_buf[0];

    // 计算输出数组的索引
    ushort outz = batch * offset_buf[1] + offset_buf[2] + feature;
    // 计算输入数组或纹理的索引
    ushort inz = batch * offset_buf[3] + feature;

    half4 intex;
    // 根据输入类型（纹理还是数组），选择读取函数
    if (append_features_in_is_arr) {
        // 从输入数组中读取数据
        intex = in_arr.read(gid_, inz);
    }
    else {
        // 从输入纹理中读取数据
        intex = in_tex.read(gid_);
    }
    // 将读取到的数据写入输出数组
    out.write(intex, gid_, outz);
}

// 定义常量，确定前置输入是否为数组（由 ushort_arg_3 和 ushort_arg_2 的比较得来）
constant bool prev_is_arr = (ushort_arg_3 > 1 || ushort_arg_2 > 4);
// 定义常量，确定前置输入是否为纹理（由 prev_is_arr 反转得来）
constant bool prev_is_tex = !prev_is_arr;
// 定义常量，确定偏移追加特征输入是否为数组（由 ushort_arg_7 和 ushort_arg_6 的比较得来）
constant bool append_features_off_in_is_arr = (ushort_arg_7 > 1 || ushort_arg_6 > 4);
// 定义常量，确定偏移追加特征输入是否为纹理（由 append_features_off_in_is_arr 反转得来）
constant bool append_features_off_in_is_tex = !append_features_off_in_is_arr;

// 偏移追加特征的核函数，从输入纹理或数组中读取数据，追加到输出数组中
kernel void append_features_off(texture2d<half, access::read> in_tex[[texture(0), function_constant(append_features_off_in_is_tex)]],
                                texture2d_array<half, access::read> in_arr[[texture(0), function_constant(append_features_off_in_is_arr)]],
                                texture2d<half, access::read> prev_tex[[texture(1), function_constant(prev_is_tex)]],
                                texture2d_array<half, access::read> prev_arr[[texture(1), function_constant(prev_is_arr)]],
                                texture2d_array<half, access::write> out[[texture(2)]],
                                constant ushort* offset_buf[[buffer(0)]],
                                ushort3 gid[[thread_position_in_grid]]) {
    // 提取二维坐标信息
    ushort2 gid_ = gid.xy;

    // 计算批次和特征索引
    ushort batch = gid.z / offset_buf[0];
    ushort feature = gid.z % offset_buf[0];
    # 计算输出索引的计算公式，基于批次、偏移缓冲区和特征值的加权和
    ushort outz = batch * offset_buf[1] + offset_buf[2] + feature;
    
    # 计算输入索引的计算公式，基于批次和特征值的加权和
    ushort inz = batch * offset_buf[3] + feature;
    
    # 初始化用于输出的纹理向量
    half4 outtex;
    
    # 根据前一个输入是否为数组，选择读取前一个数组或纹理的数据
    if (prev_is_arr)
      outtex = prev_arr.read(gid_, batch);
    else
      outtex = prev_tex.read(gid_);
    
    # 初始化用于输入的纹理向量
    half4 intex1;
    
    # 根据附加特征输入是否为数组，选择读取当前数组或纹理的数据
    if (append_features_in_is_arr)
      intex1 = in_arr.read(gid_, inz);
    else
      intex1 = in_tex.read(gid_);
    
    # 如果特征值为0，根据偏移缓冲区中的条件更新输出向量的部分分量
    if (feature == 0) {
      if (offset_buf[5] == 1)
        outtex.yzw = intex1.xyz;
      else if (offset_buf[5] == 2)
        outtex.zw = intex1.xy;
      else
        outtex.w = intex1.x;
      
      # 将更新后的输出向量写入输出数据
      out.write(outtex, gid_, outz);
    
      # 函数返回，结束处理
      return;
    }
    
    # 初始化用于输入的前一个纹理向量
    half4 intex0;
    
    # 根据附加特征输入是否为数组，选择读取上一个或当前的数组或纹理的数据
    if (append_features_in_is_arr)
      intex0 = in_arr.read(gid_, inz-1);
    else
      intex0 = intex1;
    
    # 根据偏移缓冲区中的条件更新输出向量的部分分量
    if (offset_buf[5] == 1) {
      outtex.x = intex0.w;
      outtex.yzw = intex1.xyz;
    }
    else if (offset_buf[5] == 2) {
      outtex.xy = intex0.zw;
      outtex.zw = intex1.xy;
    }
    else {
      outtex.xyz = intex0.yzw;
      outtex.w = intex1.x;
    }
    
    # 将更新后的输出向量写入输出数据
    out.write(outtex, gid_, outz);
}

constant bool clamp_is_arr = (ushort_arg_1 > 1 || ushort_arg_0 > 4);
constant bool clamp_is_tex = !clamp_is_arr;
kernel void clamp(texture2d_array<half, access::read> in_arr[[texture(0), function_constant(clamp_is_arr)]],
                  texture2d<half, access::read> in_tex[[texture(0), function_constant(clamp_is_tex)]],
                  texture2d_array<half, access::write> out_arr[[texture(1), function_constant(clamp_is_arr)]],
                  texture2d<half, access::write> out_tex[[texture(1), function_constant(clamp_is_tex)]],
                 ushort3 gid[[thread_position_in_grid]]) {
    // 根据输入参数 clamp_is_arr 或 clamp_is_tex 决定输出宽度和高度
    const ushort w = clamp_is_arr? out_arr.get_width() : out_tex.get_width();
    const ushort h = clamp_is_arr? out_arr.get_height() : out_tex.get_height();
    // 如果线程超出输出区域的范围，则结束函数
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    // 定义最小值和最大值的 float4 变量
    const float4 min_(float_arg_0, float_arg_0, float_arg_0, float_arg_0);
    const float4 max_(float_arg_1, float_arg_1, float_arg_1, float_arg_1);
    // 将 gid 的 x 和 y 分量存储到 gid_ 变量中
    ushort2 gid_ = gid.xy;
    // 根据 clamp_is_arr 的值选择从输入数组或纹理中读取数据，并进行 clamp 处理，写入到输出数组或纹理中
    if(clamp_is_arr){
        // 从输入数组中读取数据
        float4 value = (float4)in_arr.read(gid_, gid.z);
        // 对读取的值进行 clamp 处理，转换为 half4 类型
        half4 clamped = (half4)clamp(value, min_, max_);
        // 将 clamp 处理后的值写入输出数组
        out_arr.write(clamped, gid_, gid.z);
    } else {
        // 从输入纹理中读取数据
        float4 value = (float4)in_tex.read(gid_);
        // 对读取的值进行 clamp 处理，转换为 half4 类型
        half4 clamped = (half4)clamp(value, min_, max_);
        // 将 clamp 处理后的值写入输出纹理
        out_tex.write(clamped, gid_);
    }
}

constant bool hardswish_is_arr = (ushort_arg_0 > 1 || ushort_arg_1 > 4);
constant bool hardswish_is_tex = !hardswish_is_arr;
kernel void hardswish(texture2d_array<half, access::read> in_arr[[texture(0), function_constant(hardswish_is_arr)]],
                      texture2d<half, access::read> in_tex[[texture(0), function_constant(hardswish_is_tex)]],
                      texture2d_array<half, access::write> out_arr[[texture(1), function_constant(hardswish_is_arr)]],
                      texture2d<half, access::write> out_tex[[texture(1), function_constant(hardswish_is_tex)]],
                      ushort3 gid[[thread_position_in_grid]]) {
    // 从输入参数 hardswish_is_arr 和 hardswish_is_tex 中决定输出高度和宽度
    const ushort oH = ushort_arg_2;
    const ushort oW = ushort_arg_3;
    // 如果线程超出输出区域的范围，则结束函数
    if (gid.x >= oW || gid.y >= oH) {
        return;
    }
    // 将 gid 的 x 和 y 分量存储到 gid_ 变量中
    ushort2 gid_ = gid.xy;
    // 根据 hardswish_is_arr 的值选择从输入数组或纹理中读取数据，并进行 hardswish 处理，写入到输出数组或纹理中
    if (hardswish_is_arr) {
      // 从输入数组中读取数据
      half4 value = in_arr.read(gid_, gid.z);
      // 创建掩码 mask1 和 mask2 来判断是否应用 hardswish 函数
      half4 mask1 = half4(value < 3.0);
      half4 mask2 = half4(value > -3.0);
      // 计算应用 hardswish 函数后的值，并写入到输出数组
      half4 outval = mask2*(mask1*(value*(value + 3.0)/6.0) + (1 - mask1)*value);
      out_arr.write(outval, gid_, gid.z);
    } else {
      // 从输入纹理中读取数据
      half4 value = in_tex.read(gid_);
      // 创建掩码 mask1 和 mask2 来判断是否应用 hardswish 函数
      half4 mask1 = half4(value < 3);
      half4 mask2 = half4(value > -3.0);
      // 计算应用 hardswish 函数后的值，并写入到输出纹理
      half4 outval = mask2*(mask1*(value*(value + 3.0)/6.0) + (1 - mask1)*value);
      out_tex.write(outval, gid_);
    }
}

constant bool hardshrink_is_arr = (ushort_arg_0 > 1 || ushort_arg_1 > 4);
constant bool hardshrink_is_tex = !hardshrink_is_arr;
kernel void hardshrink(texture2d_array<half, access::read> in_arr[[texture(0), function_constant(hardshrink_is_arr)]],
                      texture2d<half, access::read> in_tex[[texture(0), function_constant(hardshrink_is_tex)]],
                      texture2d_array<half, access::write> out_arr[[texture(1), function_constant(hardshrink_is_arr)]],
                      texture2d<half, access::write> out_tex[[texture(1), function_constant(hardshrink_is_tex)]],
                      ushort3 gid[[thread_position_in_grid]]) {
    // 从常量参数获取输出图像的高度和宽度
    const ushort oH = ushort_arg_2;
    const ushort oW = ushort_arg_3;
    // 将浮点参数转换为半精度浮点数，作为阈值 lambda
    const half lambda = (half)float_arg_0;
    // 如果当前线程的位置超出输出图像的边界，则直接返回
    if (gid.x >= oW || gid.y >= oH) {
        return;
    }
    // 提取当前线程的二维坐标
    ushort2 gid_ = gid.xy;
    // 根据硬阈值操作的类型选择读取输入数据的方式（数组或纹理）
    if (hardshrink_is_arr) {
      // 从输入数组中读取指定位置的像素值
      half4 value = in_arr.read(gid_, gid.z);
      // 创建一个掩码，标识小于等于 lambda 的像素值
      half4 mask1 = half4(value <= lambda);
      // 创建一个掩码，标识大于等于 -lambda 的像素值
      half4 mask2 = half4(value >= -lambda);
      // 应用硬阈值函数，计算输出像素值
      half4 outval = (1 - mask1) * value + (1 - mask2) * value;
      // 将结果写入输出数组指定位置
      out_arr.write(outval, gid_, gid.z);
    } else {
      // 从输入纹理中读取指定位置的像素值
      half4 value = in_tex.read(gid_);
      // 创建一个掩码，标识小于等于 lambda 的像素值
      half4 mask1 = half4(value <= lambda);
      // 创建一个掩码，标识大于等于 -lambda 的像素值
      half4 mask2 = half4(value >= -lambda);
      // 应用硬阈值函数，计算输出像素值
      half4 outval = (1 - mask1) * value + (1 - mask2) * value;
      // 将结果写入输出纹理指定位置
      out_tex.write(outval, gid_);
    }
}

// 根据常量参数确定是否使用数组方式实现泄漏修正操作
constant bool leaky_relu_is_arr = (ushort_arg_0 > 1 || ushort_arg_1 > 4);
// 根据常量参数确定是否使用纹理方式实现泄漏修正操作
constant bool leaky_relu_is_tex = !leaky_relu_is_arr;
// 实现泄漏修正操作的内核函数
kernel void leaky_relu(texture2d_array<half, access::read> in_arr[[texture(0), function_constant(leaky_relu_is_arr)]],
                      texture2d<half, access::read> in_tex[[texture(0), function_constant(leaky_relu_is_tex)]],
                      texture2d_array<half, access::write> out_arr[[texture(1), function_constant(leaky_relu_is_arr)]],
                      texture2d<half, access::write> out_tex[[texture(1), function_constant(leaky_relu_is_tex)]],
                      ushort3 gid[[thread_position_in_grid]]) {
    // 从常量参数获取输出图像的高度和宽度
    const ushort oH = ushort_arg_2;
    const ushort oW = ushort_arg_3;
    // 将浮点参数转换为半精度浮点数，作为负斜率参数
    const half negative_slope = (half)float_arg_0;
    // 如果当前线程的位置超出输出图像的边界，则直接返回
    if (gid.x >= oW || gid.y >= oH) {
        return;
    }
    // 提取当前线程的二维坐标
    ushort2 gid_ = gid.xy;
    // 根据泄漏修正操作的类型选择读取输入数据的方式（数组或纹理）
    if (leaky_relu_is_arr) {
      // 从输入数组中读取指定位置的像素值
      half4 value = in_arr.read(gid_, gid.z);
      // 创建一个掩码，标识小于 0 的像素值
      half4 is_negative = half4(value < 0.0);
      // 应用泄漏修正函数，计算输出像素值
      half4 outval = is_negative * value * negative_slope + (1 - is_negative) * value;
      // 将结果写入输出数组指定位置
      out_arr.write(outval, gid_, gid.z);
    } else {
      // 从输入纹理中读取指定位置的像素值
      half4 value = in_tex.read(gid_);
      // 创建一个掩码，标识小于 0 的像素值
      half4 is_negative = half4(value < 0.0);
      // 应用泄漏修正函数，计算输出像素值
      half4 outval = is_negative * value * negative_slope + (1 - is_negative) * value;
      // 将结果写入输出纹理指定位置
      out_tex.write(outval, gid_);
    }
}
// 定义反射填充二维卷积核函数，处理二维数组输入和输出纹理
kernel void reflection_pad2d(texture2d_array<half, access::read> in_arr[[texture(0), function_constant(in_is_arr)]],
                             texture2d<half, access::read> in_tex[[texture(0),function_constant(in_is_tex)]],
                             texture2d_array<half, access::write> out_arr[[texture(1), function_constant(out_is_arr)]],
                             texture2d<half, access::write> out_tex[[texture(1), function_constant(out_is_tex)]],
                             ushort3 gid[[thread_position_in_grid]]) {
  // 从常量传入获取输出纹理的高度和宽度
  const ushort H2 = ushort_arg_0;
  const ushort W2 = ushort_arg_1;
  // 如果线程位置超出输出范围则返回
  if (gid.x >= W2 || gid.y >= H2) {
      return;
  }

  // 从常量传入获取填充值
  const ushort pad_left = ushort_arg_8;
  const ushort pad_right = ushort_arg_9;
  const ushort pad_top = ushort_arg_10;
  const ushort pad_bottom = ushort_arg_11;

  // 定义输出尺寸和计算反射填充的偏移量
  const ushort2 out_size = ushort2(W2, H2);
  const ushort xoff_pre  = 2 * max(pad_left - gid.x, 0);
  const ushort xoff_post = 2 * max(gid.x - (out_size.x - 1 - pad_right), 0);
  const ushort yoff_pre  = 2 * max(pad_top - gid.y, 0);
  const ushort yoff_post = 2 * max(gid.y - (out_size.y - 1 - pad_bottom), 0);
  // 计算输入位置索引，进行反射填充
  ushort2 inpos = ushort2(
      gid.x + xoff_pre - xoff_post - pad_left,
      gid.y + yoff_pre - yoff_post - pad_top);

  half4 intex;
  // 根据输入类型选择从纹理数组或纹理中读取输入数据
  if (in_is_arr) {
    intex = in_arr.read(inpos, gid.z);
  } else {
    intex = in_tex.read(inpos);
  }

  // 根据输出类型选择将数据写入纹理数组或纹理中
  if (out_is_arr) {
      out_arr.write(intex, gid.xy, gid.z);
  } else {
      out_tex.write(intex, gid.xy);
  }
}

// 定义调整形状的卷积核函数，处理输入和输出纹理数组或纹理
constant bool reshape_out_is_arr = (ushort_arg_3 > 1 || ushort_arg_2 > 4);
constant bool reshape_out_is_tex = !reshape_out_is_arr;
constant bool reshape_in_is_arr = (ushort_arg_7 > 1 || ushort_arg_6 > 4);
constant bool reshape_in_is_tex = !reshape_in_is_arr;
kernel void reshape(texture2d_array<half, access::read> in_arr[[texture(0), function_constant(reshape_in_is_arr)]],
                    texture2d<half, access::read> in_tex[[texture(0),function_constant(reshape_in_is_tex)]],
                    texture2d_array<half, access::write> out_arr[[texture(1), function_constant(reshape_out_is_arr)]],
                    texture2d<half, access::write> out_tex[[texture(1),
                        function_constant(reshape_out_is_tex)]],
                    ushort3 gid[[thread_position_in_grid]]) {
    // 从常量传入获取输出纹理的高度、宽度和通道数
    const ushort H2 = ushort_arg_0;
    const ushort W2 = ushort_arg_1;
    const ushort C2 = ushort_arg_2;
    // 如果线程位置超出输出范围则返回
    if (gid.x >= W2 || gid.y >= H2) {
        return;
    }
    // 从常量传入获取输入纹理的高度、宽度、通道数和图像数量
    const ushort H1 = ushort_arg_4;
    const ushort W1 = ushort_arg_5;
    const ushort C1 = ushort_arg_6;
    const ushort N1 = ushort_arg_7;

    // 计算输入数据的总元素数
    const size_t numel1 = H1 * W1 * C1 * N1;
    // 计算输出通道和输入通道的切片数
    const ushort slices2 = divRoundUp(C2, 4);
    const ushort slices1 = divRoundUp(C1, 4);
    // 计算图像索引和切片偏移
    const ushort n2 = gid.z / slices2; //image index
    const ushort s2 = gid.z - n2 * slices2; // slice offest
    half4 value;
    // 对于每个索引 idx 从 0 到 3 进行循环计算
    for (int idx = 0; idx < 4; ++idx){
        // 计算输出元素的“线性索引”，并将其转换为输入元素的等效“线性索引”
        ushort offset = 4 * s2 + idx;
        size_t linear_idx = n2 * C2 * H2 * W2 + offset * H2 * W2 + gid.y * W2 + gid.x;
        // 如果计算得到的线性索引超出了输入张量的元素数量，则将当前 value[idx] 设为 0 并跳过此次循环
        if(linear_idx >= numel1){
            value[idx] = 0;
            continue;
        }
        // 计算在输入张量中对应的 x1、y1、s1、n1 和 z1 的值
        auto x1 = linear_idx % W1;
        auto y1 = ((int)(linear_idx/W1)) % H1;
        auto s1 = ((int)(linear_idx/W1/H1) % C1);
        auto n1 = ((int)(linear_idx/W1/H1/C1) % N1);
        auto z1 = (int)s1 / 4 + n1 * slices1;
        auto pos = s1 % 4;
        // 根据输入是否为数组还是纹理，从相应的数据源中读取值并存入 value[idx]
        if(reshape_in_is_arr) {
            value[idx] = in_arr.read(ushort2(x1, y1), z1)[pos];
        } else {
            value[idx] = in_tex.read(ushort2(x1, y1))[pos];
        }

    }
    // 如果输出重塑后为数组，则将 value 数组写入到 out_arr 的指定位置 gid.xy 和 gid.z
    // 否则，将 value 数组写入到 out_tex 的指定位置 gid.xy
    if(reshape_out_is_arr) {
        out_arr.write(value, gid.xy, gid.z);
    } else {
        out_tex.write(value, gid.xy);
    }
}

constant bool transpose_in_is_arr = (ushort_arg_3 > 1 || ushort_arg_4 > 4);
constant bool transpose_in_is_tex = !transpose_in_is_arr;
constant bool transpose_out_is_arr = (ushort_arg_5 > 1 || ushort_arg_6 > 4);
constant bool transpose_out_is_tex = !transpose_out_is_arr;

// 定义名为 transpose 的核函数，用于数据转置操作
kernel void transpose(texture2d_array<half, access::read>in_arr[[texture(0),function_constant(transpose_in_is_arr)]],
                      // 输入数组纹理，根据 transpose_in_is_arr 的布尔值选择
                      texture2d<half, access::read> in_tex[[texture(0), function_constant(transpose_in_is_tex)]],
                      // 输入非数组纹理，根据 transpose_in_is_tex 的布尔值选择
                      texture2d_array<half, access::write>out_arr[[texture(1),function_constant(transpose_out_is_arr)]],
                      // 输出数组纹理，根据 transpose_out_is_arr 的布尔值选择
                      texture2d<half, access::write> out_tex[[texture(1), function_constant(transpose_out_is_tex)]],
                      // 输出非数组纹理，根据 transpose_out_is_tex 的布尔值选择
                      constant ushort* inSizeBuffer [[buffer(0)]],
                      // 输入尺寸缓冲区
                      constant ushort* outSizeBuffer [[buffer(1)]],
                      // 输出尺寸缓冲区
                      ushort3 gid[[thread_position_in_grid]]) {
                      // 三维线程组索引

    const ushort dim0 = ushort_arg_0;
    // 输入维度0
    const ushort dim1 = ushort_arg_1;
    // 输入维度1
    const ushort dim = ushort_arg_2;
    // 输入维度
    const ushort N1 = ushort_arg_3;
    // 输入 N1 维度
    const ushort C1 = ushort_arg_4;
    // 输入 C1 维度
    const ushort N2 = ushort_arg_5;
    // 输出 N2 维度
    const ushort C2 = ushort_arg_6;
    // 输出 C2 维度
    ushort W1,W2,H1,H2;
    // 定义变量 W1, W2, H1, H2

    if(transpose_in_is_arr) {
        // 如果输入是数组纹理
        W1 = in_arr.get_width();
        // 获取输入数组纹理的宽度
        H1 = in_arr.get_height();
        // 获取输入数组纹理的高度
    } else {
        // 否则，输入是非数组纹理
        W1 = in_tex.get_width();
        // 获取输入非数组纹理的宽度
        H1 = in_tex.get_height();
        // 获取输入非数组纹理的高度
    }

    if(transpose_out_is_arr) {
        // 如果输出是数组纹理
        W2 = out_arr.get_width();
        // 获取输出数组纹理的宽度
        H2 = out_arr.get_height();
        // 获取输出数组纹理的高度
    } else {
        // 否则，输出是非数组纹理
        W2 = out_tex.get_width();
        // 获取输出非数组纹理的宽度
        H2 = out_tex.get_height();
        // 获取输出非数组纹理的高度
    }

    if (gid.x >= W2 || gid.y >= H2) {
        // 如果当前线程索引超出输出范围，则返回
        return;
    }

    const size_t numel = H2 * W2 * C2 * N2;
    // 计算输出元素总数
    const ushort slices2 = divRoundUp(C2, 4);
    // 计算 C2 维度的切片数
    const ushort slices1 = divRoundUp(C1, 4);
    // 计算 C1 维度的切片数
    const ushort n2 = gid.z / slices2;
    // 计算当前线程组在 n2 维度的位置
    const ushort s2 = gid.z - n2 * slices2;
    // 计算当前线程组在 s2 维度的位置
    half4 value;
    // 定义 half4 类型的变量 value
    ushort4 threadIndexBufferLower{1, 1, 1, 1};
    // 定义 ushort4 类型的 threadIndexBufferLower 变量并初始化
    ushort4 threadIndexBufferUpper{1, 1, 1 ,1};
    // 定义 ushort4 类型的 threadIndexBufferUpper 变量并初始化
    // 循环遍历4次，处理每个索引
    for (int idx = 0; idx < 4; ++idx){
        // 计算当前偏移量
        ushort offset = 4 * s2 + idx;
        // 计算线性索引值
        size_t linear_idx2 = n2 * C2 * H2 * W2 + offset * H2 * W2 + gid.y * W2 + gid.x;
        // 检查线性索引是否超出总元素数，超出则将当前值设为0并继续下一个循环
        if(linear_idx2 >= numel) {
            value[idx] = 0;
            continue;
        }

        ushort d2 = 0;
        // 反向遍历维度数组
        for(int j = dim-1; j>=0; --j){
            // 获取当前维度的大小
            d2  = outSizeBuffer[j];
            // 根据维度大小对线性索引进行转换，并存储到对应的缓冲区
            if(j > 3) {
                threadIndexBufferUpper[j-3] = linear_idx2 % d2;
            } else {
                threadIndexBufferLower[j] = linear_idx2 % d2;
            }
            // 更新线性索引
            linear_idx2 /= d2;
        }

        // 交换维度
        ushort tmp;
        // 根据维度大小交换缓冲区中的索引值
        if(dim0 > 3) {
            tmp = threadIndexBufferUpper[dim0-3];
        } else {
            tmp = threadIndexBufferLower[dim0];
        }
        if(dim0 > 3 && dim1 > 3) {
            threadIndexBufferUpper[dim0-3] = threadIndexBufferUpper[dim1-3];
        } else if (dim0 > 3 && dim1 < 3) {
            threadIndexBufferUpper[dim0-3] = threadIndexBufferLower[dim1];
        } else if (dim0 < 3 && dim1 > 3) {
            threadIndexBufferLower[dim0] = threadIndexBufferUpper[dim1-3];
        } else {
            threadIndexBufferLower[dim0] = threadIndexBufferLower[dim1];
        }
        if(dim1 > 3) {
            threadIndexBufferUpper[dim1-3] = tmp;
        } else {
            threadIndexBufferLower[dim1] = tmp;
        }

        size_t linear_idx1 = 0;
        ushort m = 1;
        ushort d1 = 0;
        // 根据交换后的维度顺序计算线性索引值
        for(int k = dim-1; k>=0; --k) {
            if(k > 3) {
                d1 = threadIndexBufferUpper[k-3];
            } else {
                d1 = threadIndexBufferLower[k];
            }
            linear_idx1 += d1 * m;
            m *= inSizeBuffer[k];
        }

        // 计算在输入张量中的坐标
        auto x1 = linear_idx1 % W1;
        auto y1 = ((int)(linear_idx1/W1)) % H1;
        auto c1 = ((int)(linear_idx1/W1/H1) % C1);
        auto n1 = ((int)(linear_idx1/W1/H1/C1) % N1);
        auto z1 = (int)c1 / 4 + n1 * slices1;
        auto pos = c1 % 4;
        // 根据输入数据类型，读取相应位置的值并存储到value数组中
        if(transpose_in_is_arr) {
            value[idx] = in_arr.read(ushort2(x1, y1), z1)[pos];
        } else {
            value[idx] = in_tex.read(ushort2(x1, y1))[pos];
        }
    }
    // 根据输出数据类型，将value数组的值写入到输出数组或纹理中
    if(transpose_out_is_arr) {
        out_arr.write(value, gid.xy, gid.z);
    } else {
        out_tex.write(value, gid.xy);
    }
}
// 定义一个常量布尔值，用于指示是否需要在输入数组上拆分通道
constant bool split_channels_in_is_arr = (ushort_arg_0 > 4);
// 定义一个常量布尔值，用于指示是否需要在输入纹理上拆分通道
constant bool split_channels_in_is_tex = !split_channels_in_is_arr;
// 定义一个常量布尔值，用于指示是否需要在第一个输出数组上拆分通道
constant bool split_channels_out1_is_arr = (ushort_arg_1 > 4);
// 定义一个常量布尔值，用于指示是否需要在第一个输出纹理上拆分通道
constant bool split_channels_out1_is_tex = !split_channels_out1_is_arr;
// 定义一个常量布尔值，用于指示是否需要在第二个输出数组上拆分通道
constant bool split_channels_out2_is_arr = (ushort_arg_2 > 4);
// 定义一个常量布尔值，用于指示是否需要在第二个输出纹理上拆分通道
constant bool split_channels_out2_is_tex = !(split_channels_out2_is_arr);

// 一个简单的实现，用于将输入纹理在通道维度上分成两部分
kernel void split_channels(texture2d_array<half, access::read> in_arr[[texture(0), function_constant(split_channels_in_is_arr)]],
                           texture2d<half, access::read> in_tex[[texture(0), function_constant(split_channels_in_is_tex)]],
                           texture2d_array<half, access::write> out1_arr[[texture(1),function_constant(split_channels_out1_is_arr)]],
                           texture2d<half, access::write> out1_tex[[texture(1),function_constant(split_channels_out1_is_tex)]],
                           texture2d_array<half, access::write> out2_arr[[texture(2), function_constant(split_channels_out2_is_arr)]],
                           texture2d<half, access::write> out2_tex[[texture(2),function_constant(split_channels_out2_is_tex)]],
                           ushort3 gid[[thread_position_in_grid]]) {
    ushort W,H;
    // 根据输入类型确定宽度和高度
    if(split_channels_in_is_arr) {
        W = in_arr.get_width();
        H = in_arr.get_height();
    } else {
        W = in_tex.get_width();
        H = in_tex.get_height();
    }
    // 如果当前线程位置超出了输入尺寸，则直接返回
    if(gid.x >= W || gid.y >= H){
        return;
    }
    // 从第一个输入参数中获取通道数
    const ushort C1 = ushort_arg_1;
    // 计算需要的存储单元数量
    const ushort s1 = divRoundUp(C1, 4);
    // 计算通道的偏移量
    const ushort c_offset = C1 % 4;
    // 初始化临时变量为零
    half4 tmp1(0.0, 0.0, 0.0, 0.0);
    half4 tmp2(0.0, 0.0, 0.0, 0.0);
    // 从输入读取第一个通道数据
    half4 in41 = split_channels_in_is_arr ? in_arr.read(gid.xy, gid.z) : in_tex.read(gid.xy);
    // 从输入读取第二个通道数据，如果使用数组，则读取下一个切片；否则初始化为零
    half4 in42 = split_channels_in_is_arr ? in_arr.read(gid.xy, gid.z+1) : half4(0,0,0,0);
    // 如果通道索引小于存储单元数减一，写入第一个输出数组中
    if(gid.z < s1 - 1) {
        if(split_channels_out1_is_arr) {
            out1_arr.write(in41, gid.xy, gid.z);
        }
    }
    # 如果 gid.z 等于 s1 - 1，则执行以下操作
    else if(gid.z == s1 - 1) {
        # 如果 c_offset 等于 0
        if(c_offset == 0){
            # 如果 split_channels_out1_is_arr 为真，则向 out1_arr 写入 in41 的数据
            out1_arr.write(in41, gid.xy, gid.z);
            # 否则，向 out1_tex 写入 in41 的数据
            return;
        } else if(c_offset == 1) {
            # 将 in41.x 写入 tmp1.x
            tmp1.x = in41.x;
            # 将 in41.yzw 写入 tmp2.xyz，in42.x 写入 tmp2.w
            tmp2.xyz = in41.yzw;
            tmp2.w = in42.x;
        } else if (c_offset == 2) {
            # 将 in41.xy 写入 tmp1.xy，in41.zw 写入 tmp2.xy，in42.xy 写入 tmp2.zw
            tmp1.xy = in41.xy;
            tmp2.xy = in41.zw;
            tmp2.zw = in42.xy;
        } else {
            # 将 in41.xyz 写入 tmp1.xyz，in41.w 写入 tmp2.x，in42.xyz 写入 tmp2.yzw
            tmp1.xyz = in41.xyz;
            tmp2.x = in41.w;
            tmp2.yzw = in42.xyz;
        }
        # 如果 split_channels_out1_is_arr 为真，则向 out1_arr 写入 tmp1 的数据
        # 否则，向 out1_tex 写入 tmp1 的数据
        if(split_channels_out1_is_arr) {
            out1_arr.write(tmp1, gid.xy, gid.z);
        } else {
            out1_tex.write(tmp1, gid.xy);
        }
        # 如果 split_channels_out2_is_arr 为真，则向 out2_arr 写入 tmp2 的数据
        # 否则，向 out2_tex 写入 tmp2 的数据
        if(split_channels_out2_is_arr) {
            out2_arr.write(tmp2, gid.xy, 0);
        } else {
            out2_tex.write(tmp2, gid.xy);
        }
    }
    # 否则，执行以下操作
    else {
        # 如果 c_offset 等于 0
        if (c_offset == 0) {
            # 如果 split_channels_out2_is_arr 为真，则向 out2_arr 写入 in41 的数据，gid.z - s1
            out2_arr.write(in41, gid.xy, gid.z - s1);
            # 否则，向 out2_tex 写入 in41 的数据
            return;
        }
        # 如果 c_offset 等于 1
        else if (c_offset == 1 ){
            # 将 in41.yzw 写入 tmp2.xyz，in42.x 写入 tmp2.w
            tmp2.xyz = in41.yzw;
            tmp2.w = in42.x;
        } else if (c_offset == 2){
            # 将 in41.zw 写入 tmp2.xy，in42.xy 写入 tmp2.zw
            tmp2.xy = in41.zw;
            tmp2.zw = in42.xy;
        } else {
            # 将 in41.w 写入 tmp2.x，in42.xyz 写入 tmp2.yzw
            tmp2.x = in41.w;
            tmp2.yzw = in42.xyz;
        }
        # 如果 split_channels_out2_is_arr 为真，则向 out2_arr 写入 tmp2 的数据，gid.z - s1 + 1
        # 否则，向 out2_tex 写入 tmp2 的数据
        if(split_channels_out2_is_arr) {
            out2_arr.write(tmp2, gid.xy, gid.z - s1 + 1);
        } else {
            out2_tex.write(tmp2, gid.xy);
        }
    }
    // 确定是否存在输入数组
    constant bool ra_has_in_arr = (ushort_arg_3 > 1 ||  ushort_arg_2 > 4);
    // 确定是否存在输出数组
    constant bool ra_has_out_arr = (ushort_arg_4 > 1 || ushort_arg_2 > 4);
    // 根据输入数组的存在性确定是否存在输入纹理
    constant bool ra_has_in_tex = (!ra_has_in_arr);
    // 根据输出数组的存在性确定是否存在输出纹理
    constant bool ra_has_out_tex = (!ra_has_out_arr);

    // ROI Align 核函数定义
    kernel void roi_align(texture2d_array<half, access::sample> ina[[texture(0), function_constant(ra_has_in_arr)]],
                          texture2d<half, access::sample> in[[texture(0), function_constant(ra_has_in_tex)]],
                          texture2d_array<half, access::write> outa[[texture(1), function_constant(ra_has_out_arr)]],
                          texture2d<half, access::write> out[[texture(1), function_constant(ra_has_out_tex)]],
                          constant half4* rois[[buffer(0)]],
                          ushort3 gid[[thread_position_in_grid]]) {

        // 输出图像的宽度和高度
        ushort out_width, out_height;
        if (ra_has_out_arr) {
            out_width = outa.get_width();
            out_height = outa.get_height();
        } else {
            out_width = out.get_width();
            out_height = out.get_height();
        }

        // 如果线程超出了输出图像的尺寸，则直接返回
        if (gid.x >= out_width || gid.y >= out_height) {
            return;
        }

        // 空间比例因子和采样比率常量
        const half spatial_scale = half(ushort_arg_0) / 10000;
        const ushort sampling_ratio = ushort_arg_1;
        const ushort C = ushort_arg_2;

        // 当前像素在输出图像中的位置
        const ushort pw = gid.x;
        const ushort ph = gid.y;

        // 计算当前像素所在的ROI和通道
        const ushort n = gid.z / divRoundUp(C, 4);
        const ushort c = gid.z % divRoundUp(C, 4);

        // 根据ROI的比例因子计算ROI的起始和结束坐标
        const half4 roi_scaled = rois[n] * spatial_scale;
        const half roi_start_w = roi_scaled[0];
        const half roi_start_h = roi_scaled[1];
        const half roi_end_w = roi_scaled[2];
        const half roi_end_h = roi_scaled[3];

        // 强制不规则ROI的大小至少为1x1
        const half roi_width = max(roi_end_w - roi_start_w, (half)1.);
        const half roi_height = max(roi_end_h - roi_start_h, (half)1.);

        // 计算每个像素的bin大小
        const half bin_size_h = static_cast<half>(roi_height) / static_cast<half>(out_height);
        const half bin_size_w = static_cast<half>(roi_width) / static_cast<half>(out_width);

        // 计算ROI内部的bin网格大小
        const ushort roi_bin_grid_h = sampling_ratio > 0 ? sampling_ratio : ceil(roi_height / static_cast<half>(out_height));
        const ushort roi_bin_grid_w = sampling_ratio > 0 ? sampling_ratio : ceil(roi_width / static_cast<half>(out_width));

        // 计算每个bin的像素数量
        const half count = roi_bin_grid_h * roi_bin_grid_w;
        // 初始化输出值为0
        half4 output_val = 0.0;

        // 像素采样器配置
        constexpr sampler s2(coord::pixel, address::clamp_to_edge, filter::linear);
    // 遍历每个ROI bin的网格
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            // 将像素偏移0.5。这对于达到高精度至关重要。
            // 计算y坐标，考虑ROI起始点、像素偏移、bin大小和网格高度
            const half y =
                roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / static_cast<half>(roi_bin_grid_h);
            // 计算x坐标，考虑ROI起始点、像素偏移、bin大小和网格宽度
            const half x =
                roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / static_cast<half>(roi_bin_grid_w);
            
            // 如果ra_has_in_arr为真，则使用ina采样（考虑坐标和通道c），否则使用in采样（只考虑坐标）
            if (ra_has_in_arr) {
                output_val += ina.sample(s2, float2(x, y), c);
            } else {
                output_val += in.sample(s2, float2(x, y));
            }
        }
    }
    // 计算ROI bin的平均值
    output_val /= count;
    
    // 如果ra_has_out_arr为真，则将output_val写入outa的指定位置（考虑gid.xy和gid.z），否则写入out的指定位置（考虑gid.xy）
    if (ra_has_out_arr) {
        outa.write(static_cast<half4>(output_val), gid.xy, gid.z);
    } else {
        out.write(static_cast<half4>(output_val), gid.xy);
    }
}

)PT_METAL_SHADERS";

#endif /* MPSCNNShaders_h */


// 这部分代码片段是C++中的预处理指令和宏定义的结尾部分。
// } 表示此处是前面某个代码块的结尾
// )PT_METAL_SHADERS"; 是一个包含了 Metal shader 代码的字符串字面量的结尾
// #endif 表示条件编译指令的结束，用于关闭之前的条件编译部分
// /* MPSCNNShaders_h */ 是条件编译指令的参数，指定了当前条件编译块的名称
```