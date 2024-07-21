# `.\pytorch\aten\src\ATen\native\vulkan\glsl\texel_access.h`

```
/*
 * Texel access utility functions
 */

// 根据输出位置映射输入位置，支持广播操作
ivec3 map_output_pos_to_input_pos(
    ivec3 output_pos,        // 输出位置坐标
    ivec4 output_sizes,      // 输出张量尺寸 (H, W, N, C)
    ivec4 input_sizes) {     // 输入张量尺寸 (H, W, N, C)
  ivec3 input_pos;          // 输入位置坐标
  // 对于 H 和 W 维度，使用模运算
  input_pos.xy = output_pos.xy % input_sizes.xy;
  if (output_sizes.w == input_sizes.w && output_sizes.z != input_sizes.z) {
    // 如果输出张量的 C 维度与输入张量相同，但 N 维度不同
    // 则按照 ceil(C/4) 分组映射到输入张量的范围
    input_pos.z = output_pos.z / int(ceil(output_sizes.z / 4.0));
  } else {
    // 对于 N 维度，使用模运算
    // 输入张量的 z 范围为 batch * ceil(channel/4)
    input_pos.z =
        output_pos.z % (input_sizes.w * int(ceil(input_sizes.z / 4.0)));
  }
  return input_pos;         // 返回映射后的输入位置坐标
}

// 从三维图像纹理加载像素，应用广播操作
vec4 load_texel(
    ivec3 mapped_pos,       // 映射后的位置坐标
    ivec4 output_sizes,     // 输出张量尺寸 (H, W, N, C)
    ivec4 input_sizes,      // 输入张量尺寸 (H, W, N, C)
    sampler3D uInput) {     // 输入纹理采样器
  return (output_sizes.z != input_sizes.z)
      ? texelFetch(uInput, mapped_pos, 0).xxxx  // 如果输出张量的 C 维度与输入张量不同，返回广播的像素值
      : texelFetch(uInput, mapped_pos, 0);      // 否则直接返回像素值
}
```