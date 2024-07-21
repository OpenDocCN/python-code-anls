# `.\pytorch\aten\src\ATen\native\vulkan\glsl\indexing.h`

```
/*
 * 根据线性化的索引计算4维张量的坐标
 */
uvec4 idx_to_coord(const uint idx, const uvec4 strides, const uvec4 sizes) {
  return ivec4(mod(idx / strides, sizes));
}

/*
 * 根据4维张量的坐标计算线性化的索引
 */
uint coord_to_idx(const uvec4 coord, const uvec4 strides) {
  return int(dot(coord * strides, ivec4(1)));
}

/*
 * 将整数v向上对齐到最接近的4的倍数
 */
int align_up_4(int v) {
  return ((v + 4 - 1) / 4) * 4;
}

// 根据{n, c, h, w}索引返回通道打包的3D张量中的x, y, z和索引值。
ivec4 get_channel_packed_pos_from_index(ivec4 nchw, ivec4 sizes) {
  int n = nchw.x;
  int c = nchw.y;
  int h = nchw.z;
  int w = nchw.w;

  int aligned_c = align_up_4(sizes.y); // 对通道数进行向上对齐到4的倍数
  int c_stride = aligned_c / 4; // 计算通道步长

  return ivec4(
      w, // x：宽度坐标
      h, // y：高度坐标
      n * c_stride + c / 4, // z：通道坐标
      c % 4); // 索引值：通道内偏移
}
```