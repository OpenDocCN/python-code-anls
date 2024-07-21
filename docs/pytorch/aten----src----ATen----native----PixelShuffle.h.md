# `.\pytorch\aten\src\ATen\native\PixelShuffle.h`

```
// 定义了一个命名空间 `at::native`，包含了与像素重排相关的函数

// 检查像素重排操作的输入张量形状是否满足要求
inline void check_pixel_shuffle_shapes(const Tensor& self, int64_t upscale_factor) {
    // 使用 TORCH_CHECK 确保输入张量至少有 3 维
    TORCH_CHECK(self.dim() >= 3,
                "pixel_shuffle expects input to have at least 3 dimensions, but got input with ",
                self.dim(), " dimension(s)");
    // 使用 TORCH_CHECK 确保 upscale_factor 大于 0
    TORCH_CHECK(upscale_factor > 0,
                "pixel_shuffle expects a positive upscale_factor, but got ",
                upscale_factor);
    // 获取通道数，即张量的倒数第三维度的大小
    int64_t c = self.size(-3);
    // 计算 upscale_factor 的平方
    int64_t upscale_factor_squared = upscale_factor * upscale_factor;
    // 使用 TORCH_CHECK 确保通道数能被 upscale_factor 的平方整除
    TORCH_CHECK(c % upscale_factor_squared == 0,
                "pixel_shuffle expects its input's 'channel' dimension to be divisible by the square of "
                "upscale_factor, but input.size(-3)=", c, " is not divisible by ", upscale_factor_squared);
}

// 检查像素逆重排操作的输入张量形状是否满足要求
inline void check_pixel_unshuffle_shapes(const Tensor& self, int64_t downscale_factor) {
    // 使用 TORCH_CHECK 确保输入张量至少有 3 维
    TORCH_CHECK(
        self.dim() >= 3,
        "pixel_unshuffle expects input to have at least 3 dimensions, but got input with ",
        self.dim(),
        " dimension(s)");
    // 使用 TORCH_CHECK 确保 downscale_factor 大于 0
    TORCH_CHECK(
        downscale_factor > 0,
        "pixel_unshuffle expects a positive downscale_factor, but got ",
        downscale_factor);
    // 获取张量的倒数第二维和倒数第一维的大小，分别对应高度和宽度
    int64_t h = self.size(-2);
    int64_t w = self.size(-1);
    // 使用 TORCH_CHECK 确保高度能被 downscale_factor 整除
    TORCH_CHECK(
        h % downscale_factor == 0,
        "pixel_unshuffle expects height to be divisible by downscale_factor, but input.size(-2)=",
        h,
        " is not divisible by ",
        downscale_factor);
    // 使用 TORCH_CHECK 确保宽度能被 downscale_factor 整除
    TORCH_CHECK(
        w % downscale_factor == 0,
        "pixel_unshuffle expects width to be divisible by downscale_factor, but input.size(-1)=",
        w,
        " is not divisible by ",
        downscale_factor);
}
```