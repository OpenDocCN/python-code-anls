# `.\pytorch\test\mobile\model_test\android_api_module.py`

```
    @torch.jit.script_method
    def testNonContiguous(self):
        # 创建一个非连续的张量
        x = torch.tensor([100, 200, 300])[::2]
        # 断言张量不是连续的
        assert not x.is_contiguous()
        # 断言张量的第一个元素是100
        assert x[0] == 100
        # 断言张量的第二个元素是300
        assert x[1] == 300
        # 返回创建的非连续张量
        return x
    # 定义一个方法，用于执行二维卷积操作
    def conv2d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        # 使用 PyTorch 提供的二维卷积函数对输入张量 x 进行卷积操作，使用权重张量 w
        r = torch.nn.functional.conv2d(x, w)
        # 根据标志 toChannelsLast 决定是否转换张量 r 的存储顺序为通道在最后的格式
        if toChannelsLast:
            r = r.contiguous(memory_format=torch.channels_last)
        else:
            r = r.contiguous()
        # 返回卷积结果张量 r
        return r

    # 使用 Torch 脚本装饰器定义一个方法，用于执行三维卷积操作
    @torch.jit.script_method
    def conv3d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        # 使用 PyTorch 提供的三维卷积函数对输入张量 x 进行卷积操作，使用权重张量 w
        r = torch.nn.functional.conv3d(x, w)
        # 根据标志 toChannelsLast 决定是否转换张量 r 的存储顺序为三维通道在最后的格式
        if toChannelsLast:
            r = r.contiguous(memory_format=torch.channels_last_3d)
        else:
            r = r.contiguous()
        # 返回卷积结果张量 r
        return r

    # 使用 Torch 脚本装饰器定义一个方法，用于确保张量 x 的存储顺序是连续的
    @torch.jit.script_method
    def contiguous(self, x: Tensor) -> Tensor:
        # 返回一个连续存储顺序的张量 x
        return x.contiguous()

    # 使用 Torch 脚本装饰器定义一个方法，用于确保张量 x 的存储顺序是通道在最后的格式
    @torch.jit.script_method
    def contiguousChannelsLast(self, x: Tensor) -> Tensor:
        # 返回一个在通道维度上连续存储的张量 x
        return x.contiguous(memory_format=torch.channels_last)

    # 使用 Torch 脚本装饰器定义一个方法，用于确保张量 x 的存储顺序是三维通道在最后的格式
    @torch.jit.script_method
    def contiguousChannelsLast3d(self, x: Tensor) -> Tensor:
        # 返回一个在三维通道维度上连续存储的张量 x
        return x.contiguous(memory_format=torch.channels_last_3d)
```