# `.\pytorch\benchmarks\profiler_benchmark\resnet_memory_profiler.py`

```py
# 导入torchvision中的预定义模型，包括ResNet等
import torchvision.models as models

# 导入PyTorch核心库
import torch

# 导入PyTorch的性能分析模块
import torch.autograd.profiler as profiler

# 针对是否使用CUDA进行两次循环
for with_cuda in [False, True]:
    # 创建一个ResNet-18模型实例
    model = models.resnet18()
    
    # 创建一个形状为[5, 3, 224, 224]的随机张量作为输入
    inputs = torch.randn(5, 3, 224, 224)
    
    # 默认使用的排序键为CPU内存使用
    sort_key = "self_cpu_memory_usage"
    
    # 如果当前循环使用CUDA并且CUDA可用
    if with_cuda and torch.cuda.is_available():
        # 将模型和输入张量转移到CUDA设备上
        model = model.cuda()
        inputs = inputs.cuda()
        
        # 设置排序键为CUDA内存使用
        sort_key = "self_cuda_memory_usage"
        # 打印CUDA Resnet模型性能分析信息
        print("Profiling CUDA Resnet model")
    else:
        # 打印CPU Resnet模型性能分析信息
        print("Profiling CPU Resnet model")

    # 使用profiler.profile进行性能分析，记录内存使用和形状信息
    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        # 使用profiler.record_function记录根函数的性能
        with profiler.record_function("root"):
            # 运行模型，传入输入数据
            model(inputs)

    # 打印性能分析结果的表格，按输入形状分组，并按指定键排序
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by=sort_key, row_limit=-1
        )
    )
```