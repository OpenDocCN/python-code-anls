# `basic-computer-games\13_Bounce\csharp\IReadWriteExtensions.cs`

```

# 命名空间 Bounce 下的内部静态类 IReadWriteExtensions
namespace Bounce;

internal static class IReadWriteExtensions
{
    # 为 IReadWrite 接口添加一个扩展方法，用于读取参数并返回浮点数
    internal static float ReadParameter(this IReadWrite io, string parameter)
    {
        # 调用 IReadWrite 接口的 ReadNumber 方法读取参数的数值
        var value = io.ReadNumber(parameter);
        # 调用 IReadWrite 接口的 WriteLine 方法
        io.WriteLine();
        # 返回读取的参数值
        return value;
    }
}

```