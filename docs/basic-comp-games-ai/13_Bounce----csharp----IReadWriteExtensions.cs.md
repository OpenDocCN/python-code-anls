# `basic-computer-games\13_Bounce\csharp\IReadWriteExtensions.cs`

```
# 创建名为Bounce的命名空间
namespace Bounce;

# 创建名为IReadWriteExtensions的内部静态类
internal static class IReadWriteExtensions
{
    # 创建名为ReadParameter的内部静态方法，接收IReadWrite类型的参数io和字符串类型的参数parameter，返回浮点数
    internal static float ReadParameter(this IReadWrite io, string parameter)
    {
        # 调用io对象的ReadNumber方法，传入parameter参数，将结果赋给value变量
        var value = io.ReadNumber(parameter);
        # 调用io对象的WriteLine方法
        io.WriteLine();
        # 返回value变量的值
        return value;
    }
}
```