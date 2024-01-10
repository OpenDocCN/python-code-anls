# `basic-computer-games\34_Digits\csharp\IOExtensions.cs`

```
# 创建名为Digits的命名空间，并定义一个内部静态类IOExtensions
namespace Digits;

internal static class IOExtensions
{
    # 定义一个内部静态方法Read10Digits，接收一个IReadWrite类型的参数io，一个字符串类型的参数prompt，一个Stream类型的参数retryText，并返回一个整数类型的可枚举集合
    internal static IEnumerable<int> Read10Digits(this IReadWrite io, string prompt, Stream retryText)
    {
        # 无限循环，直到条件满足才会退出循环
        while (true)
        {
            # 创建一个包含10个浮点数的数组
            var numbers = new float[10];
            # 调用io对象的ReadNumbers方法，传入prompt和numbers数组
            io.ReadNumbers(prompt, numbers);

            # 如果数组中的所有元素都等于0、1或2，则返回将每个元素转换为整数后的可枚举集合
            if (numbers.All(n => n == 0 || n == 1 || n == 2))
            {
                return numbers.Select(n => (int)n);
            }    

            # 调用io对象的Write方法，传入retryText
            io.Write(retryText);
        }
    }
}
```