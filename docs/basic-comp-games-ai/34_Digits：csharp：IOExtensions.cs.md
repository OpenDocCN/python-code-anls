# `d:/src/tocomm/basic-computer-games\34_Digits\csharp\IOExtensions.cs`

```
namespace Digits;  # 命名空间声明

internal static class IOExtensions  # 声明一个内部静态类 IOExtensions
{
    internal static IEnumerable<int> Read10Digits(this IReadWrite io, string prompt, Stream retryText)  # 声明一个内部静态方法 Read10Digits，接受一个 IReadWrite 类型的参数 io，一个字符串类型的参数 prompt，一个 Stream 类型的参数 retryText，并返回一个 IEnumerable<int> 类型的结果
    {
        while (true)  # 进入一个无限循环
        {
            var numbers = new float[10];  # 声明一个包含 10 个元素的浮点数数组 numbers
            io.ReadNumbers(prompt, numbers);  # 调用 io 对象的 ReadNumbers 方法，传入 prompt 和 numbers 数组

            if (numbers.All(n => n == 0 || n == 1 || n == 2))  # 如果 numbers 数组中的所有元素都等于 0、1 或 2
            {
                return numbers.Select(n => (int)n);  # 返回将 numbers 数组中的每个元素转换为整数后的结果
            }    

            io.Write(retryText);  # 调用 io 对象的 Write 方法，传入 retryText
        }
    }
}
```