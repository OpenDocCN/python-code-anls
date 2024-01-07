# `basic-computer-games\34_Digits\csharp\IOExtensions.cs`

```

# 创建名为Digits的命名空间
namespace Digits;

# 创建名为IOExtensions的静态类
internal static class IOExtensions
{
    # 创建名为Read10Digits的静态方法，接收io、prompt和retryText三个参数，返回一个整数序列
    internal static IEnumerable<int> Read10Digits(this IReadWrite io, string prompt, Stream retryText)
    {
        # 创建一个无限循环
        while (true)
        {
            # 创建一个包含10个浮点数的数组
            var numbers = new float[10];
            # 调用io对象的ReadNumbers方法，传入prompt和numbers数组
            io.ReadNumbers(prompt, numbers);

            # 如果所有的数字都是0、1或2，则将浮点数转换为整数并返回
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