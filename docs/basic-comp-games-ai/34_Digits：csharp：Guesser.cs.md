# `d:/src/tocomm/basic-computer-games\34_Digits\csharp\Guesser.cs`

```
namespace Digits;  # 命名空间声明，定义了代码所在的命名空间

internal class Guesser  # 定义了一个内部类 Guesser
{
    private readonly Memory _matrices = new();  # 声明并初始化了一个名为 _matrices 的 Memory 对象
    private readonly IRandom _random;  # 声明了一个名为 _random 的 IRandom 接口类型的变量

    public Guesser(IRandom random)  # 定义了一个构造函数，接受一个 IRandom 类型的参数
    {
        _random = random;  # 将传入的 random 参数赋值给 _random 变量
    }

    public int GuessNextDigit()  # 定义了一个公共方法 GuessNextDigit，返回一个整数
    {
        var currentSum = 0;  # 声明并初始化了一个名为 currentSum 的变量，值为 0
        var guess = 0;  # 声明并初始化了一个名为 guess 的变量，值为 0

        for (int i = 0; i < 3; i++)  # 循环，i 从 0 到 2
        {
            var sum = _matrices.GetWeightedSum(i);  # 调用 _matrices 对象的 GetWeightedSum 方法，将结果赋值给 sum 变量
            if (sum > currentSum || _random.NextFloat() >= 0.5)
            {
                currentSum = sum;  # 如果当前的和大于之前记录的最大和，或者随机数大于等于0.5，则执行下面的操作
                guess = i;  # 更新猜测值为当前循环的数字
            }
        }

        return guess;  # 返回最终的猜测值
    }

    public void ObserveActualDigit(int digit) => _matrices.ObserveDigit(digit);  # 观察并记录实际的数字
}
```