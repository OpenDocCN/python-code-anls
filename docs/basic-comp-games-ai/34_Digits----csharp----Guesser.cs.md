# `basic-computer-games\34_Digits\csharp\Guesser.cs`

```

// 命名空间 Digits，包含 Guesser 类
namespace Digits;

// Guesser 类，用于猜测下一个数字
internal class Guesser
{
    // 私有成员变量 _matrices，用于存储矩阵数据
    private readonly Memory _matrices = new();
    // 私有成员变量 _random，用于生成随机数
    private readonly IRandom _random;

    // Guesser 类的构造函数，接受一个 IRandom 类型的参数
    public Guesser(IRandom random)
    {
        // 将传入的 random 参数赋值给 _random 成员变量
        _random = random;
    }

    // 猜测下一个数字的方法
    public int GuessNextDigit()
    {
        // 初始化当前总和和猜测值
        var currentSum = 0;
        var guess = 0;

        // 循环3次，计算加权总和并进行猜测
        for (int i = 0; i < 3; i++)
        {
            // 获取第 i 个矩阵的加权总和
            var sum = _matrices.GetWeightedSum(i);
            // 如果加权总和大于当前总和，或者随机数大于等于0.5，则更新当前总和和猜测值
            if (sum > currentSum || _random.NextFloat() >= 0.5)
            {
                currentSum = sum;
                guess = i;
            }
        }

        // 返回猜测值
        return guess;
    }

    // 观察实际数字的方法，将观察到的数字传递给 _matrices 对象
    public void ObserveActualDigit(int digit) => _matrices.ObserveDigit(digit);
}

```