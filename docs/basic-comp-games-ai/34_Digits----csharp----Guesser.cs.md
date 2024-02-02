# `basic-computer-games\34_Digits\csharp\Guesser.cs`

```py
namespace Digits;

internal class Guesser
{
    private readonly Memory _matrices = new();  // 创建一个内存对象用于存储矩阵数据
    private readonly IRandom _random;  // 创建一个随机数生成器对象

    public Guesser(IRandom random)  // 构造函数，接受一个随机数生成器对象作为参数
    {
        _random = random;  // 将传入的随机数生成器对象赋值给私有字段
    }

    public int GuessNextDigit()  // 猜测下一个数字的方法
    {
        var currentSum = 0;  // 初始化当前总和为0
        var guess = 0;  // 初始化猜测值为0

        for (int i = 0; i < 3; i++)  // 循环3次，遍历矩阵数据
        {
            var sum = _matrices.GetWeightedSum(i);  // 获取加权总和
            if (sum > currentSum || _random.NextFloat() >= 0.5)  // 如果加权总和大于当前总和，或者随机数大于等于0.5
            {
                currentSum = sum;  // 更新当前总和为加权总和
                guess = i;  // 更新猜测值为当前索引
            }
        }

        return guess;  // 返回猜测值
    }

    public void ObserveActualDigit(int digit) => _matrices.ObserveDigit(digit);  // 观察实际数字并更新矩阵数据
}
```