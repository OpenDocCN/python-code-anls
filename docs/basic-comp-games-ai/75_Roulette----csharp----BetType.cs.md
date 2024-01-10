# `basic-computer-games\75_Roulette\csharp\BetType.cs`

```
# 命名空间 Roulette 下的内部记录结构 BetType，包含一个整数值
internal record struct BetType(int Value)
{
    # 定义一个从整数到 BetType 的隐式转换操作符
    public static implicit operator BetType(int value) => new(value);

    # 根据不同的值计算赔付金额
    public int Payout => Value switch
        {
            # 如果值小于等于 36 或者大于等于 49，则赔付金额为 35
            <= 36 or >= 49 => 35,
            # 如果值小于等于 42，则赔付金额为 2
            <= 42 => 2,
            # 如果值小于等于 48，则赔付金额为 1
            <= 48 => 1
        };
}
```