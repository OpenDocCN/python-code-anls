# `basic-computer-games\75_Roulette\csharp\Slot.cs`

```
# 引入不可变集合的命名空间
using System.Collections.Immutable;

# 创建名为 Roulette 的命名空间
namespace Roulette;

# 创建名为 Slot 的内部类
internal class Slot
{
    # 创建只读的不可变集合 _coveringBets
    private readonly ImmutableHashSet<BetType> _coveringBets;

    # 创建构造函数，接受名称和覆盖下注类型的参数
    public Slot (string name, params BetType[] coveringBets)
    {
        # 设置名称属性
        Name = name;
        # 将覆盖下注类型参数转换为不可变集合并赋值给 _coveringBets
        _coveringBets = coveringBets.ToImmutableHashSet();
    }

    # 创建只读的名称属性
    public string Name { get; }

    # 创建方法，判断下注是否被覆盖
    public bool IsCoveredBy(Bet bet) => _coveringBets.Contains(bet.Type);
}
```