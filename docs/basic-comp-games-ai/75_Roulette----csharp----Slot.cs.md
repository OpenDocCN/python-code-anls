# `basic-computer-games\75_Roulette\csharp\Slot.cs`

```

# 使用不可变集合命名空间
using System.Collections.Immutable;

# 创建名为Roulette的命名空间
namespace Roulette;

# 创建名为Slot的内部类
internal class Slot
{
    # 创建只读的不可变的BetType集合_coveringBets
    private readonly ImmutableHashSet<BetType> _coveringBets;

    # 创建Slot类的构造函数，接受名称和覆盖下注类型的参数
    public Slot (string name, params BetType[] coveringBets)
    {
        # 设置名称属性
        Name = name;
        # 将传入的覆盖下注类型参数转换为不可变的集合并赋值给_coveringBets
        _coveringBets = coveringBets.ToImmutableHashSet();
    }

    # 创建只读的名称属性
    public string Name { get; }

    # 创建IsCoveredBy方法，用于检查下注是否被覆盖
    public bool IsCoveredBy(Bet bet) => _coveringBets.Contains(bet.Type);
}

```