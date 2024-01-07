# `basic-computer-games\07_Basketball\csharp\Probably.cs`

```

using Games.Common.Randomness; 
// 导入 Games.Common.Randomness 命名空间，用于引用随机数生成相关的功能

namespace Basketball;
// 命名空间 Basketball，用于定义相关类和结构体

/// <summary>
/// 支持基于各种概率执行一系列操作的链。原始游戏代码为每个概率检查获取一个新的随机数。对一组概率进行评估，只需一个随机数，但会产生非常不同的结果分布。这个类的目的是简化原始概率分支决策的代码。
/// </summary>
internal struct Probably
{
    private readonly float _defenseFactor;
    // 用于存储防御因子的私有只读字段
    private readonly IRandom _random;
    // 用于存储随机数生成器的私有只读字段
    private readonly bool? _result;
    // 用于存储结果的可空布尔值的私有只读字段

    internal Probably(float defenseFactor, IRandom random, bool? result = null)
    {
        _defenseFactor = defenseFactor;
        _random = random;
        _result = result;
    }
    // Probably 结构体的构造函数，用于初始化字段的值

    public Probably Do(float probability, Action action) =>
        ShouldResolveAction(probability)
            ? new Probably(_defenseFactor, _random, Resolve(action) ?? false)
            : this;
    // 执行给定概率的操作，如果满足条件则返回新的 Probably 结构体实例，否则返回当前实例

    public Probably Do(float probability, Func<bool> action) =>
        ShouldResolveAction(probability)
            ? new Probably(_defenseFactor, _random, Resolve(action) ?? false)
            : this;
    // 执行给定概率的操作，如果满足条件则返回新的 Probably 结构体实例，否则返回当前实例

    public Probably Or(float probability, Action action) => Do(probability, action);
    // 执行给定概率的操作，如果满足条件则返回新的 Probably 结构体实例，否则返回当前实例

    public Probably Or(float probability, Func<bool> action) => Do(probability, action);
    // 执行给定概率的操作，如果满足条件则返回新的 Probably 结构体实例，否则返回当前实例

    public bool Or(Action action) => _result ?? Resolve(action) ?? false;
    // 如果结果不为空，则返回结果，否则执行给定的操作并返回结果

    private bool? Resolve(Action action)
    {
        action.Invoke();
        return _result;
    }
    // 执行给定的操作并返回结果

    private bool? Resolve(Func<bool> action) => action.Invoke();
    // 执行给定的操作并返回结果

    private readonly bool ShouldResolveAction(float probability) =>
        _result is null && _random.NextFloat() <= probability * _defenseFactor;
    // 判断是否应该执行给定概率的操作
}

```