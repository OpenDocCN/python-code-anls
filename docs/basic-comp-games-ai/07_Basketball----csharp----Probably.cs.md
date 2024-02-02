# `basic-computer-games\07_Basketball\csharp\Probably.cs`

```py
using Games.Common.Randomness;  // 导入随机数生成器接口

namespace Basketball;  // 命名空间声明

/// <summary>
/// 支持基于各种概率执行一系列动作的链。原始游戏代码为每个概率检查获取一个新的随机数。对一组概率进行评估，得到一个随机数，会简化很多，但会产生非常不同的结果分布。这个类的目的是简化原始的概率分支决策的代码。
/// </summary>
internal struct Probably
{
    private readonly float _defenseFactor;  // 防守因子
    private readonly IRandom _random;  // 随机数生成器接口
    private readonly bool? _result;  // 结果

    internal Probably(float defenseFactor, IRandom random, bool? result = null)  // 构造函数
    {
        _defenseFactor = defenseFactor;
        _random = random;
        _result = result;
    }

    public Probably Do(float probability, Action action) =>  // 执行动作
        ShouldResolveAction(probability)  // 如果应该解析动作
            ? new Probably(_defenseFactor, _random, Resolve(action) ?? false)  // 创建新的 Probably 对象
            : this;  // 否则返回当前对象

    public Probably Do(float probability, Func<bool> action) =>  // 执行动作
        ShouldResolveAction(probability)  // 如果应该解析动作
            ? new Probably(_defenseFactor, _random, Resolve(action) ?? false)  // 创建新的 Probably 对象
            : this;  // 否则返回当前对象

    public Probably Or(float probability, Action action) => Do(probability, action);  // 或者执行动作

    public Probably Or(float probability, Func<bool> action) => Do(probability, action);  // 或者执行动作

    public bool Or(Action action) => _result ?? Resolve(action) ?? false;  // 或者执行动作

    private bool? Resolve(Action action)  // 解析动作
    {
        action.Invoke();  // 调用动作
        return _result;  // 返回结果
    }

    private bool? Resolve(Func<bool> action) => action.Invoke();  // 解析动作

    private readonly bool ShouldResolveAction(float probability) =>  // 是否应该解析动作
        _result is null && _random.NextFloat() <= probability * _defenseFactor;  // 结果为空且随机数小于等于概率乘以防守因子
}
```