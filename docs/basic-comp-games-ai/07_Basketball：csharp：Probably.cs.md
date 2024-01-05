# `d:/src/tocomm/basic-computer-games\07_Basketball\csharp\Probably.cs`

```
using Games.Common.Randomness;  // 导入 Games.Common.Randomness 命名空间

namespace Basketball;  // 声明 Basketball 命名空间

/// <summary>
/// 支持基于各种概率执行一系列操作的链。原始游戏代码为每个概率检查获取一个新的随机数。对一组概率进行评估，只需一个随机数，但会产生非常不同的结果分布。这个类的目的是简化原始概率分支决策的代码。
/// </summary>
internal struct Probably  // 声明 Probably 结构体
{
    private readonly float _defenseFactor;  // 声明私有的 float 类型变量 _defenseFactor
    private readonly IRandom _random;  // 声明私有的 IRandom 类型变量 _random
    private readonly bool? _result;  // 声明私有的可空的 bool 类型变量 _result

    internal Probably(float defenseFactor, IRandom random, bool? result = null)  // 声明 Probably 结构体的构造函数，接受 defenseFactor、random 和 result 参数
    {
        _defenseFactor = defenseFactor;  // 将参数 defenseFactor 赋值给 _defenseFactor
        _random = random;  // 将参数 random 赋值给 _random
        _result = result;  # 将参数 result 的值赋给成员变量 _result

    public Probably Do(float probability, Action action) =>
        ShouldResolveAction(probability)  # 调用 ShouldResolveAction 方法判断是否应该执行 action
            ? new Probably(_defenseFactor, _random, Resolve(action) ?? false)  # 如果应该执行，则创建一个新的 Probably 对象并执行 action
            : this;  # 如果不应该执行，则返回当前对象

    public Probably Do(float probability, Func<bool> action) =>
        ShouldResolveAction(probability)  # 调用 ShouldResolveAction 方法判断是否应该执行 action
            ? new Probably(_defenseFactor, _random, Resolve(action) ?? false)  # 如果应该执行，则创建一个新的 Probably 对象并执行 action
            : this;  # 如果不应该执行，则返回当前对象

    public Probably Or(float probability, Action action) => Do(probability, action);  # 调用 Do 方法执行 action

    public Probably Or(float probability, Func<bool> action) => Do(probability, action);  # 调用 Do 方法执行 action

    public bool Or(Action action) => _result ?? Resolve(action) ?? false;  # 返回 _result 的值或者执行 action 的结果

    private bool? Resolve(Action action)  # 声明一个私有方法 Resolve，参数为 Action 类型
    {
        action.Invoke();  # 调用传入的 action 函数
        return _result;   # 返回 _result 变量的值
    }

    private bool? Resolve(Func<bool> action) => action.Invoke();  # 使用传入的 action 函数进行解析，并返回结果

    private readonly bool ShouldResolveAction(float probability) =>
        _result is null && _random.NextFloat() <= probability * _defenseFactor;  # 检查 _result 是否为 null 并且随机数小于等于概率乘以防御因子
```