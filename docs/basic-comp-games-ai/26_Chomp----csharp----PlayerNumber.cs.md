# `basic-computer-games\26_Chomp\csharp\PlayerNumber.cs`

```

// 命名空间 Chomp，表示这段代码属于 Chomp 命名空间
namespace Chomp;

// 内部类 PlayerNumber，表示这是一个内部类，只能在当前命名空间中访问
internal class PlayerNumber
{
    // 只读字段 _playerCount，表示这个字段只能在构造函数中初始化，之后不能再修改
    private readonly float _playerCount;
    // 整型字段 _counter，表示这是一个整数类型的计数器
    private int _counter;
    // 浮点型字段 _number，表示这是一个浮点数类型的数字
    private float _number;

    // 构造函数，用来初始化 PlayerNumber 对象
    // 参数 playerCount 表示玩家数量
    // 注意：原始代码没有限制 playerCount 必须是整数
    public PlayerNumber(float playerCount)
    {
        // 将参数 playerCount 赋值给只读字段 _playerCount
        _playerCount = playerCount;
        // 将 _number 初始化为 0
        _number = 0;
        // 调用 Increment 方法
        Increment();
    }

    // 重载 ++ 运算符，表示对 PlayerNumber 对象进行自增操作
    public static PlayerNumber operator ++(PlayerNumber number) => number.Increment();

    // 私有方法 Increment，用来实现自增逻辑
    private PlayerNumber Increment()
    {
        // 如果 _playerCount 为 0，则抛出除零异常
        if (_playerCount == 0) { throw new DivideByZeroException(); }

        // 自增逻辑，当 _playerCount 不是整数时，会产生有趣的行为
        _counter++;
        _number = _counter - (float)Math.Floor(_counter / _playerCount) * _playerCount;
        if (_number == 0) { _number = _playerCount; }
        return this;
    }

    // 重写 ToString 方法，返回 _number 的字符串表示
    public override string ToString() => (_number >= 0 ? " " : "") + _number.ToString();
}

```