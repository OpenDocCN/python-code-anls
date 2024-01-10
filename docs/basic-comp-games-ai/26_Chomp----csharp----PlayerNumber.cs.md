# `basic-computer-games\26_Chomp\csharp\PlayerNumber.cs`

```
namespace Chomp;

internal class PlayerNumber
{
    private readonly float _playerCount; // 用于存储玩家数量的私有只读浮点数字段
    private int _counter; // 用于存储计数器的私有整数字段
    private float _number; // 用于存储数字的私有浮点数字段

    // 构造函数，用于初始化 PlayerNumber 对象，传入玩家数量
    // 原始代码没有限制 playerCount 必须是整数
    public PlayerNumber(float playerCount)
    {
        _playerCount = playerCount; // 将传入的玩家数量赋值给 _playerCount
        _number = 0; // 将 _number 初始化为 0
        Increment(); // 调用 Increment 方法
    }

    // 重载 ++ 运算符，使得 PlayerNumber 对象可以进行自增操作
    public static PlayerNumber operator ++(PlayerNumber number) => number.Increment();

    // 私有方法，用于增加数字
    private PlayerNumber Increment()
    {
        if (_playerCount == 0) { throw new DivideByZeroException(); } // 如果玩家数量为 0，则抛出 DivideByZeroException 异常

        // 这里的增加逻辑与原始程序相同，并且在 _playerCount 不是整数时表现出有趣的行为
        _counter++; // 计数器加一
        _number = _counter - (float)Math.Floor(_counter / _playerCount) * _playerCount; // 计算新的数字
        if (_number == 0) { _number = _playerCount; } // 如果数字为 0，则将其设置为玩家数量
        return this; // 返回当前对象
    }

    // 重写 ToString 方法，返回数字的字符串表示形式
    public override string ToString() => (_number >= 0 ? " " : "") + _number.ToString();
}
```