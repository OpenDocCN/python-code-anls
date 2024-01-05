# `d:/src/tocomm/basic-computer-games\26_Chomp\csharp\PlayerNumber.cs`

```
namespace Chomp;

internal class PlayerNumber
{
    private readonly float _playerCount; // 用于存储玩家数量的私有只读浮点数字段
    private int _counter; // 用于存储计数器的私有整数字段
    private float _number; // 用于存储数字的私有浮点数字段

    // 原始代码没有限制 playerCount 必须是整数
    public PlayerNumber(float playerCount) // 构造函数，接受一个浮点数参数 playerCount
    {
        _playerCount = playerCount; // 将参数 playerCount 赋值给私有字段 _playerCount
        _number = 0; // 将数字字段 _number 初始化为 0
        Increment(); // 调用 Increment 方法
    }

    public static PlayerNumber operator ++(PlayerNumber number) => number.Increment(); // 重载 ++ 运算符，调用 Increment 方法

    private PlayerNumber Increment() // 私有方法 Increment
    {
		if (_playerCount == 0) { throw new DivideByZeroException(); }  // 检查玩家数量是否为零，如果是则抛出 DivideByZeroException 异常

        // The increment logic here is the same as the original program, and exhibits
        // interesting behaviour when _playerCount is not an integer.
        _counter++;  // 计数器加一
        _number = _counter - (float)Math.Floor(_counter / _playerCount) * _playerCount;  // 根据计数器和玩家数量计算出一个浮点数
        if (_number == 0) { _number = _playerCount; }  // 如果计算出的浮点数为零，则将其赋值为玩家数量
        return this;  // 返回当前对象
    }

    public override string ToString() => (_number >= 0 ? " " : "") + _number.ToString();  // 重写 ToString 方法，返回 _number 的字符串表示形式
}
```