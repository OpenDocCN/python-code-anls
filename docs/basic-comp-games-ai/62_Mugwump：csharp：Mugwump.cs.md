# `d:/src/tocomm/basic-computer-games\62_Mugwump\csharp\Mugwump.cs`

```
namespace Mugwump;  // 命名空间声明

internal class Mugwump  // 内部类 Mugwump 声明
{
    private readonly int _id;  // 声明私有只读整型变量 _id
    private readonly Position _position;  // 声明私有只读 Position 类型变量 _position

    public Mugwump(int id, Position position)  // Mugwump 类的构造函数，接受 id 和 position 参数
    {
        _id = id;  // 将 id 参数赋值给 _id 变量
        _position = position;  // 将 position 参数赋值给 _position 变量
    }

    public (bool, Distance) FindFrom(Position guess) => (guess == _position, guess - _position);  // FindFrom 方法，返回一个元组，包含布尔值和 Distance 类型的值

    public string Reveal() => $"{this} is at {_position}";  // Reveal 方法，返回包含类名和位置信息的字符串

    public override string ToString() => $"Mugwump {_id}";  // 重写 ToString 方法，返回包含类名和 id 的字符串
}
```