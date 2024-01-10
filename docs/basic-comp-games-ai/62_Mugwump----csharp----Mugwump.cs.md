# `basic-computer-games\62_Mugwump\csharp\Mugwump.cs`

```
# 定义了一个名为Mugwump的命名空间
namespace Mugwump;

# 定义了一个名为Mugwump的内部类
internal class Mugwump
{
    # 声明了一个私有的整型变量_id
    private readonly int _id;
    # 声明了一个私有的Position类型变量_position
    private readonly Position _position;

    # 定义了一个构造函数，接受id和position作为参数
    public Mugwump(int id, Position position)
    {
        # 将传入的id赋值给_id
        _id = id;
        # 将传入的position赋值给_position
        _position = position;
    }

    # 定义了一个FindFrom方法，接受guess作为参数，返回一个元组(bool, Distance)
    public (bool, Distance) FindFrom(Position guess) => (guess == _position, guess - _position);

    # 定义了一个Reveal方法，返回一个字符串，表示Mugwump的位置
    public string Reveal() => $"{this} is at {_position}";

    # 重写了ToString方法，返回Mugwump的id
    public override string ToString() => $"Mugwump {_id}";
}
```