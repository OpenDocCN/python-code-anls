# `basic-computer-games\07_Basketball\csharp\Shot.cs`

```py
namespace Basketball;

public class Shot
{
    private readonly string _name;

    public Shot(string name)
    {
        _name = name;
    }

    // 尝试获取投篮动作对象
    public static bool TryGet(int shotNumber, out Shot? shot)
    {
        // 使用 switch 语句根据投篮编号获取对应的投篮动作对象
        shot = shotNumber switch
        {
            // 虽然游戏说明提到了两种不同的跳投动作，但原始游戏代码将它们视为相同，只是打印“跳投”
            0 => null,
            <= 2 => new JumpShot(),
            3 => new Shot("Lay up"),
            4 => new Shot("Set shot"),
            _ => null
        };
        // 返回是否成功获取投篮动作对象的布尔值
        return shotNumber == 0 || shot is not null;
    }

    // 根据浮点数获取投篮动作对象
    public static Shot Get(float shotNumber) =>
        shotNumber switch
        {
            <= 2 => new JumpShot(),
            > 3 => new Shot("Set shot"),
            > 2 => new Shot("Lay up"),
            _ => throw new Exception("Unexpected value")
        };

    // 重写 ToString 方法，返回投篮动作的名称
    public override string ToString() => _name;
}
```