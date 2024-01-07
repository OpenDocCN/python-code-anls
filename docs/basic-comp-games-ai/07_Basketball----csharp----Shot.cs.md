# `basic-computer-games\07_Basketball\csharp\Shot.cs`

```

namespace Basketball; // 命名空间声明，定义了类的作用域

public class Shot // 定义名为 Shot 的公共类
{
    private readonly string _name; // 声明一个私有的只读字符串变量 _name

    public Shot(string name) // Shot 类的构造函数，接受一个字符串参数 name
    {
        _name = name; // 将传入的 name 参数赋值给 _name 变量
    }

    public static bool TryGet(int shotNumber, out Shot? shot) // 定义一个静态方法 TryGet，接受一个整数参数 shotNumber 和一个 Shot 类型的可空输出参数 shot
    {
        shot = shotNumber switch // 使用 switch 语句根据 shotNumber 的值进行匹配
        {
            0 => null, // 当 shotNumber 为 0 时，将 shot 赋值为 null
            <= 2 => new JumpShot(), // 当 shotNumber 小于等于 2 时，创建一个 JumpShot 对象并赋值给 shot
            3 => new Shot("Lay up"), // 当 shotNumber 为 3 时，创建一个名称为 "Lay up" 的 Shot 对象并赋值给 shot
            4 => new Shot("Set shot"), // 当 shotNumber 为 4 时，创建一个名称为 "Set shot" 的 Shot 对象并赋值给 shot
            _ => null // 其他情况下，将 shot 赋值为 null
        };
        return shotNumber == 0 || shot is not null; // 返回一个布尔值，判断 shotNumber 是否为 0 或者 shot 是否不为 null
    }

    public static Shot Get(float shotNumber) => // 定义一个静态方法 Get，接受一个浮点数参数 shotNumber
        shotNumber switch // 使用 switch 语句根据 shotNumber 的值进行匹配
        {
            <= 2 => new JumpShot(), // 当 shotNumber 小于等于 2 时，创建一个 JumpShot 对象并返回
            > 3 => new Shot("Set shot"), // 当 shotNumber 大于 3 时，创建一个名称为 "Set shot" 的 Shot 对象并返回
            > 2 => new Shot("Lay up"), // 当 shotNumber 大于 2 时，创建一个名称为 "Lay up" 的 Shot 对象并返回
            _ => throw new Exception("Unexpected value") // 其他情况下，抛出一个异常
        };

    public override string ToString() => _name; // 重写 ToString 方法，返回 _name 变量的值
}

```