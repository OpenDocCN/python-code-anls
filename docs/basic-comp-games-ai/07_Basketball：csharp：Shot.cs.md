# `d:/src/tocomm/basic-computer-games\07_Basketball\csharp\Shot.cs`

```
namespace Basketball;  # 命名空间声明，定义了代码所在的命名空间

public class Shot  # 定义了一个名为Shot的公共类
{
    private readonly string _name;  # 声明了一个私有的只读字符串变量_name

    public Shot(string name)  # 定义了一个公共的构造函数，接受一个字符串参数name
    {
        _name = name;  # 将传入的name赋值给私有变量_name
    }

    public static bool TryGet(int shotNumber, out Shot? shot)  # 定义了一个公共的静态方法TryGet，接受一个整数参数shotNumber和一个out参数shot
    {
        shot = shotNumber switch  # 使用switch语句根据shotNumber的值进行匹配
        {
            // Although the game instructions reference two different jump shots,
            // the original game code treats them both the same and just prints "Jump shot"
            0 => null,  # 当shotNumber为0时，将shot赋值为null
            <= 2 => new JumpShot(),  # 当shotNumber小于等于2时，将shot赋值为一个新的JumpShot对象
            3 => new Shot("Lay up"),  # 当shotNumber为3时，将shot赋值为一个新的Shot对象，名称为"Lay up"
            4 => new Shot("Set shot"),  // 如果 shotNumber 等于 4，则创建一个新的 Shot 对象，类型为 "Set shot"
            _ => null  // 对于其他任何值，返回 null
        };
        return shotNumber == 0 || shot is not null;  // 如果 shotNumber 等于 0 或者 shot 不为 null，则返回 true，否则返回 false
    }

    public static Shot Get(float shotNumber) =>  // 定义一个静态方法 Get，接受一个浮点数参数 shotNumber
        shotNumber switch  // 使用 switch 语句根据 shotNumber 的值进行匹配
        {
            <= 2 => new JumpShot(),  // 如果 shotNumber 小于等于 2，则创建一个新的 JumpShot 对象
            > 3 => new Shot("Set shot"),  // 如果 shotNumber 大于 3，则创建一个新的 Shot 对象，类型为 "Set shot"
            > 2 => new Shot("Lay up"),  // 如果 shotNumber 大于 2，则创建一个新的 Shot 对象，类型为 "Lay up"
            _ => throw new Exception("Unexpected value")  // 对于其他任何值，抛出一个异常，提示值为意外值
        };

    public override string ToString() => _name;  // 重写 ToString 方法，返回 Shot 对象的名称
}

```