# `basic-computer-games\07_Basketball\csharp\Defense.cs`

```

# 命名空间 Basketball 下的内部类 Defense
namespace Basketball;

internal class Defense
{
    # 私有属性 _value，用于存储防守值
    private float _value;

    # 构造函数，接受一个参数 value，并调用 Set 方法进行赋值
    public Defense(float value) => Set(value);

    # 公有方法，用于设置防守值
    public void Set(float value) => _value = value;

    # 隐式转换操作符，将 Defense 类型转换为 float 类型
    public static implicit operator float(Defense defense) => defense._value;
}

```