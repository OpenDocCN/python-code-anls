# `basic-computer-games\16_Bug\csharp\Parts\Feeler.cs`

```

# 声明一个名为 BugGame.Parts 的命名空间，并在其中声明一个内部类 Feeler，实现了 IPart 接口
namespace BugGame.Parts;

# 内部类 Feeler 实现了 IPart 接口
internal class Feeler : IPart
{
    # 获取 Feeler 类的名称，并赋值给 Name 属性
    public string Name => nameof(Feeler);
}

```