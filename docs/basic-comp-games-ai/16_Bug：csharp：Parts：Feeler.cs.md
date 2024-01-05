# `d:/src/tocomm/basic-computer-games\16_Bug\csharp\Parts\Feeler.cs`

```
namespace BugGame.Parts;  // 声明命名空间 BugGame.Parts

internal class Feeler : IPart  // 声明一个内部类 Feeler，实现接口 IPart
{
    public string Name => nameof(Feeler);  // 定义属性 Name，返回 Feeler 类的名称
}
```