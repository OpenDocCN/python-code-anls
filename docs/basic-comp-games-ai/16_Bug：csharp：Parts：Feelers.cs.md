# `d:/src/tocomm/basic-computer-games\16_Bug\csharp\Parts\Feelers.cs`

```
using System.Text;  // 导入 System.Text 命名空间，用于使用 StringBuilder 类
using BugGame.Resources;  // 导入 BugGame.Resources 命名空间，用于使用 Message 类

namespace BugGame.Parts;  // 声明 BugGame.Parts 命名空间

internal class Feelers : PartCollection  // 声明 Feelers 类，继承自 PartCollection 类
{
    public Feelers()  // 声明 Feelers 类的构造函数
        : base(2, Message.FeelerAdded, Message.FeelersFull)  // 调用基类 PartCollection 的构造函数，传入参数 2, Message.FeelerAdded, Message.FeelersFull
    {
    }

    public void AppendTo(StringBuilder builder, char character) => AppendTo(builder, 10, 4, character);  // 声明 AppendTo 方法，接受 StringBuilder 和 char 类型的参数，并调用另一个重载的 AppendTo 方法
}
```