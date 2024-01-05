# `d:/src/tocomm/basic-computer-games\16_Bug\csharp\Parts\Legs.cs`

```
using System.Text;  // 导入 System.Text 命名空间，用于使用 StringBuilder 类
using BugGame.Resources;  // 导入 BugGame.Resources 命名空间，用于使用 Message 类

namespace BugGame.Parts;  // 声明 BugGame.Parts 命名空间
internal class Legs : PartCollection  // 声明 Legs 类，继承自 PartCollection 类
{
    public Legs()  // 声明 Legs 类的构造函数
        : base(6, Message.LegAdded, Message.LegsFull)  // 调用基类 PartCollection 的构造函数，传入参数 6, Message.LegAdded, Message.LegsFull
    {
    }

    public void AppendTo(StringBuilder builder) => AppendTo(builder, 6, 2, 'L');  // 声明 AppendTo 方法，将参数传递给另一个重载的 AppendTo 方法
}
```