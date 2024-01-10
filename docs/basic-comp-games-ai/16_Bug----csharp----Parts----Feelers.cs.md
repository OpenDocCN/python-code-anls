# `basic-computer-games\16_Bug\csharp\Parts\Feelers.cs`

```
# 导入 System.Text 命名空间
using System.Text;
# 导入 BugGame.Resources 命名空间
using BugGame.Resources;

# 定义 Feelers 类，继承自 PartCollection 类
namespace BugGame.Parts;
internal class Feelers : PartCollection
{
    # 定义构造函数，调用基类的构造函数，传入参数 2, Message.FeelerAdded, Message.FeelersFull
    public Feelers()
        : base(2, Message.FeelerAdded, Message.FeelersFull)
    {
    }

    # 定义 AppendTo 方法，将字符追加到 StringBuilder 对象中
    public void AppendTo(StringBuilder builder, char character) => AppendTo(builder, 10, 4, character);
}
```