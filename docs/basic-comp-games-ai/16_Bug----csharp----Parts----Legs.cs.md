# `basic-computer-games\16_Bug\csharp\Parts\Legs.cs`

```

# 导入 System.Text 和 BugGame.Resources 命名空间
using System.Text;
using BugGame.Resources;

# 声明 Legs 类，继承自 PartCollection 类
namespace BugGame.Parts
{
    internal class Legs : PartCollection
    {
        # 构造函数，初始化 Legs 类，调用基类的构造函数，传入参数 6, Message.LegAdded, Message.LegsFull
        public Legs()
            : base(6, Message.LegAdded, Message.LegsFull)
        {
        }

        # 定义 AppendTo 方法，将部件信息追加到 StringBuilder 对象中，传入参数为 builder, 6, 2, 'L'
        public void AppendTo(StringBuilder builder) => AppendTo(builder, 6, 2, 'L');
    }
}

```