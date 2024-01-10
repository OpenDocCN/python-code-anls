# `basic-computer-games\75_Roulette\csharp\Wheel.cs`

```
# 使用不可变集合命名空间
using System.Collections.Immutable;

# 创建名为Roulette的命名空间
namespace Roulette;

# 创建名为Wheel的内部类
internal class Wheel
{
    # 声明只读字段_random，类型为IRandom接口
    private readonly IRandom _random;

    # 构造函数，接受一个IRandom类型的参数，并将其赋值给_random字段
    public Wheel(IRandom random) => _random = random;

    # 定义一个名为Spin的公共方法，返回类型为Slot
    public Slot Spin() => _slots[_random.Next(_slots.Length)];
}
```