# `75_Roulette\csharp\Wheel.cs`

```
using System.Collections.Immutable;  // 导入不可变集合的命名空间

namespace Roulette  // 命名空间定义
{
    internal class Wheel  // 定义内部类 Wheel
    {
        private static readonly ImmutableArray<Slot> _slots = ImmutableArray.Create(  // 创建不可变集合 _slots
            new Slot(Strings.Red(1), 1, 37, 40, 43, 46, 47),  // 创建 Slot 对象并添加到不可变集合中
            new Slot(Strings.Black(2), 2, 37, 41, 43, 45, 48),  // 创建 Slot 对象并添加到不可变集合中
            // ... 其余 Slot 对象的创建和添加
        );
# 创建新的 Slot 对象，使用不同的颜色和数字参数
new Slot(Strings.Red(14), 14, 38, 41, 43, 45, 47),  # 创建红色的 Slot 对象，数字为14
new Slot(Strings.Black(15), 15, 38, 42, 43, 46, 48),  # 创建黑色的 Slot 对象，数字为15
# ... 以此类推，创建不同颜色和数字的 Slot 对象
        new Slot(Strings.Red(34), 34, 39, 40, 44, 45, 47),  // 创建一个红色的槽，包含了槽号和对应的位置
        new Slot(Strings.Black(35), 35, 39, 41, 44, 46, 48),  // 创建一个黑色的槽，包含了槽号和对应的位置
        new Slot(Strings.Red(36), 36, 39, 42, 44, 45, 47),  // 创建一个红色的槽，包含了槽号和对应的位置
        new Slot("0", 49),  // 创建一个槽，槽号为"0"，位置为49
        new Slot("00", 50));  // 创建一个槽，槽号为"00"，位置为50
    
    private readonly IRandom _random;  // 声明一个私有的随机数生成器接口变量

    public Wheel(IRandom random) => _random = random;  // 构造函数，接受一个随机数生成器接口变量，并将其赋值给私有变量

    public Slot Spin() => _slots[_random.Next(_slots.Length)];  // 旋转轮盘，随机返回一个槽
```