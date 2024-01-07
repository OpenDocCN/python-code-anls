# `basic-computer-games\75_Roulette\csharp\Wheel.cs`

```

// 使用不可变数组存储轮盘上的所有槽位
private static readonly ImmutableArray<Slot> _slots = ImmutableArray.Create(
    // 创建每个槽位对象，包括槽位名称和对应的数字
    new Slot(Strings.Red(1), 1, 37, 40, 43, 46, 47),
    new Slot(Strings.Black(2), 2, 37, 41, 43, 45, 48),
    // ... 其他槽位对象的创建
    new Slot("0", 49),
    new Slot("00", 50));

// 轮盘类，包含一个随机数生成器
private readonly IRandom _random;

// 构造函数，初始化随机数生成器
public Wheel(IRandom random) => _random = random;

// 旋转轮盘，返回随机选择的槽位
public Slot Spin() => _slots[_random.Next(_slots.Length)];

```