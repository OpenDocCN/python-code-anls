# `d:/src/tocomm/basic-computer-games\75_Roulette\csharp\Slot.cs`

```
using System.Collections.Immutable;  // 导入不可变集合的命名空间

namespace Roulette  // 命名空间定义
{
    internal class Slot  // 定义名为Slot的内部类
    {
        private readonly ImmutableHashSet<BetType> _coveringBets;  // 声明一个只读的不可变集合_coveringBets，存储BetType类型的对象

        public Slot (string name, params BetType[] coveringBets)  // 构造函数，接受一个名为name的字符串和一个或多个BetType类型的coveringBets参数
        {
            Name = name;  // 初始化Name属性为传入的name参数
            _coveringBets = coveringBets.ToImmutableHashSet();  // 将传入的coveringBets参数转换为不可变集合并赋值给_coveringBets
        }

        public string Name { get; }  // 只读属性Name，用于获取Slot的名称

        public bool IsCoveredBy(Bet bet) => _coveringBets.Contains(bet.Type);  // 方法IsCoveredBy，用于判断传入的Bet对象的Type是否包含在_coveringBets中
    }
}
```