# `77_Salvo\csharp\Targetting\HumanShotSelector.cs`

```
namespace Salvo.Targetting;  // 命名空间声明

internal class HumanShotSelector : ShotSelector  // 声明一个内部类 HumanShotSelector，继承自 ShotSelector 类
{
    private readonly IReadWrite _io;  // 声明一个私有只读字段 _io，类型为 IReadWrite 接口

    internal HumanShotSelector(Fleet source, IReadWrite io)  // HumanShotSelector 类的构造函数，接受 Fleet 对象和 IReadWrite 对象作为参数
        : base(source)  // 调用基类 ShotSelector 的构造函数，传入 source 参数
    {
        _io = io;  // 将传入的 io 参数赋值给 _io 字段
    }

    protected override IEnumerable<Position> GetShots()  // 重写基类的 GetShots 方法，返回一个 Position 类型的可枚举集合
    {
        var shots = new Position[NumberOfShots];  // 声明并初始化一个 Position 类型的数组 shots，长度为 NumberOfShots

        for (var i = 0; i < shots.Length; i++)  // 循环遍历 shots 数组
        {
            while (true)  // 进入一个无限循环
            {
                // 从输入/输出对象中读取有效的位置
                var position = _io.ReadValidPosition();
                // 检查该位置是否在之前的回合中已经被选择过，并返回是否被选择过的信息
                if (WasSelectedPreviously(position, out var turnTargeted)) 
                { 
                    // 如果位置在之前的回合中已经被选择过，则输出相应的提示信息并继续循环
                    _io.WriteLine($"YOU SHOT THERE BEFORE ON TURN {turnTargeted}");
                    continue;
                }
                // 将当前位置添加到射击数组中
                shots[i] = position;
                // 跳出循环
                break;
            }
        }
        // 返回射击数组
        return shots;
    }
}
```