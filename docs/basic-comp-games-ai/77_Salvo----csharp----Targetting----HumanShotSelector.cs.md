# `basic-computer-games\77_Salvo\csharp\Targetting\HumanShotSelector.cs`

```py
// 命名空间 Salvo.Targetting 下的内部类 HumanShotSelector，继承自 ShotSelector 类
internal class HumanShotSelector : ShotSelector
{
    // 只读字段 _io，类型为 IReadWrite 接口
    private readonly IReadWrite _io;

    // HumanShotSelector 类的构造函数，接受 Fleet 对象和 IReadWrite 对象作为参数
    internal HumanShotSelector(Fleet source, IReadWrite io) 
        : base(source) // 调用基类的构造函数
    {
        _io = io; // 将传入的 io 对象赋值给 _io 字段
    }

    // 重写基类的 GetShots 方法
    protected override IEnumerable<Position> GetShots()
    {
        // 创建一个包含 NumberOfShots 个元素的 Position 数组
        var shots = new Position[NumberOfShots];
        
        // 循环遍历 shots 数组
        for (var i = 0; i < shots.Length; i++)
        {
            // 无限循环，直到满足条件才跳出循环
            while (true)
            {
                // 从 _io 对象中读取有效的位置
                var position = _io.ReadValidPosition();
                // 如果该位置之前已经被选中过，则输出提示信息并继续循环
                if (WasSelectedPreviously(position, out var turnTargeted)) 
                { 
                    _io.WriteLine($"YOU SHOT THERE BEFORE ON TURN {turnTargeted}");
                    continue;
                }
                // 将位置赋值给 shots 数组的第 i 个元素，并跳出循环
                shots[i] = position;
                break;
            }
        }

        // 返回 shots 数组
        return shots;
    }
}
```