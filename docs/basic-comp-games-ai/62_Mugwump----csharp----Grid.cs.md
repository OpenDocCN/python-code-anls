# `basic-computer-games\62_Mugwump\csharp\Grid.cs`

```

// 引入必要的命名空间
using System.Collections.Generic;
using System.Linq;

// 定义 Mugwump 命名空间
namespace Mugwump;

// 定义 Grid 类
internal class Grid
{
    // 声明私有变量 _io 和 _mugwumps
    private readonly TextIO _io;
    private readonly List<Mugwump> _mugwumps;

    // 定义 Grid 类的构造函数，接受 TextIO 和 IRandom 对象作为参数
    public Grid(TextIO io, IRandom random)
    {
        // 初始化 _io 变量
        _io = io;
        // 使用 Enumerable 生成 1 到 4 的序列，对每个序列元素创建一个 Mugwump 对象，并将其添加到 _mugwumps 列表中
        _mugwumps = Enumerable.Range(1, 4).Select(id => new Mugwump(id, random.NextPosition(10, 10))).ToList();
    }

    // 定义 Check 方法，接受一个 Position 对象作为参数，返回布尔值
    public bool Check(Position guess)
    {
        // 遍历 _mugwumps 列表中的每个 Mugwump 对象
        foreach (var mugwump in _mugwumps.ToList())
        {
            // 调用 Mugwump 对象的 FindFrom 方法，返回找到的标志和距离
            var (found, distance) = mugwump.FindFrom(guess);

            // 根据找到的标志和距离输出不同的信息
            _io.WriteLine(found ? $"You have found {mugwump}" : $"You are {distance} units from {mugwump}");
            // 如果找到了 Mugwump 对象，则从 _mugwumps 列表中移除该对象
            if (found)
            {
                _mugwumps.Remove(mugwump);
            }
        }

        // 返回 _mugwumps 列表是否为空的布尔值
        return _mugwumps.Count == 0;
    }

    // 定义 Reveal 方法，用于显示每个 Mugwump 对象的信息
    public void Reveal()
    {
        // 遍历 _mugwumps 列表中的每个 Mugwump 对象，调用其 Reveal 方法并输出信息
        foreach (var mugwump in _mugwumps)
        {
            _io.WriteLine(mugwump.Reveal());
        }
    }
}

```