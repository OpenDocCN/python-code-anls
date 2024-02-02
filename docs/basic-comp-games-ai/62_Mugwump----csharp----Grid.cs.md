# `basic-computer-games\62_Mugwump\csharp\Grid.cs`

```py
// 引入必要的命名空间
using System.Collections.Generic;
using System.Linq;

// 命名空间Mugwump下的Grid类
namespace Mugwump
{
    // Grid类
    internal class Grid
    {
        // 私有字段，用于输入输出
        private readonly TextIO _io;
        // 私有字段，用于存储Mugwump对象的列表
        private readonly List<Mugwump> _mugwumps;

        // 构造函数，接受TextIO对象和IRandom对象作为参数
        public Grid(TextIO io, IRandom random)
        {
            // 初始化_io字段
            _io = io;
            // 使用Enumerable生成1到4的序列，对每个元素生成一个Mugwump对象，并存储在_mugwumps列表中
            _mugwumps = Enumerable.Range(1, 4).Select(id => new Mugwump(id, random.NextPosition(10, 10))).ToList();
        }

        // 检查玩家猜测的位置是否有Mugwump
        public bool Check(Position guess)
        {
            // 遍历_mugwumps列表的副本
            foreach (var mugwump in _mugwumps.ToList())
            {
                // 调用Mugwump对象的FindFrom方法，返回是否找到和距离
                var (found, distance) = mugwump.FindFrom(guess);

                // 根据找到与否输出不同的消息
                _io.WriteLine(found ? $"You have found {mugwump}" : $"You are {distance} units from {mugwump}");
                // 如果找到了Mugwump，则从_mugwumps列表中移除该Mugwump对象
                if (found)
                {
                    _mugwumps.Remove(mugwump);
                }
            }

            // 返回_mugwumps列表是否为空，即是否所有Mugwump都已找到
            return _mugwumps.Count == 0;
        }

        // 显示所有Mugwump的位置
        public void Reveal()
        {
            // 遍历_mugwumps列表，依次调用每个Mugwump对象的Reveal方法并输出位置信息
            foreach (var mugwump in _mugwumps)
            {
                _io.WriteLine(mugwump.Reveal());
            }
        }
    }
}
```