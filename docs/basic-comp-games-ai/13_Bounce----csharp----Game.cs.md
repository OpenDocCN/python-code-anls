# `basic-computer-games\13_Bounce\csharp\Game.cs`

```py
// 引用 Bounce.Resources.Resource 中的静态成员
using static Bounce.Resources.Resource;

// 命名空间 Bounce
namespace Bounce
{
    // 内部类 Game
    internal class Game
    {
        // 只读字段 _io，类型为 IReadWrite 接口
        private readonly IReadWrite _io;

        // 构造函数，接受 IReadWrite 类型的参数 io
        public Game(IReadWrite io)
        {
            // 将参数 io 赋值给字段 _io
            _io = io;
        }

        // 方法 Play，接受 Func<bool> 类型的参数 playAgain
        public void Play(Func<bool> playAgain)
        {
            // 输出标题
            _io.Write(Streams.Title);
            // 输出说明
            _io.Write(Streams.Instructions);

            // 当 playAgain 返回 true 时循环执行以下代码块
            while (playAgain.Invoke())
            {
                // 读取时间增量参数
                var timeIncrement = _io.ReadParameter("Time increment (sec)");
                // 读取速度参数
                var velocity = _io.ReadParameter("Velocity (fps)");
                // 读取弹性系数参数
                var elasticity = _io.ReadParameter("Coefficient");

                // 创建 Bounce 对象，传入速度参数
                var bounce = new Bounce(velocity);
                // 计算弹跳次数
                var bounceCount = (int)(Graph.Row.Width * timeIncrement / bounce.Duration);
                // 创建 Graph 对象，传入最大高度和时间增量参数
                var graph = new Graph(bounce.MaxHeight, timeIncrement);

                // 初始化时间为 0
                var time = 0f;
                // 循环执行弹跳次数次
                for (var i = 0; i < bounceCount; i++, bounce = bounce.Next(elasticity))
                {
                    // 绘制弹跳图形，更新时间
                    time = bounce.Plot(graph, time);
                }

                // 输出图形
                _io.WriteLine(graph);
            }
        }
    }
}
```