# `13_Bounce\csharp\Game.cs`

```
using static Bounce.Resources.Resource;  # 导入 Bounce 资源中的 Resource 类
namespace Bounce;  # 声明 Bounce 命名空间

internal class Game  # 内部类 Game
{
    private readonly IReadWrite _io;  # 声明私有只读字段 _io，类型为 IReadWrite 接口

    public Game(IReadWrite io)  # Game 类的构造函数，参数为 io
    {
        _io = io;  # 将参数 io 赋值给 _io 字段
    }

    public void Play(Func<bool> playAgain)  # Play 方法，参数为 playAgain 函数
    {
        _io.Write(Streams.Title);  # 调用 _io 对象的 Write 方法，传入 Streams.Title 参数
        _io.Write(Streams.Instructions);  # 调用 _io 对象的 Write 方法，传入 Streams.Instructions 参数

        while (playAgain.Invoke())  # while 循环，条件为调用 playAgain 函数的结果
        {
            // 从输入输出对象中读取时间增量参数
            var timeIncrement = _io.ReadParameter("Time increment (sec)");
            // 从输入输出对象中读取速度参数
            var velocity = _io.ReadParameter("Velocity (fps)");
            // 从输入输出对象中读取弹性系数参数
            var elasticity = _io.ReadParameter("Coefficient");

            // 创建一个弹跳对象，传入速度参数
            var bounce = new Bounce(velocity);
            // 计算弹跳次数，根据图表行宽度、时间增量和弹跳持续时间计算得出
            var bounceCount = (int)(Graph.Row.Width * timeIncrement / bounce.Duration);
            // 创建一个图表对象，传入最大高度和时间增量参数
            var graph = new Graph(bounce.MaxHeight, timeIncrement);

            // 初始化时间为0
            var time = 0f;
            // 循环进行弹跳次数次的弹跳操作
            for (var i = 0; i < bounceCount; i++, bounce = bounce.Next(elasticity))
            {
                // 在图表上绘制弹跳轨迹，并更新时间
                time = bounce.Plot(graph, time);
            }

            // 将图表输出到输入输出对象
            _io.WriteLine(graph);
        }
    }
}
```