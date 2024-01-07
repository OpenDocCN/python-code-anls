# `basic-computer-games\13_Bounce\csharp\Game.cs`

```

# 引入 Bounce.Resources.Resource 命名空间下的静态资源
using static Bounce.Resources.Resource;

# 声明 Bounce 命名空间
namespace Bounce;

# 声明 Game 类，内部可见
internal class Game
{
    # 声明私有只读字段 _io，类型为 IReadWrite 接口
    private readonly IReadWrite _io;

    # Game 类的构造函数，接受一个 IReadWrite 类型的参数 io
    public Game(IReadWrite io)
    {
        # 将参数 io 赋值给私有字段 _io
        _io = io;
    }

    # Play 方法，接受一个返回布尔值的委托 playAgain
    public void Play(Func<bool> playAgain)
    {
        # 使用 _io 输出标题
        _io.Write(Streams.Title);
        # 使用 _io 输出说明
        _io.Write(Streams.Instructions);

        # 当 playAgain 委托返回 true 时循环执行以下代码块
        while (playAgain.Invoke())
        {
            # 从 _io 读取时间增量参数
            var timeIncrement = _io.ReadParameter("Time increment (sec)");
            # 从 _io 读取速度参数
            var velocity = _io.ReadParameter("Velocity (fps)");
            # 从 _io 读取弹性系数参数
            var elasticity = _io.ReadParameter("Coefficient");

            # 创建一个新的 Bounce 对象，传入速度参数
            var bounce = new Bounce(velocity);
            # 计算弹跳次数
            var bounceCount = (int)(Graph.Row.Width * timeIncrement / bounce.Duration);
            # 创建一个新的 Graph 对象，传入最大高度和时间增量参数
            var graph = new Graph(bounce.MaxHeight, timeIncrement);

            # 初始化时间为 0
            var time = 0f;
            # 循环执行弹跳次数次以下代码块
            for (var i = 0; i < bounceCount; i++, bounce = bounce.Next(elasticity))
            {
                # 调用 bounce 对象的 Plot 方法，传入 graph 和时间参数，并更新时间
                time = bounce.Plot(graph, time);
            }

            # 使用 _io 输出 graph
            _io.WriteLine(graph);
        }
    }
}

```