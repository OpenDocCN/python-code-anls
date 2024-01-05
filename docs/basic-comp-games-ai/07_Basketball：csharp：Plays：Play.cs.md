# `d:/src/tocomm/basic-computer-games\07_Basketball\csharp\Plays\Play.cs`

```
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using Games.Common.Randomness;  # 导入 Games.Common.Randomness 模块

namespace Basketball.Plays;  # 定义 Basketball.Plays 命名空间

internal abstract class Play  # 定义一个抽象类 Play
{
    private readonly IReadWrite _io;  # 声明一个私有的只读属性 _io，类型为 IReadWrite 接口
    private readonly IRandom _random;  # 声明一个私有的只读属性 _random，类型为 IRandom 接口
    private readonly Clock _clock;  # 声明一个私有的只读属性 _clock，类型为 Clock 类

    public Play(IReadWrite io, IRandom random, Clock clock)  # 定义一个构造函数，接受 IReadWrite 类型的 io、IRandom 类型的 random 和 Clock 类型的 clock 参数
    {
        _io = io;  # 将传入的 io 参数赋值给 _io 属性
        _random = random;  # 将传入的 random 参数赋值给 _random 属性
        _clock = clock;  # 将传入的 clock 参数赋值给 _clock 属性
    }

    protected bool ClockIncrementsToHalfTime(Scoreboard scoreboard)  # 定义一个受保护的方法 ClockIncrementsToHalfTime，接受 Scoreboard 类型的 scoreboard 参数
    {
        _clock.Increment(scoreboard);  // 使用时钟对象的Increment方法来增加比分板的时间
        return _clock.IsHalfTime;  // 返回时钟对象的IsHalfTime属性，表示是否是半场时间
    }

    internal abstract bool Resolve(Scoreboard scoreboard);  // 抽象方法，用于解决比分板的问题

    protected void ResolveFreeThrows(Scoreboard scoreboard, string message) =>  // 解决罚球问题的方法
        Resolve(message)  // 调用Resolve方法，传入消息参数
            .Do(0.49f, () => scoreboard.AddFreeThrows(2, "Shooter makes both shots."))  // 如果概率小于0.49，执行得分板的AddFreeThrows方法
            .Or(0.75f, () => scoreboard.AddFreeThrows(1, "Shooter makes one shot and misses one."))  // 如果概率小于0.75，执行得分板的AddFreeThrows方法
            .Or(() => scoreboard.AddFreeThrows(0, "Both shots missed."));  // 执行得分板的AddFreeThrows方法

    protected Probably Resolve(string message) => Resolve(message, 1f);  // 重载的Resolve方法，传入消息参数和默认的防守因素

    protected Probably Resolve(string message, float defenseFactor)  // 重载的Resolve方法，传入消息参数和防守因素
    {
        _io.WriteLine(message);  // 输出消息
        return new Probably(defenseFactor, _random);  // 返回一个新的Probably对象，传入防守因素和随机数生成器
    }
}
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```