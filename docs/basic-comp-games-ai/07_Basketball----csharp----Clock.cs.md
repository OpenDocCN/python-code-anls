# `07_Basketball\csharp\Clock.cs`

```
using Basketball.Resources;  // 导入Basketball.Resources命名空间，用于访问资源文件
using Games.Common.IO;  // 导入Games.Common.IO命名空间，用于访问输入输出相关的功能

namespace Basketball;  // 声明Basketball命名空间

internal class Clock  // 声明Clock类，表示比赛计时器
{
    private readonly IReadWrite _io;  // 声明私有成员变量_io，类型为IReadWrite接口
    private int time;  // 声明私有成员变量time，表示比赛时间

    public Clock(IReadWrite io) => _io = io;  // Clock类的构造函数，接受一个IReadWrite类型的参数io，并将其赋值给成员变量_io

    public bool IsHalfTime => time == 50;  // 声明IsHalfTime属性，表示是否是半场时间，当time等于50时返回true
    public bool IsFullTime => time >= 100;  // 声明IsFullTime属性，表示是否是全场时间，当time大于等于100时返回true
    public bool TwoMinutesLeft => time == 92;  // 声明TwoMinutesLeft属性，表示是否还剩两分钟，当time等于92时返回true

    public void Increment(Scoreboard scoreboard)  // 声明Increment方法，接受一个Scoreboard类型的参数scoreboard
    {
        time += 1;  // 时间加1
        if (IsHalfTime) { scoreboard.Display(Resource.Formats.EndOfFirstHalf); }  // 如果是半场时间，调用scoreboard的Display方法显示半场结束的信息
        if (TwoMinutesLeft) { _io.Write(Resource.Streams.TwoMinutesLeft); }  // 如果剩余时间为两分钟，则向输出流写入两分钟剩余的资源数据
    }

    public void StartOvertime() => time = 93;  // 开始加时赛，将时间设置为93分钟
```