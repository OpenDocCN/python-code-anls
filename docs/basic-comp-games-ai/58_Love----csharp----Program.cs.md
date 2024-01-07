# `basic-computer-games\58_Love\csharp\Program.cs`

```

# 引入 Games.Common.IO 命名空间，以便使用其中的 ConsoleIO 类
# 引入 Love 命名空间，以便使用其中的 Resources 类
using Games.Common.IO;
using Love;
using Love.Resources;

# 创建一个 ConsoleIO 对象
var io = new ConsoleIO();

# 从资源中读取并输出 Intro 流
io.Write(Resource.Streams.Intro);

# 从控制台输入一个消息
var message = io.ReadString("Your message, please");

# 创建一个 LovePattern 对象，并将输入的消息作为参数传入
io.Write(new LovePattern(message));

```