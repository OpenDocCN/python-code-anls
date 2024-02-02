# `basic-computer-games\58_Love\csharp\Program.cs`

```py
# 引入 Games.Common.IO 命名空间，用于处理输入输出
using Games.Common.IO;
# 引入 Love 命名空间，用于创建 LovePattern 对象
using Love;
# 引入 Love.Resources 命名空间，用于访问资源文件
using Love.Resources;

# 创建控制台输入输出对象
var io = new ConsoleIO();

# 输出资源文件 "Intro" 的内容
io.Write(Resource.Streams.Intro);

# 从控制台读取用户输入的消息
var message = io.ReadString("Your message, please");

# 创建 LovePattern 对象，传入用户输入的消息
io.Write(new LovePattern(message));
```