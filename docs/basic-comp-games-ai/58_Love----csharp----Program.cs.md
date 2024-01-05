# `58_Love\csharp\Program.cs`

```
# 导入所需的模块
import Games.Common.IO
import Love
import Love.Resources

# 创建控制台输入输出对象
io = new ConsoleIO()

# 输出介绍信息
io.Write(Resource.Streams.Intro)

# 从控制台读取用户输入的消息
message = io.ReadString("Your message, please")

# 将用户输入的消息转换成 Love 模式并输出
io.Write(new LovePattern(message))
```