# `90_Tower\csharp\Program.cs`

```
# 导入所需的命名空间和类
using System;
using Tower.Resources;
using Tower.UI;

namespace Tower
{
    class Program
    {
        static void Main(string[] args)
        {
            # 在控制台输出游戏标题
            Console.Write(Strings.Title);

            # 循环执行以下代码
            do
            {
                # 在控制台输出游戏介绍
                Console.Write(Strings.Intro);

                # 尝试从用户输入中读取磁盘数量，如果失败则返回
                if (!Input.TryReadNumber(Prompt.DiskCount, out var diskCount)) { return; }

                # 创建一个新的游戏对象，传入磁盘数量作为参数
                var game = new Game(diskCount);
# 如果游戏没有成功进行，则返回
if (!game.Play()) { return; }
# 使用 do-while 循环，当输入为 Yes 时继续进行游戏，否则结束循环
} while (Input.ReadYesNo(Strings.PlayAgainPrompt, Strings.YesNoPrompt));
# 打印感谢信息
Console.Write(Strings.Thanks);
```