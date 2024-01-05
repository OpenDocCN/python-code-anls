# `30_Cube\csharp\IOExtensions.cs`

```
namespace Cube;  # 命名空间声明

internal static class IOExtensions  # 声明一个内部静态类 IOExtensions
{
    internal static float? ReadWager(this IReadWrite io, float balance)  # 声明一个内部静态方法 ReadWager，接受一个 IReadWrite 类型的参数 io 和一个 float 类型的参数 balance，返回一个可空的 float 类型
    {
        io.Write(Streams.Wager);  # 调用 io 对象的 Write 方法，传入 Streams.Wager 参数
        if (io.ReadNumber("") == 0) { return null; }  # 如果调用 io 对象的 ReadNumber 方法返回值为 0，则返回 null

        var prompt = Prompts.HowMuch;  # 声明一个变量 prompt，赋值为 Prompts.HowMuch

        while(true)  # 进入一个无限循环
        {
            var wager = io.ReadNumber(prompt);  # 声明一个变量 wager，调用 io 对象的 ReadNumber 方法，传入 prompt 参数
            if (wager <= balance) { return wager; }  # 如果 wager 小于等于 balance，则返回 wager

            prompt = Prompts.BetAgain;  # 将 prompt 赋值为 Prompts.BetAgain
        }
    }
}
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```