# `d:/src/tocomm/basic-computer-games\75_Roulette\csharp\IOExtensions.cs`

```
{
    // 读取下注数量的方法
    internal static int ReadBetCount(this IReadWrite io)
    {
        // 循环直到输入有效的下注数量
        while (true)
        {
            // 从输入中读取下注数量
            var betCount = io.ReadNumber(Prompts.HowManyBets);
            // 如果下注数量是有效的整数，就返回该数量
            if (betCount.IsValidInt(1)) { return (int)betCount; }
        }
    }

    // 读取下注的方法
    internal static Bet ReadBet(this IReadWrite io, int number)
    {
        // 循环直到输入有效的下注类型和金额
        while (true)
        {
            // 从输入中读取下注类型和金额
            var (type, amount) = io.Read2Numbers(Prompts.Bet(number));

            // 如果下注类型和金额都是有效的整数，就继续执行下一步操作
            if (type.IsValidInt(1, 50) && amount.IsValidInt(5, 500))
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流对象
    使用字节流里面内容创建 ZIP 对象  # 使用字节流内容创建一个 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流创建 ZIP 对象，以只读模式打开
    遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典  # 遍历 ZIP 对象中的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 使用字典推导式，将文件名和对应的数据组成字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```