# `d:/src/tocomm/basic-computer-games\87_3-D_Plot\csharp\Program.cs`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
# 打印程序的标题
private static void PrintTitle()
{
    Console.WriteLine("                                3D Plot");
    Console.WriteLine("               Creative Computing  Morristown, New Jersey");
    Console.WriteLine();
    Console.WriteLine();
    Console.WriteLine();
    Console.WriteLine();
}

# 在控制台上绘制一个点
private static void Plot(int z)
{
    # 获取当前光标的位置
    var x = Console.GetCursorPosition().Top;
    # 设置光标位置并在该位置打印一个星号表示点
    Console.SetCursorPosition(z, x);
    Console.Write("*");
}
```