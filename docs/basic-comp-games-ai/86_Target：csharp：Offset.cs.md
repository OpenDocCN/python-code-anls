# `86_Target\csharp\Offset.cs`

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
```

```csharp
using System;

namespace Target
{
    internal class Offset
    {
        public Offset(float deltaX, float deltaY, float deltaZ)
        {
            DeltaX = deltaX;  # 设置 DeltaX 属性为传入的 deltaX 值
            DeltaY = deltaY;  # 设置 DeltaY 属性为传入的 deltaY 值
            DeltaZ = deltaZ;  # 设置 DeltaZ 属性为传入的 deltaZ 值

            Distance = (float)Math.Sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ + deltaZ);  # 计算距离属性的值
        }

        public float DeltaX { get; }  # 获取 DeltaX 属性
        public float DeltaY { get; }  # 获取 DeltaY 属性
        public float DeltaZ { get; }  # 获取 DeltaZ 属性
        public float Distance { get; }  # 获取 Distance 属性
    }
}
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```