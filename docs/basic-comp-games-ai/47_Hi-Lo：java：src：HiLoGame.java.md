# `d:/src/tocomm/basic-computer-games\47_Hi-Lo\java\src\HiLoGame.java`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流创建 ZIP 对象
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    zip.close()  # 关闭 ZIP 对象
    return fdict  # 返回结果字典
```

需要注释的代码：

```java
public class HiLoGame {

    public static void main(String[] args) {

        HiLo hiLo = new HiLo();  // 创建 HiLo 对象
        hiLo.play();  // 调用 HiLo 对象的 play 方法开始游戏
    }
}
```