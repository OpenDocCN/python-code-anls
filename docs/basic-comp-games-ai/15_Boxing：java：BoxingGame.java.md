# `d:/src/tocomm/basic-computer-games\15_Boxing\java\BoxingGame.java`

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

需要注释的代码：

```java
public class BoxingGame {

    public static void main(String[] args) {
        new Boxing().play();
    }
}
```

注释：
- public class BoxingGame: 定义了一个名为 BoxingGame 的公共类
- public static void main(String[] args): 主方法，程序的入口点
- new Boxing().play(): 创建 Boxing 类的实例并调用 play() 方法进行游戏
```