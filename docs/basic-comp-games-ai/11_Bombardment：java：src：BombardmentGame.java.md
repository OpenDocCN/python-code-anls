# `d:/src/tocomm/basic-computer-games\11_Bombardment\java\src\BombardmentGame.java`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流创建 ZIP 对象，以只读模式打开
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象中的文件名列表，读取文件数据，组成文件名到数据的字典
    zip.close()  # 关闭 ZIP 对象
    return fdict  # 返回结果字典
```

需要注释的代码：

```java
public class BombardmentGame {

    public static void main(String[] args) {

        Bombardment bombardment = new Bombardment();  // 创建 Bombardment 类的实例对象
        bombardment.play();  // 调用实例对象的 play 方法开始游戏
    }
}
```