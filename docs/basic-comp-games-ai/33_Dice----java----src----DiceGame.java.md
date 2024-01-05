# `33_Dice\java\src\DiceGame.java`

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

在上面的代码中，我们有一个名为`DiceGame`的Java类，其中有一个名为`main`的静态方法。在`main`方法中，我们创建了一个名为`dice`的`Dice`对象，然后调用了`play`方法。

需要添加的注释：

```java
public class DiceGame {
    public static void main(String[] args) {
        // 创建一个名为dice的Dice对象
        Dice dice = new Dice();
        // 调用Dice对象的play方法
        dice.play();
    }
}
```