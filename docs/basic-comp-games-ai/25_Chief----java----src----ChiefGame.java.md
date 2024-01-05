# `25_Chief\java\src\ChiefGame.java`

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
public class ChiefGame {

    public static void main(String[] args) {
        // 创建 Chief 对象
        Chief chief = new Chief();
        // 调用 Chief 对象的 play 方法
        chief.play();
    }
}
```