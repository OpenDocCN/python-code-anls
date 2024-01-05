# `93_23_Matches\java\Messages.java`

```
# This is a utility class and contains only static members.
# Utility classes are not meant to be instantiated.
# 创建一个工具类，只包含静态成员，不应该被实例化
# 当尝试实例化该类时，抛出异常
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
    // 提示用户输入要移除的火柴数量
    public static final String REMOVE_MATCHES_QUESTION = "HOW MANY DO YOU WISH TO REMOVE? ";

    // 提示用户剩余的火柴数量
    public static final String REMAINING_MATCHES = """
            THERE ARE NOW %d MATCHES REMAINING.
            """;

    // 提示用户输入无效
    public static final String INVALID = """
            VERY FUNNY! DUMMY!
            DO YOU WANT TO PLAY OR GOOF AROUND?
            NOW, HOW MANY MATCHES DO YOU WANT?
            """;

    // 提示用户获胜
    public static final String WIN = """
            YOU WON, FLOPPY EARS !
            THINK YOU'RE PRETTY SMART !
            LETS PLAY AGAIN AND I'LL BLOW YOUR SHOES OFF !!
            """;

    // 提示轮到电脑操作
    public static final String CPU_TURN = """
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