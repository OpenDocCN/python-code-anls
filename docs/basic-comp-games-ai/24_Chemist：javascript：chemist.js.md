# `d:/src/tocomm/basic-computer-games\24_Chemist\javascript\chemist.js`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，创建字节流对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    zip.close()  # 关闭 ZIP 对象
    return fdict  # 返回结果字典
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束事件监听器的添加
});
# 结束函数定义

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串
}

// Main program
async function main()
{
    print(tab(33) + "CHEMIST\n");  // 打印带有缩进的字符串 "CHEMIST"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有缩进的字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("THE FICTITIOUS CHECMICAL KRYPTOCYANIC ACID CAN ONLY BE\n");  // 打印字符串
    print("DILUTED BY THE RATIO OF 7 PARTS WATER TO 3 PARTS ACID.\n");  // 打印字符串
    print("IF ANY OTHER RATIO IS ATTEMPTED, THE ACID BECOMES UNSTABLE\n");  // 打印字符串
    print("AND SOON EXPLODES.  GIVEN THE AMOUNT OF ACID, YOU MUST\n");  // 打印字符串
    print("DECIDE WHO MUCH WATER TO ADD FOR DILUTION.  IF YOU MISS\n");  // 打印字符串
    print("YOU FACE THE CONSEQUENCES.\n");  // 打印字符串
    t = 0;  // 初始化变量 t 为 0
    while (1) {  // 进入无限循环
        a = Math.floor(Math.random() * 50);  // 生成一个 0 到 49 之间的随机整数并赋值给变量 a
        w = 7 * a / 3;  // 根据公式计算出 w 的值
        print(a + " LITERS OF KRYPTOCYANIC ACID.  HOW MUCH WATER");  // 打印出 a 的值和一段提示信息
        r = parseFloat(await input());  // 从用户输入中获取一个浮点数并赋值给变量 r
        d = Math.abs(w - r);  // 计算 w 与 r 之差的绝对值并赋值给变量 d
        if (d > w / 20) {  // 如果 d 大于 w 的 1/20
            print(" SIZZLE!  YOU HAVE JUST BEEN DESALINATED INTO A BLOB\n");  // 打印一段提示信息
            print(" OF QUIVERING PROTOPLASM!\n");  // 打印另一段提示信息
            t++;  // t 自增 1
            if (t == 9)  // 如果 t 等于 9
                break;  // 跳出循环
            print(" HOWEVER, YOU MAY TRY AGAIN WITH ANOTHER LIFE.\n");  // 打印一段提示信息
        } else {
            print(" GOOD JOB! YOU MAY BREATHE NOW, BUT DON'T INHALE THE FUMES!\n");  // 打印一段提示信息
            print("\n");  // 打印一个空行
        }
    }
    print(" YOUR 9 LIVES ARE USED, BUT YOU WILL BE LONG REMEMBERED FOR\n");  // 打印一段提示信息
    print(" YOUR CONTRIBUTIONS TO THE FIELD OF COMIC BOOK CHEMISTRY.\n");  // 打印另一段提示信息
}
# 调用名为main的函数，但是在给定的代码中并没有定义这个函数，所以这行代码会导致错误。
```