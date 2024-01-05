# `d:/src/tocomm/basic-computer-games\54_Letter\javascript\letter.js`

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
# 结束键盘按下事件监听器的添加
});
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环执行以下操作
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串
}

// Main program
async function main()
{
    print(tab(33) + "LETTER\n");  // 在控制台打印带有33个空格的"LETTER"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在控制台打印带有15个空格的"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 在控制台打印空行
    print("\n");  // 在控制台打印空行
    print("\n");  // 在控制台打印空行
    print("LETTER GUESSING GAME\n");  // 在控制台打印"LETTER GUESSING GAME"
    print("\n");  // 在控制台打印空行
    print("I'LL THINK OF A LETTER OF THE ALPHABET, A TO Z.\n");  // 在控制台打印"I'LL THINK OF A LETTER OF THE ALPHABET, A TO Z."
    print("TRY TO GUESS MY LETTER AND I'LL GIVE YOU CLUES\n");  // 在控制台打印"TRY TO GUESS MY LETTER AND I'LL GIVE YOU CLUES"
    print("AS TO HOW CLOSE YOU'RE GETTING TO MY LETTER.\n");  // 在控制台打印"AS TO HOW CLOSE YOU'RE GETTING TO MY LETTER."
    while (1) {  // 进入无限循环
        l = 65 + Math.floor(26 * Math.random());  // 生成一个介于65和90之间的随机整数，并赋值给变量l
        g = 0;  // 将变量g的值设为0
        print("\n");  # 打印空行
        print("O.K., I HAVE A LETTER.  START GUESSING.\n");  # 打印提示信息
        while (1) {  # 进入无限循环
            print("\n");  # 打印空行
            print("WHAT IS YOUR GUESS");  # 打印提示信息
            g++;  # 猜测次数加一
            str = await input();  # 等待用户输入并将输入的字符串赋值给变量str
            a = str.charCodeAt(0);  # 获取输入字符串的第一个字符的 ASCII 值
            print("\n");  # 打印空行
            if (a == l)  # 如果输入的字符的 ASCII 值等于某个值l
                break;  # 退出循环
            if (a < l) {  # 如果输入的字符的 ASCII 值小于某个值l
                print("TOO LOW.  TRY A HIGHER LETTER.\n");  # 打印提示信息
            } else {  # 如果输入的字符的 ASCII 值大于某个值l
                print("TOO HIGH.  TRY A LOWER LETTER.\n");  # 打印提示信息
            }
        }
        print("\n");  # 打印空行
        print("YOU GOT IT IN " + g + " GUESSES!!\n");  # 打印猜测次数
        if (g > 5) {  # 如果猜测次数大于5
            print("BUT IT SHOULDN'T TAKE MORE THAN 5 GUESSES!\n");  # 打印提示信息
        } else {  # 否则
            print("GOOD JOB !!!!!\n");  # 打印祝贺信息
        }
        print("\n");  # 打印空行
        print("LET'S PLAY AGAIN.....");  # 打印提示信息
    }
}

main();  # 调用主函数
```