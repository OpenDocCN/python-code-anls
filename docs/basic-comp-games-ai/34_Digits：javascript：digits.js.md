# `34_Digits\javascript\digits.js`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，创建字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
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
    return str;  // 返回处理后的字符串
}

// Main program
async function main()
{
    print(tab(33) + "DIGITS\n");  // 打印带有33个空格的字符串和"DIGITS"，用于格式化输出
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有15个空格的字符串和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，用于格式化输出
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("THIS IS A GAME OF GUESSING.\n");  // 打印游戏说明
    print("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0'");  // 提示用户输入指令
    e = parseInt(await input());  // 将用户输入的值转换为整数
    if (e != 0) {  // 如果用户输入的值不等于0
        print("\n");  // 打印空行
        print("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN\n");  // 提示用户做笔记
        print("THE DIGITS '0', '1', OR '2' THIRTY TIMES AT RANDOM.\n");  // 提示用户写下数字0、1或2各30次
        print("ARRANGE THEM IN THREE LINES OF TEN DIGITS EACH.\n");  // 提示用户将它们排成三行，每行十个数字
        # 打印提示信息
        print("I WILL ASK FOR THEN TEN AT A TIME.\n");
        print("I WILL ALWAYS GUESS THEM FIRST AND THEN LOOK AT YOUR\n");
        print("NEXT NUMBER TO SEE IF I WAS RIGHT. BY PURE LUCK,\n");
        print("I OUGHT TO BE RIGHT TEN TIMES. BUT I HOPE TO DO BETTER\n");
        print("THAN THAT *****\n");
        print("\n");
        print("\n");
    }
    # 初始化变量
    a = 0;
    b = 1;
    c = 3;
    m = [];
    k = [];
    l = [];
    n = [];
    # 进入无限循环
    while (1) {
        # 循环初始化 m 数组
        for (i = 0; i <= 26; i++) {
            m[i] = [];
            # 循环初始化 m[i] 数组
            for (j = 0; j <= 2; j++) {
                m[i][j] = 1;
        }
        }
        for (i = 0; i <= 2; i++) {  # 循环i从0到2
            k[i] = [];  # 初始化k[i]为一个空数组
            for (j = 0; j <= 2; j++) {  # 循环j从0到2
                k[i][j] = 9;  # 将k[i][j]的值设为9
            }
        }
        for (i = 0; i <= 8; i++) {  # 循环i从0到8
            l[i] = [];  # 初始化l[i]为一个空数组
            for (j = 0; j <= 2; j++) {  # 循环j从0到2
                l[i][j] = 3;  # 将l[i][j]的值设为3
            }
        }
        l[0][0] = 2;  # 将l[0][0]的值设为2
        l[4][1] = 2;  # 将l[4][1]的值设为2
        l[8][2] = 2;  # 将l[8][2]的值设为2
        z = 26;  # 将z的值设为26
        z1 = 8;  # 将z1的值设为8
        z2 = 2;  # 将z2的值设为2
        x = 0;  // 初始化变量 x 为 0
        for (t = 1; t <= 3; t++) {  // 循环执行三次
            while (1) {  // 无限循环，直到条件满足跳出循环
                print("\n");  // 打印换行
                print("TEN NUMBERS, PLEASE");  // 打印提示信息
                str = await input();  // 等待用户输入并将输入值赋给变量 str
                for (i = 1; i <= 10; i++) {  // 循环执行十次
                    n[i] = parseInt(str);  // 将输入值转换为整数并赋给数组 n 的第 i 个元素
                    j = str.indexOf(",");  // 获取逗号的索引位置
                    if (j >= 0) {  // 如果找到逗号
                        str = str.substr(j + 1);  // 截取逗号后的字符串并赋给 str
                    }
                    if (n[i] < 0 || n[i] > 2)  // 如果输入值小于 0 或大于 2
                        break;  // 跳出循环
                }
                if (i <= 10) {  // 如果循环未执行完十次
                    print("ONLY USE THE DIGITS '0', '1', OR '2'.\n");  // 打印提示信息
                    print("LET'S TRY AGAIN.\n");  // 打印提示信息
                } else {  // 如果循环执行完十次
                    break;  // 跳出循环
                }  # 结束第一个 for 循环

            }  # 结束第二个 for 循环

            print("\n");  # 打印空行
            print("MY GUESS\tYOUR NO.\tRESULT\tNO. RIGHT\n");  # 打印表头
            print("\n");  # 打印空行

            for (u = 1; u <= 10; u++) {  # 开始第三个 for 循环，循环次数为 10
                n2 = n[u];  # 将 n[u] 赋值给 n2
                s = 0;  # 初始化变量 s 为 0
                for (j = 0; j <= 2; j++) {  # 开始第四个 for 循环，循环次数为 3
                    s1 = a * k[z2][j] + b * l[z1][j] + c * m[z][j];  # 计算 s1 的值
                    if (s > s1)  # 如果 s 大于 s1
                        continue;  # 继续下一次循环
                    if (s < s1 || Math.random() >= 0.5) {  # 如果 s 小于 s1 或者随机数大于等于 0.5
                        s = s1;  # 将 s1 赋值给 s
                        g = j;  # 将 j 赋值给 g
                    }
                }
                print("  " + g + "\t\t   " + n[u] + "\t\t");  # 打印 g 和 n[u] 的值
                if (g == n[u]) {  # 如果 g 等于 n[u]
                    x++;  # 变量 x 自增 1
                    print(" RIGHT\t " + x + "\n");  # 打印正确的信息和数字
                    m[z][n2]++;  # 将m[z][n2]的值加1
                    l[z1][n2]++;  # 将l[z1][n2]的值加1
                    k[z2][n2]++;  # 将k[z2][n2]的值加1
                    z = z % 9;  # z取余9
                    z = 3 * z + n[u];  # z重新赋值为3 * z + n[u]
                } else {
                    print(" WRONG\t " + x + "\n");  # 打印错误的信息和数字
                }
                z1 = z % 9;  # z1取余9
                z2 = n[u];  # z2赋值为n[u]
            }
        }
        print("\n");  # 打印空行
        if (x > 10) {  # 如果x大于10
            print("I GUESSED MORE THAN 1/3 OF YOUR NUMBERS.\n");  # 打印猜对1/3以上数字的信息
            print("I WIN.\n");  # 打印获胜信息
        } else if (x == 10) {  # 如果x等于10
            print("I GUESSED EXACTLY 1/3 OF YOUR NUMBERS.\n");  # 打印猜对1/3数字的信息
            print("IT'S A TIE GAME.\n");  # 打印平局信息
        } else {
            # 如果猜对的数字少于你输入的数字的三分之一，打印出相应的消息
            print("I GUESSED LESS THAN 1/3 OF YOUR NUMBERS.\n");
            # 打印出恭喜消息
            print("YOU BEAT ME.  CONGRATULATIONS *****\n");
        }
        # 打印空行
        print("\n");
        # 打印出是否想再试一次的提示
        print("DO YOU WANT TO TRY AGAIN (1 FOR YES, 0 FOR NO)");
        # 从输入中获取一个整数并赋值给变量x
        x = parseInt(await input());
        # 如果x不等于1，跳出循环
        if (x != 1)
            break;
    }
    # 打印空行
    print("\n");
    # 打印出感谢消息
    print("THANKS FOR THE GAME.\n");
}

# 调用主函数
main();
```