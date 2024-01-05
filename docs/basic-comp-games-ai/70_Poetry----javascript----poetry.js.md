# `70_Poetry\javascript\poetry.js`

```
// 定义一个名为 print 的函数，用于向页面输出内容
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个名为 input 的函数，用于获取用户输入
function input() {
    var input_element;
    var input_str;

    // 返回一个 Promise 对象，用于处理异步操作
    return new Promise(function (resolve) {
        // 创建一个 INPUT 元素
        input_element = document.createElement("INPUT");

        // 在页面上输出提示符
        print("? ");

        // 设置 INPUT 元素的类型为文本
        input_element.setAttribute("type", "text");
```
在这个示例中，我们为给定的 JavaScript 代码添加了注释，解释了每个语句的作用。这样做可以帮助其他程序员更容易地理解代码的功能和逻辑。
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时，获取输入的字符串，移除输入元素，打印输入的字符串，换行，解析输入的字符串
input_element.addEventListener("keydown", function (event) {
    if (event.keyCode == 13) {
        input_str = input_element.value;
        document.getElementById("output").removeChild(input_element);
        print(input_str);
        print("\n");
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器
});
}

# 定义一个函数 tab，参数为 space
function tab(space)
{
    # 初始化一个空字符串 str
    var str = "";
    # 当 space 大于 0 时，循环执行以下操作
    while (space-- > 0)
        str += " ";  // 将空格字符添加到字符串末尾
    return str;  // 返回处理后的字符串
}

// Main program
async function main()
{
    print(tab(30) + "POETRY\n");  // 在指定位置打印字符串"POETRY"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在指定位置打印字符串"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印换行
    print("\n");  // 打印换行
    print("\n");  // 打印换行

    times = 0;  // 初始化变量times为0

    i = 1;  // 初始化变量i为1
    j = 1;  // 初始化变量j为1
    k = 0;  // 初始化变量k为0
    u = 0;  // 初始化变量u为0
    while (1) {  // 进入无限循环
        if (j == 1):  # 如果 j 等于 1
            switch (i):  # 根据 i 的值进行判断
                case 1:  # 如果 i 等于 1
                    print("MIDNIGHT DREARY");  # 打印"MIDNIGHT DREARY"
                    break;  # 跳出 switch 语句
                case 2:  # 如果 i 等于 2
                    print("FIERY EYES");  # 打印"FIERY EYES"
                    break;  # 跳出 switch 语句
                case 3:  # 如果 i 等于 3
                    print("BIRD OF FIEND");  # 打印"BIRD OF FIEND"
                    break;  # 跳出 switch 语句
                case 4:  # 如果 i 等于 4
                    print("THING OF EVIL");  # 打印"THING OF EVIL"
                    break;  # 跳出 switch 语句
                case 5:  # 如果 i 等于 5
                    print("PROPHET");  # 打印"PROPHET"
                    break;  # 跳出 switch 语句
        else:  # 如果 j 不等于 1
            if (j == 2):  # 如果 j 等于 2
                switch (i):  # 根据 i 的值进行判断
                case 1:  # 如果 j 的值为 1
                    print("BEGUILING ME");  # 打印 "BEGUILING ME"
                    u = 2;  # 将变量 u 的值设为 2
                    break;  # 跳出 switch 语句
                case 2:  # 如果 j 的值为 2
                    print("THRILLED ME");  # 打印 "THRILLED ME"
                    break;  # 跳出 switch 语句
                case 3:  # 如果 j 的值为 3
                    print("STILL SITTING....");  # 打印 "STILL SITTING...."
                    u = 0;  # 将变量 u 的值设为 0
                    break;  # 跳出 switch 语句
                case 4:  # 如果 j 的值为 4
                    print("NEVER FLITTING");  # 打印 "NEVER FLITTING"
                    u = 2;  # 将变量 u 的值设为 2
                    break;  # 跳出 switch 语句
                case 5:  # 如果 j 的值为 5
                    print("BURNED");  # 打印 "BURNED"
                    break;  # 跳出 switch 语句
            }
        } else if (j == 3) {  # 如果 j 的值为 3
            # 开始一个 switch 语句，根据变量 i 的值进行不同的操作
            switch (i) {
                # 当 i 的值为 1 时，打印 "AND MY SOUL"，然后跳出 switch 语句
                case 1:
                    print("AND MY SOUL");
                    break;
                # 当 i 的值为 2 时，打印 "DARKNESS THERE"，然后跳出 switch 语句
                case 2:
                    print("DARKNESS THERE");
                    break;
                # 当 i 的值为 3 时，打印 "SHALL BE LIFTED"，然后跳出 switch 语句
                case 3:
                    print("SHALL BE LIFTED");
                    break;
                # 当 i 的值为 4 时，打印 "QUOTH THE RAVEN"，然后跳出 switch 语句
                case 4:
                    print("QUOTH THE RAVEN");
                    break;
                # 当 i 的值为 5 时，如果 u 的值为 0，则跳出 switch 语句；否则打印 "SIGN OF PARTING"，然后跳出 switch 语句
                case 5:
                    if (u == 0)
                        break;
                    print("SIGN OF PARTING");
                    break;
            }
        # 如果上面的 switch 语句结束后，j 的值为 4，则执行以下代码
        } else if (j == 4) {
# 开始一个 switch 语句，根据变量 i 的值进行不同的操作
            switch (i) {
                # 当 i 的值为 1 时，打印 "NOTHING MORE"，然后跳出 switch 语句
                case 1:
                    print("NOTHING MORE");
                    break;
                # 当 i 的值为 2 时，打印 "YET AGAIN"，然后跳出 switch 语句
                case 2:
                    print("YET AGAIN");
                    break;
                # 当 i 的值为 3 时，打印 "SLOWLY CREEPING"，然后跳出 switch 语句
                case 3:
                    print("SLOWLY CREEPING");
                    break;
                # 当 i 的值为 4 时，打印 "...EVERMORE"，然后跳出 switch 语句
                case 4:
                    print("...EVERMORE");
                    break;
                # 当 i 的值为 5 时，打印 "NEVERMORE"，然后跳出 switch 语句
                case 5:
                    print("NEVERMORE");
                    break;
            }
        }
        # 如果 u 的值不为 0 且生成的随机数小于等于 0.19
        if (u != 0 && Math.random() <= 0.19) {
            # 打印逗号
            print(",");
            u = 2;  # 初始化变量u为2
        }
        if (Math.random() <= 0.65) {  # 如果随机数小于等于0.65
            print(" ");  # 打印空格
            u++;  # 变量u加1
        } else {
            print("\n");  # 打印换行符
            u = 0;  # 变量u赋值为0
        }
        while (1) {  # 进入无限循环
            i = Math.floor(Math.floor(10 * Math.random()) / 2) + 1;  # 生成一个1到5之间的随机整数赋值给变量i
            j++;  # 变量j加1
            k++;  # 变量k加1
            if (u == 0 && j % 2 == 0)  # 如果u等于0且j是偶数
                print("     ");  # 打印5个空格
            if (j != 5)  # 如果j不等于5
                break;  # 跳出循环
            j = 0;  # 变量j赋值为0
            print("\n");  # 打印换行符
            if (k <= 20)  # 如果k小于等于20
                continue;  # 继续执行下一次循环
            print("\n");  # 打印换行符
            u = 0;  # 将变量 u 的值设为 0
            k = 0;  # 将变量 k 的值设为 0
            j = 2;  # 将变量 j 的值设为 2
            break;  # 跳出循环
        }
        if (u == 0 && k == 0 && j == 2 && ++times == 10)  # 如果 u、k、j 的值都为 0，且 times 自增后等于 10
            break;  # 跳出循环
    }
}

main();  # 调用 main 函数
```