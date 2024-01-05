# `08_Batnum\javascript\batnum.js`

```
// BATNUM
// 
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    // 在页面上输出文本
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    // 返回一个 Promise 对象
    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型为文本
                       input_element.setAttribute("type", "text");
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
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串
}

// Main program
async function main()
{
    print(tab(33) + "BATNUM\n");  // 在指定位置打印字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在指定位置打印字符串
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("THIS PROGRAM IS A 'BATTLE OF NUMBERS' GAME, WHERE THE\n");  // 打印字符串
    print("COMPUTER IS YOUR OPPONENT.\n");  // 打印字符串
    print("\n");  // 打印空行
    print("THE GAME STARTS WITH AN ASSUMED PILE OF OBJECTS. YOU\n");  // 打印字符串
    print("AND YOUR OPPONENT ALTERNATELY REMOVE OBJECTS FROM THE PILE.\n");  // 打印字符串
    print("WINNING IS DEFINED IN ADVANCE AS TAKING THE LAST OBJECT OR\n");  // 打印字符串
    print("NOT. YOU CAN ALSO SPECIFY SOME OTHER BEGINNING CONDITIONS.\n");  // 打印字符串
    print("DON'T USE ZERO, HOWEVER, IN PLAYING THE GAME.\n");  // 打印字符串
    # 打印提示信息，告诉用户输入负数以停止游戏
    print("ENTER A NEGATIVE NUMBER FOR NEW PILE SIZE TO STOP PLAYING.\n");
    # 打印空行
    print("\n");
    # 初始化变量 first_time 为 1
    first_time = 1;
    # 进入无限循环
    while (1) {
        # 进入内部无限循环
        while (1) {
            # 如果是第一次循环，将 first_time 设置为 0，否则打印 10 个空行
            if (first_time == 1) {
                first_time = 0;
            } else {
                for (i = 1; i <= 10; i++)
                    print("\n");
            }
            # 打印提示信息，要求用户输入堆的大小
            print("ENTER PILE SIZE");
            # 从用户输入中获取整数值赋给变量 n
            n = parseInt(await input());
            # 如果输入的值大于等于 1，则跳出循环
            if (n >= 1)
                break;
        }
        # 进入内部无限循环
        while (1) {
            # 打印提示信息，要求用户输入赢的选项
            print("ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST: ");
            # 从用户输入中获取整数值赋给变量 m
            m = parseInt(await input());
            # 如果输入的值为 1 或 2，则跳出循环
            if (m == 1 || m == 2)
        break;  # 结束当前循环，跳出循环体
        }
        while (1) {  # 进入一个无限循环
            print("ENTER MIN AND MAX ");  # 打印提示信息
            str = await input();  # 获取用户输入的字符串
            a = parseInt(str);  # 将用户输入的字符串转换为整数并赋值给变量a
            b = parseInt(str.substr(str.indexOf(",") + 1));  # 从用户输入的字符串中截取逗号后面的部分并转换为整数赋值给变量b
            if (a <= b && a >= 1)  # 如果a小于等于b且a大于等于1
                break;  # 结束当前循环，跳出循环体
        }
        while (1) {  # 进入一个无限循环
            print("ENTER START OPTION - 1 COMPUTER FIRST, 2 YOU FIRST ");  # 打印提示信息
            s = parseInt(await input());  # 获取用户输入的字符串并转换为整数赋值给变量s
            print("\n");  # 打印换行
            print("\n");  # 打印换行
            if (s == 1 || s == 2)  # 如果s等于1或者s等于2
                break;  # 结束当前循环，跳出循环体
        }
        w = 0;  # 将变量w赋值为0
        c = a + b;  # 将a和b的和赋值给变量c
        while (1) {
            // 进入循环，开始游戏
            if (s == 1) {
                // 如果是计算机的回合
                q = n;
                // 将当前剩余的物品数量赋值给 q
                if (m != 1)
                    // 如果 m 不等于 1
                    q--;
                    // 则 q 减一
                if (m != 1 && n <= a) {
                    // 如果 m 不等于 1 且 n 小于等于 a
                    w = 1;
                    // 将 w 设为 1
                    print("COMPUTER TAKES " + n + " AND LOSES.\n");
                    // 打印计算机拿走 n 个物品并且输了的消息
                } else if (m == 1 && n <= b) {
                    // 如果 m 等于 1 且 n 小于等于 b
                    w = 1;
                    // 将 w 设为 1
                    print("COMPUTER TAKES " + n + " AND WINS.\n");
                    // 打印计算机拿走 n 个物品并且赢了的消息
                } else {
                    // 否则
                    p = q - c * Math.floor(q / c);
                    // 计算 p 的值
                    if (p < a)
                        // 如果 p 小于 a
                        p = a;
                        // 则将 p 设为 a
                    if (p > b)
                        // 如果 p 大于 b
                        p = b;
                        // 则将 p 设为 b
                    n -= p;
                    // 从 n 中减去 p
                    print("COMPUTER TAKES " + p + " AND LEAVES " + n + "\n");
                    // 打印计算机拿走 p 个物品并且剩下 n 个物品的消息
                    w = 0;  # 初始化变量 w 为 0
                }
                s = 2;  # 将变量 s 赋值为 2
            }
            if (w)  # 如果变量 w 的值为真
                break;  # 跳出循环
            if (s == 2) {  # 如果变量 s 的值为 2
                while (1) {  # 进入无限循环
                    print("\n");  # 打印换行符
                    print("YOUR MOVE ");  # 打印提示信息
                    p = parseInt(await input());  # 从输入中获取整数值并赋给变量 p
                    if (p == 0) {  # 如果 p 的值为 0
                        print("I TOLD YOU NOT TO USE ZERO! COMPUTER WINS BY FORFEIT.\n");  # 打印提示信息
                        w = 1;  # 将变量 w 赋值为 1
                        break;  # 跳出循环
                    } else if (p >= a && p <= b && n - p >= 0) {  # 如果 p 大于等于 a 且小于等于 b 且 n 减去 p 大于等于 0
                        break;  # 跳出循环
                    }
                }
                if (p != 0) {  # 如果 p 不等于 0
                    n -= p;  # 从n中减去p的值
                    if (n == 0) {  # 如果n的值等于0
                        if (m != 1) {  # 如果m的值不等于1
                            print("TOUGH LUCK, YOU LOSE.\n");  # 打印“TOUGH LUCK, YOU LOSE.”
                        } else {
                            print("CONGRATULATIONS, YOU WIN.\n");  # 否则打印“CONGRATULATIONS, YOU WIN.”
                        }
                        w = 1;  # 将w的值设为1
                    } else {
                        w = 0;  # 否则将w的值设为0
                    }
                }
                s = 1;  # 将s的值设为1
            }
            if (w)  # 如果w的值为真
                break;  # 跳出循环
        }
    }
}
# 调用名为main的函数，但是在给定的代码中并没有定义main函数，所以这行代码会导致错误。
```