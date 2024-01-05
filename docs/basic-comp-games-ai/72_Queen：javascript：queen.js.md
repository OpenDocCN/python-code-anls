# `d:/src/tocomm/basic-computer-games\72_Queen\javascript\queen.js`

```
# QUEEN
# 
# 由 Oscar Toledo G. (nanochess) 从 BASIC 转换为 Javascript
#

# 定义一个打印函数，将字符串添加到输出元素中
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str));

# 定义一个输入函数，返回一个 Promise 对象
def input():
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       # 创建一个输入元素
                       input_element = document.createElement("INPUT");

                       # 打印提示符
                       print("? ");
                       # 设置输入元素的类型为文本
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
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串

}

var sa = [,81,  71,  61,  51,  41,  31,  21,  11,  // 定义一个包含数字的数组
           92,  82,  72,  62,  52,  42,  32,  22,
          103,  93,  83,  73,  63,  53,  43,  33,
          114, 104,  94,  84,  74,  64,  54,  44,
          125, 115, 105,  95,  85,  75,  65,  55,
          136, 126, 116, 106,  96,  86,  76,  66,
          147, 137, 127, 117, 107,  97,  87,  77,
          158, 148, 138, 128, 118, 108,  98,  88];

var m;  // 声明变量 m
var m1;  // 声明变量 m1
var u;  // 声明变量 u
var t;  // 声明变量 t
var u1;  // 声明变量 u1
var t1;  // 声明变量 t1
# 显示游戏说明的函数
def show_instructions():
    # 打印游戏说明
    print("WE ARE GOING TO PLAY A GAME BASED ON ONE OF THE CHESS\n")
    print("MOVES.  OUR QUEEN WILL BE ABLE TO MOVE ONLY TO THE LEFT,\n")
    print("DOWN, OR DIAGONALLY DOWN AND TO THE LEFT.\n")
    print("\n")
    print("THE OBJECT OF THE GAME IS TO PLACE THE QUEEN IN THE LOWER\n")
    print("LEFT HAND SQUARE BY ALTERNATING MOVES BETWEEN YOU AND THE\n")
    print("COMPUTER.  THE FIRST ONE TO PLACE THE QUEEN THERE WINS.\n")
    print("\n")
    print("YOU GO FIRST AND PLACE THE QUEEN IN ANY ONE OF THE SQUARES\n")
    print("ON THE TOP ROW OR RIGHT HAND COLUMN.\n")
    print("THAT WILL BE YOUR FIRST MOVE.\n")
    print("WE ALTERNATE MOVES.\n")
    print("YOU MAY FORFEIT BY TYPING '0' AS YOUR MOVE.\n")
    print("BE SURE TO PRESS THE RETURN KEY AFTER EACH RESPONSE.\n")
    print("\n")
    print("\n")
```
在这个示例中，我们为给定的代码添加了注释，解释了每个语句的作用。
# 定义名为show_map的函数，用于显示地图
function show_map()
{
    # 打印换行符
    print("\n");
    # 循环遍历地图的行
    for (var a = 0; a <= 7; a++) {
        # 循环遍历地图的列
        for (var b = 1; b <= 8; b++) {
            # 计算当前位置在数组中的索引
            i = 8 * a + b;
            # 打印当前位置的值
            print(" " + sa[i] + " ");
        }
        # 打印三个换行符
        print("\n");
        print("\n");
        print("\n");
    }
    # 打印换行符
    print("\n");
}

# 定义名为test_move的函数，用于测试移动
function test_move()
{
    # 计算移动的值
    m = 10 * t + u;
    # 如果移动的值符合特定条件，则返回true
    if (m == 158 || m == 127 || m == 126 || m == 75 || m == 73)
        return true;
    return false;  // 返回 false，表示函数执行失败

function random_move()
{
    // 随机移动
    z = Math.random();  // 生成一个 0 到 1 之间的随机数
    if (z > 0.6) {  // 如果随机数大于 0.6
        u = u1 + 1;  // u 值加 1
        t = t1 + 1;  // t 值加 1
    } else if (z > 0.3) {  // 如果随机数大于 0.3
        u = u1 + 1;  // u 值加 1
        t = t1 + 2;  // t 值加 2
    } else {  // 其他情况
        u = u1;  // u 值不变
        t = t1 + 1;  // t 值加 1
    }
    m = 10 * t + u;  // 计算 m 的值
}
# 定义一个名为 computer_move 的函数
def computer_move():
    # 如果 m1 的值等于 41、44、73、75、126 或 127 中的任意一个，就执行 random_move 函数并返回
    if (m1 == 41 or m1 == 44 or m1 == 73 or m1 == 75 or m1 == 126 or m1 == 127):
        random_move()
        return
    # 从 7 循环到 1
    for (k = 7; k >= 1; k--):
        # 将 u 的值赋为 u1
        u = u1
        # 将 t 的值赋为 t1 加上 k
        t = t1 + k
        # 如果 test_move 函数返回 True，则返回
        if (test_move()):
            return
        # 将 u 的值加上 k
        u += k
        # 如果 test_move 函数返回 True，则返回
        if (test_move()):
            return
        # 将 t 的值加上 k
        t += k
        # 如果 test_move 函数返回 True，则返回
        if (test_move()):
            return
    # 执行 random_move 函数
    random_move()
// 主程序
async function main()
{
    // 打印 QUEEN
    print(tab(33) + "QUEEN\n");
    // 打印 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印三个空行
    print("\n");
    print("\n");
    print("\n");

    // 进入循环，询问是否需要说明书
    while (1) {
        print("DO YOU WANT INSTRUCTIONS");
        // 等待用户输入
        str = await input();
        // 如果用户输入为 "YES" 或 "NO"，跳出循环
        if (str == "YES" || str == "NO")
            break;
        // 如果用户输入不是 "YES" 或 "NO"，提示用户重新输入
        print("PLEASE ANSWER 'YES' OR 'NO'.\n");
    }
    // 如果用户选择了 "YES"，显示说明书
    if (str == "YES")
        show_instructions();
    // 进入另一个循环
    while (1) {
        // 调用show_map函数，显示地图
        show_map();
        // 进入无限循环，直到条件被打破
        while (1) {
            // 打印提示信息，询问玩家想要从哪里开始
            print("WHERE WOULD YOU LIKE TO START");
            // 将输入的字符串转换为整数
            m1 = parseInt(await input());
            // 如果玩家输入0，表示放弃游戏，打印相应信息并跳出循环
            if (m1 == 0) {
                print("\n");
                print("IT LOOKS LIKE I HAVE WON BY FORFEIT.\n");
                print("\n");
                break;
            }
            // 计算输入数的十位和个位
            t1 = Math.floor(m1 / 10);
            u1 = m1 - 10 * t1;
            // 如果个位等于1或者个位等于十位，跳出循环
            if (u1 == 1 || u1 == t1)
                break;
            // 如果不符合上述条件，打印提示信息并继续循环
            print("PLEASE READ THE DIRECTIONS AGAIN.\n");
            print("YOU HAVE BEGUN ILLEGALLY.\n");
            print("\n");
        }
        // 进入循环，条件为m1的值
        while (m1) {
            // 如果m1的值等于158
            if (m1 == 158) {
                # 打印空行
                print("\n");
                # 打印祝贺消息
                print("C O N G R A T U L A T I O N S . . .\n");
                # 打印空行
                print("\n");
                # 打印胜利消息
                print("YOU HAVE WON--VERY WELL PLAYED.\n");
                # 打印对手胜利消息
                print("IT LOOKS LIKE I HAVE MET MY MATCH.\n");
                # 打印感谢消息
                print("THANKS FOR PLAYING--I CAN'T WIN ALL THE TIME.\n");
                # 打印空行
                print("\n");
                # 跳出循环
                break;
            }
            # 让计算机进行移动
            computer_move();
            # 打印计算机移动的消息
            print("COMPUTER MOVES TO SQUARE " + m + "\n");
            # 如果计算机移动到了特定的方块
            if (m == 158) {
                # 打印失败消息
                print("\n");
                print("NICE TRY, BUT IT LOOKS LIKE I HAVE WON.\n");
                print("THANKS FOR PLAYING.\n");
                print("\n");
                # 跳出循环
                break;
            }
            # 打印提示消息
            print("WHAT IS YOUR MOVE");
            # 进入循环，直到玩家输入有效的移动
            while (1) {
                # 从输入中获取整数并转换为整型
                m1 = parseInt(await input());
                # 如果输入的整数为0，则跳出循环
                if (m1 == 0)
                    break;
                # 计算输入整数的十位数
                t1 = Math.floor(m1 / 10);
                # 计算输入整数的个位数
                u1 = m1 - 10 * t1;
                # 计算个位数的差值
                p = u1 - u;
                # 计算十位数的差值
                l = t1 - t;
                # 如果输入整数小于等于上一次输入的整数，或者个位数差值为0且十位数差值小于等于0，或者个位数差值不等于十位数差值且不等于两倍的个位数差值
                if (m1 <= m || p == 0 && l <= 0 || p != 0 && l != p && l != 2 * p) {
                    # 打印提示信息
                    print("\n");
                    print("Y O U   C H E A T . . .  TRY AGAIN");
                    # 继续下一次循环
                    continue;
                }
                # 跳出循环
                break;
            }
            # 如果输入整数为0
            if (m1 == 0) {
                # 打印提示信息
                print("\n");
                print("IT LOOKS LIKE I HAVE WON BY FORFEIT.\n");
                print("\n");
                # 跳出循环
                break;
            }
        }
        while (1) {  # 进入无限循环
            print("ANYONE ELSE CARE TO TRY");  # 打印提示信息
            str = await input();  # 等待用户输入，并将输入内容赋值给变量str
            print("\n");  # 打印换行
            if (str == "YES" || str == "NO")  # 如果输入内容为"YES"或者"NO"
                break;  # 退出循环
            print("PLEASE ANSWER 'YES' OR 'NO'.\n");  # 如果输入内容不是"YES"或者"NO"，则打印提示信息
        }
        if (str != "YES")  # 如果输入内容不是"YES"
            break;  # 退出循环
    }
    print("\n");  # 打印换行
    print("OK --- THANKS AGAIN.\n");  # 打印感谢信息
}

main();  # 调用主函数
```