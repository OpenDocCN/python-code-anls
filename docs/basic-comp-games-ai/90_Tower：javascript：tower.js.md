# `d:/src/tocomm/basic-computer-games\90_Tower\javascript\tower.js`

```
// 定义名为print的函数，用于向页面输出内容
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义名为input的函数，用于获取用户输入
function input()
{
    var input_element;
    var input_str;

    // 返回一个Promise对象，用于处理异步操作
    return new Promise(function (resolve) {
                       // 创建一个input元素
                       input_element = document.createElement("INPUT");

                       // 在页面上输出提示符
                       print("? ");

                       // 设置input元素的类型为文本
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
        str += " ";  // 在字符串末尾添加一个空格
    return str;  // 返回处理后的字符串
}

var ta = [];  // 定义一个空数组

// 打印子程序
function show_towers()
{
    var z;  // 定义变量z

    for (var k = 1; k <= 7; k++) {  // 循环7次
        z = 10;  // 将变量z赋值为10
        str = "";  // 定义一个空字符串
        for (var j = 1; j <= 3; j++) {  // 循环3次
            if (ta[k][j] != 0) {  // 如果数组ta的第k个元素的第j个元素不等于0
                while (str.length < z - Math.floor(ta[k][j] / 2))  // 当字符串长度小于z减去ta[k][j]除以2的向下取整
                    str += " ";  // 在字符串末尾添加一个空格
                for (v = 1; v <= ta[k][j]; v++)  // 循环ta[k][j]次
                    str += "*";  // 在字符串末尾添加一个星号
            } else {  # 如果条件不成立
                while (str.length < z)  # 当字符串长度小于 z 时
                    str += " ";  # 在字符串末尾添加空格
                str += "*";  # 在字符串末尾添加星号
            }
            z += 21;  # z 值增加 21
        }
        print(str + "\n");  # 打印字符串并换行
    }
}

// Main control section
async function main()
{
    print(tab(33) + "TOWERS\n");  # 打印 tab(33) 和 "TOWERS" 字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印 tab(15) 和 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY" 字符串
    print("\n");  # 打印换行
    print("\n");  # 打印换行
    print("\n");  # 打印换行
    while (1) {  # 进入无限循环
        // 打印空行
        print("\n");
        // 初始化
        e = 0;
        for (d = 1; d <= 7; d++) {
            // 创建一个二维数组，用于存储每个柱子上的圆盘
            ta[d] = [];
            for (n = 1; n <= 3; n++)
                // 初始化每个柱子上的圆盘数量为0
                ta[d][n] = 0;
        }
        // 打印汉诺塔谜题的提示信息
        print("TOWERS OF HANOI PUZZLE.\n");
        print("\n");
        print("YOU MUST TRANSFER THE DISKS FROM THE LEFT TO THE RIGHT\n");
        print("TOWER, ONE AT A TIME, NEVER PUTTING A LARGER DISK ON A\n");
        print("SMALLER DISK.\n");
        print("\n");
        // 进入循环，等待用户输入移动的圆盘数量
        while (1) {
            print("HOW MANY DISKS DO YOU WANT TO MOVE (7 IS MAX)");
            // 将用户输入的字符串转换为整数
            s = parseInt(await input());
            print("\n");
            m = 0;
            // 如果用户输入的圆盘数量在1到7之间
            if (s >= 1 && s <= 7)
                break;  # 结束当前循环
            e++;  # e 自增1
            if (e < 2) {  # 如果 e 小于2
                print("SORRY, BUT I CAN'T DO THAT JOB FOR YOU.\n");  # 打印提示信息
                continue;  # 继续下一次循环
            }
            print("ALL RIGHT, WISE GUY, IF YOU CAN'T PLAY THE GAME RIGHT, I'LL\n");  # 打印提示信息
            print("JUST TAKE MY PUZZLE AND GO HOME.  SO LONG.\n");  # 打印提示信息
            return;  # 返回
        }
        // Store disks from smallest to largest  # 注释：将盘片从小到大存储
        print("IN THIS PROGRAM, WE SHALL REFER TO DISKS BY NUMERICAL CODE.\n");  # 打印提示信息
        print("3 WILL REPRESENT THE SMALLEST DISK, 5 THE NEXT SIZE,\n");  # 打印提示信息
        print("7 THE NEXT, AND SO ON, UP TO 15.  IF YOU DO THE PUZZLE WITH\n");  # 打印提示信息
        print("2 DISKS, THEIR CODE NAMES WOULD BE 13 AND 15.  WITH 3 DISKS\n");  # 打印提示信息
        print("THE CODE NAMES WOULD BE 11, 13 AND 15, ETC.  THE NEEDLES\n");  # 打印提示信息
        print("ARE NUMBERED FROM LEFT TO RIGHT, 1 TO 3.  WE WILL\n");  # 打印提示信息
        print("START WITH THE DISKS ON NEEDLE 1, AND ATTEMPT TO MOVE THEM\n");  # 打印提示信息
        print("TO NEEDLE 3.\n");  # 打印提示信息
        print("\n");  # 打印空行
        # 打印"GOOD LUCK!\n"字符串
        print("GOOD LUCK!\n");
        # 打印换行符
        print("\n");
        # 初始化变量 y 为 7
        y = 7;
        # 初始化变量 d 为 15
        d = 15;
        # 循环，从 s 到 1，每次循环将 d 赋值给 ta[y][1]，然后 d 减 2，y 减 1
        for (x = s; x >= 1; x--) {
            ta[y][1] = d;
            d -= 2;
            y--;
        }
        # 调用 show_towers() 函数
        show_towers();
        # 进入无限循环
        while (1) {
            # 打印"WHICH DISK WOULD YOU LIKE TO MOVE"字符串
            print("WHICH DISK WOULD YOU LIKE TO MOVE");
            # 初始化变量 e 为 0
            e = 0;
            # 进入无限循环
            while (1) {
                # 将输入转换为整数并赋值给变量 d
                d = parseInt(await input());
                # 如果 d 为偶数或者小于 3 或者大于 15，则打印"ILLEGAL ENTRY... YOU MAY ONLY TYPE 3,5,7,9,11,13, OR 15.\n"字符串，e 加 1，如果 e 小于等于 1，则继续循环
                if (d % 2 == 0 || d < 3 || d > 15) {
                    print("ILLEGAL ENTRY... YOU MAY ONLY TYPE 3,5,7,9,11,13, OR 15.\n");
                    e++;
                    if (e <= 1)
                        continue;
                    # 打印警告信息
                    print("STOP WASTING MY TIME.  GO BOTHER SOMEONE ELSE.\n");
                    # 返回，结束函数执行
                    return;
                } else {
                    # 跳出循环
                    break;
                }
            }
            // 检查请求的磁盘是否在另一个磁盘下面
            for (r = 1; r <= 7; r++) {
                for (c = 1; c <= 3; c++) {
                    if (ta[r][c] == d)
                        break;
                }
                if (c <= 3)
                    break;
            }
            for (q = r; q >= 1; q--) {
                if (ta[q][c] != 0 && ta[q][c] < d)
                    break;
            }
            if (q >= 1) {
                # 打印提示信息，要求用户重新选择磁盘
                print("THAT DISK IS BELOW ANOTHER ONE.  MAKE ANOTHER CHOICE.\n");
                # 继续循环，等待用户重新选择
                continue;
            }
            # 重置错误计数器
            e = 0;
            # 进入循环，等待用户输入正确的磁盘位置
            while (1) {
                # 打印提示信息，要求用户放置针头所需的磁盘
                print("PLACE DISK ON WHICH NEEDLE");
                # 读取用户输入的磁盘位置
                n = parseInt(await input());
                # 如果用户输入的磁盘位置在1到3之间，则跳出循环
                if (n >= 1 && n <= 3)
                    break;
                # 错误计数器加一
                e++;
                # 如果错误计数器小于等于1，则打印提示信息，允许用户重新输入
                if (e <= 1) {
                    print("I'LL ASSUME YOU HIT THE WRONG KEY THI TIME.  BUT WATCH IT,\n");
                    print("I ONLY ALLOW ONE MISTAKE.\n");
                    # 继续循环，等待用户重新输入
                    continue;
                } else {
                    # 如果错误计数器大于1，则打印警告信息，并结束程序
                    print("I TRIED TO WARN YOU, BUT YOU WOULDN'T LISTEN.\n");
                    print("BYE BYE, BIG SHOT.\n");
                    return;
                }
            }
            // 检查请求的磁盘是否在另一个磁盘下方
            for (r = 1; r <= 7; r++) {
                if (ta[r][n] != 0)
                    break;
            }
            if (r <= 7) {
                // 检查要放置的磁盘是否放在较大的磁盘上
                if (d >= ta[r][n]) {
                    print("YOU CAN'T PLACE A LARGER DISK ON TOP OF A SMALLER ONE,\n");
                    print("IT MIGHT CRUSH IT!\n");
                    print("NOW THEN, ");
                    continue;
                }
            }
            // 移动重新定位的磁盘
            for (v = 1; v <= 7; v++) {
                for (w = 1; w <= 3; w++) {
                    if (ta[v][w] == d)
                        break;
                }
                // 如果 w 小于等于 3，则跳出循环
                if (w <= 3)
                    break;
            }
            // 定位在 needle n 上的空位
            for (u = 1; u <= 7; u++) {
                if (ta[u][n] != 0)
                    break;
            }
            ta[--u][n] = ta[v][w];
            ta[v][w] = 0;
            // 打印当前状态
            show_towers();
            // 检查是否完成
            m++;
            for (r = 1; r <= 7; r++) {
                for (c = 1; c <= 2; c++) {
                    if (ta[r][c] != 0)
                        break;
                }
                if (c <= 2)
                    break;  # 结束当前循环
            }
            if (r > 7)  # 如果 r 大于 7
                break;  # 结束当前循环
            if (m > 128) {  # 如果 m 大于 128
                print("SORRY, BUT I HAVE ORDERS TO STOP IF YOU MAKE MORE THAN\n");  # 打印提示信息
                print("128 MOVES.\n");  # 打印提示信息
                return;  # 结束函数执行
            }
        }
        if (m == Math.pow(2, s) - 1) {  # 如果 m 等于 2 的 s 次方减 1
            print("\n");  # 打印空行
            print("CONGRATULATIONS!!\n");  # 打印祝贺信息
            print("\n");  # 打印空行
        }
        print("YOU HAVE PERFORMED THE TASK IN " + m + " MOVES.\n");  # 打印完成任务所用的步数
        print("\n");  # 打印空行
        print("TRY AGAIN (YES OR NO)");  # 打印提示信息
        while (1) {  # 进入无限循环
            str = await input();  # 等待用户输入
            if (str == "YES" || str == "NO")  # 如果输入的字符串是"YES"或者"NO"，则跳出循环
                break;
            print("\n");  # 打印空行
            print("'YES' OR 'NO' PLEASE");  # 提示用户输入"YES"或者"NO"
        }
        if (str == "NO")  # 如果输入的字符串是"NO"，则跳出循环
            break;
    }
    print("\n");  # 打印空行
    print("THANKS FOR THE GAME!\n");  # 打印感谢信息
    print("\n");  # 打印空行
}

main();  # 调用主函数
```