# `d:/src/tocomm/basic-computer-games\50_Horserace\javascript\horserace.js`

```
# HORSERACE
# 
# 由 Oscar Toledo G. (nanochess) 将 BASIC 转换为 Javascript
#

# 定义一个打印函数，将字符串添加到输出元素中
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str))

# 定义一个输入函数，返回一个 Promise 对象
def input():
    var input_element
    var input_str

    return new Promise(function (resolve):
                       # 创建一个 INPUT 元素
                       input_element = document.createElement("INPUT")

                       # 打印提示符
                       print("? ")

                       # 设置输入元素的类型为文本
                       input_element.setAttribute("type", "text")
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 为输入元素添加按键按下事件监听器
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键（keyCode 为 13）
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
# 结束添加事件监听器的函数
});
# 结束 tab 函数的定义
}

# 定义 tab 函数，参数为 space
function tab(space)
{
    # 初始化空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

var sa = [];  # 创建空数组 sa
var ws = [];  # 创建空数组 ws
var da = [];  # 创建空数组 da
var qa = [];  # 创建空数组 qa
var pa = [];  # 创建空数组 pa
var ma = [];  # 创建空数组 ma
var ya = [];  # 创建空数组 ya
var vs = [];  # 创建空数组 vs

// Main program  # 主程序开始
async function main()  # 异步函数 main 开始
{
    print(tab(31) + "HORSERACE\n");  # 打印字符串 "HORSERACE"，并在前面添加 31 个空格
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面添加 15 个空格
    print("\n");  # 打印一个空行
    print("\n");  # 打印一个空行
    # 打印空行
    print("\n");
    # 打印欢迎词
    print("WELCOME TO SOUTH PORTLAND HIGH RACETRACK\n");
    print("                      ...OWNED BY LAURIE CHEVALIER\n");
    print("DO YOU WANT DIRECTIONS");
    # 获取用户输入
    str = await input();
    # 如果用户输入为"YES"，则打印相关信息
    if (str == "YES") {
        print("UP TO 10 MAY PLAY.  A TABLE OF ODDS WILL BE PRINTED.  YOU\n");
        print("MAY BET ANY + AMOUNT UNDER 100000 ON ONE HORSE.\n");
        print("DURING THE RACE, A HORSE WILL BE SHOWN BY ITS\n");
        print("NUMBER.  THE HORSES RACE DOWN THE PAPER!\n");
        print("\n");
    }
    # 打印提示信息
    print("HOW MANY WANT TO BET");
    # 获取用户输入并转换为整数
    c = parseInt(await input());
    # 打印提示信息
    print("WHEN ? APPEARS,TYPE NAME\n");
    # 循环获取赌注者的名字
    for (a = 1; a <= c; a++) {
        ws[a] = await input();
    }
    # 打印空行
    do {
        print("\n");
        # 打印表头
        print("HORSE\t\tNUMBERS\tODDS\n");
        # 打印空行
        print("\n");
        # 初始化 sa 数组
        for (i = 1; i <= 8; i++) {
            sa[i] = 0;
        }
        # 初始化 r 变量
        r = 0;
        # 生成随机数填充 da 数组
        for (a = 1; a <= 8; a++) {
            da[a] = Math.floor(10 * Math.random() + 1);
        }
        # 计算 da 数组的和
        for (a = 1; a <= 8; a++) {
            r = r + da[a];
        }
        # 初始化 vs 数组
        vs[1] = "JOE MAN";
        vs[2] = "L.B.J.";
        vs[3] = "MR.WASHBURN";
        vs[4] = "MISS KAREN";
        vs[5] = "JOLLY";
        vs[6] = "HORSE";
        vs[7] = "JELLY DO NOT";
        vs[8] = "MIDNIGHT";
        for (n = 1; n <= 8; n++) {  # 使用循环遍历 1 到 8 的数字
            print(vs[n] + "\t\t" + n + "\t" + (r / da[n]) + ":1\n");  # 打印特定格式的字符串，包括 vs[n]、n、r/da[n] 的值
        }
        print("--------------------------------------------------\n");  # 打印分隔线
        print("PLACE YOUR BETS...HORSE # THEN AMOUNT\n");  # 提示用户下注，输入马匹号码和金额
        for (j = 1; j <= c; j++) {  # 使用循环遍历 1 到 c 的数字
            while (1) {  # 进入无限循环
                print(ws[j]);  # 打印特定马匹的信息
                str = await input();  # 等待用户输入并将其存储在变量 str 中
                qa[j] = parseInt(str);  # 将用户输入的字符串转换为整数并存储在数组 qa 中
                pa[j] = parseInt(str.substr(str.indexOf(",") + 1));  # 从用户输入的字符串中提取逗号后面的部分并转换为整数，存储在数组 pa 中
                if (pa[j] < 1 || pa[j] >= 100000) {  # 检查下注金额是否在有效范围内
                    print("  YOU CAN'T DO THAT!\N");  # 如果不在有效范围内则打印错误提示
                } else {
                    break;  # 如果在有效范围内则跳出循环
                }
            }
        }
        print("\n");  # 打印空行
        print("1 2 3 4 5 6 7 8\n");  # 打印数字 1 到 8
        t = 0;  # 初始化变量 t 为 0
        do {  # 使用 do-while 循环
            print("XXXXSTARTXXXX\n");  # 打印 "XXXXSTARTXXXX" 字符串
            for (i = 1; i <= 8; i++) {  # 使用 for 循环，i 从 1 到 8
                m = i;  # 将 m 赋值为 i
                ma[i] = m;  # 将 m 赋值给 ma[i]
                ya[ma[i]] = Math.floor(100 * Math.random() + 1);  # 生成一个 1 到 100 之间的随机整数，赋值给 ya[ma[i]]
                if (ya[ma[i]] < 10) {  # 如果 ya[ma[i]] 小于 10
                    ya[ma[i]] = 1;  # 将 ya[ma[i]] 赋值为 1
                    continue;  # 继续下一次循环
                }
                s = Math.floor(r / da[i] + 0.5);  # 计算 r 除以 da[i] 的结果，取整数部分，加上 0.5 后赋值给 s
                if (ya[ma[i]] < s + 17) {  # 如果 ya[ma[i]] 小于 s + 17
                    ya[ma[i]] = 2;  # 将 ya[ma[i]] 赋值为 2
                    continue;  # 继续下一次循环
                }
                if (ya[ma[i]] < s + 37) {  # 如果 ya[ma[i]] 小于 s + 37
                    ya[ma[i]] = 3;  # 将 ya[ma[i]] 赋值为 3
                    continue;  # 继续下一次循环
                }
                # 如果ya[ma[i]]小于s+57，则将ya[ma[i]]设置为4，并跳过当前循环
                if (ya[ma[i]] < s + 57) {
                    ya[ma[i]] = 4;
                    continue;
                }
                # 如果ya[ma[i]]小于s+77，则将ya[ma[i]]设置为5，并跳过当前循环
                if (ya[ma[i]] < s + 77) {
                    ya[ma[i]] = 5;
                    continue;
                }
                # 如果ya[ma[i]]小于s+92，则将ya[ma[i]]设置为6，并跳过当前循环
                if (ya[ma[i]] < s + 92) {
                    ya[ma[i]] = 6;
                    continue;
                }
                # 否则将ya[ma[i]]设置为7
                ya[ma[i]] = 7;
            }
            # 将m设置为i
            m = i;
            # 遍历ma数组，将sa[ma[i]]的值加上ya[ma[i]]的值
            for (i = 1; i <= 8; i++) {
                sa[ma[i]] = sa[ma[i]] + ya[ma[i]];
            }
            # 将i设置为1
            i = 1;
            # 遍历l，将l的值从1到8
            for (l = 1; l <= 8; l++) {
# 初始化循环变量 i 为 1，循环条件为 i <= 8 - l，每次循环 i 自增 1
for (i = 1; i <= 8 - l; i++) {
    # 如果 sa[ma[i]] 小于 sa[ma[i + 1]]，则跳过当前循环，继续下一次循环
    if (sa[ma[i]] < sa[ma[i + 1]])
        continue;
    # 将 ma[i] 的值赋给 h
    h = ma[i];
    # 将 ma[i + 1] 的值赋给 ma[i]
    ma[i] = ma[i + 1];
    # 将 h 的值赋给 ma[i + 1]
    ma[i + 1] = h;
}

# 将 sa[ma[8]] 的值赋给 t
t = sa[ma[8]];
# 初始化循环变量 i 为 1，循环条件为 i <= 8，每次循环 i 自增 1
for (i = 1; i <= 8; i++) {
    # 计算 b 的值为 sa[ma[i]] 减去 sa[ma[i - 1]]
    b = sa[ma[i]] - sa[ma[i - 1]];
    # 如果 b 不等于 0
    if (b != 0) {
        # 初始化循环变量 a 为 1，循环条件为 a <= b，每次循环 a 自增 1
        for (a = 1; a <= b; a++) {
            # 打印换行符
            print("\n");
            # 如果 sa[ma[i]] 大于 27，则跳出当前循环
            if (sa[ma[i]] > 27)
                break;
        }
        # 如果 a 小于等于 b，则跳出当前循环
        if (a <= b)
            break;
    }
}
                print(" " + ma[i] + " ");  # 打印马匹编号
            }
            for (a = 1; a < 28 - t; a++) {  # 循环打印换行符
                print("\n");
            }
            print("XXXXFINISHXXXX\n");  # 打印比赛结束标志
            print("\n");  # 打印换行
            print("\n");  # 打印两个换行
            print("---------------------------------------------\n");  # 打印分隔线
            print("\n");  # 打印换行
        } while (t < 28) ;  # 循环条件
        print("THE RACE RESULTS ARE:\n");  # 打印比赛结果标题
        z9 = 1;  # 初始化变量z9为1
        for (i = 8; i >= 1; i--) {  # 循环打印比赛结果
            f = ma[i];  # 获取马匹编号
            print("\n");  # 打印换行
            print("" + z9 + " PLACE HORSE NO. " + f + " AT " + (r / da[f]) + ":1\n");  # 打印比赛名次和相关信息
            z9++;  # z9自增1
        }
        for (j = 1; j <= c; j++) {  # 循环条件
            if (qa[j] != ma[8])  # 如果 qa[j] 不等于 ma[8]
                continue;  # 继续下一次循环
            n = qa[j];  # 将 qa[j] 赋值给 n
            print("\n");  # 打印换行
            print(ws[j] + " WINS $" + (r / da[n]) * pa[j] + "\n");  # 打印 ws[j] + " WINS $" + (r / da[n]) * pa[j] + 换行
        }
        print("DO YOU WANT TO BET ON THE NEXT RACE ?\n");  # 打印提示信息
        print("YES OR NO");  # 打印提示信息
        str = await input();  # 等待用户输入并赋值给 str
    } while (str == "YES") ;  # 当 str 等于 "YES" 时继续循环
}

main();  # 调用 main 函数
```