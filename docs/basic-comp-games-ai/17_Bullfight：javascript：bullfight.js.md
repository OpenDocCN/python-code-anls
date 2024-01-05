# `17_Bullfight\javascript\bullfight.js`

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
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串变量str的末尾
    return str;  # 返回处理后的字符串

}

var a;  # 声明变量a
var b;  # 声明变量b
var c;  # 声明变量c
var l;  # 声明变量l
var t;  # 声明变量t
var as;  # 声明变量as
var bs;  # 声明变量bs
var d = [];  # 声明变量d并初始化为空数组
var ls = [, "SUPERB", "GOOD", "FAIR", "POOR", "AWFUL"];  # 声明变量ls并初始化为包含字符串元素的数组

function af(k)  # 定义名为af的函数，接受参数k
{
    return Math.floor(Math.random() * 2 + 1);  # 返回一个随机整数，范围为1到2
}

function cf(q)  # 定义名为cf的函数，接受参数q
{
    return df(q) * Math.random();  # 返回参数 q 经过 df 函数处理后的结果与一个随机数的乘积
}

function df(q)
{
    return (4.5 + l / 6 - (d[1] + d[2]) * 2.5 + 4 * d[4] + 2 * d[5] - Math.pow(d[3], 2) / 120 - a);  # 返回一个复杂的数学表达式的结果
}

function setup_helpers()
{
    b = 3 / a * Math.random();  # 计算 b 的值
    if (b < 0.37)  # 如果 b 小于 0.37
        c = 0.5;  # 则 c 赋值为 0.5
    else if (b < 0.5)  # 否则如果 b 小于 0.5
        c = 0.4;  # 则 c 赋值为 0.4
    else if (b < 0.63)  # 否则如果 b 小于 0.63
        c = 0.3;  # 则 c 赋值为 0.3
    else if (b < 0.87)  # 否则如果 b 小于 0.87
        c = 0.2;  # 则 c 赋值为 0.2
}
    else
        c = 0.1;  # 如果条件不满足，则将变量c赋值为0.1
    t = Math.floor(10 * c + 0.2);  # 根据c的值计算t，取整数部分
    print("THE " + as + bs + " DID A " + ls[t] + " JOB.\n");  # 打印特定格式的字符串
    if (4 <= t) {  # 如果t大于等于4
        if (5 != t) {  # 如果t不等于5
            // Lines 1800 and 1810 of original program are unreachable  # 注释：原始程序的1800和1810行是无法到达的
            switch (af(0)) {  # 根据af(0)的值进行不同的操作
                case 1:
                    print("ONE OF THE " + as + bs + " WAS KILLED.\n");  # 打印特定格式的字符串
                    break;
                case 2:
                    print("NO " + as + b + " WERE KILLED.\n");  # 打印特定格式的字符串
                    break;
            }
        } else {
            if (as != "TOREAD")  # 如果as不等于"TOREAD"
                print(af(0) + " OF THE HORSES OF THE " + as + bs + " KILLED.\n");  # 打印特定格式的字符串
            print(af(0) + " OF THE " + as + bs + " KILLED.\n");  # 打印特定格式的字符串
        }
    }
    # 打印换行
    print("\n");
}

// 主程序
async function main()
{
    # 打印制表符和"BULL"
    print(tab(34) + "BULL\n");
    # 打印制表符和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    # 打印三个换行
    print("\n");
    print("\n");
    print("\n");
    # 初始化变量l为1
    l = 1;
    # 打印"Do you want instructions"并等待用户输入
    print("DO YOU WANT INSTRUCTIONS");
    str = await input();
    # 如果用户输入不是"NO"，则执行以下代码块
    if (str != "NO") {
        # 打印欢迎词
        print("HELLO, ALL YOU BLOODLOVERS AND AFICIONADOS.\n");
        print("HERE IS YOUR BIG CHANCE TO KILL A BULL.\n");
        print("\n");
        print("ON EACH PASS OF THE BULL, YOU MAY TRY\n");
# 打印第一条消息
print("0 - VERONICA (DANGEROUS INSIDE MOVE OF THE CAPE)\n");
# 打印第二条消息
print("1 - LESS DANGEROUS OUTSIDE MOVE OF THE CAPE\n");
# 打印第三条消息
print("2 - ORDINARY SWIRL OF THE CAPE.\n");
# 打印空行
print("\n");
# 打印第四条消息
print("INSTEAD OF THE ABOVE, YOU MAY TRY TO KILL THE BULL\n");
# 打印第五条消息
print("ON ANY TURN: 4 (OVER THE HORNS), 5 (IN THE CHEST).\n");
# 打印第六条消息
print("BUT IF I WERE YOU,\n");
# 打印第七条消息
print("I WOULDN'T TRY IT BEFORE THE SEVENTH PASS.\n");
# 打印空行
print("\n");
# 打印第八条消息
print("THE CROWD WILL DETERMINE WHAT AWARD YOU DESERVE\n");
# 打印第九条消息
print("(POSTHUMOUSLY IF NECESSARY).\n");
# 打印第十条消息
print("THE BRAVER YOU ARE, THE BETTER THE AWARD YOU RECEIVE.\n");
# 打印空行
print("\n");
# 打印第十一条消息
print("THE BETTER THE JOB THE PICADORES AND TOREADORES DO,\n");
# 打印第十二条消息
print("THE BETTER YOUR CHANCES ARE.\n");
# 打印两个空行
print("\n");
print("\n");
# 将字典 d 中键为 5 的值设为 1
d[5] = 1;
# 将字典 d 中键为 4 的值设为 1
d[4] = 1;
    d[3] = 0;  // 将数组 d 的第三个元素赋值为 0
    a = Math.floor(Math.random() * 5 + 1);  // 生成一个 1 到 5 之间的随机整数赋值给变量 a
    print("YOU HAVE DRAWN A " + ls[a] + " BULL.\n");  // 打印出字符串 "YOU HAVE DRAWN A " 后面接变量 ls 中索引为 a 的元素再接上字符串 " BULL.\n"
    if (a > 4) {  // 如果 a 大于 4
        print("YOU'RE LUCKY.\n");  // 打印出字符串 "YOU'RE LUCKY.\n"
    } else if (a < 2) {  // 否则如果 a 小于 2
        print("GOOD LUCK.  YOU'LL NEED IT.\n");  // 打印出字符串 "GOOD LUCK.  YOU'LL NEED IT.\n"
        print("\n");  // 打印一个空行
    }
    print("\n");  // 打印一个空行
    as = "PICADO";  // 将字符串 "PICADO" 赋值给变量 as
    bs = "RES";  // 将字符串 "RES" 赋值给变量 bs
    setup_helpers();  // 调用 setup_helpers 函数
    d[1] = c;  // 将变量 c 的值赋给数组 d 的第一个元素
    as = "TOREAD";  // 将字符串 "TOREAD" 赋值给变量 as
    bs = "ORES";  // 将字符串 "ORES" 赋值给变量 bs
    setup_helpers();  // 调用 setup_helpers 函数
    d[2] = c;  // 将变量 c 的值赋给数组 d 的第二个元素
    print("\n");  // 打印一个空行
    print("\n");  // 打印一个空行
    z = 0;  # 初始化变量 z 为 0
    while (z == 0) {  # 当 z 等于 0 时执行循环
        d[3]++;  # 数组 d 的第四个元素加一
        print("PASS NUMBER " + d[3] + "\n");  # 打印当前 PASS 的次数
        if (d[3] >= 3) {  # 如果 PASS 次数大于等于 3
            print("HERE COMES THE BULL.  TRY FOR A KILL");  # 打印提示信息
            while (1) {  # 无限循环
                str = await input();  # 等待用户输入
                if (str != "YES" && str != "NO")  # 如果输入不是 "YES" 且不是 "NO"
                    print("INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO'.\n");  # 打印提示信息
                else
                    break;  # 跳出循环
            }
            z1 = (str == "YES") ? 1 : 2;  # 如果输入是 "YES"，则 z1 为 1，否则为 2
            if (z1 != 1) {  # 如果 z1 不等于 1
                print("CAPE MOVE");  # 打印提示信息
            }
        } else {  # 如果 PASS 次数小于 3
            print("THE BULL IS CHARGING AT YOU!  YOU ARE THE MATADOR--\n");  # 打印提示信息
            print("DO YOU WANT TO KILL THE BULL");  # 打印提示信息
            while (1) {  # 进入一个无限循环
                str = await input();  # 从输入中获取字符串
                if (str != "YES" && str != "NO")  # 如果输入不是"YES"且不是"NO"
                    print("INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO'.\n");  # 打印错误提示信息
                else
                    break;  # 否则跳出循环
            }
            z1 = (str == "YES") ? 1 : 2;  # 如果输入是"YES"，则z1为1，否则为2
            if (z1 != 1) {  # 如果z1不等于1
                print("WHAT MOVE DO YOU MAKE WITH THE CAPE");  # 打印提示信息
            }
        }
        gore = 0;  # 初始化gore为0
        if (z1 != 1) {  # 如果z1不等于1
            while (1) {  # 进入一个无限循环
                e = parseInt(await input());  # 从输入中获取整数
                if (e >= 3) {  # 如果输入大于等于3
                    print("DON'T PANIC, YOU IDIOT!  PUT DOWN A CORRECT NUMBER\n");  # 打印错误提示信息
                } else {
                    break;  # 否则跳出循环
            }
        }
        if (e == 0)  # 如果 e 等于 0
            m = 3;  # 则 m 等于 3
        else if (e == 1)  # 否则如果 e 等于 1
            m = 2;  # 则 m 等于 2
        else  # 否则
            m = 0.5;  # m 等于 0.5
        l += m;  # l 加上 m
        f = (6 - a + m / 10) * Math.random() / ((d[1] + d[2] + d[3] / 10) * 5);  # 计算 f 的值
        if (f < 0.51)  # 如果 f 小于 0.51
            continue;  # 继续下一次循环
        gore = 1;  # gore 等于 1
    } else {  # 否则
        z = 1;  # z 等于 1
        print("\n");  # 打印换行
        print("IT IS THE MOMENT OF THE TRUTH.\n");  # 打印提示信息
        print("\n");  # 打印换行
        print("HOW DO YOU TRY TO KILL THE BULL");  # 打印提示信息
        h = parseInt(await input());  # 将输入转换为整数并赋值给 h
            if (h != 4 && h != 5) {  // 如果 h 不等于 4 且不等于 5
                print("YOU PANICKED.  THE BULL GORED YOU.\n");  // 打印“你惊慌了。公牛刺伤了你。”
                gore = 2;  // gore 变量赋值为 2
            } else {
                k = (6 - a) * 10 * Math.random() / ((d[1] + d[2]) * 5 * d[3]);  // 计算 k 的值
                if (h != 4) {   // 如果 h 不等于 4
                    if (k > 0.2)  // 如果 k 大于 0.2
                        gore = 1;  // gore 变量赋值为 1
                } else {
                    if (k > 0.8)  // 如果 k 大于 0.8
                        gore = 1;  // gore 变量赋值为 1
                }
                if (gore == 0) {  // 如果 gore 等于 0
                    print("YOU KILLED THE BULL!\n");  // 打印“你杀死了公牛！”
                    d[5] = 2;  // d 数组的第五个元素赋值为 2
                    break;  // 跳出循环
                }
            }
        }
        if (gore) {  // 如果 gore 为真
            # 如果 gore 等于 1，则打印“THE BULL HAS GORED YOU!”
            if (gore == 1)
                print("THE BULL HAS GORED YOU!\n");
            # 将 kill 设为 false
            kill = false;
            # 进入无限循环
            while (1) {
                # 如果 af(0) 等于 1，则打印“YOU ARE DEAD.”，将 d[4] 设为 1.5，将 kill 设为 true，然后跳出循环
                if (af(0) == 1) {
                    print("YOU ARE DEAD.\n");
                    d[4] = 1.5;
                    kill = true;
                    break;
                }
                # 打印“YOU ARE STILL ALIVE.”和换行符
                print("YOU ARE STILL ALIVE.\n");
                print("\n");
                # 打印“DO YOU RUN FROM THE RING”
                print("DO YOU RUN FROM THE RING");
                # 进入内部循环
                while (1) {
                    # 等待输入并将结果赋给 str
                    str = await input();
                    # 如果输入不是“YES”和“NO”，则打印“INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO’.”，否则跳出内部循环
                    if (str != "YES" && str != "NO")
                        print("INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO'.\n");
                    else
                        break;
                }
# 使用条件运算符判断字符串是否等于"YES"，如果是则赋值1，否则赋值2
z1 = (str == "YES") ? 1 : 2;
# 如果z1不等于2，则打印"COWARD"，将d[4]赋值为0，将kill赋值为True，然后跳出循环
if (z1 != 2) {
    print("COWARD\n");
    d[4] = 0;
    kill = true;
    break;
}
# 如果z1等于2，则打印"YOU ARE BRAVE.  STUPID, BUT BRAVE."
print("YOU ARE BRAVE.  STUPID, BUT BRAVE.\n");
# 如果af(0)的返回值等于1，则将d[4]赋值为2，将kill赋值为False，然后跳出循环
if (af(0) == 1) {
    d[4] = 2;
    kill = false;
    break;
}
# 如果af(0)的返回值不等于1，则打印"YOU ARE GORED AGAIN!"
print("YOU ARE GORED AGAIN!\n");
# 如果kill为True，则跳出循环
if (kill)
    break;
# 继续下一次循环
continue;
    # 打印三个空行
    print("\n");
    print("\n");
    print("\n");
    # 如果列表 d 的第四个元素等于 0
    if (d[4] == 0) {
        # 打印以下内容
        print("THE CROWD BOOS FOR TEN MINUTES.  IF YOU EVER DARE TO SHOW\n");
        print("YOUR FACE IN A RING AGAIN, THEY SWEAR THEY WILL KILL YOU--\n");
        print("UNLESS THE BULL DOES FIRST.\n");
    } else {
        # 如果列表 d 的第四个元素等于 2
        if (d[4] == 2) {
            # 打印以下内容
            print("THE CROWD CHEERS WILDLY!\n");
        } else if (d[5] == 2) {
            # 如果列表 d 的第五个元素等于 2，打印以下内容
            print("THE CROWD CHEERS!\n");
            print("\n");
        }
        # 打印以下内容
        print("THE CROWD AWARDS YOU\n");
        # 如果函数 cf 的返回值小于 2.4，打印以下内容
        if (cf(0) < 2.4) {
            print("NOTHING AT ALL.\n");
        } else if (cf(0) < 4.9) {
            # 如果函数 cf 的返回值小于 4.9，打印以下内容
            print("ONE EAR OF THE BULL.\n");
        } else if (cf(0) < 7.4) {
            # 如果函数 cf 的返回值小于 7.4，打印以下内容
            print("BOTH EARS OF THE BULL!\n");  # 打印"BOTH EARS OF THE BULL!\n"字符串
            print("OLE!\n");  # 打印"OLE!\n"字符串
        } else {
            print("OLE!  YOU ARE 'MUY HOMBRE'!! OLE!  OLE!\n");  # 如果条件不满足，打印"OLE!  YOU ARE 'MUY HOMBRE'!! OLE!  OLE!\n"字符串
        }
        print("\n");  # 打印一个空行
        print("ADIOS\n");  # 打印"ADIOS\n"字符串
        print("\n");  # 打印一个空行
        print("\n");  # 打印两个空行
        print("\n");
    }
}


main();  # 调用main函数
```