# `43_Hammurabi\javascript\hammurabi.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 input 元素，用于用户输入
// 在页面上打印提示符 "? "
// 设置 input 元素的类型为文本输入类型
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

var a;  // 声明变量a
var s;  // 声明变量s

function exceeded_grain()  // 定义函数exceeded_grain
{
    print("HAMURABI: THINK AGAIN.  YOU HAVE ONLY\n");  // 打印提示信息
    print(s + " BUSHELS OF GRAIN.  NOW THEN,\n");  // 打印变量s的值和提示信息
}

function exceeded_acres()  // 定义函数exceeded_acres
{
    print("HAMURABI: THINK AGAIN.  YOU OWN ONLY " + a + " ACRES.  NOW THEN,\n");  // 打印提示信息和变量a的值
}

// Main control section  // 主控制部分
async function main()
{
    # 打印游戏标题
    print(tab(32) + "HAMURABI\n");
    # 打印游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    # 打印游戏提示
    print("TRY YOUR HAND AT GOVERNING ANCIENT SUMERIA\n");
    print("FOR A TEN-YEAR TERM OF OFFICE.\n");
    print("\n");

    # 初始化变量
    d1 = 0;  # 灾害人口死亡数
    p1 = 0;  # 疫病人口死亡数
    z = 0;   # 灾害数
    p = 95;  # 初始人口
    s = 2800;  # 初始粮食数
    h = 3000;  # 初始粮食数
    e = h - s;  # 剩余粮食数
    y = 3;   # 初始年数
    a = h / y;  # 平均每年粮食数
    i = 5;  # 初始化变量 i 为 5
    q = 1;  # 初始化变量 q 为 1
    d = 0;  # 初始化变量 d 为 0
    while (1) {  # 进入无限循环
        print("\n");  # 打印空行
        print("\n");  # 打印空行
        print("\n");  # 打印空行
        print("HAMURABI:  I BEG TO REPORT TO YOU,\n");  # 打印报告信息
        z++;  # 变量 z 自增
        print("IN YEAR " + z + ", " + d + " PEOPLE STARVED, " + i + " CAME TO THE CITY,\n");  # 打印年份、饥饿人数和迁入人数信息
        p += i;  # 变量 p 增加 i 的值
        if (q <= 0) {  # 如果 q 小于等于 0
            p = Math.floor(p / 2);  # 将 p 减半
            print("A HORRIBLE PLAGUE STRUCK!  HALF THE PEOPLE DIED.\n");  # 打印瘟疫信息
        }
        print("POPULATION IS NOW " + p + "\n");  # 打印当前人口数量
        print("THE CITY NOW OWNS " + a + " ACRES.\n");  # 打印城市拥有的土地面积
        print("YOU HARVESTED " + y + " BUSHELS PER ACRE.\n");  # 打印每英亩收获的谷物数量
        print("THE RATS ATE " + e + " BUSHELS.\n");  # 打印老鼠吃掉的谷物数量
        print("YOU NOW HAVE " + s + " BUSHELS IN STORE.\n");  # 打印当前存储的谷物数量
        print("\n");  # 打印空行
        if (z == 11) {  # 如果 z 等于 11
            q = 0;  # 将 q 设为 0
            break;  # 跳出循环
        }
        c = Math.floor(10 * Math.random());  # 生成一个 0 到 9 之间的随机数，并向下取整
        y = c + 17;  # 将随机数加上 17 赋值给 y
        print("LAND IS TRADING AT " + y + " BUSHELS PER ACRE.\n");  # 打印土地每英亩的价格
        while (1) {  # 进入无限循环
            print("HOW MANY ACRES DO YOU WISH TO BUY");  # 打印提示信息
            q = parseInt(await input());  # 将输入的值转换为整数并赋值给 q
            if (q < 0)  # 如果 q 小于 0
                break;  # 跳出循环
            if (y * q > s) {  # 如果购买的土地总价大于玩家拥有的粮食总量
                exceeded_grain();  # 调用函数处理粮食不足的情况
            } else
                break;  # 跳出循环
        }
        if (q < 0)  # 如果 q 小于 0
            break;  # 跳出循环
        if (q != 0) {  # 如果 q 不等于 0
            a += q;  # 将 a 增加 q
            s -= y * q;  # 将 s 减去 y 乘以 q
            c = 0;  # 将 c 设为 0
        } else {  # 否则
            while (1) {  # 进入无限循环
                print("HOW MANY ACRES DO YOU WISH TO SELL");  # 打印提示信息
                q = parseInt(await input());  # 从输入中获取 q 的值
                if (q < 0)  # 如果 q 小于 0
                    break;  # 退出循环
                if (q >= a) {  # 如果 q 大于等于 a
                    exceeded_acres();  # 调用 exceeded_acres 函数
                } else {  # 否则
                    break;  # 退出循环
                }
            }
            if (q < 0)  # 如果 q 小于 0
                break;  # 退出循环
            a -= q;  # 将 a 减去 q
            s += y * q;  # 将 s 增加 y 乘以 q
            c = 0;  // 初始化变量 c 为 0
        }
        print("\n");  // 打印换行符
        while (1) {  // 进入无限循环
            print("HOW MANY BUSHELS DO YOU WISH TO FEED YOUR PEOPLE");  // 打印提示信息
            q = parseInt(await input());  // 从用户输入中获取整数值并赋给变量 q
            if (q < 0)  // 如果输入值小于 0
                break;  // 退出循环
            if (q > s)  // 如果尝试使用的粮食多于仓库中的粮食
                exceeded_grain();  // 调用函数处理超出粮食的情况
            else
                break;  // 退出循环
        }
        if (q < 0)  // 如果输入值小于 0
            break;  // 退出循环
        s -= q;  // 从仓库中减去使用的粮食数量
        c = 1;  // 将变量 c 设置为 1
        print("\n");  // 打印换行符
        while (1) {  // 进入无限循环
            print("HOW MANY ACRES DO YOU WISH TO PLANT WITH SEED");  // 打印提示信息
            # 从输入中获取一个整数并转换为数字
            d = parseInt(await input());
            # 如果输入的数字不等于0
            if (d != 0) {
                # 如果输入的数字小于0，则跳出循环
                if (d < 0)
                    break;
                # 如果输入的数字大于拥有的土地数量
                if (d > a) {    // Trying to plant more acres than you own?
                    # 调用超出土地数量的函数
                    exceeded_acres();
                } else {
                    # 如果种植的数量除以2大于拥有的种子数量
                    if (Math.floor(d / 2) > s)  // Enough grain for seed?
                        # 调用超出种子数量的函数
                        exceeded_grain();
                    else {
                        # 如果输入的数量大于等于10倍的人口数量
                        if (d >= 10 * p) {
                            # 打印警告信息
                            print("BUT YOU HAVE ONLY " + p + " PEOPLE TO TEND THE FIELDS!  NOW THEN,\n");
                        } else {
                            # 否则跳出循环
                            break;
                        }
                    }
                }
            }
        }
        # 如果输入的数字小于0
        if (d < 0) {
            q = -1;  // 初始化变量 q 为 -1
            break;  // 跳出循环
        }
        s -= Math.floor(d / 2);  // s 减去 d 除以 2 的向下取整
        c = Math.floor(Math.random() * 5) + 1;  // 生成一个 1 到 5 之间的随机整数赋值给 c
        // A bountiful harvest!（丰收了！）
        if (c % 2 == 0) {  // 如果 c 是偶数
            // Rats are running wild!!（老鼠疯狂地奔跑！！）
            e = Math.floor(s / c);  // e 等于 s 除以 c 的向下取整
        }
        s = s - e + h;  // s 减去 e 加上 h
        c = Math.floor(Math.random() * 5) + 1;  // 生成一个 1 到 5 之间的随机整数赋值给 c
        // Let's have some babies（让我们生些孩子吧）
        i = Math.floor(c * (20 * a + s) / p / 100 + 1);  // i 等于 c 乘以（20 乘以 a 加上 s）除以 p 再除以 100 加上 1 的向下取整
        // How many people had full tummies?（有多少人吃饱了肚子？）
        c = Math.floor(q / 20);  // c 等于 q 除以 20 的向下取整
        // Horros, a 15% chance of plague（可怕，15% 的几率会发生瘟疫）
        q = Math.floor(10 * (2 * Math.random() - 0.3));  // q 等于 10 乘以（2 乘以 生成的随机数 减去 0.3）的向下取整
        if (p < c) {  // 如果 p 小于 c
            d = 0;  // d 等于 0
            continue;  // 继续执行下一次循环
        }
        // Starve enough for impeachment?  // 是否饿死足够多的人导致弹劾？
        d = p - c;  // 计算人口减少的数量
        if (d <= 0.45 * p) {  // 如果人口减少的数量小于等于总人口的45%
            p1 = ((z - 1) * p1 + d * 100 / p) / z;  // 更新平均粮食每人的数量
            p = c;  // 更新总人口数量
            d1 += d;  // 更新总饿死人口数量
            continue;  // 继续执行下一次循环
        }
        print("\n");  // 打印空行
        print("YOU STARVED " + d + " PEOPLE IN ONE YEAR!!!\n");  // 打印饿死人口数量
        q = 0;  // 重置变量q为0
        p1 = 34;  // 重置平均粮食每人的数量为34
        p = 1;  // 重置总人口数量为1
        break;  // 跳出循环
    }
    if (q < 0) {  // 如果变量q小于0
        print("\n");  // 打印空行
        print("HAMURABI:  I CANNOT DO WHAT YOU WISH.\n");  // 打印提示信息
        print("GET YOURSELF ANOTHER STEWARD!!!!!\n");
```
这行代码是一个条件语句的结尾，表示如果条件不满足，则执行这行代码。

```
    } else {
```
这行代码是一个条件语句的开始，表示如果条件不满足，则执行下面的代码块。

```
        print("IN YOUR 10-YEAR TERM OF OFFICE, " + p1 + " PERCENT OF THE\n");
```
这行代码用于打印一条消息，其中包含变量p1的值。

```
        print("POPULATION STARVED PER YEAR ON THE AVERAGE, I.E. A TOTAL OF\n");
```
这行代码用于打印一条消息。

```
        print(d1 + " PEOPLE DIED!!\n");
```
这行代码用于打印一条消息，其中包含变量d1的值。

```
        l = a / p;
```
这行代码用于计算变量l的值，通过将变量a除以变量p得到。

```
        print("YOU STARTED WITH 10 ACRES PER PERSON AND ENDED WITH\n");
```
这行代码用于打印一条消息。

```
        print(l + " ACRES PER PERSON.\n");
```
这行代码用于打印一条消息，其中包含变量l的值。

```
        print("\n");
```
这行代码用于打印一个空行。

```
        if (p1 > 33 || l < 7) {
```
这行代码是一个条件语句的开始，表示如果条件满足，则执行下面的代码块。

```
            print("DUE TO THIS EXTREME MISMANAGEMENT YOU HAVE NOT ONLY\n");
```
这行代码用于打印一条消息。

```
            print("BEEN IMPEACHED AND THROWN OUT OF OFFICE BUT YOU HAVE\n");
```
这行代码用于打印一条消息。

```
            print("ALSO BEEN DECLARED NATIONAL FINK!!!!\n");
```
这行代码用于打印一条消息。

```
        } else if (p1 > 10 || l < 9) {
```
这行代码是一个条件语句的开始，表示如果上一个条件不满足且这个条件满足，则执行下面的代码块。

```
            print("YOUR HEAVY-HANDED PERFORMANCE SMACKS OF NERO AND IVAN IV.\n");
```
这行代码用于打印一条消息。

```
            print("THE PEOPLE (REMAINING) FIND YOU AN UNPLEASANT RULER, AND,\n");
```
这行代码用于打印一条消息。

```
            print("FRANKLY, HATE YOUR GUTS!!\n");
```
这行代码用于打印一条消息。

```
        } else if (p1 > 3 || l < 10) {
```
这行代码是一个条件语句的开始，表示如果上一个条件不满足且这个条件满足，则执行下面的代码块。

```
            print("YOUR PERFORMANCE COULD HAVE BEEN SOMEWHAT BETTER, BUT\n");
```
这行代码用于打印一条消息。

```
            print("REALLY WASN'T TOO BAD AT ALL. " + Math.floor(p * 0.8 * Math.random()) + " PEOPLE\n");
```
这行代码用于打印一条消息，其中包含一个数学表达式的计算结果。
            print("WOULD DEARLY LIKE TO SEE YOU ASSASSINATED BUT WE ALL HAVE OUR\n");
            # 打印一条消息，表达了愤怒和不满的情绪
            print("TRIVIAL PROBLEMS.\n");
            # 打印一条消息，表达了对问题的轻视和不重视
        } else {
            # 如果条件不满足
            print("A FANTASTIC PERFORMANCE!!!  CHARLEMANGE, DISRAELI, AND\n");
            # 打印一条消息，表达了对表现的赞美
            print("JEFFERSON COMBINED COULD NOT HAVE DONE BETTER!\n");
            # 打印一条消息，表达了对表现的赞美
        }
    }
    # 打印一个空行
    print("\n");
    # 打印一条消息
    print("SO LONG FOR NOW.\n");
    # 打印一个空行
    print("\n");
}

main();
# 调用主函数
```