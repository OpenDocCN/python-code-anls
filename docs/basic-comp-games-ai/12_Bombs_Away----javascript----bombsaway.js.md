# `12_Bombs_Away\javascript\bombsaway.js`

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
# 结束函数定义

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串
}

// Main program
async function main()
{
    s = 0;  // 初始化变量 s
    t = 0;  // 初始化变量 t
    while (1) {  // 进入无限循环
        print("YOU ARE A PILOT IN A WORLD WAR II BOMBER.\n");  // 打印提示信息
        while (1) {  // 进入内部无限循环
            print("WHAT SIDE -- ITALY(1), ALLIES(2), JAPAN(3), GERMANY(4)");  // 打印提示信息
            a = parseInt(await input());  // 获取用户输入并转换为整数赋值给变量 a
            if (a < 1 || a > 4)  // 判断用户输入是否在指定范围内
                print("TRY AGAIN...\n");  // 如果不在范围内则打印提示信息
            else
                break;  // 如果在范围内则跳出内部循环
        }
        if (a == 1) {  // 判断用户输入的值
            while (1) {
                # 打印目标选项，让用户选择 ALBANIA(1), GREECE(2), NORTH AFRICA(3)
                print("YOUR TARGET -- ALBANIA(1), GREECE(2), NORTH AFRICA(3)");
                # 将用户输入的内容转换为整数并赋值给变量 b
                b = parseInt(await input());
                # 如果用户输入的数字小于1或大于3，则打印提示信息并让用户重新输入
                if (b < 1 || b > 3)
                    print("TRY AGAIN...\n");
                else
                    break;
            }
            # 打印换行符
            print("\n");
            # 根据用户选择的目标进行不同的打印提示
            if (b == 1) {
                print("SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE.\n");
            } else if (b == 2) {
                print("BE CAREFUL!!!\n");
            } else {
                print("YOU'RE GOING FOR THE OIL, EH?\n");
            }
        } else if (a == 2) {
            while (1) {
                # 打印飞机选项，让用户选择 LIBERATOR(1), B-29(2), B-17(3), LANCASTER(4)
                print("AIRCRAFT -- LIBERATOR(1), B-29(2), B-17(3), LANCASTER(4)");
                # 将用户输入的内容转换为整数并赋值给变量 g
                g = parseInt(await input());
                # 如果输入的值小于1或者大于4，则打印提示信息并重新输入
                if (g < 1 || g > 4)
                    print("TRY AGAIN...\n");
                else
                    break;
            }
            # 打印换行符
            print("\n");
            # 根据用户输入的值，打印相应的信息
            if (g == 1) {
                print("YOU'VE GOT 2 TONS OF BOMBS FLYING FOR PLOESTI.\n");
            } else if (g == 2) {
                print("YOU'RE DUMPING THE A-BOMB ON HIROSHIMA.\n");
            } else if (g == 3) {
                print("YOU'RE CHASING THE BISMARK IN THE NORTH SEA.\n");
            } else {
                print("YOU'RE BUSTING A GERMAN HEAVY WATER PLANT IN THE RUHR.\n");
            }
        } else if (a == 3) {
            # 打印相应的信息
            print("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.\n");
            print("YOUR FIRST KAMIKAZE MISSION(Y OR N)");
            # 等待用户输入
            str = await input();
            # 如果用户输入为"N"，则执行相应的操作
            if (str == "N") {
                s = 0;  // 初始化变量 s 为 0
            } else {  // 如果条件不成立
                s = 1;  // 将变量 s 赋值为 1
                print("\n");  // 打印换行符
            }
        } else {  // 如果条件不成立
            while (1) {  // 进入无限循环
                print("A NAZI, EH?  OH WELL.  ARE YOU GOING FOR RUSSIA(1),\n");  // 打印提示信息
                print("ENGLAND(2), OR FRANCE(3)");  // 打印提示信息
                m = parseInt(await input());  // 从用户输入中获取整数并赋值给变量 m
                if (m < 1 || m > 3)  // 如果 m 小于 1 或者大于 3
                    print("TRY AGAIN...\n");  // 打印提示信息
                else  // 如果条件成立
                    break;  // 退出循环
            }
            print("\n");  // 打印换行符
            if (m == 1) {  // 如果条件成立
                print("YOU'RE NEARING STALINGRAD.\n");  // 打印提示信息
            } else if (m == 2) {  // 如果条件不成立，但是 m 等于 2
                print("NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR.\n");  // 打印提示信息
            } else if (m == 3) {  # 如果 m 的值为 3
                print("NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS.\n");  # 打印特定的消息
            }
        }
        if (a != 3) {  # 如果 a 的值不等于 3
            print("\n");  # 打印换行符
            while (1) {  # 进入无限循环
                print("HOW MANY MISSIONS HAVE YOU FLOWN");  # 打印提示消息
                d = parseInt(await input());  # 从输入中获取值并转换为整数赋给变量 d
                if (d < 160)  # 如果 d 小于 160
                    break;  # 退出循环
                print("MISSIONS, NOT MILES...\n");  # 打印特定的消息
                print("150 MISSIONS IS HIGH EVEN FOR OLD-TIMERS.\n");  # 打印特定的消息
                print("NOW THEN, ");  # 打印特定的消息
            }
            print("\n");  # 打印换行符
            if (d >= 100) {  # 如果 d 大于等于 100
                print("THAT'S PUSHING THE ODDS!\n");  # 打印特定的消息
            } else if (d < 25) {  # 如果 d 小于 25
                print("FRESH OUT OF TRAINING, EH?\n");  # 打印特定的消息
            }  # 结束 if-else 语句块
            print("\n");  # 打印空行
            if (d >= 160 * Math.random())  # 如果 d 大于等于 160 乘以一个随机数
                hit = true;  # 则命中为真
            else  # 否则
                hit = false;  # 命中为假
        } else {  # 如果不满足上面的条件
            if (s == 0) {  # 如果 s 等于 0
                hit = false;  # 命中为假
            } else if (Math.random() > 0.65) {  # 否则如果随机数大于 0.65
                hit = true;  # 命中为真
            } else {  # 否则
                hit = false;  # 命中为假
                s = 100;  # s 赋值为 100
            }
        }
        if (hit) {  # 如果命中为真
            print("DIRECT HIT!!!! " + Math.floor(100 * Math.random()) + " KILLED.\n");  # 打印直接命中并且随机数乘以 100 向下取整再加上 KILLED
            print("MISSION SUCCESSFUL.\n");  # 打印任务成功
        } else {  # 如果命中为假
            t = 0;  # 初始化变量 t 为 0
            if (a != 3):  # 如果变量 a 不等于 3
                print("MISSED TARGET BY " + Math.floor(2 + 30 * Math.random()) + " MILES!\n");  # 打印未击中目标的距离
                print("NOW YOU'RE REALLY IN FOR IT !!\n");  # 打印警告信息
                print("\n");  # 打印空行
                while (1):  # 进入无限循环
                    print("DOES THE ENEMY HAVE GUNS(1), MISSILE(2), OR BOTH(3)");  # 打印提示信息
                    r = parseInt(await input());  # 获取用户输入并转换为整数赋值给变量 r
                    if (r < 1 || r > 3)  # 如果 r 小于 1 或者大于 3
                        print("TRY AGAIN...\n");  # 打印提示信息
                    else:  # 否则
                        break;  # 退出循环
                print("\n");  # 打印空行
                if (r != 2):  # 如果 r 不等于 2
                    print("WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS (10 TO 50)");  # 打印提示信息
                    s = parseInt(await input());  # 获取用户输入并转换为整数赋值给变量 s
                    if (s < 10)  # 如果 s 小于 10
                        print("YOU LIE, BUT YOU'LL PAY...\n");  # 打印警告信息
                    print("\n");  # 打印空行
                }  # 结束内层循环
                print("\n");  # 打印空行
                if (r > 1)  # 如果 r 大于 1
                    t = 35;  # 则 t 等于 35
            }  # 结束外层循环
            if (s + t <= 100 * Math.random()) {  # 如果 s + t 小于等于 100 乘以一个随机数
                print("YOU MADE IT THROUGH TREMENDOUS FLAK!!\n");  # 打印通过了强烈的火力
            } else {
                print("* * * * BOOM * * * *\n");  # 否则打印爆炸
                print("YOU HAVE BEEN SHOT DOWN.....\n");  # 打印你已被击落
                print("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR\n");  # 打印亲爱的，今天我们聚集在这里悼念
                print("LAST TRIBUTE...\n");  # 打印最后的致敬
            }
        }  # 结束循环
        print("\n");  # 打印空行
        print("\n");  # 打印两个空行
        print("\n");
        print("ANOTHER MISSION (Y OR N)");  # 打印另一个任务（Y 或 N）
        str = await input();  # 获取用户输入
        if (str != "Y")  # 如果输入不是 Y
            break;  # 结束当前的循环或者 switch 语句
    }
    print("CHICKEN !!!\n");  # 打印 "CHICKEN !!!" 到控制台
    print("\n");  # 打印一个空行到控制台
}

main();  # 调用名为 main 的函数
```