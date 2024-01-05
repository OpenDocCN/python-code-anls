# `d:/src/tocomm/basic-computer-games\86_Target\javascript\target.js`

```
// 定义一个名为print的函数，用于在页面上输出文本
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个名为input的函数，用于获取用户输入
function input() {
    var input_element;
    var input_str;

    // 返回一个Promise对象，表示异步操作的最终完成或失败
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
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串

}

// Main program
async function main()
{
    print(tab(33) + "TARGET\n");  // 打印带有缩进的字符串"TARGET"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有缩进的字符串"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    r = 0;  // 初始化变量r为0（原始代码中为1）
    r1 = 57.296;  // 初始化变量r1为57.296
    p = Math.PI;  // 将圆周率赋值给变量p
    print("YOU ARE THE WEAPONS OFFICER ON THE STARSHIP ENTERPRISE\n");  // 打印字符串"YOU ARE THE WEAPONS OFFICER ON THE STARSHIP ENTERPRISE"
    print("AND THIS IS A TEST TO SEE HOW ACCURATE A SHOT YOU\n");  // 打印字符串"AND THIS IS A TEST TO SEE HOW ACCURATE A SHOT YOU"
    print("ARE IN A THREE-DIMENSIONAL RANGE.  YOU WILL BE TOLD\n");  // 打印字符串"ARE IN A THREE-DIMENSIONAL RANGE.  YOU WILL BE TOLD"
    print("THE RADIAN OFFSET FOR THE X AND Z AXES, THE LOCATION\n");  // 打印字符串"THE RADIAN OFFSET FOR THE X AND Z AXES, THE LOCATION"
    print("OF THE TARGET IN THREE DIMENSIONAL RECTANGULAR COORDINATES,\n");  // 打印字符串"OF THE TARGET IN THREE DIMENSIONAL RECTANGULAR COORDINATES,"
    # 打印提示信息：从X和Z轴的大致角度和到目标的大致距离
    print("THE APPROXIMATE NUMBER OF DEGREES FROM THE X AND Z\n");
    print("AXES, AND THE APPROXIMATE DISTANCE TO THE TARGET.\n");
    # 打印提示信息：射击目标直到摧毁
    print("YOU WILL THEN PROCEEED TO SHOOT AT THE TARGET UNTIL IT IS\n");
    print("DESTROYED!\n");
    print("\n");
    # 打印祝你好运的信息
    print("GOOD LUCK!!\n");
    print("\n");
    print("\n");
    # 进入无限循环
    while (1) {
        # 生成随机的角度a和b
        a = Math.random() * 2 * p;
        b = Math.random() * 2 * p;
        # 对a和b进行取整操作
        q = Math.floor(a * r1);
        w = Math.floor(b * r1);
        # 打印目标的大致坐标
        print("RADIANS FROM X AXIS = " + a + "   FROM Z AXIS = " + b + "\n");
        # 生成随机的p1，并根据a和b计算x、y、z坐标
        p1 = 100000 * Math.random() + Math.random();
        x = Math.sin(b) * Math.cos(a) * p1;
        y = Math.sin(b) * Math.sin(a) * p1;
        z = Math.cos(b) * p1;
        # 打印目标的大致坐标
        print("TARGET SIGHTED: APPROXIMATE COORDINATES:  X=" + x + "  Y=" + y + "  Z=" + z + "\n");
        # 进入内部无限循环
        while (1) {
            r++;  // 增加变量 r 的值
            switch (r) {  // 根据变量 r 的值进行不同的操作
                case 1:  // 当 r 的值为 1 时
                    p3 = Math.floor(p1 * 0.05) * 20;  // 计算 p3 的值
                    break;  // 结束当前 case
                case 2:  // 当 r 的值为 2 时
                    p3 = Math.floor(p1 * 0.1) * 10;  // 计算 p3 的值
                    break;  // 结束当前 case
                case 3:  // 当 r 的值为 3 时
                    p3 = Math.floor(p1 * 0.5) * 2;  // 计算 p3 的值
                    break;  // 结束当前 case
                case 4:  // 当 r 的值为 4 时
                    p3 = Math.floor(p1);  // 计算 p3 的值
                    break;  // 结束当前 case
                case 5:  // 当 r 的值为 5 时
                    p3 = p1;  // 将 p3 的值设置为 p1 的值
                    break;  // 结束当前 case
            }
            print("     ESTIMATED DISTANCE: " + p3 + "\n");  // 打印估计距离的值
            print("\n");  // 打印换行符
            # 打印提示信息，要求输入角度偏差和距离
            print("INPUT ANGLE DEVIATION FROM X, DEVIATION FROM Z, DISTANCE");
            # 从用户输入中获取字符串
            str = await input();
            # 将输入的字符串转换为整数，表示角度偏差
            a1 = parseInt(str);
            # 从输入的字符串中获取逗号后的字符串，并转换为整数，表示角度偏差
            b1 = parseInt(str.substr(str.indexOf(",") + 1));
            # 从输入的字符串中获取最后一个逗号后的字符串，并转换为整数，表示距离
            p2 = parseInt(str.substr(str.lastIndexOf(",") + 1));
            # 打印换行符
            print("\n");
            # 如果距离小于20，打印提示信息并结束循环
            if (p2 < 20) {
                print("YOU BLEW YOURSELF UP!!\n");
                break;
            }
            # 将角度偏差除以r1
            a1 /= r1;
            # 将角度偏差除以r1
            b1 /= r1;
            # 打印从X轴的弧度和从Z轴的弧度
            print("RADIANS FROM X AXIS = " + a1 + "  ");
            print("FROM Z AXIS = " + b1 + "\n");
            # 根据公式计算x1, y1, z1
            x1 = p2 * Math.sin(b1) * Math.cos(a1);
            y1 = p2 * Math.sin(b1) * Math.sin(a1);
            z1 = p2 * Math.cos(b1);
            # 计算新的点与原点的距离
            d = Math.sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y) + (z1 - z) * (z1 - z));
            # 如果距离小于等于20，打印换行符
            if (d <= 20) {
                print("\n");
                # 打印目标非功能的提示信息
                print(" * * * HIT * * *   TARGET IS NON-FUNCTIONAL\n");
                # 打印爆炸距离目标的距离
                print("\n");
                print("DISTANCE OF EXPLOSION FROM TARGET WAS " + d + " KILOMETERS.");
                print("\n");
                # 打印完成任务所需的射击次数
                print("MISSION ACCOMPLISHED IN " + r + " SHOTS.\n");
                # 重置射击次数
                r = 0;
                # 打印空行
                for (i = 1; i <= 5; i++)
                    print("\n");
                # 打印下一个目标的提示信息
                print("NEXT TARGET...\n");
                print("\n");
                # 跳出循环
                break;
            }
            # 计算目标与爆炸点的距离
            x2 = x1 - x;
            y2 = y1 - y;
            z2 = z1 - z;
            # 根据距离的正负打印相应的提示信息
            if (x2 >= 0)
                print("SHOT IN FRONT OF TARGET " + x2 + " KILOMETERS.\n");
            else
                print("SHOT BEHIND TARGET " + -x2 + " KILOMETERS.\n");
            if (y2 >= 0)
                print("SHOT TO LEFT OF TARGET " + y2 + " KILOMETERS.\n");  # 打印目标左侧的距离
            else
                print("SHOT TO RIGHT OF TARGET " + -y2 + " KILOMETERS.\n");  # 打印目标右侧的距离
            if (z2 >= 0)
                print("SHOT ABOVE TARGET " + z2 + " KILOMETERS.\n");  # 打印目标上方的距离
            else
                print("SHOT BELOW TARGET " + -z2 + " KILOMETERS.\n");  # 打印目标下方的距离
            print("APPROX POSITION OF EXPLOSION:  X=" + x1 + "   Y=" + y1 + "   Z=" + z1 + "\n");  # 打印爆炸的大致位置
            print("     DISTANCE FROM TARGET = " + d + "\n");  # 打印与目标的距离
            print("\n");  # 打印空行
            print("\n");  # 打印空行
            print("\n");  # 打印空行
        }
    }
}

main();  # 调用主函数
```