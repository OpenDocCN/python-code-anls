# `d:/src/tocomm/basic-computer-games\13_Bounce\javascript\bounce.js`

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
    # 如果按下的是回车键
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
    return str;  // 返回修改后的字符串
}

// Main program
async function main()
{
    print(tab(33) + "BOUNCE\n");  // 在指定位置打印字符串"BOUNCE"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在指定位置打印字符串"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    ta = [];  // 初始化一个空数组
    print("THIS SIMULATION LETS YOU SPECIFY THE INITIAL VELOCITY\n");  // 打印提示信息
    print("OF A BALL THROWN STRAIGHT UP, AND THE COEFFICIENT OF\n");  // 打印提示信息
    print("ELASTICITY OF THE BALL.  PLEASE USE A DECIMAL FRACTION\n");  // 打印提示信息
    print("COEFFICIENCY (LESS THAN 1).\n");  // 打印提示信息
    print("\n");  // 打印空行
    print("YOU ALSO SPECIFY THE TIME INCREMENT TO BE USED IN\n");  // 打印提示信息
    print("'STROBING' THE BALL'S FLIGHT (TRY .1 INITIALLY).\n");  // 打印提示信息
    print("\n");  # 打印空行
    while (1) {  # 进入无限循环
        print("TIME INCREMENT (SEC)");  # 打印提示信息
        s2 = parseFloat(await input());  # 获取用户输入并转换为浮点数赋值给变量s2
        print("\n");  # 打印空行
        print("VELOCITY (FPS)");  # 打印提示信息
        v = parseFloat(await input());  # 获取用户输入并转换为浮点数赋值给变量v
        print("\n");  # 打印空行
        print("COEFFICIENT");  # 打印提示信息
        c = parseFloat(await input());  # 获取用户输入并转换为浮点数赋值给变量c
        print("\n");  # 打印空行
        print("FEET\n");  # 打印提示信息和换行符
        s1 = Math.floor(70 / (v / (16 * s2)));  # 计算并赋值给变量s1
        for (i = 1; i <= s1; i++)  # 进入循环，i从1到s1
            ta[i] = v * Math.pow(c, i - 1) / 16;  # 计算并赋值给数组ta的第i个元素
        for (h = Math.floor(-16 * Math.pow(v / 32, 2) + Math.pow(v, 2) / 32 + 0.5); h >= 0; h -= 0.5) {  # 进入循环，h从初始值到0递减0.5
            str = "";  # 初始化字符串变量
            if (Math.floor(h) == h)  # 判断条件
                str += " " + h + " ";  # 拼接字符串
# 初始化变量 l 为 0
l = 0;
# 循环遍历 s1 次
for (i = 1; i <= s1; i++) {
    # 循环遍历直到 t 大于等于 ta[i]
    for (t = 0; t <= ta[i]; t += s2) {
        # l 增加 s2
        l += s2;
        # 如果 h 与计算得到的高度差的绝对值小于等于 0.25
        if (Math.abs(h - (0.5 * (-32) * Math.pow(t, 2) + v * Math.pow(c, i - 1) * t)) <= 0.25) {
            # 当 str 的长度小于 l/s2 时，添加空格
            while (str.length < l / s2)
                str += " ";
            # 添加 "0" 到 str
            str += "0";
        }
    }
    # 将 t 设为 ta[i+1] 的一半
    t = ta[i + 1] / 2;
    # 如果 -16 * Math.pow(t, 2) + v * Math.pow(c, i - 1) * t 小于 h，跳出循环
    if (-16 * Math.pow(t, 2) + v * Math.pow(c, i - 1) * t < h)
        break;
}
# 打印 str 和换行符
print(str + "\n");
# 初始化 str 为一个空格
str = " ";
# 循环遍历直到 i 小于等于 Math.floor(l + 1) / s2 + 1
for (i = 1; i < Math.floor(l + 1) / s2 + 1; i++)
    # 添加 "." 到 str
    str += ".";
# 打印 str 和换行符
print(str + "\n");
        str = " 0";  # 初始化一个字符串变量str，赋值为" 0"
        for (i = 1; i < Math.floor(l + 0.9995); i++) {  # 使用for循环，从1开始遍历到l的向下取整加0.9995
            while (str.length < Math.floor(i / s2))  # 使用while循环，当str的长度小于i除以s2的向下取整时
                str += " ";  # 给str添加空格
            str += i;  # 将i添加到str的末尾
        }
        print(str + "\n");  # 打印str并换行
        print(tab(Math.floor(l + 1) / (2 * s2) - 2) + "SECONDS\n");  # 调用tab函数并打印结果，然后打印"SECONDS"并换行
    }
}

main();  # 调用main函数
```