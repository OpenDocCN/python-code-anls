# `d:/src/tocomm/basic-computer-games\25_Chief\javascript\chief.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 input 元素，用于用户输入
// 在页面上打印问号，提示用户输入
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
# 结束函数定义

# 定义一个名为 tab 的函数，接受一个参数 space
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
    print(tab(30) + "CHIEF\n");  // 打印带有缩进的字符串 "CHIEF"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有缩进的字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("I AM CHIEF NUMBERS FREEK, THE GREAT INDIAN MATH GOD.\n");  // 打印字符串
    print("ARE YOU READY TO TAKE THE TEST YOU CALLED ME OUT FOR");  // 打印字符串
    a = await input();  // 等待用户输入并将结果存储在变量 a 中
    if (a.substr(0, 1) != "Y")  // 如果用户输入的第一个字符不是 "Y"
        print("SHUT UP, PALE FACE WITH WIE TONGUE.\n");  // 打印字符串
    print(" TAKE A NUMBER AND ADD 3. DIVIDE THIS NUMBER BY 5 AND\n");  // 打印字符串
    print("MULTIPLY BY 8. DIVIDE BY 5 AND ADD THE SAME. SUBTRACT 1.\n");  // 打印字符串
    print("  WHAT DO YOU HAVE");  // 打印字符串
    # 从输入中获取浮点数并赋值给变量b
    b = parseFloat(await input());
    # 根据数学表达式计算变量c的值
    c = (b + 1 - 5) * 5 / 8 * 5 - 3;
    # 打印带有计算结果的字符串
    print("I BET YOUR NUMBER WAS " + Math.floor(c + 0.5) + ". AM I RIGHT");
    # 从输入中获取值并赋给变量d
    d = await input();
    # 如果输入的值的第一个字符不是"Y"，则执行下面的代码块
    if (d.substr(0, 1) != "Y") {
        # 打印提示信息
        print("WHAT WAS YOUR ORIGINAL NUMBER");
        # 从输入中获取浮点数并赋值给变量k
        k = parseFloat(await input());
        # 计算f的值
        f = k + 3;
        # 计算g的值
        g = f / 5;
        # 计算h的值
        h = g * 8;
        # 计算i的值
        i = h / 5 + 5;
        # 计算j的值
        j = i - 1;
        # 打印一系列计算结果的字符串
        print("SO YOU THINK YOU'RE SO SMART, EH?\n");
        print("NOW WATCH.\n");
        print(k + " PLUS 3 EQUALS " + f + ". THIS DIVIDED BY 5 EQUALS " + g + ";\n");
        print("THIS TIMES 8 EQUALS " + h + ". IF WE DIVIDE BY 5 AND ADD 5,\n");
        print("WE GET " + i + ", WHICH, MINUS 1, EQUALS " + j + ".\n");
        print("NOW DO YOU BELIEVE ME");
        # 从输入中获取值并赋给变量z
        z = await input();
        # 如果输入的值的第一个字符不是"Y"，则执行下面的代码块
        if (z.substr(0, 1) != "Y") {
# 打印警告信息
print("YOU HAVE MADE ME MAD!!!\n")
print("THERE MUST BE A GREAT LIGHTNING BOLT!\n")
print("\n")
print("\n")
# 循环打印特定格式的字符串
for (x = 30; x >= 22; x--)
    print(tab(x) + "X X\n")
print(tab(21) + "X XXX\n")
print(tab(20) + "X   X\n")
print(tab(19) + "XX X\n")
# 循环打印特定格式的字符串
for (y = 20; y >= 13; y--)
    print(tab(y) + "X X\n")
print(tab(12) + "XX\n")
print(tab(11) + "X\n")
print(tab(10) + "*\n")
print("\n")
print("#########################\n")
print("\n")
# 打印警告信息
print("I HOPE YOU BELIEVE ME NOW, FOR YOUR SAKE!!\n")
# 返回函数
return
    }
    # 打印“BYE!!!”并换行
    print("BYE!!!\n");
}

# 调用主函数
main();
```