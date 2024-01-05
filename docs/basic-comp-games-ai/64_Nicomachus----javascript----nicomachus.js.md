# `64_Nicomachus\javascript\nicomachus.js`

```
# NICOMACHUS
# 
# Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
# 

# 定义一个打印函数，将字符串添加到输出元素中
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str))

# 定义一个输入函数，返回一个 Promise 对象
def input():
    var input_element
    var input_str

    return new Promise(function (resolve):
                       # 创建一个输入元素
                       input_element = document.createElement("INPUT")

                       # 在输出元素中打印提示符
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
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

var str;  # 声明变量str
var b;  # 声明变量b

// 主程序
async function main()
{
    print(tab(33) + "NICOMA\n");  # 在输出中打印"NICOMA"，并在前面添加33个空格
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 在输出中打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面添加15个空格
    print("\n");  # 打印一个空行
    print("\n");  # 打印一个空行
    print("\n");  # 打印一个空行
    print("BOOMERANG PUZZLE FROM ARITHMETICA OF NICOMACHUS -- A.D. 90!\n");  # 在输出中打印"BOOMERANG PUZZLE FROM ARITHMETICA OF NICOMACHUS -- A.D. 90!"
    while (1) {  # 进入无限循环
        print("\n");  # 打印一个空行
        print("PLEASE THINK OF A NUMBER BETWEEN 1 AND 100.\n");  # 在输出中打印"PLEASE THINK OF A NUMBER BETWEEN 1 AND 100."
        print("YOUR NUMBER DIVIDED BY 3 HAS A REMAINDER OF");  # 在输出中打印"YOUR NUMBER DIVIDED BY 3 HAS A REMAINDER OF"
        # 从用户输入中获取整数并赋值给变量a
        a = parseInt(await input());
        # 打印提示信息
        print("YOUR NUMBER DIVIDED BY 5 HAS A REMAINDER OF");
        # 从用户输入中获取整数并赋值给变量b
        b = parseInt(await input());
        # 打印提示信息
        print("YOUR NUMBER DIVIDED BY 7 HAS A REMAINDER OF");
        # 从用户输入中获取整数并赋值给变量c
        c = parseInt(await input());
        # 打印换行
        print("\n");
        # 打印提示信息
        print("LET ME THINK A MOMENT...\n");
        # 打印换行
        print("\n");
        # 根据给定的公式计算d的值
        d = 70 * a + 21 * b + 15 * c;
        # 当d大于105时，循环减去105，直到d小于等于105
        while (d > 105)
            d -= 105;
        # 打印结果
        print("YOUR NUMBER WAS " + d + ", RIGHT");
        # 无限循环，等待用户输入
        while (1) {
            # 从用户输入中获取字符串并赋值给变量str
            str = await input();
            # 打印换行
            print("\n");
            # 如果用户输入为"YES"，打印提示信息并跳出循环
            if (str == "YES") {
                print("HOW ABOUT THAT!!\n");
                break;
            # 如果用户输入为"NO"，打印提示信息
            } else if (str == "NO") {
                print("I FEEL YOUR ARITHMETIC IS IN ERROR.\n");
                break;  # 如果用户输入的是 YES 或 NO 之外的内容，跳出循环
            } else {
                print("EH?  I DON'T UNDERSTAND '" + str + "'  TRY 'YES' OR 'NO'.\n");  # 如果用户输入的不是 YES 或 NO，打印提示信息
            }
        }
        print("\n");  # 打印空行
        print("LET'S TRY ANOTHER.\n");  # 提示用户尝试另一个选项
    }
}

main();  # 调用主函数
```