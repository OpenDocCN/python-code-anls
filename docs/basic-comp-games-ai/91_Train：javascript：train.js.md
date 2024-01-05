# `d:/src/tocomm/basic-computer-games\91_Train\javascript\train.js`

```
# 定义一个名为print的函数，用于向页面输出文本
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str));

# 定义一个名为input的函数，用于获取用户输入
def input():
    var input_element;
    var input_str;

    # 返回一个Promise对象，用于处理异步操作
    return new Promise(function (resolve) {
                       # 创建一个input元素
                       input_element = document.createElement("INPUT");

                       # 在页面上输出提示符
                       print("? ");

                       # 设置input元素的类型为文本
                       input_element.setAttribute("type", "text");
// 设置输入框的长度为50
input_element.setAttribute("length", "50");
// 将输入框添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
// 让输入框获得焦点
input_element.focus();
// 初始化输入字符串为 undefined
input_str = undefined;
// 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    // 如果按下的是回车键
    if (event.keyCode == 13) {
        // 将输入框中的值赋给 input_str
        input_str = input_element.value;
        // 移除输入框
        document.getElementById("output").removeChild(input_element);
        // 打印输入的字符串
        print(input_str);
        // 打印换行符
        print("\n");
        // 解析输入的字符串
        resolve(input_str);
    }
});
// 结束添加事件监听器的语句块
// 结束函数定义
});

// 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    // 初始化一个空字符串
    var str = "";
    // 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串
}

// Main control section
async function main()
{
    print(tab(33) + "TRAIN\n");  // 打印带有缩进的字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有缩进的字符串
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("TIME - SPEED DISTANCE EXERCISE\n");  // 打印标题
    print("\n ");  // 打印空行
    while (1) {  // 进入无限循环
        c = Math.floor(25 * Math.random()) + 40;  // 生成随机数并赋值给变量c
        d = Math.floor(15 * Math.random()) + 5;  // 生成随机数并赋值给变量d
        t = Math.floor(19 * Math.random()) + 20;  // 生成随机数并赋值给变量t
        print(" A CAR TRAVELING " + c + " MPH CAN MAKE A CERTAIN TRIP IN\n");  // 打印带有变量的字符串
        print(d + " HOURS LESS THAN A TRAIN TRAVELING AT " + t + " MPH.\n");  // 打印带有变量的字符串
        # 打印提示信息，询问汽车行驶时间
        print("HOW LONG DOES THE TRIP TAKE BY CAR");
        # 从用户输入中获取汽车行驶时间并转换为浮点数
        a = parseFloat(await input());
        # 根据公式计算预期时间
        v = d * t / (c - t);
        # 计算误差百分比并取绝对值，四舍五入到整数
        e = Math.floor(Math.abs((v - a) * 100 / a) + 0.5);
        # 如果误差大于5%，打印提示信息
        if (e > 5) {
            print("SORRY.  YOU WERE OFF BY " + e + " PERCENT.\n");
        } else {
            # 如果误差在5%以内，打印提示信息
            print("GOOD! ANSWER WITHIN " + e + " PERCENT.\n");
        }
        # 打印正确答案
        print("CORRECT ANSWER IS " + v + " HOURS.\n");
        # 打印空行
        print("\n");
        # 询问用户是否继续
        print("ANOTHER PROBLEM (YES OR NO)\n");
        # 获取用户输入
        str = await input();
        # 打印空行
        print("\n");
        # 如果用户输入的第一个字符不是"Y"，跳出循环
        if (str.substr(0, 1) != "Y")
            break;
    }
}

main();
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```