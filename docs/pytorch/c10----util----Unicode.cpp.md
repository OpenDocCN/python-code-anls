# `.\pytorch\c10\util\Unicode.cpp`

```
// 包含 C10 库中的 Unicode 头文件
#include <c10/util/Unicode.h>

// 定义 c10 命名空间
namespace c10 {

// 如果是在 Windows 平台下编译
#if defined(_WIN32)

// 将 UTF-8 编码的 std::string 转换为 UTF-16 编码的 std::wstring
std::wstring u8u16(const std::string& str) {
    // 如果输入字符串为空，则返回空的 std::wstring
    if (str.empty()) {
        return std::wstring();
    }
    
    // 计算转换后需要的宽字符数
    int size_needed = MultiByteToWideChar(
        CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), NULL, 0);
    // 检查转换是否成功
    TORCH_CHECK(size_needed > 0, "Error converting the content to Unicode");

    // 分配足够大小的 std::wstring
    std::wstring wstr(size_needed, 0);
    
    // 执行实际的 UTF-8 到 UTF-16 转换
    MultiByteToWideChar(
        CP_UTF8,
        0,
        str.c_str(),
        static_cast<int>(str.size()),
        &wstr[0],
        size_needed);
    
    // 返回转换后的 std::wstring
    return wstr;
}

// 将 UTF-16 编码的 std::wstring 转换为 UTF-8 编码的 std::string
std::string u16u8(const std::wstring& wstr) {
    // 如果输入宽字符串为空，则返回空的 std::string
    if (wstr.empty()) {
        return std::string();
    }
    
    // 计算转换后需要的字节大小
    int size_needed = WideCharToMultiByte(
        CP_UTF8,
        0,
        wstr.c_str(),
        static_cast<int>(wstr.size()),
        NULL,
        0,
        NULL,
        NULL);
    // 检查转换是否成功
    TORCH_CHECK(size_needed > 0, "Error converting the content to UTF8");

    // 分配足够大小的 std::string
    std::string str(size_needed, 0);
    
    // 执行实际的 UTF-16 到 UTF-8 转换
    WideCharToMultiByte(
        CP_UTF8,
        0,
        wstr.c_str(),
        static_cast<int>(wstr.size()),
        &str[0],
        size_needed,
        NULL,
        NULL);
    
    // 返回转换后的 std::string
    return str;
}

#endif // _WIN32

} // namespace c10
```