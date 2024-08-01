# `.\DB-GPT-src\web\components\knowledge\doc-icon.tsx`

```py
import { FileTextFilled, FileWordTwoTone, IeCircleFilled } from '@ant-design/icons';

// 默认导出的函数组件，根据传入的 type 参数渲染不同的文档图标
export default function DocIcon({ type }: { type: string }) {
    // 如果 type 参数为 'TEXT'，返回文本文件图标
    if (type === 'TEXT') {
        return <FileTextFilled className="text-[#2AA3FF] mr-2 !text-lg" />;
    } else if (type === 'DOCUMENT') {
        // 如果 type 参数为 'DOCUMENT'，返回文档文件图标
        return <FileWordTwoTone className="text-[#2AA3FF] mr-2 !text-lg" />;
    } else {
        // 其他情况下，返回未知文件类型的默认图标
        return <IeCircleFilled className="text-[#2AA3FF] mr-2 !text-lg" />;
    }
}
```