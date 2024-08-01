# `.\DB-GPT-src\web\components\chat\doc-upload.tsx`

```py
// 导入所需模块和库
import { ChatContext } from '@/app/chat-context';
import { apiInterceptors, uploadDocument } from '@/client/api';

// 导入自定义 hooks 和 Ant Design 相关组件
import useSummary from '@/hooks/use-summary';
import { PaperClipOutlined } from '@ant-design/icons';
import { Button, Upload } from 'antd';
import React, { useContext, useState } from 'react';

// 定义组件接收的 props 类型
interface IProps {
  className?: string;        // 可选的类名
  handleFinish?: (data: boolean) => void;   // 处理完成时的回调函数
  onUploadFinish: () => void;  // 上传完成时的回调函数
}

// 导出组件
export default function DocUpload(props: IProps) {
  // 使用 useContext 获取 ChatContext 中的数据
  const { dbParam, setDocId } = useContext(ChatContext);
  // 从 props 中解构出需要使用的回调函数
  const { onUploadFinish, handleFinish } = props;
  // 使用自定义 hook 获取摘要数据
  const summary = useSummary();
  // 定义 loading 状态及其更新函数
  const [loading, setLoading] = useState<boolean>(false);

  // 处理上传事件的函数
  const handleUpload = async (data: any) => {
    setLoading(true);  // 设置 loading 状态为 true

    // 创建 FormData 对象，并添加上传的文件信息
    const formData = new FormData();
    formData.append('doc_name', data.file.name);
    formData.append('doc_file', data.file);
    formData.append('doc_type', 'DOCUMENT');

    // 调用 API 请求函数进行上传，并添加拦截器处理返回结果
    const res = await apiInterceptors(uploadDocument(dbParam || 'default', formData));

    // 若上传失败，则恢复 loading 状态为 false 并返回
    if (!res[1]) {
      setLoading(false);
      return;
    }

    // 设置 ChatContext 中的 docId
    setDocId(res[1]);
    // 调用上传完成的回调函数
    onUploadFinish();
    // 恢复 loading 状态为 false
    setLoading(false);

    // 调用摘要生成函数并传入 docId
    await summary(res[1]);
    // 调用处理完成的回调函数，传入 false 表示完成
    handleFinish?.(false);
  };

  // 渲染上传组件
  return (
    <Upload
      customRequest={handleUpload}  // 指定自定义上传函数
      showUploadList={false}        // 不显示上传列表
      maxCount={1}                  // 最大上传数量为 1
      multiple={false}              // 不允许多选
      className="absolute z-10 top-2 left-2"  // 自定义样式类名
      accept=".pdf,.ppt,.pptx,.xls,.xlsx,.doc,.docx,.txt,.md"  // 允许上传的文件类型
    >
      {/* 渲染一个上传按钮，包含 loading 状态和图标 */}
      <Button loading={loading} size="small" shape="circle" icon={<PaperClipOutlined />}></Button>
    </Upload>
  );
}
```