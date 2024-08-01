# `.\DB-GPT-src\web\components\chat\header\excel-upload.tsx`

```py
import { PropsWithChildren, useContext, useState } from 'react';
import { Upload, UploadProps, Button, message, UploadFile, Tooltip } from 'antd';
import { LinkOutlined, SelectOutlined, UploadOutlined } from '@ant-design/icons';
import { apiInterceptors, postChatModeParamsFileLoad } from '@/client/api';
import { ChatContext } from '@/app/chat-context';

interface Props {
  convUid: string;
  chatMode: string;
  onComplete?: () => void;
}

function ExcelUpload({ convUid, chatMode, onComplete, ...props }: PropsWithChildren<Props & UploadProps>) {
  // 状态管理：控制上传过程中的 loading 状态
  const [loading, setLoading] = useState(false);
  // 使用 Ant Design 的 message 钩子，用于消息提示
  const [messageApi, contextHolder] = message.useMessage();
  // 状态管理：存储上传文件列表
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  // 状态管理：存储上传进度百分比
  const [percent, setPercent] = useState<number>();
  // 使用 useContext 获取 ChatContext 中的 model 数据
  const { model } = useContext(ChatContext);

  // 上传组件的 onChange 回调函数
  const onChange: UploadProps['onChange'] = async (info) => {
    if (!info) {
      // 如果 info 为空，提示用户选择文件
      message.error('Please select the *.(csv|xlsx|xls) file');
      return;
    }
    if (!/\.(csv|xlsx|xls)$/.test(info.file.name ?? '')) {
      // 如果文件类型不符合要求，提示用户文件必须为 csv、xlsx 或 xls 格式
      message.error('File type must be csv, xlsx or xls');
      return;
    }

    // 更新文件列表为当前选择的文件
    setFileList([info.file]);
  };

  // 触发上传动作的函数
  const onUpload = async () => {
    setLoading(true); // 设置 loading 状态为 true，显示上传中的状态
    try {
      const formData = new FormData();
      formData.append('doc_file', fileList[0] as any); // 将文件添加到 formData 中
      // 打开消息提示，显示上传中的文件名
      messageApi.open({ content: `Uploading ${fileList[0].name}`, type: 'loading', duration: 0 });
      // 发起上传请求，调用 apiInterceptors 函数
      const [err] = await apiInterceptors(
        postChatModeParamsFileLoad({
          convUid,
          chatMode,
          data: formData,
          model,
          config: {
            /** timeout 1h */ // 设置请求超时时间为 1 小时
            timeout: 1000 * 60 * 60,
            // 上传进度回调函数，更新上传进度百分比
            onUploadProgress: (progressEvent) => {
              const progress = Math.ceil((progressEvent.loaded / (progressEvent.total || 0)) * 100);
              setPercent(progress);
            },
          },
        }),
      );
      if (err) return; // 如果有错误，直接返回
      message.success('success'); // 提示上传成功
      onComplete?.(); // 如果定义了 onComplete 回调函数，则执行
    } catch (e: any) {
      // 捕获异常并提示用户上传错误
      message.error(e?.message || 'Upload Error');
    } finally {
      setLoading(false); // 上传结束，设置 loading 状态为 false
      messageApi.destroy(); // 销毁消息提示
    }
  };

  return (
    <div className="flex items-start gap-2">
        {/* 渲染上传文件状态的上下文组件 */}
        {contextHolder}
        {/* 提示信息：上传后文件不可更改 */}
        <Tooltip placement="bottom" title="File cannot be changed after upload">
            {/* 文件上传组件 */}
            <Upload
                // 如果处于加载状态，禁用上传功能
                disabled={loading}
                className="mr-1"
                // 在上传前阻止自动上传
                beforeUpload={() => false}
                // 允许上传的文件类型为 .csv、.xlsx 和 .xls
                accept=".csv,.xlsx,.xls"
                // 只允许单个文件上传
                multiple={false}
                // 上传文件变化时的回调函数
                onChange={onChange}
                // 不显示下载、预览和删除图标
                showUploadList={{
                    showDownloadIcon: false,
                    showPreviewIcon: false,
                    showRemoveIcon: false,
                }}
                // 自定义文件项的渲染方式
                itemRender={() => <></>}
                // 将额外的属性传递给 Upload 组件
                {...props}
            >
                {/* 上传文件的按钮 */}
                <Button className="flex justify-center items-center" type="primary" disabled={loading} icon={<SelectOutlined />}>
                    Select File
                </Button>
            </Upload>
        </Tooltip>
        {/* 执行上传操作的按钮 */}
        <Button
            type="primary"
            // 如果处于加载状态，显示加载状态
            loading={loading}
            className="flex justify-center items-center"
            // 如果文件列表为空，禁用上传按钮
            disabled={!fileList.length}
            icon={<UploadOutlined />}
            // 点击上传按钮时的回调函数
            onClick={onUpload}
        >
            {/* 根据加载状态和上传进度显示不同的文本 */}
            {loading ? (percent === 100 ? 'Analysis' : 'Uploading') : 'Upload'}
        </Button>
        {/* 如果文件列表不为空，显示文件链接 */}
        {!!fileList.length && (
            <div className="mt-2 text-gray-500 text-sm flex items-center">
                {/* 显示文件链接的图标 */}
                <LinkOutlined className="mr-2" />
                {/* 显示第一个文件的名称 */}
                <span>{fileList[0]?.name}</span>
            </div>
        )}
    </div>
}

export default ExcelUpload;
```