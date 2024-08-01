# `.\DB-GPT-src\web\components\agent\my-plugins.tsx`

```py
import { apiInterceptors, postAgentMy, postAgentUninstall, postAgentUpload } from '@/client/api';
import { IMyPlugin } from '@/types/agent';
import { useRequest } from 'ahooks';
import { Button, Card, Spin, Tag, Tooltip, Upload, UploadProps, message } from 'antd';
import { useCallback, useState } from 'react';
import MyEmpty from '../common/MyEmpty';
import { ClearOutlined, LoadingOutlined, UploadOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

function MyPlugins() {
  const { t } = useTranslation();  // 使用 i18n 国际化钩子函数获取 t 函数

  const [messageApi, contextHolder] = message.useMessage();  // 使用 Ant Design 的 message 钩子函数

  const [uploading, setUploading] = useState(false);  // 上传状态标志位，默认为 false
  const [isError, setIsError] = useState(false);  // 错误状态标志位，默认为 false
  const [actionIndex, setActionIndex] = useState<number | undefined>();  // 当前操作的插件索引

  const {
    data = [],  // 插件列表数据，默认为空数组
    loading,  // 加载状态
    refresh,  // 刷新数据的函数
  } = useRequest(async () => {
    const [err, res] = await apiInterceptors(postAgentMy());  // 调用 API 获取插件列表数据
    setIsError(!!err);  // 根据错误情况设置错误状态
    return res ?? [];  // 返回获取的数据或空数组
  });

  // 卸载插件的异步函数
  const uninstall = async (name: string, index: number) => {
    if (actionIndex) return;  // 如果已经有正在进行的操作，直接返回
    setActionIndex(index);  // 设置当前操作的插件索引
    const [err] = await apiInterceptors(postAgentUninstall(name));  // 调用 API 卸载插件
    message[err ? 'error' : 'success'](err ? 'failed' : 'success');  // 根据返回结果显示消息类型
    !err && refresh();  // 如果操作成功，则刷新插件列表
    setActionIndex(undefined);  // 清除当前操作的插件索引
  };

  // 渲染操作按钮的回调函数
  const renderAction = useCallback(
    (item: IMyPlugin, index: number) => {
      if (index === actionIndex) {  // 如果当前索引与操作索引相同，显示加载中图标
        return <LoadingOutlined />;
      }
      return (
        <Tooltip title="Uninstall">  // 提示用户功能为卸载
          <div
            className="w-full h-full"
            onClick={() => {
              uninstall(item.name, index);  // 点击时执行卸载操作
            }}
          >
            <ClearOutlined />  // 卸载图标
          </div>
        </Tooltip>
      );
    },
    [actionIndex],  // 依赖于当前操作的插件索引
  );

  // 上传文件的 onChange 回调函数
  const onChange: UploadProps['onChange'] = async (info) => {
    if (!info) {  // 如果没有信息，提示用户选择正确的文件格式
      message.error('Please select the *.zip,*.rar file');
      return;
    }
    try {
      const file = info.file;  // 获取上传的文件信息
      setUploading(true);  // 设置上传状态为 true
      const formData = new FormData();  // 创建 FormData 对象
      formData.append('doc_file', file as any);  // 向表单数据中添加文件
      messageApi.open({ content: `Uploading ${file.name}`, type: 'loading', duration: 0 });  // 显示上传中的消息
      const [err] = await apiInterceptors(postAgentUpload(undefined, formData, { timeout: 60000 }));  // 调用 API 上传文件
      if (err) return;  // 如果上传失败，直接返回
      message.success('success');  // 显示上传成功消息
      refresh();  // 刷新插件列表
    } catch (e: any) {
      message.error(e?.message || 'Upload Error');  // 捕获并显示上传过程中的错误消息
    } finally {
      setUploading(false);  // 无论上传成功或失败，设置上传状态为 false
      messageApi.destroy();  // 销毁消息显示
    }
  };

  return (
    {/* 根据 loading 状态显示加载动画 */}
    <Spin spinning={loading}>
      {/* 显示全局消息提示器 */}
      {contextHolder}
      <div>
        {/* 文件上传组件 */}
        <Upload
          // 如果正在加载中，则禁用上传功能
          disabled={loading}
          className="mr-1"
          // 阻止文件自动上传
          beforeUpload={() => false}
          name="file"
          // 接受的文件类型限制为 .zip 和 .rar
          accept=".zip,.rar"
          // 仅允许单文件上传
          multiple={false}
          // 文件选择或移除时的回调函数
          onChange={onChange}
          // 自定义上传列表项的显示方式，不显示下载、预览和删除图标
          showUploadList={{
            showDownloadIcon: false,
            showPreviewIcon: false,
            showRemoveIcon: false,
          }}
          // 自定义上传列表项的渲染方式为空
          itemRender={() => <></>}
        >
          {/* 上传按钮，显示上传状态和图标 */}
          <Button loading={uploading} type="primary" icon={<UploadOutlined />}>
            {/* 上传按钮的文本内容，国际化 */}
            {t('Upload')}
          </Button>
        </Upload>
      </div>
      {/* 如果数据为空且不在加载状态下，则显示自定义的空状态组件 */}
      {!data.length && !loading && <MyEmpty error={isError} refresh={refresh} />}
      <div className="flex gap-2 md:gap-4">
        {/* 遍历数据数组，渲染每个项目卡片 */}
        {data.map((item, index) => (
          <Card className="w-full md:w-1/2 lg:w-1/3 xl:w-1/4" key={item.id} actions={[renderAction(item, index)]}>
            {/* 显示项目名称，支持鼠标悬停显示完整名称 */}
            <Tooltip title={item.name}>
              <h2 className="mb-2 text-base font-semibold line-clamp-1">{item.name}</h2>
            </Tooltip>
            {/* 如果项目有版本信息，则显示版本标签 */}
            {item.version && <Tag>v{item.version}</Tag>}
            {/* 如果项目有类型信息，则显示类型标签 */}
            {item.type && <Tag>Type {item.type}</Tag>}
            {/* 显示项目描述，支持鼠标悬停显示完整描述 */}
            <Tooltip title={item.description}>
              <p className="mt-2 line-clamp-2 text-gray-400 text-sm">{item.description}</p>
            </Tooltip>
          </Card>
        ))}
      </div>
    </Spin>
  );
}

export default MyPlugins;
```