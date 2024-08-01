# `.\DB-GPT-src\web\components\knowledge\segmentation.tsx`

```py
import { apiInterceptors, getChunkStrategies, getDocumentList, syncBatchDocument } from '@/client/api';
import { File, IChunkStrategyResponse, ISyncBatchParameter, StepChangeParams } from '@/types/knowledge';
import { Alert, Button, Collapse, Form, Spin, message } from 'antd';
import Icon from '@ant-design/icons';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import StrategyForm from './strategy-form';
import { DoneIcon, PendingIcon, SyncIcon, FileError } from '@/components/icons';

type IProps = {
  spaceName: string;
  docType: string;
  handleStepChange: (params: StepChangeParams) => void;
  uploadFiles: Array<File>;
};

type FieldType = {
  fileStrategies: Array<ISyncBatchParameter>;
};

// 用于存储定时器的 ID
let intervalId: string | number | NodeJS.Timeout | undefined;

export default function Segmentation(props: IProps) {
  // 解构传入的 props
  const { spaceName, docType, uploadFiles, handleStepChange } = props;
  // 获取国际化翻译函数
  const { t } = useTranslation();
  // 创建表单实例
  const [form] = Form.useForm();
  // 管理上传的文件列表状态
  const [files, setFiles] = useState(uploadFiles);
  // 控制加载状态的状态钩子
  const [loading, setLoading] = useState<boolean>();
  // 存储获取的分块策略列表
  const [strategies, setStrategies] = useState<Array<IChunkStrategyResponse>>([]);
  // 存储同步状态，初始为空字符串
  const [syncStatus, setSyncStatus] = useState<string>('');

  // 异步函数，获取分块策略列表
  async function getStrategies() {
    setLoading(true);
    // 发起 API 请求获取分块策略
    const [, allStrategies] = await apiInterceptors(getChunkStrategies());
    setLoading(false);
    // 过滤出与当前文档类型相关的策略
    setStrategies((allStrategies || [])?.filter((i) => i.type.indexOf(docType) > -1));
  }

  // 组件挂载时获取分块策略，并在组件卸载时清除定时器
  useEffect(() => {
    getStrategies();
    return () => {
      // 清除定时器
      intervalId && clearInterval(intervalId);
    };
  }, []);

  // 处理表单提交
  const handleFinish = async (data: FieldType) => {
    // 检查提交参数的有效性
    if (checkParameter(data)) {
      setLoading(true);
      // 发起文档批量同步请求
      const [, result] = await apiInterceptors(syncBatchDocument(spaceName, data.fileStrategies));
      setLoading(false);
      if (result?.tasks && result?.tasks?.length > 0) {
        // 显示成功消息，并更新同步状态为运行中
        message.success(`Segemation task start successfully. task id: ${result?.tasks.join(',')}`);
        setSyncStatus('RUNNING');
        // 提取文档 ID 列表
        const docIds = data.fileStrategies.map((i) => i.doc_id);
        // 启动定时器，每 3 秒更新同步状态
        intervalId = setInterval(async () => {
          const status = await updateSyncStatus(docIds);
          if (status === 'FINISHED') {
            // 当同步完成时，清除定时器，更新同步状态为完成，并显示成功消息
            clearInterval(intervalId);
            setSyncStatus('FINISHED');
            message.success('Congratulation, All files sync successfully.');
            // 触发步骤变更回调函数
            handleStepChange({
              label: 'finish',
            });
          }
        }, 3000);
      }
    }
  };

  // 检查提交参数的有效性
  function checkParameter(data: FieldType) {
    let checked = true;
    // 如果当前同步状态为运行中，则阻止再次提交，并显示警告消息
    if (syncStatus === 'RUNNING') {
      checked = false;
      message.warning('The task is still running, do not submit it again.');
    }
    // 返回参数检查结果
    const { fileStrategies } = data;
    return checked;
  }
    fileStrategies.map((item) => {
      const name = item?.chunk_parameters?.chunk_strategy;
      // 检查是否存在 chunk_strategy，如果不存在，则设置为默认值 'Automatic'
      if (!name) {
        // 设置默认策略
        item.chunk_parameters = { chunk_strategy: 'Automatic' };
      }
      // 查找与 item.chunk_parameters.chunk_strategy 匹配的策略
      const strategy = strategies.filter((item) => item.strategy === name)[0];
      const newParam: any = {
        chunk_strategy: item?.chunk_parameters?.chunk_strategy,
      };
      // 如果存在匹配的策略并且该策略有参数，则更新参数
      if (strategy && strategy.parameters) {
        // 遍历策略的参数列表，将有效参数复制到 newParam 中
        strategy.parameters.forEach((param) => {
          const paramName = param.param_name;
          newParam[paramName] = (item?.chunk_parameters as any)[paramName];
        });
      }
      // 更新 item 的 chunk_parameters
      item.chunk_parameters = newParam;
    });
    // 返回最终的 checked
    return checked;
  }

  async function updateSyncStatus(docIds: Array<number>) {
    // 调用 apiInterceptors 获取文档列表，并解构获取 docs
    const [, docs] = await apiInterceptors(
      getDocumentList(spaceName, {
        doc_ids: docIds,
      }),
    );
    // 如果 docs 包含有效数据
    if (docs?.data && docs?.data.length > 0) {
      // 创建副本以避免直接修改 state
      const copy = [...files!];
      // 遍历 docs.data，更新 files 中匹配的文档状态
      docs?.data.map((doc) => {
        const file = copy?.filter((file) => file.doc_id === doc.id)?.[0];
        if (file) {
          file.status = doc.status;
        }
      });
      // 更新状态后的文件列表
      setFiles(copy);
      // 如果所有文档的状态均为 'FINISHED' 或 'FAILED'，则返回 'FINISHED'
      if (docs?.data.every((item) => item.status === 'FINISHED' || item.status === 'FAILED')) {
        return 'FINISHED';
      }
    }
  }

  function renderStrategy() {
    // 如果 strategies 不存在或为空数组，返回警告信息
    if (!strategies || !strategies.length) {
      return <Alert message={`Cannot find one strategy for ${docType} type knowledge.`} type="warning" />;
    }
    // 渲染表单列表，根据 docType 返回不同的 UI 结构
    return (
      <Form.List name="fileStrategies">
        {(fields) => {
          switch (docType) {
            // 对于 'TEXT' 或 'URL' 类型的文档，渲染 StrategyForm 组件
            case 'TEXT':
            case 'URL':
              return fields?.map((field) => (
                <StrategyForm strategies={strategies} docType={docType} fileName={files![field.name].name} field={field} />
              ));
            // 对于 'DOCUMENT' 类型的文档，使用 Collapse 渲染折叠面板
            case 'DOCUMENT':
              return (
                <Collapse defaultActiveKey={0} size={files.length > 5 ? 'small' : 'middle'}>
                  {fields?.map((field) => (
                    // 渲染折叠面板，显示文档名称及同步状态
                    <Collapse.Panel header={`${field.name + 1}. ${files![field.name].name}`} key={field.key} extra={renderSyncStatus(field.name)}>
                      <StrategyForm strategies={strategies} docType={docType} fileName={files![field.name].name} field={field} />
                    </Collapse.Panel>
                  ))}
                </Collapse>
              );
          }
        }}
      </Form.List>
    );
  }

  function renderSyncStatus(index: number) {
    // 获取指定索引处文件的同步状态
    const status = files![index].status;
    // 返回同步状态

    const status = files![index].status;
    // 返回文件的同步状态
    return status;
  }


这些注释涵盖了给定代码中每个语句的作用和功能描述，确保代码结构和逻辑清晰可读。
    switch (status) {
      case 'FINISHED':
        // 如果状态为 'FINISHED'，显示一个完成图标
        return <Icon component={DoneIcon} />;
      case 'RUNNING':
        // 如果状态为 'RUNNING'，显示一个旋转动画的同步图标
        return <Icon className="animate-spin animate-infinite" component={SyncIcon} />;
      case 'FAILED':
        // 如果状态为 'FAILED'，显示一个文件错误图标
        return <Icon component={FileError} />;
      default:
        // 默认情况下显示一个待处理图标
        return <Icon component={PendingIcon} />;
    }
  }

  return (
    <Spin spinning={loading}>
      <Form
        labelCol={{ span: 6 }}
        wrapperCol={{ span: 18 }}
        labelAlign="right"
        form={form}
        size="large"
        className="mt-4"
        layout="horizontal"
        name="basic"
        autoComplete="off"
        initialValues={{
          fileStrategies: files,
        }}
        onFinish={handleFinish}
      >
        {renderStrategy()}
        <Form.Item className="mt-4">
          <Button
            onClick={() => {
              // 处理点击返回按钮事件，触发步骤切换到前一个步骤
              handleStepChange({ label: 'back' });
            }}
            className="mr-4"
          >{`${t('Back')}`}</Button>
          <Button type="primary" htmlType="submit" loading={loading || syncStatus === 'RUNNING'}>
            {t('Process')}
          </Button>
        </Form.Item>
      </Form>
    </Spin>
  );
}



# 这行代码关闭了一个代码块。在程序中，关闭代码块通常用于结束一个函数、循环、条件语句或其他代码块的定义和执行。
```