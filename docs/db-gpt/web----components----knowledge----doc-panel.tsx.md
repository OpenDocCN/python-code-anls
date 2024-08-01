# `.\DB-GPT-src\web\components\knowledge\doc-panel.tsx`

```py
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button, Card, Space, Divider, Empty, Spin, Tag, Tooltip, Modal } from 'antd';
import { DeleteFilled, InteractionFilled, PlusOutlined, ToolFilled, EyeFilled, WarningOutlined, DeploymentUnitOutlined} from '@ant-design/icons';
import { apiInterceptors, delDocument, getDocumentList, syncDocument } from '@/client/api';
import { IDocument, ISpace } from '@/types/knowledge';
import moment from 'moment';
import ArgumentsModal from './arguments-modal';
import { useTranslation } from 'react-i18next';
import { useRouter } from 'next/router';
import DocIcon from './doc-icon';

// 接口声明，定义组件属性
interface IProps {
  space: ISpace;                          // 空间对象，包含空间信息
  onAddDoc: (spaceName: string) => void;  // 添加文档的回调函数，传入空间名称
  onDeleteDoc: () => void;                // 删除文档的回调函数
}

const { confirm } = Modal;

// 文档面板组件
export default function DocPanel(props: IProps) {
  const { space } = props;  // 从属性中解构空间对象
  const { t } = useTranslation();  // 获取国际化文本处理函数
  const router = useRouter();  // 获取路由对象
  const page_size = 18;  // 每页显示的文档数量

  // 状态管理
  const [isLoading, setIsLoading] = useState<boolean>(false);  // 是否加载中
  const [documents, setDocuments] = useState<any>([]);  // 文档列表
  const [argumentsShow, setArgumentsShow] = useState<boolean>(false);  // 是否显示参数模态框
  const [total, setTotal] = useState<number>(0);  // 文档总数
  const currentPageRef = useRef(1);  // 当前页数的引用

  // 计算属性，判断是否还有更多文档未加载
  const hasMore = useMemo(() => {
    return documents.length < total;
  }, [documents.length, total]);

  // 弹出确认删除对话框
  const showDeleteConfirm = (row: any) => {
    confirm({
      title: t('Tips'),  // 对话框标题，显示国际化文本
      icon: <WarningOutlined />,  // 对话框图标，警告图标
      content: `${t('Del_Document_Tips')}?`,  // 对话框内容，显示国际化文本
      okText: 'Yes',  // 确认按钮文本
      okType: 'danger',  // 确认按钮类型，危险按钮
      cancelText: 'No',  // 取消按钮文本
      async onOk() {  // 确认按钮点击回调函数，异步执行删除操作
        await handleDelete(row);
      },
    });
  };

  // 异步获取文档列表
  async function fetchDocuments() {
    setIsLoading(true);  // 设置加载状态为true
    const [_, data] = await apiInterceptors(  // 调用API拦截器，获取文档列表数据
      getDocumentList(space.name, {
        page: currentPageRef.current,  // 当前页数
        page_size,  // 每页显示数量
      }),
    );
    setDocuments(data?.data);  // 设置文档列表数据
    setTotal(data?.total || 0);  // 设置文档总数
    setIsLoading(false);  // 设置加载状态为false
  }

  // 加载更多文档
  const loadMoreDocuments = async () => {
    if (!hasMore) {  // 如果没有更多文档
      return;
    }
    setIsLoading(true);  // 设置加载状态为true
    currentPageRef.current += 1;  // 当前页数加1
    const [_, data] = await apiInterceptors(  // 调用API拦截器，获取更多文档数据
      getDocumentList(space.name, {
        page: currentPageRef.current,  // 当前页数
        page_size,  // 每页显示数量
      }),
    );
    setDocuments([...documents, ...data!.data]);  // 添加新加载的文档到文档列表
    setIsLoading(false);  // 设置加载状态为false
  };

  // 处理文档同步
  const handleSync = async (spaceName: string, id: number) => {
    await apiInterceptors(syncDocument(spaceName, { doc_ids: [id] }));  // 调用API拦截器，执行文档同步操作
  };

  // 处理文档删除
  const handleDelete = async (row: any) => {
    await apiInterceptors(delDocument(space.name, { doc_name: row.doc_name }));  // 调用API拦截器，执行文档删除操作
    fetchDocuments();  // 删除后重新获取文档列表
    props.onDeleteDoc();  // 调用父组件传入的删除文档回调函数
  };

  // 处理添加文档
  const handleAddDocument = () => {
    props.onAddDoc(space.name);  // 调用父组件传入的添加文档回调函数
  };

  // 处理显示参数模态框
  const handleArguments = () => {
    setArgumentsShow(true);  // 设置显示参数模态框状态为true
  };

  // 打开知识图谱可视化页面
  const openGraphVisualPage = () => {
    router.push(`/knowledge/graph/?spaceName=${space.name}`);  // 跳转到知识图谱可视化页面
  }

  // 渲染结果标签
  const renderResultTag = (status: string, result: string) => {
    let color;
    // 根据状态（status）选择合适的颜色（color）来显示标签（Tag）
    switch (status) {
      case 'TODO':
        color = 'gold';  // 如果状态为 'TODO'，设置颜色为金色
        break;
      case 'RUNNING':
        color = '#2db7f5';  // 如果状态为 'RUNNING'，设置颜色为浅蓝色
        break;
      case 'FINISHED':
        color = 'cyan';  // 如果状态为 'FINISHED'，设置颜色为青色
        break;
      case 'FAILED':
        color = 'red';  // 如果状态为 'FAILED'，设置颜色为红色
        break;
      default:
        color = 'red';  // 默认情况下，设置颜色为红色（防止未知状态）
        break;
    }
    // 返回一个带有提示工具的标签，提示内容为 result，颜色为根据状态确定的 color
    return (
      <Tooltip title={result}>
        <Tag color={color}>{status}</Tag>  // 显示状态文本，使用根据状态确定的颜色标签
      </Tooltip>
    );
  };

  useEffect(() => {
    fetchDocuments();  // 在组件加载或者 space 发生变化时，调用 fetchDocuments 函数
  }, [space]);

  const renderDocumentCard = () => {
    // 如果 documents 数组存在并且长度大于 0，则执行以下代码块
    if (documents?.length > 0) {
      // 返回一个包含多个 Card 组件的 div 元素，显示文档列表
      return (
        <div className="max-h-96 overflow-auto max-w-3/4">
          {/* 栅格布局，根据屏幕大小不同显示不同列数的卡片 */}
          <div className="mt-3 grid grid-cols-1 gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3 xl:gap-x-5">
            {/* 遍历 documents 数组中的每个 document 对象 */}
            {documents.map((document: IDocument) => {
              // 返回一个 Card 组件，展示文档的详细信息和操作按钮
              return (
                <Card
                  key={document.id} // 使用 document 的 id 作为唯一键
                  className=" dark:bg-[#484848] relative  shrink-0 grow-0 cursor-pointer rounded-[10px] border border-gray-200 border-solid w-full"
                  title={
                    // 文档标题部分，包含文件名和文件类型图标的 Tooltip 提示
                    <Tooltip title={document.doc_name}>
                      <div className="truncate ">
                        <DocIcon type={document.doc_type} /> {/* 根据文档类型显示不同的图标 */}
                        <span>{document.doc_name}</span> {/* 显示文档名 */}
                      </div>
                    </Tooltip>
                  }
                  extra={
                    // Card 右上角的操作按钮区域
                    <div className="mx-3">
                      {/* 查看按钮，点击后跳转到文档详情页面 */}
                      <Tooltip title={'detail'}>
                        <EyeFilled
                          className="mr-2 !text-lg"
                          style={{ color: '#1b7eff', fontSize: '20px' }}
                          onClick={() => {
                            router.push(`/knowledge/chunk/?spaceName=${space.name}&id=${document.id}`);
                          }}
                        />
                      </Tooltip>
                      {/* 同步按钮，点击后执行 handleSync 函数 */}
                      <Tooltip title={'Sync'}>
                        <InteractionFilled
                          className="mr-2 !text-lg"
                          style={{ color: '#1b7eff', fontSize: '20px' }}
                          onClick={() => {
                            handleSync(space.name, document.id);
                          }}
                        />
                      </Tooltip>
                      {/* 删除按钮，点击后显示删除确认对话框 */}
                      <Tooltip title={'Delete'}>
                        <DeleteFilled
                          className="text-[#ff1b2e] !text-lg"
                          onClick={() => {
                            showDeleteConfirm(document);
                          }}
                        />
                      </Tooltip>
                    </div>
                  }
                >
                  {/* Card 主体内容部分 */}
                  <p className="mt-2 font-semibold ">{t('Size')}:</p> {/* 显示文档大小标签 */}
                  <p>{document.chunk_size} chunks</p> {/* 显示文档的分块数 */}
                  <p className="mt-2 font-semibold ">{t('Last_Sync')}:</p> {/* 显示上次同步时间标签 */}
                  <p>{moment(document.last_sync).format('YYYY-MM-DD HH:MM:SS')}</p> {/* 格式化显示上次同步时间 */}
                  <p className="mt-2 mb-2">{renderResultTag(document.status, document.result)}</p> {/* 根据文档状态和结果渲染相应的标签 */}
                </Card>
              );
            })}
          </div>
          {/* 如果还有更多文档未加载，则显示加载更多按钮 */}
          {hasMore && (
            <Divider>
              <span className="cursor-pointer" onClick={loadMoreDocuments}>
                {t('Load_more')} {/* 加载更多按钮文本 */}
              </span>
            </Divider>
          )}
        </div>
      );
    }
    return (
      // 返回一个 JSX 元素，包含一个 Empty 组件和一个 Button 组件
      <Empty image={Empty.PRESENTED_IMAGE_DEFAULT}>
        {/* 创建一个带有加号图标的主按钮，点击时触发 handleAddDocument 函数 */}
        <Button type="primary" className="flex items-center mx-auto" icon={<PlusOutlined />} onClick={handleAddDocument}>
          Create Now
        </Button>
      </Empty>
    );
    };
    
    return (
      // 返回一个包含多个组件的 div 容器，设置了样式类名为 "collapse-container pt-2 px-4"
      <div className="collapse-container pt-2 px-4">
        <Space>
          {/* 添加数据源的按钮，点击时触发 handleAddDocument 函数 */}
          <Button size="middle" type="primary" className="flex items-center" icon={<PlusOutlined />} onClick={handleAddDocument}>
            {t('Add_Datasource')}
          </Button>
          {/* 展示参数的按钮，点击时触发 handleArguments 函数 */}
          <Button size="middle" className="flex items-center mx-2" icon={<ToolFilled />} onClick={handleArguments}>
            Arguments
          </Button>
          {/* 如果 space.vector_type 为 'KnowledgeGraph'，显示查看图谱的按钮，点击时触发 openGraphVisualPage 函数 */}
          {
            space.vector_type === 'KnowledgeGraph' && (<Button size="middle" className="flex items-center mx-2" icon={<DeploymentUnitOutlined />} onClick={openGraphVisualPage}>{t('View_Graph')}</Button>)
          }
        </Space>
        <Divider />
        {/* 当 isLoading 为 true 时显示加载中的 Spin 组件，否则渲染文档卡片的内容 */}
        <Spin spinning={isLoading}>{renderDocumentCard()}</Spin>
        {/* 显示参数模态框，传递 space、argumentsShow 和 setArgumentsShow 作为 props */}
        <ArgumentsModal space={space} argumentsShow={argumentsShow} setArgumentsShow={setArgumentsShow} />
      </div>
    );
}



# 这行代码是一个代码块的结束标志，匹配之前的代码块的开始标志 {
```