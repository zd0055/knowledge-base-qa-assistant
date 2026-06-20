# 知识库问答助手

基于 LangChain 框架和 RAG（检索增强生成）技术的本地私有知识库问答系统。
支持 PDF / TXT / Markdown / DOCX / CSV 多种文档格式，数据完全本地化。

## 功能特点

- **多格式文档支持**：PDF / TXT / Markdown / DOCX / CSV
- **本地向量存储**：基于 ChromaDB，数据完全本地化，支持增量索引和去重
- **中英文双语嵌入**：使用 BAAI/bge-small-zh-v1.5 模型（国内 hf-mirror 自动回退）
- **双 LLM 后端**：支持 DeepSeek API 和本地 Ollama 模型
- **流式回答**：实时逐字生成答案
- **来源引用**：每条回答均标明参考文档来源及内容预览
- **Web 界面**：基于 Streamlit，开箱即用
- **HTTP REST API**：提供 POST /api/v1/query 和 GET /api/v1/health 端点
- **身份认证**：可选登录认证，环境变量配置用户名密码
- **文件安全**：路径遍历防护、文件大小限制、PDF 魔数校验
- **成本控制**：每日 API 调用次数和 Token 上限限制
- **文档去重**：基于内容 MD5 哈希的增量索引，避免重复入库
- **多样性搜索**：支持 MMR 最大边际相关性检索，提高结果多样性
- **知识库管理**：支持上传、索引、清空、查看统计和文档列表
- **技术细节隐藏**：生产环境可关闭侧边栏配置信息展示


## 环境要求

- Python 3.10+
- 至少 4GB 可用内存（建议 8GB 以使用本地 Ollama 模型）
- （可选）Ollama — 使用本地 LLM 时安装

## 快速开始

### 1. 安装依赖

```
pip install -r requirements.txt
```

### 2. 配置密钥

复制 `.env.example` 为 `.env`，填入 DeepSeek API Key：

```ini
DEEPSEEK_API_KEY=sk-your-key-here
```

注册 DeepSeek 开发者账号获取 API Key：[platform.deepseek.com](https://platform.deepseek.com)

> `settings.yaml` 中的敏感字段会自动从环境变量注入，无需硬编码。

### 3. 放入文档

将需要索引的文档放入 `data/documents/` 目录，支持格式：
- **PDF**（含魔数校验，跳过无效文件）
- **TXT / Markdown**
- **DOCX**
- **CSV**

### 4. 启动应用

#### Streamlit Web 界面（推荐）

```bash
streamlit run src/app.py
```

#### HTTP REST API 服务

```bash
python src/api.py
```

API 默认运行在 `http://localhost:8765`：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/query` | POST | 提问，返回答案与来源 |
| `/api/v1/health` | GET | 健康检查 |

请求示例：

```bash
curl -X POST http://localhost:8765/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 RAG？"}'
```

### 5. 索引文档

在左侧边栏点击「索引文档」按钮，系统将自动加载、分块、嵌入并存入向量库。
已索引的文档片段会自动去重（基于 MD5 内容哈希），不会重复入库。

### 6. 开始提问

在聊天输入框中输入问题，系统将检索相关文档内容并生成回答。

## 项目结构

```
├── config/
│   └── settings.yaml              # 全局配置文件
├── data/
│   ├── chroma_db/                 # 向量数据库持久化目录
│   └── documents/                 # 文档存放目录
├── src/
│   ├── api.py                     # HTTP REST API 服务
│   ├── app.py                     # Streamlit Web 入口
│   ├── config_loader.py           # 配置加载模块（YAML + 环境变量注入）
│   ├── embedding/
│   │   └── embedder.py            # Embedding 模型管理（镜像回退）
│   ├── generation/
│   │   └── generator.py           # LLM 生成（DeepSeek / Ollama + 配额控制）
│   ├── ingestion/
│   │   ├── loader.py              # 多格式文档加载器
│   │   ├── splitter.py            # 中文优化递归文本分块器
│   │   └── pipeline.py            # 入库管道（去重 + 增量索引）
│   ├── rag/
│   │   └── chain.py               # RAG 查询链（流式 + 同步）
│   └── retrieval/
│       └── retriever.py           # 向量检索器（MMR + 统计 + 清空）
├── tests/
│   ├── conftest.py                # 测试共享 fixture
│   ├── test_config_loader.py      # 配置加载测试
│   ├── test_embedding.py          # 嵌入模型测试
│   ├── test_generator.py          # LLM 生成及配额测试
│   ├── test_loader.py             # 文档加载分块测试
│   ├── test_pipeline.py           # 入库流水线测试
│   ├── test_rag_chain.py          # RAG 链测试
│   └── test_retriever.py          # 检索器测试
├── .env                           # 环境变量（gitignored）
├── .env.example                   # 环境变量模板
├── .gitignore
├── requirements.txt
└── README.md
```

## 配置说明

编辑 `config/settings.yaml` 可调整参数。以下为完整配置项：

### LLM 配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `llm.provider` | LLM 后端（deepseek / ollama） | `ollama` |
| `llm.max_daily_calls` | 每日最大 API 调用次数 | `100` |
| `llm.max_daily_tokens` | 每日最大 Token 消耗 | `100000` |
| `llm.deepseek.model` | DeepSeek 模型名 | `deepseek-chat` |
| `llm.deepseek.temperature` | DeepSeek 生成温度 | `0.1` |
| `llm.deepseek.max_tokens` | DeepSeek 最大 Token 数 | `4096` |
| `llm.deepseek.top_p` | DeepSeek 采样参数 | `0.9` |
| `llm.ollama.base_url` | Ollama 服务地址 | `http://localhost:11434` |
| `llm.ollama.model` | Ollama 模型名 | `qwen3:14b` |
| `llm.ollama.temperature` | Ollama 生成温度 | `0.1` |
| `llm.ollama.max_tokens` | Ollama 最大 Token 数 | `4096` |

### 嵌入与向量库

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `embedding.model_name` | 嵌入模型 | `BAAI/bge-small-zh-v1.5` |
| `embedding.device` | 推理设备 | `cpu` |
| `embedding.batch_size` | 批量嵌入大小 | `32` |
| `embedding.max_file_size_mb` | 上传文件大小上限 | `50` |
| `vector_store.persist_directory` | 向量库持久化路径 | `data/chroma_db` |
| `vector_store.top_k` | 检索返回文档数 | `5` |
| `vector_store.use_mmr` | 使用 MMR 多样性搜索 | `true` |
| `vector_store.mmr_fetch_multiplier` | MMR 候选倍数 | `2` |
| `vector_store.mmr_diversity` | MMR 多样性参数 | `0.3` |

### 文本分块

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `text_splitter.chunk_size` | 文本分块大小 | `512` |
| `text_splitter.chunk_overlap` | 分块重叠大小 | `128` |

### UI 与认证

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `ui.page_title` | 页面标题 | `知识库问答助手` |
| `ui.chat_history_length` | 聊天历史保留条数 | `50` |
| `ui.show_technical_details` | 侧边栏是否显示技术配置 | `false` |
| `auth.enabled` | 是否启用登录认证 | `true` |

### 文档与日志

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `documents.directory` | 文档存放目录 | `data/documents` |
| `logging.level` | 日志级别 | `INFO` |
| `logging.format` | 日志格式 | 见 settings.yaml |

## 身份认证

系统默认开启登录认证。启动应用后首先看到登录页面，输入正确的用户名和密码才能进入。

认证通过 `config/settings.yaml` 中的 `auth.enabled` 控制：

```yaml
auth:
  enabled: true          # true=需要登录，false=跳过登录
```

关闭登录后，退出登录按钮也不显示。

用户名和密码通过环境变量配置，编辑 `.env` 文件：

```ini
APP_USERNAME=admin       # 登录用户名
APP_PASSWORD=admin       # 登录密码
```

不设置则默认用户名密码均为 `admin`。

## 切换为 Ollama 本地模型

1. 安装 [Ollama](https://ollama.com) 并拉取模型：
   ```bash
   ollama pull qwen3:14b
   ```
2. 修改 `config/settings.yaml` 中 `llm.provider` 为 `ollama`
3. 重启应用

也可通过环境变量覆盖 Ollama 地址：

```ini
OLLAMA_BASE_URL=http://localhost:11434
```

## 运行测试

```bash
pytest tests/ -v
```

测试覆盖配置加载、文档加载分块、嵌入模型、LLM 配额控制、RAG 链、入库流水线等模块。

## 技术栈

| 组件 | 技术 |
|------|------|
| RAG 框架 | LangChain |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 (HuggingFace, hf-mirror 回退) |
| 向量数据库 | ChromaDB |
| LLM 后端 | DeepSeek API / Ollama (OpenAI 兼容协议) |
| 文档解析 | LangChain Community Loaders |
| Web 界面 | Streamlit |
| REST API | Python http.server |
| 开发语言 | Python 3.10+ |

## 许可

MIT
