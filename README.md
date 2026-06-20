 # 知识库问答助手
 
 基于 LangChain 框架和 RAG（检索增强生成）技术的本地私有知识库问答系统。
 支持 PDF / TXT / Markdown / DOCX / CSV 多种文档格式，数据完全本地化。
 
 ## 功能特点
 
 - 多格式文档支持：PDF / TXT / Markdown / DOCX / CSV
 - 本地向量存储：基于 ChromaDB，数据完全本地化
 - 中英文双语嵌入：使用 BAAI/bge-small-zh-v1.5 模型
 - 双 LLM 后端：支持 DeepSeek API 和本地 Ollama 模型
 - 流式回答：实时生成答案（JSON 结构化输出）
 - 来源引用：每条回答均标明参考文档来源
 - Web 界面：基于 Streamlit，开箱即用
 - 安全性：可选登录认证、文件路径遍历防护、文件大小限制、PDF 魔数校验
 - 成本控制：每日 API 调用次数和 Token 上限限制
 - 技术细节隐藏：生产环境可关闭配置信息展示
 
 ## 环境要求
 
 - Python 3.10+
 - 至少 4GB 可用内存（建议 8GB 以使用本地 Ollama 模型）
 - （可选）Ollama — 使用本地 LLM 时安装
 
 ## 项目结构
 
 ```
 ├── config/
 │   └── settings.yaml          # 全局配置文件
 ├── data/
 │   └── documents/             # 文档存放目录
 ├── src/
 │   ├── app.py                 # Streamlit 入口
 │   ├── config_loader.py       # 配置加载模块
 │   ├── embedding/
 │   │   └── embedder.py        # Embedding 模型管理
 │   ├── generation/
 │   │   └── generator.py       # LLM 生成（DeepSeek/Ollama）
 │   ├── ingestion/
 │   │   ├── loader.py          # 文档加载器
 │   │   ├── splitter.py        # 文本分块器
 │   │   └── pipeline.py        # 入库流水线
 │   ├── rag/
 │   │   └── chain.py           # RAG 查询链
 │   └── retrieval/
 │       └── retriever.py       # 向量检索器
 ├── tests/
 │   ├── conftest.py            # 测试共享 fixture
 │   ├── test_config_loader.py  # 配置加载测试
 │   ├── test_embedding.py      # 嵌入模型测试
 │   ├── test_generator.py      # LLM 生成及配额测试
 │   ├── test_loader.py         # 文档加载分块测试
 │   ├── test_pipeline.py       # 入库流水线测试
 │   ├── test_rag_chain.py      # RAG 链测试
 │   └── test_retriever.py      # 检索器测试
 ├── .env.example               # 环境变量模板
 ├── .gitignore
 ├── requirements.txt
 └── README.md
 ```
 
 ## 快速开始
 
 ### 1. 安装依赖
 
 ```bash
 pip install -r requirements.txt
 ```
 
 ### 2. 配置密钥
 
 复制 `.env.example` 为 `.env`，填入 DeepSeek API Key：
 
 ```ini
 DEEPSEEK_API_KEY=sk-your-key-here
 ```
 
 注册 DeepSeek 开发者账号获取 API Key：[platform.deepseek.com](https://platform.deepseek.com)
 
 > `settings.yaml` 中 `deepseek.api_key` 会自动从环境变量 `DEEPSEEK_API_KEY` 注入，无需在配置文件中硬编码。
 
 ### 3. 放入文档
 
 将需要索引的文档放入 `data/documents/` 目录，支持格式：
 - PDF（含魔数校验，跳过无效文件）
 - TXT / Markdown
 - DOCX
 - CSV
 
 ### 4. 启动应用
 
 ```bash
 streamlit run src/app.py
 ```
 
 ### 5. 索引文档
 
 在左侧边栏点击「索引文档」按钮，系统将自动加载、分块、嵌入并存入向量库。
 
 ### 6. 开始提问
 
 在聊天输入框中输入问题，系统将检索相关文档内容并生成回答。
 
 ## 配置说明
 
 编辑 `config/settings.yaml` 可调整参数。
 
 | 配置项 | 说明 | 默认值 |
 |--------|------|--------|
 | `llm.provider` | LLM 后端（deepseek / ollama） | `ollama` |
 | `llm.max_daily_calls` | 每日最大 API 调用次数 | `100` |
 | `llm.max_daily_tokens` | 每日最大 Token 消耗 | `100000` |
 | `llm.deepseek.model` | DeepSeek 模型名 | `deepseek-chat` |
 | `llm.ollama.model` | Ollama 模型名 | `qwen3:14b` |
 | `embedding.model_name` | 嵌入模型 | `BAAI/bge-small-zh-v1.5` |
 | `embedding.max_file_size_mb` | 上传文件大小上限 | `50` |
 | `vector_store.top_k` | 检索返回文档数 | `5` |
 | `vector_store.use_mmr` | 使用 MMR 多样性搜索 | `true` |
 | `text_splitter.chunk_size` | 文本分块大小 | `512` |
 | `text_splitter.chunk_overlap` | 分块重叠大小 | `128` |
 | `ui.show_technical_details` | 侧边栏是否显示技术配置 | `false` |
 
 ## 身份认证
 
 系统默认关闭认证。启用后所有页面需要登录才能访问：
 
 ```bash
 # Windows PowerShell
 $env:APP_AUTH_ENABLED="true"
 $env:APP_USERNAME="your-username"
 $env:APP_PASSWORD="your-password"
 streamlit run src/app.py
 ```
 
 ## 切换为 Ollama 本地模型
 
 1. 安装 [Ollama](https://ollama.com) 并拉取模型：
    ```bash
    ollama pull qwen3:14b
    ```
 2. 修改 `config/settings.yaml` 中 `llm.provider` 为 `ollama`
 3. 重启应用
 
 ## 运行测试
 
 ```bash
 pytest tests/ -v
 ```
 
 测试覆盖配置加载、文档加载分块、嵌入模型、LLM 配额控制、RAG 链、入库流水线等模块。
 
 ## 技术栈
 
 | 组件 | 技术 |
 |------|------|
 | RAG 框架 | LangChain |
 | 嵌入模型 | BAAI/bge-small-zh-v1.5 (HuggingFace) |
 | 向量数据库 | ChromaDB |
 | LLM 后端 | DeepSeek API / Ollama |
 | 文档解析 | LangChain Community Loaders |
 | Web 界面 | Streamlit |
 | 开发语言 | Python 3.10+ |
 
 ## 许可
 
 MIT
