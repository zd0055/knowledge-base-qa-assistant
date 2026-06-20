"""Knowledge Base Q&A Assistant - Streamlit Web Interface"""

import logging
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
from langchain_chroma import Chroma

from src.config_loader import load_config, resolve_path
from src.embedding import EmbeddingManager
from src.ingestion import IngestionPipeline
from src.rag import RAGChain
from src.retrieval import Retriever

logger = logging.getLogger(__name__)

st.set_page_config(page_title="知识库问答助手", page_icon=":mag:", layout="wide", initial_sidebar_state="expanded")


def check_auth():
    """Simple auth check using env vars."""
    auth_enabled = os.getenv("APP_AUTH_ENABLED", "false").lower() == "true"
    if not auth_enabled:
        st.session_state.authenticated = True
        return
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return
    st.title(":lock: 登录")
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", use_container_width=True)
        if submitted:
            expected_user = os.getenv("APP_USERNAME", "admin")
            expected_pass = os.getenv("APP_PASSWORD", "admin")
            if username == expected_user and password == expected_pass:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("用户名或密码错误")
        st.stop()

check_auth()


@st.cache_resource
def init_services():
    config = load_config()
    log_cfg = config.get("logging", {})
    logging.basicConfig(level=getattr(logging, log_cfg.get("level", "INFO")), format=log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    embedding_manager = EmbeddingManager(config)
    retriever = Retriever(config, embedding_manager)
    rag_chain = RAGChain(config, retriever)
    ingestion_pipeline = IngestionPipeline(config)
    return config, rag_chain, ingestion_pipeline, retriever, embedding_manager


def get_vector_store(embedding_manager, persist_dir):
    embedding_fn = embedding_manager.get_embeddings()
    if not Path(persist_dir).exists():
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(persist_directory=persist_dir, embedding_function=embedding_fn)


def safe_filename(filename):
    return Path(filename).name


if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = True
if "show_confirm_clear" not in st.session_state:
    st.session_state.show_confirm_clear = False


with st.sidebar:
    st.title(":mag: 知识库问答助手")
    try:
        config, rag_chain, ingestion_pipeline, retriever, embedding_manager = init_services()
        st.success(":white_check_mark: 服务已就绪")
    except Exception:
        st.error("服务初始化失败，请查看日志")
        logger.exception("init failed")
        st.stop()

    show_tech = config.get("ui", {}).get("show_technical_details", False)
    if show_tech:
        llm_cfg = config.get("llm", {})
        provider = llm_cfg.get("provider", "deepseek")
        model_name = llm_cfg.get(provider, {}).get("model", "unknown") if provider in llm_cfg else "unknown"
        embed_model = config.get("embedding", {}).get("model_name", "N/A")
        chunk_size_val = config.get("text_splitter", {}).get("chunk_size", 512)
        st.markdown("### 当前配置")
        st.markdown(f"- **LLM**: {provider} ({model_name})")
        st.markdown(f"- **嵌入模型**: {embed_model}")
        st.markdown(f"- **分块大小**: {chunk_size_val}")

    st.divider()
    st.markdown("### :inbox_tray: 文档管理")
    doc_dir = resolve_path(config, "documents.directory", "data/documents")
    persist_dir = resolve_path(config, "vector_store.persist_directory", "data/chroma_db")
    max_file_size_mb = config.get("embedding", {}).get("max_file_size_mb", 50)
    uploaded_files = st.file_uploader("上传文档到知识库", type=["pdf", "txt", "md", "docx", "csv"], accept_multiple_files=True)
    if uploaded_files:
        saved = 0
        os.makedirs(doc_dir, exist_ok=True)
        for uf in uploaded_files:
            safe_name = safe_filename(uf.name)
            file_path = Path(doc_dir) / safe_name
            file_bytes = uf.getbuffer()
            if len(file_bytes) > max_file_size_mb * 1024 * 1024:
                st.warning(f"已跳过 {uf.name}: 超过 {max_file_size_mb}MB 限制")
                continue
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            saved += 1
        if saved > 0:
            st.success(f"已保存 {saved} 个文件到 {doc_dir}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(":page_facing_up: 索引文档", use_container_width=True):
            with st.spinner("正在处理文档..."):
                try:
                    vs = get_vector_store(embedding_manager, persist_dir)
                    result = ingestion_pipeline.run(vs)
                    loaded = result["loaded"]
                    chunks_count = result["chunks"]
                    new_count = result["new"]
                    st.success(f"加载 {loaded} 文档 - {chunks_count} 分块 - {new_count} 新增")
                except Exception:
                    st.error("索引失败，请查看日志")
                    logger.exception("index failed")

    with col2:
        if st.button(":wastebasket: 清空知识库", use_container_width=True, type="secondary"):
            st.session_state.show_confirm_clear = True
        if st.session_state.get("show_confirm_clear", False):
            st.warning("确定要删除所有已索引的数据吗？")
            c1, c2 = st.columns(2)
            with c1:
                if st.button(":white_check_mark: 确认清空", use_container_width=True, type="primary"):
                    retriever.clear()
                    st.session_state.show_confirm_clear = False
                    st.success("知识库已清空")
                    st.rerun()
            with c2:
                if st.button(":x: 取消", use_container_width=True):
                    st.session_state.show_confirm_clear = False
                    st.rerun()

    st.divider()
    st.markdown("### :bar_chart: 知识库统计")
    stats = retriever.get_collection_stats()
    total_chunks_val = stats["total_chunks"]
    st.markdown(f"- **总片段数**: {total_chunks_val}")
    num_docs_val = stats["num_documents"]
    st.markdown(f"- **文档数**: {num_docs_val}")
    if stats["sources"]:
        with st.expander("查看文档列表"):
            for src, count in sorted(stats["sources"].items()):
                st.markdown(f"- {src} ({count} 片)")

    st.divider()
    st.caption("知识库问答助手 v1.0 | 本地私有 RAG 系统")
    if st.button(":key: 退出登录"):
        st.session_state.authenticated = False
        st.rerun()


st.title(":speech_balloon: 知识库问答")
st.caption("基于本地文档的智能问答系统 | 答案来源于已索引的文档内容")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(":satellite: 参考来源"):
                for i, src in enumerate(msg["sources"], 1):
                    src_name = src.get("source", "未知")
                    st.markdown(f"**[{i}]** {src_name}")
                    preview = src.get("content", "")[:200]
                    st.caption(preview + ("..." if len(src.get("content", "")) > 200 else ""))

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                result = rag_chain.query(prompt)
                full_answer = result["answer"]
                docs = result["sources"]
                st.markdown(full_answer)
                if docs:
                    sources_for_msg = []
                    with st.expander(":satellite: 参考来源", expanded=True):
                        for i, doc in enumerate(docs, 1):
                            source = doc.metadata.get("source", "未知")
                            content_preview = doc.page_content[:200]
                            sources_for_msg.append({"source": source, "content": doc.page_content})
                            st.markdown(f"**[{i}]** {source}")
                            st.markdown(content_preview + ("..." if len(doc.page_content) > 200 else ""))
                    st.session_state.messages.append({"role": "assistant", "content": full_answer, "sources": sources_for_msg})
                else:
                    st.info("没有检索到相关文档，请先上传并索引文档")
                    st.session_state.messages.append({"role": "assistant", "content": full_answer})
            except Exception:
                st.error("查询失败，请稍后重试")
                logger.exception("query failed")

if not st.session_state.messages:
    st.info(
        "**:speech_left: 开始使用**\n\n"
        "1. 在左侧上传您的文档（PDF/TXT/MD/DOCX/CSV）\n"
        "2. 点击「索引文档」将内容入库\n"
        "3. 在输入框中提问，系统将基于文档内容回答\n\n"
        ":bulb: **提示**: 首次使用需要先上传文档并索引，之后的提问都会自动在文档中检索相关信息。"
    )