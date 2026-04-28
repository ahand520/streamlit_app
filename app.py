import time
from qa_utils import (
    handle_streaming_response_openai,
    handle_non_streaming_response_openai,
    handle_streaming_response_other,
    handle_non_streaming_response_other,
)
import os
import json
import streamlit as st
from openai import BadRequestError, OpenAI,DefaultHttpxClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


env = st.secrets.get("RUN_ENV", "local")
# 取得變數，預設為 cloud
if env == "local":
    api_base = st.secrets.get("LOCAL_API_BASE", "http://localhost:8000")
    api_base_embedding = st.secrets.get("LOCAL_API_BASE", "http://localhost:8000")
    chat_model_options = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "google/gemma-3-27b-it",
        "google/gemma-3-1b-it",
        "mistralai/ministral-3-3b-instruct-2512",
        "mistralai/mistral-small-3.2-24b-instruct"
    ]
    chat_model = st.sidebar.selectbox("選擇 Chat 模型", chat_model_options, index=0)
    embedding_model = st.secrets.get("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
    api_key = st.secrets.get("VLLM_API_KEY", "EMPTY")
    api_key_embedding = st.secrets.get("VLLM_API_KEY", "EMPTY")
    check_embedding_ctx_length = False
    ssl_verify = False
elif env == "cloud":
    api_base_embedding = st.secrets.get("LOCAL_API_BASE", "http://localhost:8888/v1")
    api_base = st.secrets.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    # 模型選單
    chat_model_options = [
        "openai/gpt-oss-20b:free",
        "openai/gpt-oss-120b",
        "google/gemma-3-27b-it:free",
        "openai/gpt-4o",
        "mistralai/mistral-small-3.2-24b-instruct:free"
    ]
    chat_model = st.sidebar.selectbox("選擇 Chat 模型", chat_model_options, index=0)
    embedding_model = st.secrets.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    api_key_embedding = st.secrets.get("OPENAI_API_KEY", "EMPTY")
    api_key = st.secrets.get("OPENROUTER_API_KEY", "EMPTY")
    check_embedding_ctx_length = True
    ssl_verify = True
else:
    api_base_embedding = st.secrets.get("LOCAL_API_BASE", "http://localhost:8888/v1")
    api_base = st.secrets.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    # 模型選單
    chat_model_options = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "google/gemma-4-26b-a4b-it"
    ]
    chat_model = st.sidebar.selectbox("選擇 Chat 模型", chat_model_options, index=0)
    embedding_model = st.secrets.get("OPENAI_EMBEDDING_MODEL", "google/embeddinggemma-300m")
    api_key_embedding = st.secrets.get("VLLM_API_KEY", "EMPTY")
    api_key = st.secrets.get("OPENROUTER_API_KEY", "EMPTY")
    check_embedding_ctx_length = False
    ssl_verify = False
    # 設定代理伺服器
    proxy = st.secrets.get('proxy', 'http://sproxy.cht.com.tw:8080')
    os.environ['http_proxy'] = proxy
    os.environ['https_proxy'] = proxy
    # 指定不走 proxy 的 domain 或 IP
    no_proxy_list = [
        'jkm-llm.nat.fia.gov.tw',  # 只寫 domain，不要加 https:// 與路徑
        'localhost',
        '127.0.0.1',
        '10.97.59.123'
    ]
    os.environ['no_proxy'] = ','.join(no_proxy_list)

def single_qa():
    # 新增 stream 選項
    stream_response = st.sidebar.selectbox("回應模式", ["串流 (stream=True)", "一次輸出 (stream=False)"], index=0)
    use_stream = stream_response == "串流 (stream=True)"
    is_gemma_reasoning_model = chat_model == "google/gemma-4-26b-a4b-it"
    is_gpt_oss_reasoning_model = chat_model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
    is_reasoning_mandatory_model = is_gemma_reasoning_model
    reasoning_enabled = st.sidebar.checkbox(
        "啟用 reasoning",
        value=True,
        disabled=True,
        help="google/gemma-4-26b-a4b-it 的目前 endpoint 強制啟用 reasoning；其他模型目前也固定為啟用。",
    )
    if is_gemma_reasoning_model:
        st.sidebar.caption("google/gemma-4-26b-a4b-it 目前無法關閉 reasoning。")
    st.title("單次問答")
    question = st.text_input("請輸入您的問題")
    # 選擇要搜尋的 Top-k 文件數
    top_k = st.sidebar.selectbox("選擇相似文件數量 (Top-k)", [5, 6, 10, 15], index=1)
    # 允許使用者在側邊欄自訂 Prompt 範本
    st.sidebar.header("Prompt 設定")
    default_prompt = """相關文字內容：
``` 
{context_text}
```

使用者問題：{question}

指令：
你是一個知識管理系統搜尋助理，根據「相關文字內容」，回答使用者的問題。回答問題時請注意以下重點：
1.只使用提供的「相關文字內容」來回答問題，如果文字內容中沒有相關資訊，請說明無法回答，沒有明確關聯之文字也不要引用。
2.使用繁體中文回答，針對問題中的各種可能答案，利用參考的文字內容提供詳盡、完整、條理清晰的回覆
3.說明盡量以原始內容進行引述，不要修改原始文字以避免表達錯誤。
4.說明時附上其提供之來源標記，格式：(來源編號： ,來源檔案：xxxxx, 頁數起迄： )。
"""
    # 加入 key 以確保唯一性，並顯示在 Prompt 設定區塊下
    custom_prompt = st.sidebar.text_area(
        label="修改 Prompt 範本",
        value=default_prompt,
        height=300,
        key="custom_prompt_template"
    )
    # 在側邊欄選擇 Vector DB 資料夾，並僅於切換時載入
    db_base = os.path.join(os.path.dirname(__file__), "vector_db")
    folder_name_map = {
        "embeddinggemma-300m_c1000_o200_all": "全部-embeddinggemma",
    }
    try:
        db_folders = [name for name in os.listdir(db_base) if os.path.isdir(os.path.join(db_base, name))]
    except FileNotFoundError:
        db_folders = []
    display_names = [folder_name_map.get(name, name) for name in db_folders]
    display_to_folder = {folder_name_map.get(name, name): name for name in db_folders}
    # 用 session_state 儲存已載入的 vector db
    if "selected_db_display" not in st.session_state:
        st.session_state["selected_db_display"] = display_names[0] if display_names else ""
    selected_display = st.sidebar.selectbox("選擇向量資料庫資料夾", display_names, key="db_select_box")
    selected_db = display_to_folder.get(selected_display, "")
    # 若切換資料夾，重新載入 vector db
    if ("vector_store" not in st.session_state) or (st.session_state["selected_db_display"] != selected_display):
        if selected_db:
            with st.spinner(f"正在載入向量資料庫：{selected_display} ..."):
                http_client=DefaultHttpxClient(verify=ssl_verify)
                embeddings = OpenAIEmbeddings(
                    check_embedding_ctx_length = check_embedding_ctx_length,
                    openai_api_base = api_base_embedding,
                    openai_api_key= api_key_embedding,
                    model= embedding_model,
                    http_client=http_client
                )
                db_path = os.path.join(os.path.dirname(__file__), "vector_db", selected_db)
                store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
                st.session_state["vector_store"] = store
                st.session_state["selected_db_display"] = selected_display
        else:
            st.session_state["vector_store"] = None
    store = st.session_state.get("vector_store", None)
    # reasoning_effort 僅在 gpt-oss 模型可調整
    reasoning_effort = st.sidebar.selectbox(
        "推理細緻度 (reasoning_effort)",
        ["low", "medium", "high"],
        index=1,
        disabled=not is_gpt_oss_reasoning_model,
        help="僅 openai/gpt-oss-20b 與 openai/gpt-oss-120b 會使用此選項。",
    )

    if st.button("提交"):
        total_start_time = time.perf_counter()
        if not question:
            st.warning("請輸入問題")
            return
        if store is None:
            st.error("尚未載入向量資料庫，請先選擇並載入。")
            return
        with st.spinner("搜尋中..."):
            retrieval_start_time = time.perf_counter()
            # 取得文件與相似度分數
            results = store.similarity_search_with_score(question, k=top_k)
            #st.write("[DEBUG] similarity_search_with_score results:", results)
            # 格式化結果
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
            # 構建 context_items 與 context_text，使用 JSON 格式
            context_items = [
                {
                    "來源編號": idx + 1,
                    "文字內容": item["content"],
                    "來源標記": {
                        "來源檔案": item["metadata"].get("source", "未知"),
                        "頁數起迄": item["metadata"].get("page_range", "未知")
                    },
                    "相似度": float(item["score"])
                }
                for idx, item in enumerate(formatted_results)
            ]
            context_text = json.dumps(
                context_items, ensure_ascii=False, indent=2, separators=(',', ': ')
            )
            #st.write("[DEBUG] context_text:", context_text)
            # 組成 prompt，使用自訂或預設 Prompt 範本
            prompt = custom_prompt.format(context_text=context_text, question=question)
            retrieval_elapsed = time.perf_counter() - retrieval_start_time
            #st.write("[DEBUG] prompt:", prompt)
            http_client=DefaultHttpxClient(verify=ssl_verify)
            client = OpenAI(
                api_key = api_key,
                base_url= api_base, 
                http_client=http_client
            )
            msg = []
            msg.append({
                    "role": "system",
                    "content": f"你是一個有推理能力的知識管理系統搜尋助理，請根據相關文字內容回答問題。"
                })
                
            msg.append({"role": "user", "content": prompt})

            extra_body = {
                "provider": {"order": ["novita/bf16"]},
                "reasoning": {"enabled": True if is_reasoning_mandatory_model else reasoning_enabled},
            }
            if is_gpt_oss_reasoning_model:
                extra_body["reasoning"]["effort"] = reasoning_effort
           
            api_request_start_time = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model = chat_model,
                    messages= msg,
                    temperature=0.1,        
                    extra_body=extra_body,
                    stream=use_stream
                )
            except BadRequestError as exc:
                st.error(f"API 請求失敗：{exc}")
                return
            api_request_elapsed = time.perf_counter() - api_request_start_time

            # 回答顯示
            if chat_model.startswith("openai/gpt-oss") or chat_model.startswith("google/gemma-4-26b-a4b-it"):
                if use_stream:
                    response_metrics = handle_streaming_response_openai(
                        response,
                        request_start_time=api_request_start_time,
                    )
                else:
                    response_metrics = handle_non_streaming_response_openai(response)
            else:
                if use_stream:
                    response_metrics = handle_streaming_response_other(
                        response,
                        request_start_time=api_request_start_time,
                    )
                else:
                    response_metrics = handle_non_streaming_response_other(response)

            total_elapsed = time.perf_counter() - total_start_time

            timing_lines = [
                f"整體等待時間：{total_elapsed:.2f} 秒",
                f"檢索與提示組裝：{retrieval_elapsed:.2f} 秒",
                f"API 建立回應：{api_request_elapsed:.2f} 秒",
            ]
            if use_stream and response_metrics.get("first_token_latency") is not None:
                timing_lines.append(f"串流首字出現：{response_metrics['first_token_latency']:.3f} 秒")
                timing_lines.append(f"串流完整輸出：{response_metrics['response_render_time']:.2f} 秒")
            else:
                timing_lines.append(f"回應渲染時間：{response_metrics['response_render_time']:.2f} 秒")
            st.info("\n".join(timing_lines))
            st.subheader("使用的 Prompt")
            st.markdown(
                f"<div style='white-space:pre-wrap; word-break:break-all; font-family:monospace; background:#f0f0f0; padding:8px; border-radius:4px'>{prompt}</div>",
                unsafe_allow_html=True
            )
def main():
    single_qa()

if __name__ == "__main__":
    main()
