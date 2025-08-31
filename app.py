import os
import json
import streamlit as st
from openai import OpenAI,DefaultHttpxClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 取得變數，預設為 cloud
env = st.secrets.get("RUN_ENV", "local")

if env == "local":
    api_base = st.secrets.get("LOCAL_API_BASE", "http://localhost:8000")
    api_base_embedding = st.secrets.get("LOCAL_API_BASE", "http://localhost:8000")
    chat_model_options = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "google/gemma-3-27b-it",
        "mistralai/mistral-small-3.2-24b-instruct"
    ]
    chat_model = st.sidebar.selectbox("選擇 Chat 模型", chat_model_options, index=0)
    embedding_model = st.secrets.get("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
    api_key = st.secrets.get("VLLM_API_KEY", "EMPTY")
    api_key_embedding = st.secrets.get("VLLM_API_KEY", "EMPTY")
    check_embedding_ctx_length = False
    ssl_verify = False
elif env == "cloud":
    api_base_embedding = st.secrets.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_base = st.secrets.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    chat_model = st.secrets.get("OPENAI_CHAT_MODEL", "gpt-4o")
    embedding_model = st.secrets.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    api_key_embedding = st.secrets.get("OPENAI_API_KEY", "EMPTY")
    api_key = st.secrets.get("OPENAI_API_KEY", "EMPTY")
    check_embedding_ctx_length = True
    ssl_verify = True
else:
    api_base_embedding = st.secrets.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_base = st.secrets.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    # 模型選單
    chat_model_options = [
        "openai/gpt-oss-20b:free",
        "google/gemma-3-27b-it:free",
        "mistralai/mistral-small-3.2-24b-instruct:free"
    ]
    chat_model = st.sidebar.selectbox("選擇 Chat 模型", chat_model_options, index=0)
    embedding_model = st.secrets.get("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-large")
    api_key_embedding = st.secrets.get("OPENAI_API_KEY", "EMPTY")
    api_key = st.secrets.get("OPENROUTER_API_KEY", "EMPTY")
    check_embedding_ctx_length = True
    ssl_verify = True


# 載入 JSON 資料
def load_json_data(filename):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def browse_json():
    st.title("JSON 問題瀏覽")
    uploaded_file = st.sidebar.file_uploader("上傳 JSON 檔案 (檔名勿帶有中文)", type=["json"])
    if uploaded_file is None:
        st.info("請在側邊欄上傳 JSON 檔案")
        return
    try:
        data = json.load(uploaded_file)
    except Exception as e:
        st.error(f"讀取 JSON 失敗: {e}")
        return
    questions = [item.get("question", "") for item in data]
    if not questions:
        st.warning("此 JSON 檔案無任何問題")
        return
    selected_q = st.sidebar.selectbox("選擇問題", questions)
    idx = questions.index(selected_q)
    item = data[idx]
    st.header("問題")
    st.write(selected_q)
    st.header("回答")
    st.write(item.get("answer", ""))
    st.header("相關內容")
    st.write(item.get("context_texts", ""))
    #for ctx in item.get("context_text", []):
    #    st.text(ctx)
    #    st.markdown("-------------------------------------------")

def single_qa():
    # 新增 stream 選項
    stream_response = st.sidebar.selectbox("回應模式", ["串流 (stream=True)", "一次輸出 (stream=False)"], index=0)
    use_stream = stream_response == "串流 (stream=True)"
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
        "e5-mistral-7b-instruct_c1000_o200_all": "全部-e5-mistral",
        "text-embedding-3-large_c1000_o200_KS": "高雄國稅局-text-embedding-3-large",
        "text-embedding-3-large_c1000_o200_all": "全部-text-embedding-3-large"
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
    # reasoning_effort UI 選項
    reasoning_effort = st.sidebar.selectbox("推理細緻度 (reasoning_effort)", ["low", "medium", "high"], index=1)

    if st.button("提交"):
        if not question:
            st.warning("請輸入問題")
            return
        if store is None:
            st.error("尚未載入向量資料庫，請先選擇並載入。")
            return
        with st.spinner("搜尋中..."):
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
                    }
                }
                for idx, item in enumerate(formatted_results)
            ]
            context_text = json.dumps(
                context_items, ensure_ascii=False, indent=2, separators=(',', ': ')
            )
            #st.write("[DEBUG] context_text:", context_text)
            # 組成 prompt，使用自訂或預設 Prompt 範本
            prompt = custom_prompt.format(context_text=context_text, question=question)
            #st.write("[DEBUG] prompt:", prompt)
            http_client=DefaultHttpxClient(verify=ssl_verify)
            client = OpenAI(
                api_key = api_key,
                base_url= api_base, 
                http_client=http_client
            )
            msg = []
            if chat_model.startswith("openai/gpt-oss"):
                msg.append({
                    "role": "system",
                    "content": f"你是一個有推理能力的知識管理系統搜尋助理，請根據相關文字內容回答問題。reasoning_effort:{reasoning_effort}"
                })
            msg.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model = chat_model,
                messages= msg,
                temperature=0.0,                
                extra_body={
                    "reasoning": { "effort": reasoning_effort, "exclude": False },  # low/medium/high
                    "include_reasoning": True                              # 回傳 <think/> 區塊
                },
                stream=use_stream
            )
            import re
            def parse_reasoning_from_message(msg_obj):
                """同時相容舊欄位與新格式（content 內含 analysis / assistantfinal）。"""
                # 1) 先試舊欄位：reasoning_context / reasoning_content / reasoning
                for key in ("reasoning_context", "reasoning_content", "reasoning"):
                    val = getattr(msg_obj, key, None)
                    if isinstance(val, str) and val.strip():
                        return val.strip(), (getattr(msg_obj, "content", None) or "").strip()

                # 2) 若沒有舊欄位，就從 content 裡切 analysis / final
                content = (getattr(msg_obj, "content", None) or "")
                if not content:
                    return "", ""

                # 常見的新樣式：
                #   "analysis....\n\nassistantfinal...."
                # 有些模型會在最前面加 "analysis" 字頭
                # 我們用 'assistantfinal' 當分隔點，再把前段的 "analysis" 字頭拿掉
                marker = "assistantfinal"
                i = content.find(marker)
                if i != -1:
                    analysis_part = content[:i]
                    final_part = content[i + len(marker):]
                    # 去掉前段開頭的 "analysis" 字頭（大小寫寬鬆、前後空白）
                    analysis_part = re.sub(r"^\s*analysis\s*", "", analysis_part, flags=re.IGNORECASE).strip()
                    final_part = final_part.strip()
                    return analysis_part, final_part

                # 3) 若也找不到 marker，就嘗試以 "analysis" 開頭粗略切分
                m = re.match(r"^\s*analysis\s*(.*)", content, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    # 只有 analysis，沒有明確 final；此時把整段視為推理，final 留空
                    return m.group(1).strip(), ""

                # 4) 完全沒有可辨識的分段：把整段視為 final
                return "", content.strip()
            reasoning_key_used = None
            if chat_model.startswith("openai/gpt-oss"):
                if use_stream:
                    # UI 區塊
                    reasoning_container = st.container()
                    answer_container = st.container()
                    reasoning_text = ""
                    answer_text = ""

                    with reasoning_container:
                        reasoning_header_placeholder = st.empty()
                        reasoning_placeholder = st.empty()
                    with answer_container:
                        st.header("最終答案")
                        answer_placeholder = st.empty()

                    # 狀態旗標
                    reasoning_key_used = None            # 舊式 reasoning 欄位的實際鍵名
                    in_final_mode = False               # 是否已經遇到 'assistantfinal'
                    stripped_analysis_prefix = False    # 是否已經把推理開頭的 'analysis' 去掉
                    pending_content_buf = ""            # 用來暫存還未確定要丟哪邊的 content（在找到分隔符前）

                    # 用來在第一次出現推理內容時，去除 'analysis' 前綴（大小寫寬鬆）
                    def strip_leading_analysis_prefix(s: str) -> str:
                        return re.sub(r"^\s*analysis\s*", "", s, flags=re.IGNORECASE).strip()

                    # 主要串流處理
                    for chunk in response:
                        if not getattr(chunk, "choices", None):
                            continue
                        delta = chunk.choices[0].delta

                        # 1) 先吃舊欄位（若模型有回）
                        key_found = None
                        for key in ("reasoning_context", "reasoning_content", "reasoning"):
                            val = getattr(delta, key, None)
                            if val:
                                key_found = key
                                break
                        if key_found and reasoning_key_used is None:
                            reasoning_key_used = key_found

                        # 1a) 舊式 reasoning 欄位增量
                        if key_found:
                            r = getattr(delta, key_found, None)
                            if r:
                                if not stripped_analysis_prefix:
                                    # 保險：有些模型仍可能從 "analysis" 開頭
                                    r = strip_leading_analysis_prefix(r)
                                    stripped_analysis_prefix = True
                                reasoning_text += r
                                reasoning_header_placeholder.header(f"推理過程（{len(reasoning_text)} 字）")
                                reasoning_placeholder.write(reasoning_text)

                        # 2) 新式：推理與答案都混在 delta.content，以 'assistantfinal' 分隔
                        c = getattr(delta, "content", None)
                        if c:
                            if in_final_mode:
                                # 已進入 final 區段：持續累加到答案
                                answer_text += c
                                answer_placeholder.write(answer_text)
                            else:
                                # 還沒遇到分隔符：先放入暫存
                                pending_content_buf += c

                                # 嘗試尋找分隔符（大小寫不敏感）
                                # 例： "...analysis.....assistantfinal....."
                                pattern = re.compile(r"assistantfinal", re.IGNORECASE)
                                m = pattern.search(pending_content_buf)

                                if m:
                                    # 分成「推理片段」與「答案起始」
                                    before = pending_content_buf[:m.start()]
                                    after = pending_content_buf[m.end():]
                                    in_final_mode = True
                                    pending_content_buf = ""  # 已切開，清掉暫存

                                    # 推理片段：去除開頭 'analysis'
                                    if not stripped_analysis_prefix:
                                        before = strip_leading_analysis_prefix(before)
                                        stripped_analysis_prefix = True
                                    reasoning_text += before.strip()
                                    if reasoning_text:
                                        reasoning_header_placeholder.header(f"推理過程（{len(reasoning_text)} 字）")
                                        reasoning_placeholder.write(reasoning_text)

                                    # 從分隔符之後開始，都是答案
                                    answer_text += after
                                    answer_placeholder.write(answer_text)
                                else:
                                    # 還沒找到分隔符：把目前暫存視為正在產生的推理
                                    # 但為避免每一小塊都刷新整段，僅在「有合理長度或遇到換行」時刷新 UI
                                    # 你也可以改成每次都刷新（更即時、但更耗效能）
                                    if len(pending_content_buf) >= 64 or "\n" in pending_content_buf:
                                        tmp = pending_content_buf
                                        # 不要消耗 buffer，因為可能將來前段要併回（若永遠找不到分隔符，最後會把它當作推理）
                                        if not stripped_analysis_prefix:
                                            tmp = strip_leading_analysis_prefix(tmp)
                                            stripped_analysis_prefix = True
                                            # 注意：不改動 buffer 本身，以免日後切分位移
                                        reasoning_text = (tmp if not in_final_mode else reasoning_text)
                                        reasoning_header_placeholder.header(f"推理過程（{len(reasoning_text)} 字）")
                                        reasoning_placeholder.write(reasoning_text)

                    # 串流結束：若整段都沒遇到分隔符，就把暫存的 content 視為推理或答案
                    if not in_final_mode and pending_content_buf:
                        # 若到最後都沒有 'assistantfinal'，那就把暫存當成推理（去 analysis 前綴）
                        tail = pending_content_buf
                        if not stripped_analysis_prefix:
                            tail = strip_leading_analysis_prefix(tail)
                        reasoning_text += tail.strip()
                        if reasoning_text:
                            reasoning_header_placeholder.header(f"推理過程（{len(reasoning_text)} 字）")
                            reasoning_placeholder.write(reasoning_text)
                else:
                    # 一次性取得完整回應
                    st.header("最終答案")
                    answer_placeholder = st.empty()
                    reasoning_text = ""
                    answer_text = ""
                    # 呼叫 API 並取得完整 response
                    result = response
                    if result.choices:
                        msg_obj = result.choices[0].message
                        print(result)
                        reasoning_text, answer_text = parse_reasoning_from_message(msg_obj)
                    if reasoning_text:
                        st.subheader(f"推理過程（{len(reasoning_text)} 字）")
                        st.write(reasoning_text)
                    answer_placeholder.write(answer_text or "")
            else:
                if use_stream:
                    st.header("回答")
                    answer_placeholder = st.empty()
                    answer_text = ""
                    for chunk in response:
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                answer_text += delta.content
                                answer_placeholder.write(answer_text)
                else:
                    st.header("回答")
                    answer_placeholder = st.empty()
                    answer_text = ""
                    result = response
                    if result.choices:
                        msg_obj = result.choices[0].message
                        if hasattr(msg_obj, "content") and msg_obj.content:
                            answer_text = msg_obj.content
                    answer_placeholder.write(answer_text)
                if reasoning_key_used:
                    st.write(f"{reasoning_key_used}")
                else:
                    st.write("本次未取得推理欄位內容")

            st.subheader("使用的 Prompt")
            st.code(prompt)  

def main():
    mode = st.sidebar.selectbox("選擇功能", ["單次問答", "JSON 批次結果瀏覽"])
    if mode == "JSON 批次結果瀏覽":
        browse_json()
    else:
        single_qa()

if __name__ == "__main__":
    main()
