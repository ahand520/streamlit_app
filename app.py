import os
import json
import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 取得變數，預設為 cloud
env = st.secrets.get("RUN_ENV", "local")

if env == "local":
    api_base = st.secrets.get("LOCAL_API_BASE", "http://localhost:8000")
    chat_model = st.secrets.get("LOCAL_CHAT_MODEL", "google/gemma-3-1b-it")
    embedding_model = st.secrets.get("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
    api_key = st.secrets.get("VLLM_API_KEY", "EMPTY")
    check_embedding_ctx_length = False
else:
    api_base = st.secrets.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    chat_model = st.secrets.get("OPENAI_CHAT_MODEL", "gpt-4o")
    embedding_model = st.secrets.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    api_key = st.secrets.get("OPENAI_API_KEY", "EMPTY")
    check_embedding_ctx_length = True


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
4.說明時附上其提供之來源標記，格式：(來源編號： ,來源檔案：xxxxx)。
"""
    # 加入 key 以確保唯一性，並顯示在 Prompt 設定區塊下
    custom_prompt = st.sidebar.text_area(
        label="修改 Prompt 範本",
        value=default_prompt,
        height=300,
        key="custom_prompt_template"
    )
    # 在側邊欄選擇 Vector DB 資料夾
    db_base = os.path.join(os.path.dirname(__file__), "vector_db")
    # 建立英文資料夾名稱與中文名稱對照表
    folder_name_map = {
        "e5-mistral-7b-instruct_c1000_o200_all": "全部-e5-mistral",
        "multilingual-e5-large-instruct_c500_o100": "北區國稅局-e5-large",
        "e5-mistral-7b-instruct_c500_o100": "北區國稅局-e5-mistral",
        "text-embedding-3-large_c500_o100_ND": "北區國稅局-text-embedding-3-large"
    }
    try:
        db_folders = [name for name in os.listdir(db_base) if os.path.isdir(os.path.join(db_base, name))]
    except FileNotFoundError:
        db_folders = []
    # 只顯示有對應中文名稱的資料夾
    display_names = [folder_name_map.get(name, name) for name in db_folders]
    # 建立中文名稱到英文資料夾名稱的反查表
    display_to_folder = {folder_name_map.get(name, name): name for name in db_folders}
    selected_display = st.sidebar.selectbox("選擇向量資料庫資料夾", display_names)
    selected_db = display_to_folder[selected_display]
    if st.button("提交"):
        if not question:
            st.warning("請輸入問題")
            return
        with st.spinner("搜尋中..."):
            # 進行 embedding
            embeddings = OpenAIEmbeddings(
                check_embedding_ctx_length = check_embedding_ctx_length,
                openai_api_base = api_base,
                openai_api_key= api_key,
                model= embedding_model,                
                #show_progress_bar=True  # 顯示進度列
            )
            # 根據使用者選擇的子資料夾組成 db_path
            db_path = os.path.join(
                os.path.dirname(__file__),
                "vector_db",
                selected_db
            )
            store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            # 取得文件與相似度分數
            results = store.similarity_search_with_score(question, k=top_k)
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
            # 組成 prompt，使用自訂或預設 Prompt 範本
            prompt = custom_prompt.format(context_text=context_text, question=question)
            # 呼叫 OpenAI ChatCompletion
            client = OpenAI(
                api_key = api_key,
                base_url= api_base
            )
            response = client.chat.completions.create(
                model = chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                stream=True
            )
            def stream_answer():
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            #answer = response.choices[0].message.content
            
        st.header("回答")
        st.write(stream_answer)
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
