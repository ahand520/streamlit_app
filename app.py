import os
import json
import streamlit as st
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# 載入 JSON 資料
def load_json_data(filename):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def browse_json():
    st.title("JSON 問題瀏覽")
    uploaded_file = st.sidebar.file_uploader("上傳 JSON 檔案", type=["json"])
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
    for ctx in item.get("context_text", []):
        st.text(ctx)
        st.markdown("-------------------------------------------")

def single_qa():
    st.title("單次問答")
    question = st.text_input("請輸入您的問題")
    if st.button("提交"):
        if not question:
            st.warning("請輸入問題")
            return
        with st.spinner("搜尋中..."):
            # 使用 OpenAI 進行 embedding
            embeddings = OpenAIEmbeddings(
                openai_api_key=st.secrets["OPENAI_API_KEY"],
                model="text-embedding-3-large"
            )
            db_path = os.path.join(
                os.path.dirname(__file__),
                "vector_db",
                "text-embedding-3-large_c1000_o200"
            )
            store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            # 取得文件與相似度分數
            results = store.similarity_search_with_score(question, k=15)
            # 格式化結果
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
            # 組成 context_text，包括來源與頁碼範圍
            context_text = "\n\n".join([
                f"========【以下文字來源: {item['metadata'].get('source', '未知')}, 範圍: {item['metadata'].get('page_range', '')}】========\n{item['content']} \n\n ================"
                for item in formatted_results
            ])
            # 組成 prompt
            prompt = f"""根據以下相關文字內容，回答使用者的問題。
只使用提供的文字內容來回答問題，如果文字內容中沒有相關資訊，請說明無法回答。
請使用繁體中文回答，請針對問題中的各種可能答案，利用參考的文字內容提供詳盡、完整、條理清晰的回覆。
請盡量以提供的文字內容作為說明，不要進行過多的修改。
請利用相關文字內容中實際有出現的來源，提供引用來源註記（格式: 來源: 檔案名, 範圍: 頁碼範圍】）標示參考到的相關文字內容之資料來源。

相關文字內容：
{context_text}

使用者問題：{question}
"""
            # 呼叫 OpenAI ChatCompletion
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.0
            )
            answer = response.choices[0].message.content
            
        st.header("回答")
        st.write(answer)
        st.subheader("使用的 Prompt")
        st.code(prompt)

def main():
    mode = st.sidebar.selectbox("選擇功能", ["JSON 問題瀏覽", "單次問答"])
    if mode == "JSON 問題瀏覽":
        browse_json()
    else:
        single_qa()

if __name__ == "__main__":
    main()
