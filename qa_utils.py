import streamlit as st
import re

def strip_leading_analysis_prefix(s: str) -> str:
    """去除開頭的 'analysis' 前綴（大小寫寬鬆）"""
    return re.sub(r"^\s*analysis\s*", "", s, flags=re.IGNORECASE).strip()

def parse_reasoning_from_message(msg_obj):
    """同時相容舊欄位與新格式（content 內含 analysis / assistantfinal）。"""
    for key in ("reasoning_context", "reasoning_content", "reasoning"):
        val = getattr(msg_obj, key, None)
        if isinstance(val, str) and val.strip():
            return val.strip(), (getattr(msg_obj, "content", None) or "").strip()
    content = (getattr(msg_obj, "content", None) or "")
    if not content:
        return "", ""
    marker = "assistantfinal"
    i = content.find(marker)
    if i != -1:
        analysis_part = content[:i]
        final_part = content[i + len(marker):]
        analysis_part = re.sub(r"^\s*analysis\s*", "", analysis_part, flags=re.IGNORECASE).strip()
        final_part = final_part.strip()
        return analysis_part, final_part
    m = re.match(r"^\s*analysis\s*(.*)", content, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), ""
    return "", content.strip()

def handle_streaming_response_openai(response):
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
    in_final_mode = False
    stripped_analysis_prefix = False
    pending_content_buf = ""
    for chunk in response:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        key_found = None
        for key in ("reasoning_context", "reasoning_content", "reasoning"):
            val = getattr(delta, key, None)
            if val:
                key_found = key
                break
        if key_found:
            r = getattr(delta, key_found, None)
            if r:
                if not stripped_analysis_prefix:
                    r = strip_leading_analysis_prefix(r)
                    stripped_analysis_prefix = True
                reasoning_text += r
                reasoning_header_placeholder.header(f"推理過程（{len(reasoning_text)} 字）")
                reasoning_placeholder.write(reasoning_text)
        c = getattr(delta, "content", None)
        if c:
            if in_final_mode:
                answer_text += c
                answer_placeholder.write(answer_text)
            else:
                pending_content_buf += c
                pattern = re.compile(r"assistantfinal", re.IGNORECASE)
                m = pattern.search(pending_content_buf)
                if m:
                    before = pending_content_buf[:m.start()]
                    after = pending_content_buf[m.end():]
                    in_final_mode = True
                    pending_content_buf = ""
                    if not stripped_analysis_prefix:
                        before = strip_leading_analysis_prefix(before)
                        stripped_analysis_prefix = True
                    reasoning_text += before.strip()
                    if reasoning_text:
                        reasoning_header_placeholder.header(f"推理過程（{len(reasoning_text)} 字）")
                        reasoning_placeholder.write(reasoning_text)
                    answer_text += after
                    answer_placeholder.write(answer_text)
                else:
                    if len(pending_content_buf) >= 64 or "\n" in pending_content_buf:
                        tmp = pending_content_buf
                        if not stripped_analysis_prefix:
                            tmp = strip_leading_analysis_prefix(tmp)
                            stripped_analysis_prefix = True
                        reasoning_text = (tmp if not in_final_mode else reasoning_text)
                        reasoning_header_placeholder.header(f"推理過程（{len(reasoning_text)} 字）")
                        reasoning_placeholder.write(reasoning_text)
    if not in_final_mode and pending_content_buf:
        tail = pending_content_buf
        if not stripped_analysis_prefix:
            tail = strip_leading_analysis_prefix(tail)
        reasoning_text += tail.strip()
        if reasoning_text:
            reasoning_header_placeholder.header(f"推理過程（{len(reasoning_text)} 字）")
            reasoning_placeholder.write(reasoning_text)

def handle_non_streaming_response_openai(response):
    st.header("最終答案")
    answer_placeholder = st.empty()
    reasoning_text = ""
    answer_text = ""
    result = response
    if result.choices:
        msg_obj = result.choices[0].message
        reasoning_text, answer_text = parse_reasoning_from_message(msg_obj)
    if reasoning_text:
        st.subheader(f"推理過程（{len(reasoning_text)} 字）")
        st.write(reasoning_text)
    answer_placeholder.write(answer_text or "")

def handle_streaming_response_other(response):
    st.header("回答")
    answer_placeholder = st.empty()
    answer_text = ""
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                answer_text += delta.content
                answer_placeholder.write(answer_text)

def handle_non_streaming_response_other(response):
    st.header("回答")
    answer_placeholder = st.empty()
    answer_text = ""
    result = response
    if result.choices:
        msg_obj = result.choices[0].message
        if hasattr(msg_obj, "content") and msg_obj.content:
            answer_text = msg_obj.content
    answer_placeholder.write(answer_text)
