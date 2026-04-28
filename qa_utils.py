import streamlit as st
import re
import time

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

def handle_streaming_response_openai(response, request_start_time=None):
    stream_start = time.perf_counter()
    latency_start = request_start_time if request_start_time is not None else stream_start
    first_token_latency = None
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
    is_reasoning_seperate = False
    stripped_analysis_prefix = False
    pending_content_buf = ""
    combined_reasoning_mode = False

    def starts_with_analysis(text: str) -> bool:
        return bool(re.match(r"^\s*analysis\b", text, flags=re.IGNORECASE))

    for chunk in response:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        key_found = None
        for key in ("reasoning_context", "reasoning_content", "reasoning"):
            val = getattr(delta, key, None)
            if val:
                key_found = key
                if not is_reasoning_seperate:
                    is_reasoning_seperate = True
                break
        if key_found:
            r = getattr(delta, key_found, None)
            if r:
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - latency_start
                if not stripped_analysis_prefix:
                    r = strip_leading_analysis_prefix(r)
                    stripped_analysis_prefix = True
                reasoning_text += r
                reasoning_header_placeholder.header(f"推理過程（{len(reasoning_text)} 字）")
                reasoning_placeholder.write(reasoning_text)
        
        c = getattr(delta, "content", None)
        if c:
            if first_token_latency is None:
                first_token_latency = time.perf_counter() - latency_start
            if in_final_mode or is_reasoning_seperate:
                answer_text += c
                answer_placeholder.write(answer_text)
            else:
                pending_content_buf += c
                pattern = re.compile(r"assistantfinal", re.IGNORECASE)
                m = pattern.search(pending_content_buf)
                if m:
                    combined_reasoning_mode = True
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
                    if starts_with_analysis(pending_content_buf):
                        tmp = pending_content_buf
                        if not stripped_analysis_prefix:
                            tmp = strip_leading_analysis_prefix(tmp)
                            stripped_analysis_prefix = True
                        reasoning_text = (tmp if not in_final_mode else reasoning_text)
                        reasoning_header_placeholder.header(f"推理過程（{len(reasoning_text)} 字）")
                        reasoning_placeholder.write(reasoning_text)
                    elif len(pending_content_buf) >= 64 or "\n" in pending_content_buf:
                        in_final_mode = True
                        answer_text += pending_content_buf
                        pending_content_buf = ""
                        answer_placeholder.write(answer_text)
    if not in_final_mode and pending_content_buf:
        if combined_reasoning_mode or starts_with_analysis(pending_content_buf):
            tail = pending_content_buf
            if not stripped_analysis_prefix:
                tail = strip_leading_analysis_prefix(tail)
            reasoning_text += tail.strip()
            if reasoning_text:
                reasoning_header_placeholder.header(f"推理過程（{len(reasoning_text)} 字）")
                reasoning_placeholder.write(reasoning_text)
        else:
            answer_text += pending_content_buf
            answer_placeholder.write(answer_text)
    return {
        "first_token_latency": first_token_latency,
        "response_render_time": time.perf_counter() - stream_start,
    }

def handle_non_streaming_response_openai(response):
    render_start = time.perf_counter()
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
    return {
        "first_token_latency": None,
        "response_render_time": time.perf_counter() - render_start,
    }

def handle_streaming_response_other(response, request_start_time=None):
    stream_start = time.perf_counter()
    latency_start = request_start_time if request_start_time is not None else stream_start
    first_token_latency = None
    st.header("回答")
    answer_placeholder = st.empty()
    answer_text = ""
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - latency_start
                answer_text += delta.content
                answer_placeholder.write(answer_text)
    return {
        "first_token_latency": first_token_latency,
        "response_render_time": time.perf_counter() - stream_start,
    }

def handle_non_streaming_response_other(response):
    render_start = time.perf_counter()
    st.header("回答")
    answer_placeholder = st.empty()
    answer_text = ""
    result = response
    if result.choices:
        msg_obj = result.choices[0].message
        if hasattr(msg_obj, "content") and msg_obj.content:
            answer_text = msg_obj.content
    answer_placeholder.write(answer_text)
    return {
        "first_token_latency": None,
        "response_render_time": time.perf_counter() - render_start,
    }
