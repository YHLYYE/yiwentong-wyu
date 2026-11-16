"""邑问通 Streamlit 演示界面：学生问答 + 辅导员工作台。

运行：
    cd 05-项目源码
    streamlit run ui/wyu_app.py

可选登录门：设置 WYU_DEMO_PASSWORD 后才会要求密码；不设置则直接进入。
问答日志默认写入 data/logs/qa_log.jsonl，可用 WYU_LOG_PATH 修改。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]  # 05-项目源码
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_streamlit_secrets() -> None:
    """把 Streamlit Cloud 的 secrets 注入环境变量，便于复用现有配置。"""
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in (
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL",
        "WYU_USE_DENSE",
        "WYU_OFFLINE",
        "WYU_DEVICE",
        "WYU_BM25_MIN",
        "WYU_DENSE_MIN",
        "WYU_COVERAGE_MIN",
        "WYU_MODEL_PATH",
        "WYU_LOG_PATH",
        "WYU_DEMO_PASSWORD",
    ):
        try:
            value = secrets[key]
        except Exception:
            continue
        if key not in os.environ and value != "":
            os.environ[key] = str(value)


load_streamlit_secrets()

from interaction_log import InteractionLog  # noqa: E402
from wyu_qa import HybridQA  # noqa: E402

SAMPLE_QUESTIONS = [
    "补考怎么报名",
    "缓考怎么申请",
    "国家奖学金怎么申请",
    "食堂几点开门",
]

BOUNDARY_IN = [
    "学业：补考、缓考、选课、校历、作息、成绩复核、教室借用、学籍异动、转专业、休学复学",
    "资助：国家助学奖学金、困难认定、资助细则、广东省资助政策",
]
BOUNDARY_OUT = [
    "就业：三方协议、档案、报到证、招聘、就业推荐表",
    "校园生活：校园卡、宿舍、食堂、校车、快递、WiFi、水电报修",
    "个人业务：查成绩、查学籍状态、查审批进度",
    "承诺类：能否通融、能否一定批准、帮我改成绩",
]


@st.cache_resource(show_spinner="正在加载知识库和检索模型…")
def get_qa() -> HybridQA:
    return HybridQA()


@st.cache_resource
def get_log() -> InteractionLog:
    path = os.environ.get(
        "WYU_LOG_PATH", str(ROOT / "data" / "logs" / "qa_log.jsonl")
    )
    return InteractionLog(path)


def login_gate() -> None:
    password = os.environ.get("WYU_DEMO_PASSWORD")
    if not password:
        return
    if st.session_state.get("authenticated"):
        return
    st.title("邑问通")
    st.caption("五邑大学学业与资助政策答疑智能体 · 有据才答，无据即拒")
    entered = st.text_input("演示密码", type="password")
    if st.button("进入"):
        if entered == password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("密码错误")
    st.stop()


def render_answer(qa: HybridQA, question: str, answer: str, bm25, dense):
    response_type = getattr(qa, "last_response_type", "answer")
    if response_type == "out_of_scope":
        st.warning("超范围问题")
    elif response_type == "missing_source":
        st.warning("资料库暂未收录")
    elif response_type == "stale_warning":
        st.info("资料可能过时")
    else:
        st.success("回答")
    st.markdown(answer)

    dense_text = (
        f"向量 {dense:.3f}"
        if qa.dense_available
        else "向量未加载（BM25-only）"
    )
    st.caption(
        f"BM25 {bm25:.2f} · {dense_text} · "
        f"覆盖度 {qa.last_coverage:.2f} · "
        f"判定 {qa.last_verdict or '-'} · {qa.mode}"
    )

    if qa.last_hits:
        with st.expander("检索到的资料", expanded=True):
            for hit in qa.last_hits:
                st.markdown(
                    f"- **{hit.chunk.source_doc}** "
                    f"（融合分 {hit.score:.4f}）"
                )
                snippet = hit.chunk.text.strip()
                if len(snippet) > 300:
                    snippet = snippet[:300] + "…"
                st.code(snippet, language="text")


def student_view(qa: HybridQA, log: InteractionLog) -> None:
    with st.sidebar:
        st.subheader("示例问题")
        for question in SAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True):
                st.session_state.pending_query = question
                st.session_state.ask = True
                st.rerun()

        st.divider()
        with st.expander("服务边界"):
            st.markdown("**范围内：**")
            for item in BOUNDARY_IN:
                st.markdown(f"- {item}")
            st.markdown("**范围外：**")
            for item in BOUNDARY_OUT:
                st.markdown(f"- {item}")
            st.caption("超范围会明确拒答，并给出对应部门/官网入口。")

    st.title("邑问通")
    st.caption(
        "五邑大学学业与资助政策答疑智能体 · "
        "有据才答，无据即拒；超出范围，给你指路。"
    )

    if st.session_state.get("pending_query"):
        st.session_state.query_value = st.session_state.pending_query
        st.session_state.pending_query = ""

    question = st.text_input(
        "请输入你的问题",
        key="query_value",
        placeholder="例如：补考怎么报名？",
    )
    ask = st.button("提问", type="primary")

    if ask or st.session_state.get("ask"):
        st.session_state.ask = False
        question = (question or "").strip()
        if not question:
            st.info("请先输入一个问题。")
            return

        with st.spinner("正在检索资料并生成回答…"):
            answer, bm25, dense = qa.answer(question)

        sources = [
            hit.chunk.source_doc for hit in qa.last_hits
        ]
        event = log.log(
            question=question,
            answer=answer,
            verdict=qa.last_verdict,
            bm25=bm25,
            dense=dense,
            coverage=qa.last_coverage,
            sources=sources,
            mode=qa.mode,
        )
        st.session_state.last_event = event
        render_answer(qa, question, answer, bm25, dense)

        event_id = event["id"]
        st.caption("这条回答对你有帮助吗？反馈会进入辅导员工作台，用于补知识库。")
        col1, col2, col3 = st.columns(3)
        if col1.button("有用", key=f"up_{event_id}"):
            log.feedback(event_id, "有用")
            st.success("已记录：有用")
        if col2.button("没用", key=f"down_{event_id}"):
            log.feedback(event_id, "没用")
            st.warning("已记录：没用，会进入知识缺口")
        if col3.button("需要人工", key=f"human_{event_id}"):
            log.feedback(event_id, "需要人工", escalated=True)
            st.info("已标记为需要人工，辅导员工作台可见")


def counselor_view(qa: HybridQA, log: InteractionLog) -> None:
    st.title("辅导员工作台")
    st.caption(
        "高频问题、未回答/知识缺口、用户反馈、转人工和 CSV 导出。"
    )

    summary = log.summary()
    total = summary["total"]
    answered = summary["answered"]
    refused = summary["refused"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总提问", total)
    col2.metric("已回答", answered)
    col3.metric("拒答/兜底", refused)
    col4.metric("已转人工", summary["escalated"])

    col5, col6 = st.columns(2)
    col5.metric("有用反馈", summary["positive"])
    col6.metric("没用反馈", summary["negative"])

    st.divider()
    st.subheader("高频问题")
    if summary["top_questions"]:
        st.table(
            [
                {"问题": question, "次数": count}
                for question, count in summary["top_questions"]
            ]
        )
    else:
        st.info("暂无提问记录。学生端提问后会自动出现在这里。")

    st.subheader("知识缺口 / 需要人工")
    gaps = summary["gaps"]
    if gaps:
        st.table(
            [
                {
                    "时间": event.get("ts", "")[:19],
                    "问题": event.get("question", ""),
                    "判定": event.get("verdict", ""),
                    "反馈": event.get("feedback", ""),
                    "状态": "已转人工" if event.get("escalated") else "待处理",
                    "来源": " | ".join(event.get("sources", [])),
                }
                for event in reversed(gaps)
            ]
        )
        options = {
            f"{event.get('ts', '')[:19]} | {event.get('question', '')[:30]}": event["id"]
            for event in reversed(gaps)
        }
        selected = st.selectbox("选择一条标记为已转人工", list(options))
        if st.button("标记为已转人工"):
            log.update(options[selected], escalated=True)
            st.success("已标记")
            st.rerun()
    else:
        st.info("暂无知识缺口。")

    st.subheader("最近提问")
    recent = summary["recent"]
    if recent:
        st.table(
            [
                {
                    "时间": event.get("ts", "")[:19],
                    "问题": event.get("question", ""),
                    "判定": event.get("verdict", ""),
                    "反馈": event.get("feedback", ""),
                    "模式": event.get("mode", ""),
                }
                for event in reversed(recent)
            ]
        )
        st.download_button(
            "下载问答日志 CSV",
            data=log.to_csv(log.read_all()),
            file_name="qa_log.csv",
            mime="text/csv",
        )
    else:
        st.info("暂无最近提问。")

    st.divider()
    st.subheader("知识库状态")
    col7, col8, col9 = st.columns(3)
    col7.metric("切片", len(qa.chunks))
    col8.metric("资料来源", len({c.source_doc for c in qa.chunks}))
    col9.metric("当前模式", qa.mode)
    if qa.dense_error:
        st.warning(f"稠密检索未加载：{qa.dense_error}")
    with st.expander("资料来源清单"):
        for source in sorted({c.source_doc for c in qa.chunks}):
            st.markdown(f"- {source}")

    st.caption(
        "隐私提示：日志只记录问题、答案、来源和反馈；"
        "请勿在问题中输入姓名、学号、身份证等个人敏感信息。"
    )


def main() -> None:
    st.set_page_config(
        page_title="邑问通",
        page_icon="🎓",
        layout="wide",
    )
    login_gate()

    try:
        qa = get_qa()
    except Exception as exc:
        st.error(f"知识库加载失败：{type(exc).__name__}: {exc}")
        st.stop()

    log = get_log()

    with st.sidebar:
        st.header("邑问通")
        st.caption("五邑大学学业与资助政策答疑智能体")
        view = st.radio("视图", ["学生问答", "辅导员工作台"])
        st.divider()
        st.metric("知识库切片", len(qa.chunks))
        st.metric(
            "资料来源",
            len({chunk.source_doc for chunk in qa.chunks}),
        )
        st.caption(f"当前模式：{qa.mode}")
        if qa.dense_error:
            st.warning(f"稠密检索未加载：{qa.dense_error}")

    if view == "辅导员工作台":
        counselor_view(qa, log)
    else:
        student_view(qa, log)

    st.divider()
    st.caption(
        "资料来自五邑大学教务处、学生工作部等公开页面；"
        "非官方答复，以学校最新通知为准。"
    )


if __name__ == "__main__":
    main()
