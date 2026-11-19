"""Intelligent Student Assignment Evaluation Assistant - Streamlit app.
Run:  streamlit run app.py
"""
from pathlib import Path
import io

import pandas as pd
import streamlit as st
from PIL import Image

from evaluator import nlp_engine, ocr_engine

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="Assignment Evaluation Assistant", page_icon="📝", layout="wide")


@st.cache_resource
def load_data():
    return nlp_engine.FacultyData(DATA_DIR)


@st.cache_data(show_spinner=False)
def cached_ocr(image_bytes: bytes, engine: str):
    return ocr_engine.extract_from_image(Image.open(io.BytesIO(image_bytes)), engine)


data = load_data()
engines = ocr_engine.available_engines()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📝 Evaluation Assistant")
    st.caption("MCA · NLP & Computer Vision")
    if not engines:
        st.error("No OCR engine found. Install Tesseract (and `pip install pytesseract`) or `pip install easyocr`.")
        st.stop()
    engine = st.selectbox("OCR engine", engines)
    source = st.radio("Image source", ["Upload an image", "Pick a sample from dataset"])
    st.divider()
    st.info("Marks are **preliminary**. Final scores must be reviewed by faculty.")

if "run_id" not in st.session_state:
    st.session_state.run_id = 0

tab_eval, tab_bench, tab_data = st.tabs(["🧾 Evaluate answer sheet", "📊 Validation benchmark", "📚 Faculty data"])

# ============================================================ evaluate tab
with tab_eval:
    st.header("Intelligent Student Assignment Evaluation Assistant")
    image_bytes, label = None, ""

    # ---- Step 1: upload ----------------------------------------------------
    st.subheader("Step 1 · Upload student answer image")
    if source == "Upload an image":
        up = st.file_uploader("Answer sheet (PNG/JPG)", type=["png", "jpg", "jpeg"],
                              key=f"uploader_{st.session_state.run_id}")
        if up:
            image_bytes, label = up.getvalue(), up.name
    else:
        samples = sorted((DATA_DIR / "answer_images").glob("*.png"))
        choice = st.selectbox("Sample image (still processed through OCR)", [p.name for p in samples])
        image_bytes, label = (DATA_DIR / "answer_images" / choice).read_bytes(), choice

    if image_bytes is None:
        st.caption("Upload an answer-sheet image to begin.")
        st.stop()

    c_img, c_info = st.columns([3, 2])
    c_img.image(image_bytes, caption=label, width="stretch")

    with st.spinner("Running OCR…"):
        ocr = cached_ocr(image_bytes, engine)

    with c_info:
        st.markdown("**Identified student details**")
        st.metric("Student ID", ocr.student_id or "—")
        st.metric("Question number", ocr.question_id or "—")
        for w in ocr.warnings:
            st.warning(w)

    # ---- Step 2: OCR text ----------------------------------------------------
    st.subheader("Step 2 · Text extracted by OCR")
    st.caption(f"Engine: {ocr.engine}. Edit the text below if OCR made a mistake (e.g. on a poor scan).")
    col1, col2 = st.columns(2)
    q_text = col1.text_area("Extracted question", ocr.question_text, height=80)
    a_text = col2.text_area("Extracted student answer", ocr.answer_text, height=80)
    with st.expander("Raw OCR output"):
        st.code(ocr.raw_text or "(empty)")

    # ---- Step 3: identify question --------------------------------------------
    st.subheader("Step 3 · Identify the question")
    qid = ocr.question_id
    q_row = data.get_question(qid) if qid else None
    if q_row is None:
        guess, score = data.identify_by_text(q_text)
        if guess:
            st.warning(f"Question ID '{qid or '?'}' not found; matched by question text instead "
                       f"({guess}, similarity {score:.2f}).")
            qid, q_row = guess, data.get_question(guess)
    else:
        guess, score = data.identify_by_text(q_text)
        if guess and guess != qid and score > 0.6:
            st.warning(f"Header says {qid} but the question text looks like {guess}. Please verify.")
    if q_row is None:
        st.error("Could not identify the question. Check the image or edit the text above.")
        st.stop()

    m1, m2 = st.columns([4, 1])
    m1.markdown(f"**{qid} · {q_row['subject']}** — {q_row['question']}")
    m1.markdown(f"**Reference answer:** {q_row['reference_answer']}")
    m2.metric("Max marks", f"{q_row['maximum_marks']:g}")

    if not a_text.strip():
        st.error("No answer text available to evaluate.")
        st.stop()

    ev = nlp_engine.evaluate(ocr.student_id, qid, a_text, data)

    # ---- Step 4: preprocess -------------------------------------------------------
    st.subheader("Step 4 · Preprocessed answer")
    p = ev.pre
    st.markdown(f"**Cleaned text:** `{p.cleaned}`")
    c1, c2 = st.columns(2)
    c1.markdown("**Tokens (stop-words removed)**")
    c1.write(", ".join(p.content_tokens) or "—")
    c2.markdown("**Stemmed key terms**")
    c2.write(", ".join(dict.fromkeys(p.stems)) or "—")

    # ---- Step 5: comparison ----------------------------------------------------------
    st.subheader("Step 5 · Compare with reference answer")
    s1, s2 = st.columns([1, 3])
    s1.metric("TF-IDF cosine similarity", f"{ev.similarity:.2f}")
    s1.progress(min(max(ev.similarity, 0.0), 1.0))
    matched = [c for c in ev.concepts if c.status != "missing"]
    s2.markdown("**Matching concepts**")
    if matched:
        for c in matched:
            icon = "✅" if c.status == "covered" else "🟡"
            s2.write(f"{icon} **{c.concept}** — matched via: {', '.join(dict.fromkeys(c.matched_terms))}")
    else:
        s2.write("No rubric concepts matched.")
    st.caption("Similarity alone does not decide marks; rubric concepts do.")

    # ---- Step 6: rubric ----------------------------------------------------------------
    st.subheader("Step 6 · Rubric-based evaluation")
    table = pd.DataFrame({
        "Expected concept": [c.concept for c in ev.concepts],
        "Identified in answer?": [{"covered": "Yes", "partial": "Partly", "missing": "No"}[c.status] for c in ev.concepts],
        "Marks available": [c.max_marks for c in ev.concepts],
        "Marks awarded": [c.awarded for c in ev.concepts],
    })
    st.dataframe(table, hide_index=True, width="stretch")
    st.markdown(f"**Rubric total:** {ev.raw_marks:g} / {ev.max_marks:g}")
    if ev.relevance_note:
        st.warning(ev.relevance_note)

    # ---- Step 7: feedback -----------------------------------------------------------------
    st.subheader("Step 7 · Feedback")
    f1, f2 = st.columns(2)
    f1.markdown("**Concepts covered**")
    f1.write("\n".join(f"- ✅ {c}" for c in ev.covered + [f"{x} (partial)" for x in ev.partial]) or "None")
    f2.markdown("**Concepts missing**")
    f2.write("\n".join(f"- ❌ {c}" for c in ev.missing) or "None")
    st.info(ev.feedback)

    # ---- Step 8: report ----------------------------------------------------------------------
    st.subheader("Step 8 · Evaluation report")
    with st.container(border=True):
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Student ID", ev.student_id or "—")
        r2.metric("Question ID", ev.question_id)
        r3.metric("Maximum marks", f"{ev.max_marks:g}")
        r4.metric("Preliminary marks", f"{ev.marks:g} / {ev.max_marks:g}")
        st.markdown(f"**Assessment:** {ev.category}  ·  **Similarity:** {ev.similarity:.2f}")
        st.markdown(f"**Feedback:** {ev.feedback}")
        final = st.number_input("Faculty-reviewed final marks (optional override)", 0.0, float(ev.max_marks),
                                float(ev.marks), step=0.5, key=f"final_{label}_{qid}")
    report = pd.DataFrame([{
        "student_id": ev.student_id, "question_id": ev.question_id, "max_marks": ev.max_marks,
        "preliminary_marks": ev.marks, "faculty_final_marks": final, "similarity": round(ev.similarity, 3),
        "category": ev.category, "concepts_covered": "; ".join(ev.covered),
        "concepts_partial": "; ".join(ev.partial), "concepts_missing": "; ".join(ev.missing),
        "feedback": ev.feedback}])
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download report (CSV)", report.to_csv(index=False), f"report_{ev.student_id}_{ev.question_id}.csv")
    if d2.button("🔄 Evaluate another image"):
        st.session_state.run_id += 1
        st.rerun()

# ============================================================ benchmark tab
with tab_bench:
    st.header("Validation against student_answers.csv")
    st.caption("Every image is run through OCR and the full pipeline. `student_answers.csv` is used only "
               "afterwards to check OCR accuracy and compare marks - never as evaluation input.")
    if st.button("▶ Run benchmark on all 24 images"):
        import difflib
        df = pd.read_csv(DATA_DIR / "student_answers.csv", encoding="utf-8-sig")
        rows, bar = [], st.progress(0.0)
        for i, r in enumerate(df.itertuples()):
            o = ocr_engine.extract_from_image(Image.open(DATA_DIR / r.answer_image), engine)
            qid_ = o.question_id if data.get_question(o.question_id) is not None else data.identify_by_text(o.question_text)[0]
            e = nlp_engine.evaluate(o.student_id, qid_, o.answer_text, data)
            rows.append({
                "Image": Path(r.answer_image).name,
                "ID ✓": o.student_id == r.student_id, "Q ✓": e.question_id == r.question_id,
                "OCR match": round(difflib.SequenceMatcher(None, o.answer_text.lower(),
                                   r.student_answer_text_for_validation_only.lower()).ratio(), 3),
                "Similarity": round(e.similarity, 2), "Predicted": e.marks, "Expected": r.expected_marks_out_of_5,
                "Diff": e.marks - r.expected_marks_out_of_5, "Category": e.category,
                "Expected category": r.expected_category})
            bar.progress((i + 1) / len(df))
        res = pd.DataFrame(rows)
        a, b, c, d = st.columns(4)
        a.metric("ID / question detection", f"{(res['ID ✓'] & res['Q ✓']).mean():.0%}")
        b.metric("Avg OCR text match", f"{res['OCR match'].mean():.1%}")
        c.metric("Mean abs. mark error", f"{res['Diff'].abs().mean():.2f}")
        d.metric("Within ±1 mark", f"{(res['Diff'].abs() <= 1).mean():.0%}")
        st.dataframe(res, hide_index=True, width="stretch")

# ============================================================ data tab
with tab_data:
    st.subheader("questions.csv")
    st.dataframe(data.questions, hide_index=True, width="stretch")
    st.subheader("marking_rubric.csv")
    st.dataframe(data.rubric, hide_index=True, width="stretch")
