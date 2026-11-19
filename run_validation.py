"""Headless benchmark: OCR every image, evaluate, compare with expected marks.
expected_marks / validation text are used ONLY for scoring the system, never as inputs."""
import sys, difflib
from pathlib import Path
import pandas as pd
from PIL import Image
from evaluator import ocr_engine, nlp_engine

def run(data_dir="data", engine="tesseract"):
    data = nlp_engine.FacultyData(data_dir)
    df = pd.read_csv(Path(data_dir) / "student_answers.csv", encoding="utf-8-sig")
    rows = []
    for r in df.itertuples():
        ocr = ocr_engine.extract_from_image(Image.open(Path(data_dir) / r.answer_image), engine)
        qid = ocr.question_id
        if data.get_question(qid) is None:
            qid, _ = data.identify_by_text(ocr.question_text)
        ev = nlp_engine.evaluate(ocr.student_id, qid, ocr.answer_text, data)
        ocr_sim = difflib.SequenceMatcher(None, ocr.answer_text.lower(), r.student_answer_text_for_validation_only.lower()).ratio()
        rows.append(dict(image=Path(r.answer_image).name, id_ok=ocr.student_id == r.student_id,
                         qid_ok=ev.question_id == r.question_id, ocr_match=round(ocr_sim, 3),
                         sim=round(ev.similarity, 2), marks=ev.marks, expected=r.expected_marks_out_of_5,
                         diff=ev.marks - r.expected_marks_out_of_5, category=ev.category,
                         exp_cat=r.expected_category, cat_ok=ev.category == r.expected_category))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"\nID ok {out.id_ok.mean():.0%} | QID ok {out.qid_ok.mean():.0%} | mean OCR match {out.ocr_match.mean():.3f}")
    print(f"Mean abs error {out['diff'].abs().mean():.2f} | within 1 mark {(out['diff'].abs()<=1).mean():.0%} | category agreement {out.cat_ok.mean():.0%}")
