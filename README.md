# Intelligent Student Assignment Evaluation Assistant

Streamlit prototype: answer-sheet image -> OCR -> question lookup -> NLP preprocessing ->
TF-IDF similarity + rubric concept matching -> preliminary marks -> feedback -> report.

## Setup
1. Install the Tesseract binary (Windows: UB-Mannheim installer; macOS: `brew install tesseract`; Ubuntu: `sudo apt install tesseract-ocr`).
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
4. Headless benchmark: `python run_validation.py`

## Structure
- `app.py` - Streamlit UI (Steps 1-8 + benchmark tab)
- `evaluator/ocr_engine.py` - image preprocessing (OpenCV), Tesseract/EasyOCR, field parsing
- `evaluator/nlp_engine.py` - question lookup, preprocessing, TF-IDF, concept matching, feedback
- `run_validation.py` - OCR + evaluate all 24 images and compare with expected marks
- `data/` - faculty questions, rubric, answers CSV and images

## How marking works
1. OCR text is lowercased, stripped of punctuation, tokenised, stop-words removed, Porter-stemmed.
2. Each rubric concept (alternatives separated by `/`) is matched on stems, using a synonym table
   (e.g. divide/split/separate) and fuzzy matching to tolerate OCR typos.
   Full coverage = full mark, >=50% coverage of a multi-word concept = half mark.
3. Relevance gate: if fewer than 2 marks are matched (and similarity is low) the answer is treated
   as off-topic and given 0, since TF-IDF/keywords alone do not prove correctness.
4. Cosine similarity is shown for information; marks come from the rubric.
5. `student_answers.csv` is used only to validate OCR and marks, never as evaluation input.
Marks are preliminary and require faculty review (override field provided).

Partial credit (>=50% of a multi-word concept) is only given when the match rests on a synonym or a word
that is not merely repeated from the question text (so "language" alone does not earn marks for
"interpret language"). On the worked example in the brief ("NLP helps computers understand and process
human language") the system awards 3/5, as in the brief.
