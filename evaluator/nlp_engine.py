"""Steps 3-7: question lookup, preprocessing, similarity, rubric scoring, feedback."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_stem = PorterStemmer().stem

STOP_WORDS = set("""a an the is are was were be been being am of in on at to for from by with as into
onto over under it its this that these those and or but so such which who whom whose what when where
how do does did done can could may might will would shall should has have had having there their
they them he she his her we our you your i me my also than then very just about used use using
helps help enables enable allows allow one some any each more most other not no""".split())

# Words that are stop-words generally but meaningful as rubric content
KEEP = {"new", "unseen"}

# Synonym / paraphrase groups (matched on stems) - lets "helps computers understand
# language" match "enables computers to understand human language", etc.
SYNONYM_GROUPS = [
    "split divide separate partition break segment cut",
    "segment part region section portion piece area",
    "remove eliminate reduce clean denoise suppress",
    "improve enhance better boost",
    "resize scale rescale",
    "normalize standardize scale",
    "filter smooth blur sharpen",
    "label known annotated tagged supervised",
    "unseen new unknown test future",
    "predict forecast estimate infer",
    "output result prediction outcome answer",
    "understand comprehend grasp",
    "interpret analyze analyse decode",
    "process handle manipulate",
    "identify detect find determine recognize classify",
    "correct fix repair handle",
    "duplicate repeated redundant copy",
    "incorrect wrong erroneous invalid inconsistent error",
    "inconsistency inconsistent inconsistencies",
    "missing null empty absent",
    "opinion sentiment view attitude",
    "emotion feeling mood",
    "object item entity",
    "feature pattern characteristic",
    "image picture photo",
    "architecture network model structure",
    "learn train",
    "unit piece token chunk",
    "human natural",
    "technique method approach",
    "dataset data",
    "quality clarity",
    "layer layers",
    "convolution convolutional",
    "interest important relevant",
    "ai artificial",
]


def _build_synonyms() -> dict[str, set[str]]:
    groups = [{_stem(w) for w in g.split()} for g in SYNONYM_GROUPS]
    table: dict[str, set[str]] = {}
    for g in groups:
        for s in g:
            table.setdefault(s, set()).update(g)
    return table


SYN = _build_synonyms()


# --------------------------------------------------------------------------
# Step 4: preprocessing
# --------------------------------------------------------------------------
@dataclass
class Preprocessed:
    original: str
    lowered: str
    cleaned: str
    tokens: list[str]
    content_tokens: list[str]
    stems: list[str]


def preprocess(text: str) -> Preprocessed:
    # OCR commonly reads the capital "AI" as "Al" (lowercase L); correct that one known confusion.
    text = re.sub(r"\bAl\b", "AI", text)
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = cleaned.split()
    content = [t for t in tokens if t not in STOP_WORDS or t in KEEP]
    stems = [_stem(t) for t in content]
    return Preprocessed(text, lowered, cleaned, tokens, content, stems)


def stems_of(text: str) -> list[str]:
    return preprocess(text).stems


# --------------------------------------------------------------------------
# Data access (Step 3)
# --------------------------------------------------------------------------
class FacultyData:
    def __init__(self, data_dir: str | Path):
        d = Path(data_dir)
        self.questions = pd.read_csv(d / "questions.csv", encoding="utf-8-sig")
        self.rubric = pd.read_csv(d / "marking_rubric.csv", encoding="utf-8-sig")
        self.questions["question_id"] = self.questions["question_id"].str.strip().str.upper()
        self.rubric["question_id"] = self.rubric["question_id"].str.strip().str.upper()

    def get_question(self, qid: str):
        row = self.questions[self.questions["question_id"] == qid.upper()]
        return None if row.empty else row.iloc[0]

    def get_rubric(self, qid: str) -> pd.DataFrame:
        return self.rubric[self.rubric["question_id"] == qid.upper()].reset_index(drop=True)

    def identify_by_text(self, question_text: str) -> tuple[str | None, float]:
        """Fallback: identify the question by similarity to questions.csv text."""
        if not question_text.strip():
            return None, 0.0
        corpus = self.questions["question"].tolist()
        vec = TfidfVectorizer(analyzer=lambda t: stems_of(t)).fit(corpus + [question_text])
        sims = cosine_similarity(vec.transform([question_text]), vec.transform(corpus))[0]
        i = int(sims.argmax())
        return self.questions.iloc[i]["question_id"], float(sims[i])


# --------------------------------------------------------------------------
# Step 5: similarity
# --------------------------------------------------------------------------
def tfidf_similarity(student: str, reference: str, corpus: list[str]) -> float:
    vec = TfidfVectorizer(analyzer=lambda t: stems_of(t), sublinear_tf=True)
    vec.fit(corpus + [student, reference])
    m = vec.transform([student, reference])
    return float(cosine_similarity(m[0], m[1])[0][0])


# --------------------------------------------------------------------------
# Step 6: rubric / concept matching
# --------------------------------------------------------------------------
@dataclass
class ConceptResult:
    concept: str
    max_marks: float
    awarded: float
    status: str               # "covered" | "partial" | "missing"
    matched_alt: str = ""
    matched_terms: list[str] = field(default_factory=list)
    coverage: float = 0.0
    informative: bool = False  # True if a match is a synonym or a word not repeated from the question


def _fuzzy_has(stem: str, answer_stems: set[str]) -> bool:
    if len(stem) < 4:
        return False
    return any(SequenceMatcher(None, stem, a).ratio() >= 0.86 for a in answer_stems if len(a) >= 4)


def _token_matches(stem: str, answer_stems: set[str]) -> tuple[bool, str]:
    if stem in answer_stems:
        return True, stem
    for syn in SYN.get(stem, ()):
        if syn in answer_stems:
            return True, syn
    if _fuzzy_has(stem, answer_stems):
        return True, stem
    return False, ""


def match_concept(concept: str, marks: float, answer_stems: set[str],
                  question_stems: frozenset = frozenset()) -> ConceptResult:
    """A rubric entry may list alternatives separated by '/'. The best alternative wins.
    Coverage = share of the alternative's content stems found (directly, via synonym, or
    via fuzzy match for OCR typos). 100% -> full marks, >=50% -> half marks."""
    best = ConceptResult(concept, marks, 0.0, "missing")
    for alt in [a.strip() for a in concept.split("/") if a.strip()]:
        alt_stems = stems_of(alt) or [_stem(alt.lower())]
        hits, terms, informative = 0, [], False
        for s in dict.fromkeys(alt_stems):
            ok, via = _token_matches(s, answer_stems)
            if ok:
                hits += 1
                terms.append(via)
                if via != s or s not in question_stems:
                    informative = True
        cov = hits / len(set(alt_stems))
        if cov > best.coverage:
            best.coverage, best.matched_alt, best.matched_terms = cov, alt, terms
            best.informative = informative
    if best.coverage >= 0.99:
        best.awarded, best.status = marks, "covered"
    elif (best.coverage >= 0.5 and len(set(stems_of(best.matched_alt))) > 1
          # partial credit must rest on at least one word that is NOT merely repeated from the
          # question itself (e.g. the word "language" in "What is NLP... language?")
          and best.informative):
        best.awarded, best.status = round(marks * 0.5, 2), "partial"
    return best


@dataclass
class Evaluation:
    student_id: str
    question_id: str
    question: str
    reference: str
    max_marks: float
    similarity: float
    concepts: list[ConceptResult]
    raw_marks: float
    marks: float
    category: str
    relevance_note: str
    covered: list[str]
    partial: list[str]
    missing: list[str]
    feedback: str
    pre: Preprocessed


MIN_CONCEPT_HITS = 2.0   # relevance gate: fewer matched marks than this -> off-topic
MIN_SIM_FOR_SINGLE = 0.30


def evaluate(student_id: str, qid: str, answer_text: str, data: FacultyData) -> Evaluation:
    q = data.get_question(qid)
    if q is None:
        raise KeyError(f"Question ID '{qid}' not found in questions.csv")
    rubric = data.get_rubric(qid)
    max_marks = float(q["maximum_marks"])

    pre = preprocess(answer_text)
    ans_stems = set(pre.stems)
    sim = tfidf_similarity(answer_text, q["reference_answer"], data.questions["reference_answer"].tolist())

    q_stems = frozenset(stems_of(q["question"]))
    results = [match_concept(r.concept_or_expected_point, float(r.marks), ans_stems, q_stems)
               for r in rubric.itertuples()]
    raw = min(sum(c.awarded for c in results), max_marks)

    # Relevance gate: "similarity alone does not establish correctness", and a single
    # incidental keyword (e.g. 'words' in an answer about stemming) is not an answer.
    marks, note = raw, ""
    if raw < MIN_CONCEPT_HITS and not (raw >= 1 and sim >= MIN_SIM_FOR_SINGLE):
        if raw > 0:
            note = (f"Only {raw:g} incidental concept match(es) and low similarity ({sim:.2f}) - "
                    "answer appears off-topic, so marks set to 0. Faculty may override.")
        marks = 0.0
    if marks == 0:
        category = "Irrelevant"
    elif marks >= 0.8 * max_marks:
        category = "Relevant"
    else:
        category = "Partially relevant"

    covered = [c.concept for c in results if c.status == "covered"]
    partial = [c.concept for c in results if c.status == "partial"]
    missing = [c.concept for c in results if c.status == "missing"]
    if marks == 0:  # off-topic: nothing counts as covered
        missing = [c.concept for c in results]
        covered, partial = [], []

    fb = make_feedback(q["question"], covered, partial, missing, category, marks, max_marks)
    return Evaluation(student_id, qid.upper(), q["question"], q["reference_answer"], max_marks, sim,
                      results, raw, marks, category, note, covered, partial, missing, fb, pre)


# --------------------------------------------------------------------------
# Step 7: feedback
# --------------------------------------------------------------------------
def _clean_concept(c: str) -> str:
    return c.split("/")[0].strip()


def _join(items: list[str]) -> str:
    items = [_clean_concept(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def make_feedback(question: str, covered, partial, missing, category, marks, max_marks) -> str:
    if category == "Irrelevant":
        return ("Your answer does not address the question asked. Re-read the question and explain the "
                f"key idea of: \"{question}\" Make sure to mention: {_join(missing[:4])}.")
    parts = []
    if covered:
        parts.append(f"Good - your answer correctly covers {_join(covered)}.")
    if partial:
        parts.append(f"You touched on {_join(partial)} but not completely; state it more precisely.")
    if missing:
        parts.append(f"To improve, also include {_join(missing)}.")
    elif not partial:
        parts.append("Excellent, all expected points are covered.")
    return " ".join(parts)
