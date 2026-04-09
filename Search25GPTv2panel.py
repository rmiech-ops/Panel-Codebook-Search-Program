# -*- coding: utf-8 -*-
# =====================================================
# MTF Codebook Streamlit Browser -- Accessible Version + AI Search Gating
# Backend from current planner/embedding version
# UI/startup behavior merged from prior UI version
# AI search simplified to broad retrieval + ranking (less brittle)
# ASCII only, Emacs safe, unique widget keys
# =====================================================

import os
import sys
import re
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from Search25GPT_llm_upgrade import enhanced_parse

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml
import numpy as np
from openai import AzureOpenAI
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

# =====================================================
# PATH RESOLUTION (PyInstaller-safe)
# =====================================================
def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

BASE_DIR = app_base_dir()

# =====================================================
# Get secrets / env vars
# =====================================================
def _secret(name: str, default: str = "") -> str:
    if name in os.environ and str(os.environ[name]).strip():
        return str(os.environ[name]).strip()
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return default


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="MTF Panel Codebook Search",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# ACCESSIBILITY CSS
# =====================================================
st.markdown(
    '''
<style>
html, body, [class*="css"]  {
  font-size: 18px !important;
  line-height: 1.45 !important;
}
:focus {
  outline: 3px solid #000 !important;
  outline-offset: 2px !important;
}
.skip-link {
  position:absolute;
  left:-999px;
}
.skip-link:focus {
  left:10px;
  top:10px;
  background:white;
  border:2px solid black;
  padding:6px;
  z-index:9999;
}
</style>

<a class="skip-link" href="#results">Skip to results</a>
''',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
div[data-testid="stTextInput"] input {
    font-size: 18px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("MTF Panel Codebook Search")

st.markdown("""
<style>
.block-container {
    padding-top: 1.10rem !important;
}
h1 {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Startup status banner
# -----------------------------------------------------
if "startup_done" not in st.session_state:
    startup_status = st.empty()

    startup_status.info(
        "Initializing MTF Panel Codebook Search...\n\n"
        "- Loading codebook\n"
        "- Preparing AI search index\n"
        "- Building search structures"
    )

    st.session_state.startup_banner = startup_status

if "startup_done" in st.session_state:
    st.markdown(
        '<div style="font-weight:600; margin-bottom:0.15rem;">AI-assisted search</div>',
        unsafe_allow_html=True
    )

    ai_query = st.text_input(
        "AI-assisted search",
        placeholder="Example: Show me questions on perceived risk of LSD use",
        help=('Examples: "perceived risk of MDMA", "disapproval of LSD", '
              '"questions about mother\'s education", '
              '"when students first started using marijuana"'),
        key="ui_ai_query",
        label_visibility="collapsed",
    )

    st.caption(
        "Tip: Use AI-assisted search to find relevant questions. "
        "Then use Exact Word Search (upper left) with a distinctive phrase "
        "from the survey question text—or the other filters—to locate that question "
        "and related ones across the codebooks."
    )
else:
    ai_query = ""

# =====================================================
# AGE / FORM LABELS
# =====================================================
AGE_FILTER_OPTIONS = ["18", "19-30", "35", "40", "45", "50", "55", "60", "65"]
FORM_FILTER_OPTIONS = [str(i) for i in range(1, 8)] + ["n/a, age 35+"]

PANEL_FIELD_ORDER = [
    "by_panel",
    "fu1_panel",
    "fu2_panel",
    "fu3_panel",
    "fu4_panel",
    "fu5_panel",
    "fu6_panel",
    "fz1_panel",
    "fz2_panel",
    "fz3_panel",
    "fz4_panel",
    "fz5_panel",
    "fz6_panel",
    "fz7_panel",
]

PANEL_PRETTY_COLS = {
    "by_panel": "Age 18\n(BY)",
    "fu1_panel": "Age 19-20\n(FU1)",
    "fu2_panel": "Age 21-22\n(FU2)",
    "fu3_panel": "Age 23-24\n(FU3)",
    "fu4_panel": "Age 25-26\n(FU4)",
    "fu5_panel": "Age 27-28\n(FU5)",
    "fu6_panel": "Age 29-30\n(FU6)",
    "fz1_panel": "Age 35\n(FZ1)",
    "fz2_panel": "Age 40\n(FZ2)",
    "fz3_panel": "Age 45\n(FZ3)",
    "fz4_panel": "Age 50\n(FZ4)",
    "fz5_panel": "Age 55\n(FZ5)",
    "fz6_panel": "Age 60\n(FZ6)",
    "fz7_panel": "Age 65\n(FZ7)",
}
PANEL_PRETTY_TO_INTERNAL = {v: k for k, v in PANEL_PRETTY_COLS.items()}


def branch_form_to_age(branch, form) -> str:
    b = str(branch).strip().upper()
    f = str(form).strip()
    if b == "BY":
        return "18"
    if b == "FU":
        return "19-30"
    if b == "FZ":
        return {
            "1": "35",
            "2": "40",
            "3": "45",
            "4": "50",
            "5": "55",
            "6": "60",
            "7": "65",
        }.get(f, "")
    return ""

def result_form_label(branch, form) -> str:
    b = str(branch).strip().upper()
    if b == "FZ":
        return "n/a"
    s = str(form).strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    return s

def form_filter_label(branch, form) -> str:
    b = str(branch).strip().upper()
    if b == "FZ":
        return "n/a, age 35+"
    s = str(form).strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    return s

def origq_to_yes_no(x) -> str:
    s = str(x).strip().lower()
    if s in ("1", "1.0", "yes", "y", "true"):
        return "yes"
    if s in ("0", "0.0", "no", "n", "false"):
        return "no"
    if s in ("", "nan", "none"):
        return ""
    return str(x)

ENV_YAML = os.environ.get("MTF_YAML_PATH", "").strip()
CANDIDATES = []
if ENV_YAML:
    CANDIDATES.append(Path(ENV_YAML))
CANDIDATES.extend(
    [
        BASE_DIR / "PanelAlliaqToYAMLv3.yaml",
        Path(sys.executable).resolve().parent / "PanelAlliaqToYAMLv3.yaml"
        if getattr(sys, "frozen", False)
        else BASE_DIR / "PanelAlliaqToYAMLv3.yaml",
    ]
)
FILE_PATH = None
for p in CANDIDATES:
    try:
        if p.exists():
            FILE_PATH = p
            break
    except Exception:
        continue

def read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            pass
    return path.read_text(encoding="cp1252", errors="replace")

def load_yaml_records(path: Path):
    raw = read_text(path)
    data = yaml.safe_load(raw)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError("YAML root must be a list of records.")
    out = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        norm = {}
        for k, v in rec.items():
            if k is None:
                continue
            norm[str(k).strip().upper()] = v
        out.append(norm)
    return out

@st.cache_data(show_spinner=False)
def load_data(path_str: str, mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    recs = load_yaml_records(path)
    df = pd.DataFrame.from_records(recs)

    if "CATEGORY TEXT" in df.columns and "CATEGORYTEXT" not in df.columns:
        df["CATEGORYTEXT"] = df["CATEGORY TEXT"]

    return df

def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].apply(lambda x: isinstance(x, (list, dict))).any():
            def norm(x):
                if isinstance(x, list):
                    return " | ".join(str(i) for i in x)
                if isinstance(x, dict):
                    return str(x)
                return x
            out[col] = out[col].apply(norm)
    return out

def normalize_for_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _dedupe_terms(terms: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in terms:
        t = normalize_for_match(str(t))
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

def _count_hits(cat_norm: str, phrases: List[str]) -> int:
    n = 0
    for p in phrases:
        pn = normalize_for_match(p)
        if pn and pn in cat_norm:
            n += 1
    return n

SCALE_DEFS: Dict[str, Dict[str, object]] = {
    "RISK_4": {"phrases": ["no risk", "slight risk", "moderate risk", "great risk"], "min_hits": 3},
    "DISAPPROVAL": {"phrases": ["disapprove", "don t disapprove", "strongly disapprove", "approve"], "min_hits": 1},
    "AVAILABILITY_4": {"phrases": ["very difficult", "fairly difficult", "fairly easy", "very easy"], "min_hits": 3},
    "FRIENDS_USE_5": {"phrases": ["none", "a few", "some", "most", "all"], "min_hits": 4},
    "EDU_7": {
        "phrases": [
            "completed grade school",
            "grade school or less",
            "some high school",
            "completed high school",
            "some college",
            "completed college",
            "graduate or professional school",
            "don t know",
            "does not apply",
        ],
        "min_hits": 4,
    },
    "INIT_GRADE": {
        "phrases": ["grade 6 or below", "grade 7", "grade 8", "grade 9", "grade 10", "grade 11", "grade 12", "never"],
        "min_hits": 2,
    },
    "INIT_AGE": {
        "phrases": ["10 or younger", "11", "12", "13", "14", "15", "16", "17", "18 or older", "never"],
        "min_hits": 2,
    },
}

def detect_scale_from_category(cat_text: str) -> str:
    c = normalize_for_match(cat_text)
    if not c:
        return ""
    best = ""
    best_score = 0
    for name, d in SCALE_DEFS.items():
        phrases = list(d.get("phrases", []))
        min_hits = int(d.get("min_hits", 1))
        hits = _count_hits(c, phrases)
        if hits >= min_hits and hits > best_score:
            best = str(name)
            best_score = hits
    return best

_STOP_SIG = {
    "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "the",
    "is", "are", "be", "been", "do", "does", "did", "don", "t", "yes", "no",
}

def category_signature(cat_text: str, max_tokens: int = 12) -> str:
    c = normalize_for_match(cat_text)
    if not c:
        return ""
    toks = [t for t in c.split() if t and t not in _STOP_SIG]
    if not toks:
        return ""
    seen = set()
    uniq = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    uniq = uniq[:max_tokens]
    s = "|".join(uniq)
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]

SUBSTANCE_MAP: Dict[str, List[str]] = {
    "lsd": ["lsd", "l s d", "acid"],
    "mdma": ["mdma", "ecstasy", "molly"],
    "marijuana": ["marijuana", "marihuana", "cannabis", "pot", "weed", "grass", "hashish"],
    "cocaine": ["cocaine", "coke", "crack"],
    "heroin": ["heroin", "smack"],
    "alcohol": ["alcohol", "alcoholic", "beer", "wine", "liquor", "drink", "drinking", "drinks", "drunk"],
    "cigarettes": ["cigarette", "cigarettes", "smoke cigarettes", "smoking cigarettes"],
    "vaping nicotine": ["vape", "vaping", "e cigarette", "e cigarettes", "juul"],
    "inhalants": ["inhalant", "inhalants", "glue", "aerosol", "spray", "gasoline"],
}

NON_SUBSTANCE_HINTS = {
    "mother", "father", "parent", "parents", "education", "schooling", "school",
    "grade", "grades", "college", "plans", "religion", "religious", "church",
    "race", "ethnicity", "ethnic", "black", "white", "hispanic", "latino",
    "asian", "region", "urban", "rural", "city", "suburb", "farm", "politics",
    "political", "party", "ideology", "conservative", "liberal", "gradeschool",
    "job", "work", "military", "marriage", "marital", "children", "family",
    "sex", "gender", "friends", "peer", "achievement", "homework", "truancy",
    "absent", "absentee", "attendance"
}

SUBSTANCE_TERMS_FLAT = _dedupe_terms(
    [term for variants in SUBSTANCE_MAP.values() for term in variants] + list(SUBSTANCE_MAP.keys())
)

_STOP_ENTITY = {"drugs", "drug", "substances", "substance", "someone", "something", "anything", "it", "them", "this", "that"}

def query_mentions_substance(user_text: str) -> bool:
    q = normalize_for_match(user_text)
    if not q:
        return False
    for v in SUBSTANCE_TERMS_FLAT:
        if v and v in q:
            return True
    return False

def infer_query_domain(user_text: str) -> str:
    q = normalize_for_match(user_text)
    if not q:
        return "ambiguous"
    substance_hits = sum(1 for v in SUBSTANCE_TERMS_FLAT if v and v in q)
    non_sub_hits = sum(1 for v in NON_SUBSTANCE_HINTS if v and v in q)
    if substance_hits > 0 and non_sub_hits == 0:
        return "substance"
    if non_sub_hits > 0 and substance_hits == 0:
        return "non_substance"
    return "ambiguous"

def _clean_entity_phrase(p: str) -> str:
    p = normalize_for_match(p)
    p = re.sub(r"\b(once or twice|occasionally|regularly|every day|daily)\b", "", p)
    p = re.sub(r"\b(in the last 30 days|during the last 30 days|during the last 12 months|in the last 12 months)\b", "", p)
    p = re.sub(r"\s+", " ", p).strip()
    return p

def _entity_from_question_text(q: str) -> List[str]:
    qn = normalize_for_match(q)
    if not qn:
        return []
    out: List[str] = []
    patterns = [
        r"\btry\s+([a-z0-9 ]{2,60}?)\s+(once|occasionally|regularly|every|daily)\b",
        r"\buse\s+([a-z0-9 ]{2,60}?)\s+(once|occasionally|regularly|every|daily)\b",
        r"\bsmoke\s+([a-z0-9 ]{2,60}?)\s+(once|occasionally|regularly|every|daily)\b",
        r"\btake\s+([a-z0-9 ]{2,60}?)\s+(once|occasionally|regularly|every|daily)\b",
        r"\bget\s+([a-z0-9 ]{2,60}?)\s+if\s+you\s+wanted\b",
        r"\bused\s+([a-z0-9 ]{2,60}?)\s+(during|in)\s+the\s+last\b",
        r"\bfirst\s+(?:use|used)\s+([a-z0-9 ]{2,60}?)\b",
        r"\bwhen\s+(?:if\s+ever\s+)?did\s+you\s+first\s+use\s+([a-z0-9 ]{2,60}?)\b",
        r"\bgrade\s+of\s+first\s+use\s+of\s+([a-z0-9 ]{2,60}?)\b",
        r"\bfriends\b.*?\buse\s+([a-z0-9 ]{2,60}?)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, qn):
            e = _clean_entity_phrase(m.group(1))
            if not e or e in _STOP_ENTITY or len(e) < 3:
                continue
            out.append(e)
    seen = set()
    uniq = []
    for e in out:
        if e in seen:
            continue
        seen.add(e)
        uniq.append(e)
    return uniq

@st.cache_data(show_spinner=False)
def build_entity_lexicon(path_str: str, mtime: float) -> Dict[str, Dict[str, object]]:
    _df = load_data(path_str, mtime)
    qtexts = _df.get("QTEXTALL", pd.Series([""] * len(_df))).astype(str).tolist()
    counts: Dict[str, int] = {}
    for q in qtexts:
        for e in _entity_from_question_text(q):
            counts[e] = counts.get(e, 0) + 1
    kept = {e: c for e, c in counts.items() if c >= 2}
    lex: Dict[str, Dict[str, object]] = {}
    for e, c in sorted(kept.items(), key=lambda x: (-x[1], x[0])):
        lex[e] = {"variants": [e], "count": int(c)}
    for canonical, variants in SUBSTANCE_MAP.items():
        canon_norm = normalize_for_match(canonical)
        if canon_norm and canon_norm not in lex:
            lex[canon_norm] = {"variants": _dedupe_terms(variants), "count": 999999}
    return lex

def detect_entity_terms(user_text: str, entity_lex: Dict[str, Dict[str, object]]) -> List[str]:
    q = normalize_for_match(user_text)
    if not q:
        return []
    for _, variants in SUBSTANCE_MAP.items():
        for v in variants:
            vn = normalize_for_match(v)
            if vn and vn in q:
                return _dedupe_terms(variants)
    best = ""
    best_len = 0
    for ent, meta in entity_lex.items():
        if ent and ent in q:
            L = len(ent.split())
            if L > best_len:
                best = ent
                best_len = L
    if best:
        meta = entity_lex.get(best, {}) or {}
        variants = meta.get("variants", []) or [best]
        return _dedupe_terms([str(x) for x in variants])
    return []

def parse_role_from_text(text: str) -> Optional[str]:
    t = normalize_for_match(text)
    if not t:
        return None
    if "mother" in t or "mom" in t or "mom s" in t:
        return "MOTHER"
    if "father" in t or "dad" in t or "dad s" in t:
        return "FATHER"
    if "parent" in t or "parents" in t:
        return "PARENT"
    return None

def parse_scale_from_ai_text(text: str) -> Optional[str]:
    t = normalize_for_match(text)
    if not t:
        return None
    if "risk" in t or "harm" in t or "perceived risk" in t:
        return "RISK_4"
    if "disapprove" in t or "disapproval" in t or "approve" in t:
        return "DISAPPROVAL"
    if "availability" in t or "easy to get" in t or "difficult to get" in t or "how difficult" in t:
        return "AVAILABILITY_4"
    if "friends" in t:
        return "FRIENDS_USE_5"
    if (
        "first started" in t or "first start" in t or "first used" in t or "first use" in t or
        "when did you first" in t or "when did students first" in t or "when students first" in t or
        "grade of 1st use" in t or "grade of first use" in t or "age of first use" in t or
        "age first" in t or "initiation" in t
    ):
        return "INITIATION"
    if ("mother" in t or "father" in t or "parent" in t or "parents" in t) and (
        "education" in t or "schooling" in t or "school" in t or "highest level" in t or "completed" in t
    ):
        return "EDU_7"
    return None

def parse_timeframe_from_ai_text(text: str) -> Optional[str]:
    t = normalize_for_match(text)
    if not t:
        return None
    if (
        "past year" in t or "last year" in t or "past 12 months" in t or "last 12 months" in t or
        "during the last 12 months" in t or "in the last 12 months" in t or
        "during the past 12 months" in t or "in the past 12 months" in t or
        "12 month" in t or "12 months" in t
    ):
        return "PAST_YEAR"
    if (
        "past month" in t or "last month" in t or "past 30 days" in t or "last 30 days" in t or
        "during the last 30 days" in t or "in the last 30 days" in t or
        "30 day" in t or "30 days" in t
    ):
        return "PAST_30D"
    if "lifetime" in t or "ever" in t:
        return "LIFETIME"
    return None

_STOP_TEXT = {"show", "me", "questions", "question", "about", "on", "of", "the", "a", "an", "to", "please", "all", "any", "find", "give"}

def leftover_text_terms(ai_text: str, scale: Optional[str], role: Optional[str], entity_terms: List[str]) -> List[str]:
    t = normalize_for_match(ai_text)
    if not t:
        return []
    toks = [x for x in t.split() if x and x not in _STOP_TEXT]
    toks = [x for x in toks if x not in ("year", "years", "month", "months", "12", "30", "past", "last", "during")]
    if role == "MOTHER":
        toks = [x for x in toks if x not in ("mother", "mom", "mom s")]
    if role == "FATHER":
        toks = [x for x in toks if x not in ("father", "dad", "dad s")]
    if role == "PARENT":
        toks = [x for x in toks if x not in ("parent", "parents")]
    if scale == "RISK_4":
        toks = [x for x in toks if x not in ("risk", "harm", "perceived")]
    if scale == "DISAPPROVAL":
        toks = [x for x in toks if x not in ("disapprove", "disapproval", "approve")]
    if scale == "AVAILABILITY_4":
        toks = [x for x in toks if x not in ("availability", "easy", "difficult")]
    if scale == "FRIENDS_USE_5":
        toks = [x for x in toks if x not in ("friends",)]
    if scale == "EDU_7":
        toks = [x for x in toks if x not in ("education", "schooling", "school", "highest", "level", "completed")]
    if scale == "INITIATION":
        toks = [x for x in toks if x not in ("first", "started", "start", "use", "used", "when", "grade", "age")]
    entity_tokens = set()
    for e in entity_terms:
        for tt in e.split():
            entity_tokens.add(tt)
    toks = [x for x in toks if x not in entity_tokens]
    seen = set()
    out = []
    for x in toks:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out[:6]

QTEXT_GATE_PHRASES: Dict[str, List[str]] = {
    "RISK_4": ["how much do you think people risk", "how much do you think you risk", "great risk", "risk"],
    "DISAPPROVAL": ["do you disapprove", "how wrong do you think", "wrong", "disapprove"],
    "AVAILABILITY_4": [
        "how difficult do you think it would be for you to get",
        "how difficult would it be for you to get",
        "how difficult would it be to get",
        "very difficult",
        "fairly easy",
        "very easy",
    ],
    "FRIENDS_USE_5": ["how many of your friends", "friends"],
    "INITIATION": ["when", "first", "first use", "first used", "first time", "grade of first", "age when", "if ever"],
}

QTEXT_TIMEFRAME_INCLUDE: Dict[str, List[str]] = {
    "PAST_YEAR": [
        "during the last 12 months", "during the past 12 months", "in the last 12 months", "in the past 12 months",
        "during the last year", "in the last year", "past year", "last year", "12 months", "12 month",
    ],
    "PAST_30D": ["during the last 30 days", "in the last 30 days", "past 30 days", "last 30 days", "30 days", "30 day"],
    "LIFETIME": ["lifetime", "ever", "in your lifetime"],
}

QTEXT_TIMEFRAME_EXCLUDE: Dict[str, List[str]] = {
    "PAST_YEAR": ["lifetime", "in your lifetime", "ever"],
    "PAST_30D": ["lifetime", "in your lifetime", "ever", "12 months", "last year", "past year"],
}

def _safe_apply_qtext_phrase_gate(df_in: pd.DataFrame, qtext_col: str, phrases: List[str]) -> pd.DataFrame:
    if df_in.empty or not phrases:
        return df_in
    qn = df_in[qtext_col].astype(str)
    m = pd.Series(False, index=df_in.index)
    for p in phrases:
        pn = normalize_for_match(p)
        if pn:
            m = m | qn.str.contains(re.escape(pn), na=False)
    gated = df_in[m]
    if len(gated) == 0:
        return df_in
    return gated

def parse_search_terms(query: str, phrase_mode=True):
    if not query:
        return [], None
    q = query.strip()
    q_ops = re.sub(r'"[^"]*"', " ", q).upper()
    explicit_op = None
    if " OR " in q_ops:
        explicit_op = "OR"
    elif " AND " in q_ops:
        explicit_op = "AND"
    terms = []
    if phrase_mode:
        quoted = re.findall(r'"([^"]+)"', q)
        terms.extend(quoted)
    q = re.sub(r'"[^"]+"', " ", q)
    q = re.sub(r"\bAND\b|\bOR\b", " ", q, flags=re.I)
    terms.extend(q.split())
    return terms, explicit_op

# =====================================================
# TABLE RENDERER
# =====================================================
def render_wrapped_html_table(df_in: pd.DataFrame, height_px: int = 800) -> None:
    df = df_in.copy()
    for i, col in enumerate(df.columns):
        s = df.iloc[:, i].astype(str)
        s = s.str.replace(r"\\n", " ", regex=True)
        s = s.str.replace("\r\n", " ", regex=False).str.replace("\n", " ", regex=False).str.replace("\r", " ", regex=False)
        s = s.str.replace(r"\s+$", "", regex=True)
        col = df.columns[i]
        df[col] = s.astype("object")

    cols = list(df.columns)
    col_width_px = {
        "Question\nID": 73,
        "Variable\nlabel": 135,
        "Age": 85,
        "Form": 55,
        "First\nyear": 65,
        "Latest\nyear": 65,
        "Original\nQuestion": 70,
        "Year Question\nChanged": 85,
        "Type of\nQuestion Change": 85,
        "Question\ntext": 350,
        "Response\nCategories": 230,
        "Version": 65,
        "Age 18\n(BY)": 130,
        "Age 19-20\n(FU1)": 130,
        "Age 21-22\n(FU2)": 130,
        "Age 23-24\n(FU3)": 130,
        "Age 25-26\n(FU4)": 130,
        "Age 27-28\n(FU5)": 130,
        "Age 29-30\n(FU6)": 130,
        "Age 35\n(FZ1)": 130,
        "Age 40\n(FZ2)": 130,
        "Age 45\n(FZ3)": 130,
        "Age 50\n(FZ4)": 130,
        "Age 55\n(FZ5)": 130,
        "Age 60\n(FZ6)": 130,
        "Age 65\n(FZ7)": 130,
        "Gap_years": 90,
        "Notes": 220,
    }
    colgroup = "<colgroup>\n"
    for c in cols:
        w = col_width_px.get(str(c), 140)
        colgroup += f'  <col style="width:{int(w)}px;">\n'
    colgroup += "</colgroup>\n"
    html_table = df.to_html(index=False, escape=True)
    html_table = re.sub(r"(<table[^>]*>)", r"\1\n" + colgroup, html_table, count=1)
    for c in cols:
        if "\n" in str(c):
            html_table = html_table.replace(f"<th>{str(c)}</th>", "<th>" + str(c).replace("\n", "<br>") + "</th>")

    def _th_repl(match):
        inner = match.group(1)
        return '<th><div class="th-wrap"><span class="th-label" role="button" tabindex="0">' + inner + '</span><span class="sort-ind" aria-hidden="true"></span><span class="resizer" aria-hidden="true"></span></div></th>'

    html_table = re.sub(r"<th>(.*?)</th>", _th_repl, html_table, count=0)
    css_lines = []
    center_all = {
        "Question\nID",
        "Age",
        "Form",
        "First\nyear",
        "Latest\nyear",
        "Original\nQuestion",
        "Version",
        "Year Question\nChanged",
        "Type of\nQuestion Change",
        "BY_Panl",
        "FU1_Panl",
        "FU2_Panl",
        "FU3_Panl",
        "FU4_Panl",
        "FU5_Panl",
        "FU6_Panl",
        "FZ1_Panl",
        "FZ2_Panl",
        "FZ3_Panl",
        "FZ4_Panl",
        "FZ5_Panl",
        "FZ6_Panl",
        "FZ7_Panl",
        "Gap_years",
    }
    header_center_only = {
        "Variable\nlabel",
        "Question\ntext",
        "Response\nCategories",
    }
    no_wrap_cols = {"Age"}
    for idx, c in enumerate(cols):
        col_index = idx + 1
        if c in center_all:
            css_lines.append(f".mtf-wrap td:nth-child({col_index}), .mtf-wrap th:nth-child({col_index}) {{ text-align: center; }}")
        if c in header_center_only:
            css_lines.append(f".mtf-wrap th:nth-child({col_index}) {{ text-align: center; }}")
            css_lines.append(f".mtf-wrap td:nth-child({col_index}) {{ text-align: left; }}")
        if c in no_wrap_cols:
            css_lines.append(f".mtf-wrap td:nth-child({col_index}), .mtf-wrap th:nth-child({col_index}) {{ white-space: nowrap; }}")
    alignment_css = "\n".join(css_lines)
    html = f'''
<style>
.mtf-wrap {{ height: {int(height_px)}px; overflow: auto; border: 1px solid #ddd; border-radius: 6px; }}
.mtf-wrap table {{ border-collapse: collapse; width: 100%; table-layout: fixed; font-size: 16px; }}
.mtf-wrap th, .mtf-wrap td {{ border: 1px solid #e5e5e5; padding: 6px 8px; vertical-align: top; }}
.mtf-wrap th {{ position: sticky; top: 0; background: #f8f8f8; z-index: 2; white-space: pre-line; overflow-wrap: normal; word-break: normal; hyphens: none; text-align: center; line-height: 1.15; }}
.mtf-wrap td {{ white-space: normal; overflow-wrap: break-word; word-break: normal; line-height: 1.2; }}
.mtf-wrap .th-wrap {{ position: relative; padding-right: 20px; user-select: none; display: flex; align-items: center; gap: 4px; }}
.mtf-wrap .th-label {{ flex: 1 1 auto; cursor: pointer; outline: none; }}
.mtf-wrap .th-label:focus {{ outline: 2px solid #000; outline-offset: 2px; }}
.mtf-wrap .sort-ind {{ flex: 0 0 auto; width: 12px; text-align: center; opacity: 0.7; font-size: 11px; }}
.mtf-wrap .resizer {{ position: absolute; right: -8px; top: 0; width: 16px; height: 100%; cursor: col-resize; z-index: 3; }}
.mtf-wrap .resizer:hover {{ background: rgba(0,0,0,0.08); }}
{alignment_css}
</style>
<div class="mtf-wrap" id="mtf_wrap">{html_table}</div>
<script>
(function() {{
 const wrap = document.getElementById("mtf_wrap");
 if (!wrap) return;
 const table = wrap.querySelector("table");
 if (!table) return;
 const colgroup = table.querySelector("colgroup");
 const colEls = colgroup ? colgroup.querySelectorAll("col") : null;
 const thead = table.querySelector("thead");
 const tbody = table.querySelector("tbody");
 if (!thead || !tbody) return;
 const headers = Array.from(thead.querySelectorAll("th"));
 const originalRows = Array.from(tbody.querySelectorAll("tr"));
 if (colEls && colEls.length === headers.length) {{
   let startX = 0; let startWidth = 0; let activeCol = null;
   function onMouseMove(e) {{ if (!activeCol) return; const dx = e.clientX - startX; const newW = Math.max(40, startWidth + dx); activeCol.style.width = newW + "px"; }}
   function onMouseUp() {{ activeCol = null; document.removeEventListener("mousemove", onMouseMove); document.removeEventListener("mouseup", onMouseUp); }}
   headers.forEach((th, idx) => {{
     const handle = th.querySelector(".resizer");
     if (!handle) return;
     handle.addEventListener("mousedown", (e) => {{
       e.preventDefault(); startX = e.clientX; activeCol = colEls[idx];
       const w = activeCol.style.width || window.getComputedStyle(activeCol).width;
       startWidth = parseFloat(w) || th.getBoundingClientRect().width;
       document.addEventListener("mousemove", onMouseMove);
       document.addEventListener("mouseup", onMouseUp);
     }});
   }});
 }}
 let sortState = {{ col: -1, dir: 0 }};
 function cellText(tr, idx) {{ const td = tr.children[idx]; if (!td) return ""; return (td.textContent || "").trim(); }}
 function isNumericColumn(idx) {{
   let seen = 0; let ok = 0; const rows = Array.from(tbody.querySelectorAll("tr"));
   for (let i = 0; i < rows.length; i++) {{
     const t = cellText(rows[i], idx); if (!t) continue; seen++;
     const v = parseFloat(t.replace(/,/g, "")); if (!isNaN(v)) ok++;
     if (seen >= 25) break;
   }}
   return (seen > 0 && ok / seen >= 0.8);
 }}
 function setIndicators(activeIdx, dir) {{
   headers.forEach((th, i) => {{
     const ind = th.querySelector(".sort-ind");
     if (!ind) return;
     if (i !== activeIdx || dir === 0) ind.textContent = "";
     else if (dir === 1) ind.textContent = "▲";
     else ind.textContent = "▼";
   }});
 }}
 function applySort(idx) {{
   let dir = 1;
   if (sortState.col === idx && sortState.dir === 1) dir = -1;
   else if (sortState.col === idx && sortState.dir === -1) dir = 0;
   sortState = {{ col: idx, dir }};
   setIndicators(idx, dir);
   if (dir === 0) {{ tbody.innerHTML = ""; originalRows.forEach(r => tbody.appendChild(r)); return; }}
   const numeric = isNumericColumn(idx);
   const rows = Array.from(tbody.querySelectorAll("tr"));
   rows.sort((a, b) => {{
     const ta = cellText(a, idx); const tb = cellText(b, idx);
     if (numeric) {{
       const va = parseFloat(ta.replace(/,/g, "")); const vb = parseFloat(tb.replace(/,/g, ""));
       const na = isNaN(va); const nb = isNaN(vb);
       if (na && nb) return 0; if (na) return 1; if (nb) return -1;
       return (va - vb) * dir;
     }}
     return ta.localeCompare(tb, undefined, {{ numeric: true, sensitivity: "base" }}) * dir;
   }});
   tbody.innerHTML = ""; rows.forEach(r => tbody.appendChild(r));
 }}
 headers.forEach((th, idx) => {{
   const label = th.querySelector(".th-label"); if (!label) return;
   label.addEventListener("click", () => applySort(idx));
   label.addEventListener("keydown", (e) => {{ if (e.key === "Enter" || e.key === " ") {{ e.preventDefault(); applySort(idx); }} }});
 }});
}})();
</script>
'''
    components.html(html, height=height_px + 40, scrolling=True)

if FILE_PATH is None:
    searched = "\n".join([str(p) for p in CANDIDATES])
    st.error(
        "YAML file not found.\n\n"
        "The app searched these locations:\n"
        f"{searched}\n\n"
        "Fix options:\n"
        "1) Put AlliaqToYAMLv2.yaml in the same folder as the app's bundled files.\n"
        "2) Or set the environment variable MTF_YAML_PATH to the YAML full path."
    )
    st.stop()

try:
    mtime = FILE_PATH.stat().st_mtime
except FileNotFoundError:
    st.error(f"File not found: {FILE_PATH}")
    st.stop()

df = load_data(str(FILE_PATH), mtime)

# --- FIX: remove .0 from integer-like numeric fields ---
def coerce_integer_like_series(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    mask = num.notna() & (num % 1 == 0)
    out = s.astype("object").copy()
    out.loc[mask] = num.loc[mask].astype("Int64").astype(str)
    return out

int_like_cols = [
    "FORM", "FIRST_YR", "LATEST_YR",
    "CHG_YR", "VERSION",
    "BY_PANL", "FU1_PANL", "FU2_PANL", "FU3_PANL", "FU4_PANL", "FU5_PANL", "FU6_PANL",
    "FZ1_PANL", "FZ2_PANL", "FZ3_PANL", "FZ4_PANL", "FZ5_PANL", "FZ6_PANL", "FZ7_PANL",
    "GAP_YEARS"
]

for col in int_like_cols:
    if col in df.columns:
        df[col] = coerce_integer_like_series(df[col])
        
expected_cols = [
    "ITEMREFNO", "QNAME", "BY_X_FU_X", "FORM", "FIRST_YR", "LATEST_YR", "ORIGQ", "CHG_YR", "CHG_TYPE",
    "QTEXTALL", "CATEGORYTEXT", "VERSION",
    "BY_PANL", "FU1_PANL", "FU2_PANL", "FU3_PANL", "FU4_PANL", "FU5_PANL", "FU6_PANL",
    "FZ1_PANL", "FZ2_PANL", "FZ3_PANL", "FZ4_PANL", "FZ5_PANL", "FZ6_PANL", "FZ7_PANL",
    "GAP_YEARS", "NOTES",
    "SUBJ_1", "SUBJ_1_TEXT_LEV1", "SUBJ_1_TEXT_LEV2", "SUBJ_1_TEXT_LEV3",
    "SUBJ_2", "SUBJ_2_TEXT_LEV1", "SUBJ_2_TEXT_LEV2", "SUBJ_2_TEXT_LEV3",
    "SUBJ_3", "SUBJ_3_TEXT_LEV1", "SUBJ_3_TEXT_LEV2", "SUBJ_3_TEXT_LEV3",
]
for c in expected_cols:
    if c not in df.columns:
        df[c] = ""

df["FIRST_YR_NUM"] = pd.to_numeric(df["FIRST_YR"], errors="coerce")
df["LATEST_YR_NUM"] = pd.to_numeric(df["LATEST_YR"], errors="coerce")

@st.cache_data(show_spinner=False)
def build_cached_fields(path_str: str, mtime: float):
    _df = load_data(path_str, mtime)

    extra_cols = [
        "BY_PANL", "FU1_PANL", "FU2_PANL", "FU3_PANL", "FU4_PANL", "FU5_PANL", "FU6_PANL",
        "FZ1_PANL", "FZ2_PANL", "FZ3_PANL", "FZ4_PANL", "FZ5_PANL", "FZ6_PANL", "FZ7_PANL",
        "GAP_YEARS", "NOTES",
    ]
    subj_cols = [
        "SUBJ_1_TEXT_LEV1", "SUBJ_1_TEXT_LEV2", "SUBJ_1_TEXT_LEV3",
        "SUBJ_2_TEXT_LEV1", "SUBJ_2_TEXT_LEV2", "SUBJ_2_TEXT_LEV3",
        "SUBJ_3_TEXT_LEV1", "SUBJ_3_TEXT_LEV2", "SUBJ_3_TEXT_LEV3",
    ]

    for c in extra_cols + subj_cols:
        if c not in _df.columns:
            _df[c] = ""

    if "CATEGORYTEXT" not in _df.columns:
        if "CATEGORY TEXT" in _df.columns:
            _df["CATEGORYTEXT"] = _df["CATEGORY TEXT"]
        else:
            _df["CATEGORYTEXT"] = ""

    cat_series = _df["CATEGORYTEXT"].astype(str)

    blob = (
        _df["QTEXTALL"].astype(str) + "\n" +
        cat_series + "\n" +
        _df["QNAME"].astype(str) + "\n" +
        _df["SUBJ_1_TEXT_LEV1"].astype(str) + "\n" +
        _df["SUBJ_1_TEXT_LEV2"].astype(str) + "\n" +
        _df["SUBJ_1_TEXT_LEV3"].astype(str) + "\n" +
        _df["SUBJ_2_TEXT_LEV1"].astype(str) + "\n" +
        _df["SUBJ_2_TEXT_LEV2"].astype(str) + "\n" +
        _df["SUBJ_2_TEXT_LEV3"].astype(str) + "\n" +
        _df["SUBJ_3_TEXT_LEV1"].astype(str) + "\n" +
        _df["SUBJ_3_TEXT_LEV2"].astype(str) + "\n" +
        _df["SUBJ_3_TEXT_LEV3"].astype(str)
    ).apply(normalize_for_match)

    qnorm = _df["QTEXTALL"].astype(str).apply(normalize_for_match)
    cnorm = cat_series.apply(normalize_for_match)
    scale = cat_series.apply(detect_scale_from_category)
    sig = cat_series.apply(category_signature)
    subj_norm = {c: _df[c].astype(str).apply(normalize_for_match) for c in subj_cols}

    return blob, qnorm, cnorm, scale, sig, subj_norm

blob, qnorm, cnorm, scale_series, sig_series, subj_norm = build_cached_fields(str(FILE_PATH), mtime)
df = df.copy()
df["__BLOB_NORM"] = blob
df["__QTEXT_NORM"] = qnorm
df["__CAT_NORM"] = cnorm
df["__SCALE"] = scale_series
df["__CAT_SIG"] = sig_series
for k, v in {
    "__SUBJ_1_L1": "SUBJ_1_TEXT_LEV1",
    "__SUBJ_1_L2": "SUBJ_1_TEXT_LEV2",
    "__SUBJ_1_L3": "SUBJ_1_TEXT_LEV3",
    "__SUBJ_2_L1": "SUBJ_2_TEXT_LEV1",
    "__SUBJ_2_L2": "SUBJ_2_TEXT_LEV2",
    "__SUBJ_2_L3": "SUBJ_2_TEXT_LEV3",
    "__SUBJ_3_L1": "SUBJ_3_TEXT_LEV1",
    "__SUBJ_3_L2": "SUBJ_3_TEXT_LEV2",
    "__SUBJ_3_L3": "SUBJ_3_TEXT_LEV3",
}.items():
    df[k] = subj_norm[v]

ENTITY_LEXICON = build_entity_lexicon(str(FILE_PATH), mtime)
AI_MAX_HITS_TARGET_DEFAULT = 60

with st.sidebar:
    st.header("Filters")

    search_query = st.text_input(
        "Exact word search (no AI assistance)",
        placeholder='Example: risk lsd',
        help=("Search includes question_text, category_text, and variable label. "
              "Use quotes and AND/OR."),
        key="ui_search_query",
    )

    with st.container():
        st.caption("Options for Exact word search only")

        opt_left, opt_right = st.columns([0.08, 0.92])

        with opt_right:
            search_mode = st.radio(
                "Match mode (default)",
                ["AND", "OR"],
                horizontal=True,
                key="ui_search_mode",
            )

            phrase_mode = st.checkbox(
                "Keep quoted phrases",
                True,
                key="ui_phrase_mode",
            )

    selected_ages = st.pills(
       "Age",
        options=AGE_FILTER_OPTIONS,
        default=AGE_FILTER_OPTIONS,
        selection_mode="multi",
        key="ui_age_labels",
    )
    selected_forms = st.pills(
        "Form",
        options=FORM_FILTER_OPTIONS,
        default=FORM_FILTER_OPTIONS,
        selection_mode="multi",
        key="ui_forms",
    )
    irn = st.text_input("Question ID", key="ui_irn")
    first_vals = df["FIRST_YR_NUM"].dropna()
    latest_vals = df["LATEST_YR_NUM"].dropna()

    st.subheader("Year filters (optional)")
    use_first = st.checkbox("Filter by first_yr range", value=False, key="ui_use_first")
    first_range = None
    if use_first and not first_vals.empty:
        fmin, fmax = int(first_vals.min()), int(first_vals.max())
        first_range = st.slider("first_yr range", fmin, fmax, (fmin, fmax), key="ui_first_range")
    elif use_first and first_vals.empty:
        st.info("No numeric first_yr values found.")

    use_latest = st.checkbox("Filter by latest_yr range", value=False, key="ui_use_latest")
    latest_range = None
    if use_latest and not latest_vals.empty:
        lmin, lmax = int(latest_vals.min()), int(latest_vals.max())
        latest_range = st.slider("latest_yr range", lmin, lmax, (lmin, lmax), key="ui_latest_range")
    elif use_latest and latest_vals.empty:
        st.info("No numeric latest_yr values found.")

    st.subheader("Results display")
    page_size = st.selectbox("Results per page", [25, 50, 100, 250, 500], index=1, key="ui_page_size")

# =====================================================
# EMBEDDINGS + LLM HELPERS
# =====================================================
def _get_azure_client() -> AzureOpenAI:
    endpoint = _secret("AZURE_OPENAI_ENDPOINT")
    api_version = _secret("AZURE_OPENAI_API_VERSION")
    api_key = (
        _secret("AZURE_OPENAI_API_KEY")
        or _secret("API_KEY")
        or _secret("OPENAI_API_KEY")
    )
    shortcode = _secret("SHORTCODE")

    if not endpoint or not api_version or not api_key:
        raise RuntimeError(
            "Missing AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_VERSION / "
            "AZURE_OPENAI_API_KEY (or API_KEY / OPENAI_API_KEY)."
        )
    if not shortcode:
        raise RuntimeError("Missing SHORTCODE (required by UM GPT gateway).")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        api_key=api_key,
        organization=shortcode,
    )

