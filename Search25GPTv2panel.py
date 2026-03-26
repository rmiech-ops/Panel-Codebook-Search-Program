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
from dotenv import load_dotenv

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
# Get API key
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

api_key_test = (
    _secret("AZURE_OPENAI_API_KEY")
    or _secret("API_KEY")
    or _secret("OPENAI_API_KEY")
)
st.write("final api_key_test is nonempty:", bool(api_key_test))

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="MTF Panel Codebooks Search",
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

st.title("MTF Codebook Search")

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
        "Initializing MTF Codebook Search...\n\n"
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
        BASE_DIR / "PanelAlliaqToYAMLv2.yaml",
        Path(sys.executable).resolve().parent / "PanelAlliaqToYAMLv2.yaml"
        if getattr(sys, "frozen", False)
        else BASE_DIR / "PanelAlliaqToYAMLv2.yaml",
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
    return pd.DataFrame.from_records(recs)


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
    qtexts = _df.get("QUESTION_TEXT", pd.Series([""] * len(_df))).astype(str).tolist()
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
        df.iloc[:, i] = s

    rename_map = {"VNUM_CONCAT": "VNUM_\nCONCAT", "VNUM_CONCAT\nCORE": "VNUM_\nCONCATCORE"}
    df = df.rename(columns=rename_map)
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
        "VNUM_\nCONCAT": 95,
        "VNUM_\nCONCATCORE": 115,
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

expected_cols = [
    "ITEMREFNO", "QNAME", "BRANCH", "FORM", "FIRST_YR", "LATEST_YR", "ORIGQ", "CHG_YR", "CHG_TYPE",
    "QUESTION_TEXT", "CATEGORY_TEXT", "VERSION", "VNUM_CONCAT", "VNUM_CONCAT_CORE",
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
    subj_cols = [
        "SUBJ_1_TEXT_LEV1", "SUBJ_1_TEXT_LEV2", "SUBJ_1_TEXT_LEV3",
        "SUBJ_2_TEXT_LEV1", "SUBJ_2_TEXT_LEV2", "SUBJ_2_TEXT_LEV3",
        "SUBJ_3_TEXT_LEV1", "SUBJ_3_TEXT_LEV2", "SUBJ_3_TEXT_LEV3",
    ]
    for c in subj_cols:
        if c not in _df.columns:
            _df[c] = ""
    blob = (
        _df["QUESTION_TEXT"].astype(str) + "\n" +
        _df["CATEGORY_TEXT"].astype(str) + "\n" +
        _df["QNAME"].astype(str) + "\n" +
        _df["VNUM_CONCAT"].astype(str) + "\n" +
        _df["VNUM_CONCAT_CORE"].astype(str) + "\n" +
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
    qnorm = _df["QUESTION_TEXT"].astype(str).apply(normalize_for_match)
    cnorm = _df["CATEGORY_TEXT"].astype(str).apply(normalize_for_match)
    scale = _df["CATEGORY_TEXT"].astype(str).apply(detect_scale_from_category)
    sig = _df["CATEGORY_TEXT"].astype(str).apply(category_signature)
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
    vnum_concat = st.text_input("VNUM_CONCAT exact", key="ui_vnum_concat")
    vnum_concat_core = st.text_input("VNUM_CONCAT_CORE exact", key="ui_vnum_concat_core")
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

@st.cache_data(show_spinner=False)
def llm_expand_for_lexical_rerank(user_query: str, chat_deployment: str) -> Dict[str, List[str]]:
    q = (user_query or "").strip()
    dep = (chat_deployment or "").strip()
    if not q or not dep:
        return {"terms": [], "phrases": []}
    client = _get_azure_client()
    system = (
        "You expand a search query for searching survey QUESTION_TEXT. "
        "Return ONLY valid JSON with keys: terms, phrases. "
        "terms: 6-12 single words (lowercase) including close variants, inflections, and likely survey wording. "
        "phrases: 6-12 short phrases (2-6 words) likely to appear verbatim or nearly verbatim in survey questions. "
        "Prefer terms and phrases that would help match how survey questions are actually worded. "
        "Return only the most relevant expansions for the user's query."
    )
    user = "User query:\n" + q + "\n\nReturn ONLY JSON like:\n{\"terms\":[\"...\"],\"phrases\":[\"...\"]}"
    resp = client.chat.completions.create(
        model=dep,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=250,
    )
    txt = (resp.choices[0].message.content or "").strip()
    txt = re.sub(r"^```json\s*|\s*```$", "", txt.strip(), flags=re.I)
    try:
        obj = json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            return {"terms": [], "phrases": []}
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return {"terms": [], "phrases": []}

    def _clean_list(xs, maxn):
        out = []
        if not isinstance(xs, list):
            return out
        for x in xs:
            if not isinstance(x, str):
                continue
            s = normalize_for_match(x)
            if s:
                out.append(s)
        seen = set()
        out2 = []
        for s in out:
            if s in seen:
                continue
            seen.add(s)
            out2.append(s)
        return out2[:maxn]

    terms = _clean_list(obj.get("terms", []), 12)
    phrases = _clean_list(obj.get("phrases", []), 12)
    return {"terms": terms, "phrases": phrases}

@st.cache_data(show_spinner=False)
def llm_plan_search(user_query: str, chat_deployment: str) -> Dict[str, object]:
    q = (user_query or "").strip()
    dep = (chat_deployment or "").strip()
    empty = {
        "intent": "",
        "semantic_queries": [],
        "include_terms": [],
        "include_phrases": [],
        "exclude_terms": [],
        "exclude_phrases": [],
        "entity_aliases": [],
        "scale_hint": "",
        "role_hint": "",
        "timeframe_hint": "",
    }
    if not q or not dep:
        return empty
    client = _get_azure_client()
    system = (
        "You are a search planner for a survey codebook search engine. "
        "Return ONLY valid JSON with these keys: intent, semantic_queries, include_terms, include_phrases, "
        "exclude_terms, exclude_phrases, entity_aliases, scale_hint, role_hint, timeframe_hint. "
        "semantic_queries should contain 2 to 5 rewritten search queries for semantic embedding retrieval. "
        "include_terms and include_phrases should capture likely survey wording. "
        "exclude_terms and exclude_phrases should capture nearby but wrong constructs to downweight. "
        "scale_hint should be one of: RISK_4, DISAPPROVAL, AVAILABILITY_4, FRIENDS_USE_5, EDU_7, INITIATION, or empty. "
        "role_hint should be one of: MOTHER, FATHER, PARENT, or empty. "
        "timeframe_hint should be one of: PAST_YEAR, PAST_30D, LIFETIME, or empty. "
        "entity_aliases should include alternate names for the main substance or entity. "
        "Do not explain anything. Return only JSON."
    )
    resp = client.chat.completions.create(
        model=dep,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": q},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    txt = (resp.choices[0].message.content or "").strip()
    txt = re.sub(r"^```json\s*|\s*```$", "", txt.strip(), flags=re.I)
    try:
        obj = json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            return empty
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return empty

    def clean_list(val, maxn=12):
        if not isinstance(val, list):
            return []
        out = []
        for x in val:
            if not isinstance(x, str):
                continue
            s = normalize_for_match(x)
            if s:
                out.append(s)
        return _dedupe_terms(out)[:maxn]

    plan = {
        "intent": str(obj.get("intent", "") or "").strip(),
        "semantic_queries": clean_list(obj.get("semantic_queries", []), 5),
        "include_terms": clean_list(obj.get("include_terms", []), 12),
        "include_phrases": clean_list(obj.get("include_phrases", []), 12),
        "exclude_terms": clean_list(obj.get("exclude_terms", []), 12),
        "exclude_phrases": clean_list(obj.get("exclude_phrases", []), 12),
        "entity_aliases": clean_list(obj.get("entity_aliases", []), 12),
        "scale_hint": str(obj.get("scale_hint", "") or "").strip(),
        "role_hint": str(obj.get("role_hint", "") or "").strip(),
        "timeframe_hint": str(obj.get("timeframe_hint", "") or "").strip(),
    }
    if not plan["semantic_queries"]:
        plan["semantic_queries"] = [normalize_for_match(q)]
    return plan


def _embedding_deployment() -> str:
    dep = _secret("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if not dep:
        raise RuntimeError("Missing AZURE_OPENAI_EMBEDDING_DEPLOYMENT (Azure deployment name).")
    return dep

def _truncate_for_embed(s: str, max_chars: int = 2500) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars]


def _build_doc_text(df_in: pd.DataFrame) -> List[str]:
    q = df_in.get("QUESTION_TEXT", "").astype(str)
    c = df_in.get("CATEGORY_TEXT", "").astype(str)
    n = df_in.get("QNAME", "").astype(str)
    out = []
    for i in range(len(df_in)):
        txt = (
            "VARIABLE: " + str(n.iloc[i]) + "\n"
            "QUESTION: " + str(q.iloc[i]) + "\n"
            "RESPONSES: " + str(c.iloc[i])
        )
        out.append(_truncate_for_embed(txt, 2500))
    return out


def _embed_texts_azure(texts: List[str]) -> np.ndarray:
    client = _get_azure_client()
    deployment = _embedding_deployment()
    batch_sz = int(os.environ.get("MTF_EMBED_BATCH", "128").strip() or "128")
    vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_sz):
        chunk = texts[i:i + batch_sz]
        resp = client.embeddings.create(model=deployment, input=chunk)
        for item in resp.data:
            vecs.append(np.asarray(item.embedding, dtype=np.float32))
    return np.vstack(vecs)


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return X / denom


def _embed_cache_path(path_str: str) -> Path:
    return Path(path_str).with_suffix(".embeddings.npz")


def _embed_cache_meta_path(path_str: str) -> Path:
    return Path(path_str).with_suffix(".embeddings.meta.txt")


@st.cache_resource(show_spinner=True)
def build_embedding_index(path_str: str, mtime: float) -> Dict[str, object]:
    cache_npz = _embed_cache_path(path_str)
    cache_meta = _embed_cache_meta_path(path_str)
    try:
        if cache_npz.exists() and cache_meta.exists():
            meta = cache_meta.read_text(encoding="utf-8").strip()
            if meta == str(mtime):
                data = np.load(cache_npz)
                Xn = data["Xn"].astype(np.float32)
                return {"Xn": Xn}
    except Exception:
        pass
    _df = load_data(path_str, mtime)
    texts = _build_doc_text(_df)
    X = _embed_texts_azure(texts)
    Xn = _l2_normalize_rows(X).astype(np.float32)
    try:
        np.savez_compressed(cache_npz, Xn=Xn)
        cache_meta.write_text(str(mtime), encoding="utf-8")
    except Exception:
        pass
    return {"Xn": Xn}


def semantic_topk_indices(path_str: str, mtime: float, query: str, topk: int) -> np.ndarray:
    idx = build_embedding_index(path_str, mtime)
    Xn = idx["Xn"]

    try:
        chat_dep = _secret("AZURE_OPENAI_CHAT_DEPLOYMENT")
        if not chat_dep:
            raise RuntimeError("Missing AZURE_OPENAI_CHAT_DEPLOYMENT.")
        exp = llm_expand_for_lexical_rerank(query, chat_dep)
        expanded_parts = [query]
        expanded_parts += exp.get("terms", [])[:8]
        expanded_parts += exp.get("phrases", [])[:6]
        expanded_query = " ".join(expanded_parts)
    except Exception:
        expanded_query = query

    qvec = _embed_texts_azure([_truncate_for_embed(expanded_query, 2000)])
    qn = _l2_normalize_rows(qvec.astype(np.float32))[0]
    sims = Xn @ qn
    topk = int(max(1, min(topk, sims.shape[0])))
    cand = np.argpartition(-sims, kth=topk - 1)[:topk]
    cand = cand[np.argsort(-sims[cand])]
    return cand

def _subject_match_score(df_in: pd.DataFrame, terms: List[str], phrases: List[str]) -> pd.Series:
    if df_in is None or df_in.empty:
        return pd.Series(dtype=float)
    score = pd.Series(0.0, index=df_in.index)
    subj_weights = {
        "__SUBJ_1_L1": 1.5, "__SUBJ_1_L2": 3.5, "__SUBJ_1_L3": 4.5,
        "__SUBJ_2_L1": 0.8, "__SUBJ_2_L2": 1.5, "__SUBJ_2_L3": 2.0,
        "__SUBJ_3_L1": 0.3, "__SUBJ_3_L2": 0.5, "__SUBJ_3_L3": 0.8,
    }
    all_terms = [normalize_for_match(t) for t in (terms or []) if normalize_for_match(t)]
    all_phrases = [normalize_for_match(p) for p in (phrases or []) if normalize_for_match(p)]
    for col, wt in subj_weights.items():
        if col not in df_in.columns:
            continue
        s = df_in[col].astype(str)
        for t in all_terms[:12]:
            if len(t) >= 4:
                score += s.str.contains(re.escape(t), na=False).astype(float) * wt
        for p in all_phrases[:12]:
            if len(p.split()) >= 2:
                score += s.str.contains(re.escape(p), na=False).astype(float) * (wt * 1.5)
    return score


def broad_lexical_candidate_indices(
    df_in: pd.DataFrame,
    user_query: str,
    include_terms: Optional[List[str]] = None,
    include_phrases: Optional[List[str]] = None,
    subject_hints: Optional[List[str]] = None,
    topk: int = 2500,
) -> np.ndarray:
    if df_in is None or df_in.empty:
        return np.array([], dtype=int)

    qn = normalize_for_match(user_query)
    include_terms = include_terms or []
    include_phrases = include_phrases or []
    subject_hints = subject_hints or []

    tokens = [t for t in qn.split() if len(t) >= 3 and t not in _STOP_TEXT][:12]
    phrases = _dedupe_terms(include_phrases[:12])
    more_terms = _dedupe_terms(include_terms[:12] + subject_hints[:8])

    blob = df_in["__BLOB_NORM"].astype(str)
    qtext = df_in["__QTEXT_NORM"].astype(str)
    cat = df_in["__CAT_NORM"].astype(str)
    qname = df_in["QNAME"].astype(str).apply(normalize_for_match)

    score = pd.Series(0.0, index=df_in.index)

    for t in tokens:
        score += qtext.str.contains(re.escape(t), na=False).astype(float) * 4.0
        score += qname.str.contains(re.escape(t), na=False).astype(float) * 3.5
        score += blob.str.contains(re.escape(t), na=False).astype(float) * 1.25

    for t in more_terms:
        if len(t) >= 3:
            score += qtext.str.contains(re.escape(t), na=False).astype(float) * 3.0
            score += qname.str.contains(re.escape(t), na=False).astype(float) * 2.5
            score += cat.str.contains(re.escape(t), na=False).astype(float) * 1.25
            score += blob.str.contains(re.escape(t), na=False).astype(float) * 1.0

    for p in phrases:
        if len(p.split()) >= 2:
            score += qtext.str.contains(re.escape(p), na=False).astype(float) * 7.0
            score += cat.str.contains(re.escape(p), na=False).astype(float) * 3.0
            score += blob.str.contains(re.escape(p), na=False).astype(float) * 2.0

    score += _subject_match_score(df_in, terms=(tokens + more_terms), phrases=phrases)

    positive = score[score > 0].sort_values(ascending=False)
    if positive.empty:
        return np.array([], dtype=int)
    return positive.head(int(topk)).index.to_numpy()


def rerank_with_expansions(
    df_in: pd.DataFrame,
    user_query: str,
    expansions: Dict[str, List[str]],
    planner: Optional[Dict[str, object]] = None,
    entity_terms: Optional[List[str]] = None,
    keep: int = 60,
    query_domain: str = "ambiguous",
    explicit_substance: bool = False,
    role_hint: str = "",
    scale_hint_override: str = "",
    timeframe_hint_override: str = "",
) -> pd.DataFrame:
    if df_in is None or df_in.empty or keep <= 0:
        return df_in

    planner = planner or {}
    entity_terms = entity_terms or []

    qn = normalize_for_match(user_query)
    blob = df_in["__BLOB_NORM"].astype(str)
    qtext = df_in["__QTEXT_NORM"].astype(str)
    cat = df_in["__CAT_NORM"].astype(str)
    qname = df_in["QNAME"].astype(str).apply(normalize_for_match)

    q_toks = [t for t in qn.split() if len(t) >= 4][:10]
    exp_terms = (expansions or {}).get("terms", []) or []
    exp_phrases = (expansions or {}).get("phrases", []) or []

    plan_inc_terms = planner.get("include_terms", []) or []
    plan_inc_phrases = planner.get("include_phrases", []) or []
    timeframe_hint = timeframe_hint_override or str(planner.get("timeframe_hint", "") or "")
    scale_hint = scale_hint_override or str(planner.get("scale_hint", "") or "")

    score = pd.Series(0.0, index=df_in.index)

    for t in q_toks:
        score += qtext.str.contains(re.escape(t), na=False).astype(float) * 3.0
        score += qname.str.contains(re.escape(t), na=False).astype(float) * 2.0
        score += blob.str.contains(re.escape(t), na=False).astype(float) * 1.0

    for t in exp_terms[:12]:
        if len(t) >= 4:
            score += qtext.str.contains(re.escape(t), na=False).astype(float) * 2.5
            score += qname.str.contains(re.escape(t), na=False).astype(float) * 1.5
            score += blob.str.contains(re.escape(t), na=False).astype(float) * 1.25

    for p in exp_phrases[:12]:
        if len(p.split()) >= 2:
            score += qtext.str.contains(re.escape(p), na=False).astype(float) * 6.0
            score += cat.str.contains(re.escape(p), na=False).astype(float) * 2.5
            score += blob.str.contains(re.escape(p), na=False).astype(float) * 2.0

    for t in plan_inc_terms[:12]:
        if len(t) >= 4:
            score += qtext.str.contains(re.escape(t), na=False).astype(float) * 2.5
            score += qname.str.contains(re.escape(t), na=False).astype(float) * 1.5
            score += blob.str.contains(re.escape(t), na=False).astype(float) * 1.25

    for p in plan_inc_phrases[:12]:
        if len(p.split()) >= 2:
            score += qtext.str.contains(re.escape(p), na=False).astype(float) * 6.5
            score += cat.str.contains(re.escape(p), na=False).astype(float) * 2.5
            score += blob.str.contains(re.escape(p), na=False).astype(float) * 2.0

    if entity_terms:
        entity_mask = pd.Series(False, index=df_in.index)
        for t in entity_terms:
            tn = normalize_for_match(t)
            if tn:
                entity_mask = entity_mask | blob.str.contains(re.escape(tn), na=False)
        score += entity_mask.astype(float) * 2.0

    if timeframe_hint and timeframe_hint in QTEXT_TIMEFRAME_INCLUDE:
        for p in QTEXT_TIMEFRAME_INCLUDE.get(timeframe_hint, []):
            pn = normalize_for_match(p)
            score += qtext.str.contains(re.escape(pn), na=False).astype(float) * 1.5

    if scale_hint:
        score += (df_in["__SCALE"].astype(str) == scale_hint).astype(float) * 2.0
        for p in QTEXT_GATE_PHRASES.get(scale_hint, []):
            pn = normalize_for_match(p)
            if pn:
                score += qtext.str.contains(re.escape(pn), na=False).astype(float) * 1.25

    role_hint = str(role_hint or "").strip()
    if role_hint == "MOTHER":
        score += qtext.str.contains(r"\bmother\b|\bmom\b", na=False).astype(float) * 2.5
    elif role_hint == "FATHER":
        score += qtext.str.contains(r"\bfather\b|\bdad\b", na=False).astype(float) * 2.5
    elif role_hint == "PARENT":
        score += qtext.str.contains(r"\bparent\b|\bparents\b", na=False).astype(float) * 2.0

    score += _subject_match_score(
        df_in,
        terms=(q_toks + exp_terms + plan_inc_terms),
        phrases=(exp_phrases + plan_inc_phrases),
    )

    if query_domain == "non_substance" and not explicit_substance:
        substance_mask = pd.Series(False, index=df_in.index)
        for t in SUBSTANCE_TERMS_FLAT:
            if t:
                substance_mask = substance_mask | blob.str.contains(re.escape(t), na=False)
        score -= substance_mask.astype(float) * 2.5

    if float(score.max()) <= 0.0:
        return df_in.head(keep)

    out = df_in.assign(__RERANK_SCORE=score).sort_values("__RERANK_SCORE", ascending=False)
    out = out.drop(columns=["__RERANK_SCORE"], errors="ignore")
    return out.head(keep)


def _env_on(name: str, default: bool = True) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v == "":
        return default
    return v in ("1", "true", "t", "yes", "y", "on")


PREWARM = _env_on("MTF_PREWARM_EMBEDDINGS", True)
if PREWARM and "startup_done" not in st.session_state:
    try:
        with st.spinner("Preparing AI search index (first run may take ~30 seconds)..."):
            _ = build_embedding_index(str(FILE_PATH), mtime)

        banner = st.session_state.get("startup_banner")
        if banner is not None:
            banner.success("MTF Codebook Search ready.")
            time.sleep(1)
            banner.empty()

        st.session_state.startup_done = True
        st.rerun()

    except Exception as e:
        st.warning(
            "Semantic index prewarm failed; searches will still work but may be slower. "
            f"{type(e).__name__}: {e}"
        )


@st.cache_data(show_spinner=False)
def apply_filters_cached(
    path_str: str,
    mtime: float,
    ai_query: str,
    search_query: str,
    search_mode: str,
    phrase_mode: bool,
    selected_ages: tuple,
    selected_forms: tuple,
    irn: str,
    vnum_concat: str,
    vnum_concat_core: str,
    first_range,
    latest_range,
    ai_max_hits_target: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    _df = load_data(path_str, mtime).copy()
    blob, qnorm, cnorm, scale_series, sig_series, subj_norm = build_cached_fields(path_str, mtime)
    _df["__BLOB_NORM"] = blob
    _df["__QTEXT_NORM"] = qnorm
    _df["__CAT_NORM"] = cnorm
    _df["__SCALE"] = scale_series
    _df["__CAT_SIG"] = sig_series
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
        _df[k] = subj_norm[v]

    entity_lex = build_entity_lexicon(path_str, mtime)
    debug: Dict[str, object] = {
        "used_ai": False,
        "ai_query": (ai_query or "").strip(),
        "scale_gate": "",
        "role_gate": "",
        "timeframe_gate": "",
        "entity_terms": [],
        "tighten_terms": [],
        "qtext_phrase_gate_used": False,
        "cat_sig_gate": "",
        "stage_hits_after_ai": None,
        "notes": "",
        "planner_used": False,
        "planner_intent": "",
        "plan_include_terms": [],
        "plan_include_phrases": [],
        "plan_exclude_terms": [],
        "plan_subject_hints": [],
        "query_domain": "",
        "explicit_substance": False,
        "embed_hits": 0,
        "lexical_hits": 0,
        "union_hits": 0,
    }

    filtered = _df
    has_ai = bool((ai_query or "").strip())
    has_lit = bool((search_query or "").strip())
    has_id = bool(str(irn or "").strip() or str(vnum_concat or "").strip() or str(vnum_concat_core or "").strip())
    ALL_AGES_DEFAULT = tuple(AGE_FILTER_OPTIONS)
    ALL_FORMS_DEFAULT = tuple(FORM_FILTER_OPTIONS)
    ages_tuple = tuple(selected_ages) if selected_ages else tuple()
    forms_tuple = tuple(selected_forms) if selected_forms else tuple()
    age_changed = set(ages_tuple) != set(ALL_AGES_DEFAULT)
    form_changed = set(forms_tuple) != set(ALL_FORMS_DEFAULT)
    year_filter_on = (first_range is not None) or (latest_range is not None)
    has_any_filter_intent = bool(has_id or age_changed or form_changed or year_filter_on)

    if not (has_ai or has_lit or has_any_filter_intent):
        return _df.iloc[0:0].copy(), debug

    aiq = (ai_query or "").strip()

    if aiq:
        debug["used_ai"] = True
        exp = {"terms": [], "phrases": []}
        plan = {
            "semantic_queries": [],
            "include_terms": [],
            "include_phrases": [],
            "exclude_terms": [],
            "exclude_phrases": [],
            "entity_aliases": [],
            "scale_hint": "",
            "role_hint": "",
            "timeframe_hint": "",
        }
        filtered_embed = None
        chat_dep = _secret("AZURE_OPENAI_CHAT_DEPLOYMENT")
        embed_query = aiq

        role = None
        scale_guess = None
        timeframe = None
        entity_terms = []
        plan_include_terms = []
        plan_include_phrases = []
        plan_exclude_terms = []
        plan_subject_hints = []

        explicit_substance = query_mentions_substance(aiq)
        query_domain = infer_query_domain(aiq)
        debug["query_domain"] = query_domain
        debug["explicit_substance"] = explicit_substance

        try:
            plan = llm_plan_search(aiq, chat_dep)
            debug["planner_used"] = True
            debug["planner_intent"] = plan.get("intent", "") or ""
            semantic_queries = plan.get("semantic_queries", []) or []
            embed_pieces = [aiq] + semantic_queries[:4] + list(plan.get("include_phrases", []) or [])[:4]
            embed_query = " | ".join([x for x in embed_pieces if x])

        except Exception as e:
            debug["notes"] = f"Planner unavailable ({type(e).__name__}: {e})"

        try:
            topk = int(os.environ.get("MTF_EMBED_TOPK", "2000").strip() or "2000")
            top_idx = semantic_topk_indices(path_str, mtime, embed_query, topk=topk)
            filtered_embed = _df.iloc[top_idx].copy()
            debug["embed_hits"] = int(len(filtered_embed))
            note = f"Embeddings prefilter: topK={len(filtered_embed)}"
            if debug["notes"]:
                debug["notes"] += " | " + note
            else:
                debug["notes"] = note
            exp = llm_expand_for_lexical_rerank(aiq, chat_dep)

        except Exception as e:
            debug["notes"] = (
                (debug.get("notes", "") + " | " if debug.get("notes") else "")
                + f"Embeddings prefilter unavailable; using lexical retrieval. ({type(e).__name__}: {e})"
            )

        plan2 = enhanced_parse(
            aiq,
            lambda q: detect_entity_terms(q, entity_lex) if query_mentions_substance(q) else [],
            parse_role_from_text,
            parse_scale_from_ai_text,
            parse_timeframe_from_ai_text
        )

        role = plan2.get("role")
        scale_guess = plan2.get("scale")
        timeframe = plan2.get("timeframe")
        entity_terms = plan2.get("entity") or []
        plan_include_terms = plan2.get("include_terms") or []
        plan_include_phrases = plan2.get("include_phrases") or []
        plan_exclude_terms = plan2.get("exclude_terms") or []
        plan_subject_hints = plan2.get("subject_hints") or []

        if not explicit_substance and query_domain == "non_substance":
            entity_terms = []

        exp["terms"] = list(dict.fromkeys((exp.get("terms", []) or []) + plan_include_terms))
        exp["phrases"] = list(dict.fromkeys((exp.get("phrases", []) or []) + plan_include_phrases))

        if query_domain == "non_substance" and not explicit_substance:
            exp["terms"] = [t for t in exp["terms"] if t not in SUBSTANCE_TERMS_FLAT]
            exp["phrases"] = [p for p in exp["phrases"] if not query_mentions_substance(p)]
            plan_include_terms = [t for t in plan_include_terms if t not in SUBSTANCE_TERMS_FLAT]
            plan_include_phrases = [p for p in plan_include_phrases if not query_mentions_substance(p)]

        debug["notes"] = (debug.get("notes", "") + " | " if debug.get("notes") else "") + "Broad AI retrieval mode"
        debug["timeframe_gate"] = (timeframe or "")
        debug["plan_include_terms"] = plan_include_terms
        debug["plan_include_phrases"] = plan_include_phrases
        debug["plan_exclude_terms"] = plan_exclude_terms
        debug["plan_subject_hints"] = plan_subject_hints
        debug["scale_gate"] = (scale_guess or "")
        debug["role_gate"] = (role or "")
        debug["entity_terms"] = entity_terms

        lexical_idx = broad_lexical_candidate_indices(
            _df,
            aiq,
            include_terms=plan_include_terms + list(plan.get("include_terms", []) or []),
            include_phrases=plan_include_phrases + list(plan.get("include_phrases", []) or []),
            subject_hints=plan_subject_hints,
            topk=int(os.environ.get("MTF_LEXICAL_TOPK", "2500").strip() or "2500"),
        )
        debug["lexical_hits"] = int(len(lexical_idx))

        candidate_index = []
        if filtered_embed is not None and len(filtered_embed) > 0:
            candidate_index.extend(list(filtered_embed.index))
        if len(lexical_idx) > 0:
            candidate_index.extend(list(lexical_idx))

        if candidate_index:
            seen = set()
            ordered_union = []
            for idx0 in candidate_index:
                if idx0 in seen:
                    continue
                seen.add(idx0)
                ordered_union.append(idx0)
            filtered = _df.loc[ordered_union].copy()
        else:
            filtered = _df.copy()

        debug["union_hits"] = int(len(filtered))

        keep_n = int(os.environ.get("MTF_AI_KEEP", "60").strip() or "60")
        pretruncate_hits = int(len(filtered))

        filtered = rerank_with_expansions(
            filtered,
            aiq,
            exp,
            planner=plan,
            entity_terms=entity_terms,
            keep=keep_n,
            query_domain=query_domain,
            explicit_substance=explicit_substance,
            role_hint=(role or str(plan.get("role_hint", "") or "")),
            scale_hint_override=(scale_guess or str(plan.get("scale_hint", "") or "")),
            timeframe_hint_override=(timeframe or str(plan.get("timeframe_hint", "") or "")),
        )

        debug["stage_hits_after_ai"] = pretruncate_hits
        debug["notes"] = (debug.get("notes", "") + " | " if debug.get("notes") else "") + f"Reranked top {keep_n}"

    else:
        if search_query:
            terms, explicit_op = parse_search_terms(search_query, phrase_mode)
            op = explicit_op if explicit_op else search_mode
            masks = []
            for t in terms:
                tn = normalize_for_match(t)
                if tn:
                    masks.append(filtered["__BLOB_NORM"].str.contains(re.escape(tn), na=False))
            if masks:
                mask = masks[0]
                for m in masks[1:]:
                    mask = mask & m if op == "AND" else mask | m
                filtered = filtered[mask]

    if selected_ages:
        age_mask = filtered["BRANCH"].astype(str).combine(filtered["FORM"].astype(str), branch_form_to_age).isin(list(selected_ages))
        filtered = filtered[age_mask]
    if selected_forms:
        form_mask = filtered["BRANCH"].astype(str).combine(filtered["FORM"].astype(str), form_filter_label).isin(list(selected_forms))
        filtered = filtered[form_mask]
    if irn:
        irn_s = str(irn).strip()
        col_s = filtered["ITEMREFNO"].astype(str).str.strip()
        if irn_s.isdigit():
            irn_int = int(irn_s)
            col_int = pd.to_numeric(col_s, errors="coerce")
            filtered2 = filtered[col_int == irn_int]
            if len(filtered2) == 0:
                filtered2 = filtered[col_s == irn_s]
            filtered = filtered2
        else:
            filtered = filtered[col_s == irn_s]
    if vnum_concat:
        filtered = filtered[filtered["VNUM_CONCAT"].astype(str) == str(vnum_concat).strip()]
    if vnum_concat_core:
        filtered = filtered[filtered["VNUM_CONCAT_CORE"].astype(str) == str(vnum_concat_core).strip()]
    if first_range is not None:
        y0, y1 = first_range
        s = pd.to_numeric(filtered["FIRST_YR"], errors="coerce")
        filtered = filtered[s.notna() & (s >= y0) & (s <= y1)]
    if latest_range is not None:
        y0, y1 = latest_range
        s = pd.to_numeric(filtered["LATEST_YR"], errors="coerce")
        filtered = filtered[s.notna() & (s >= y0) & (s <= y1)]

    return filtered, debug


filtered, ai_debug = apply_filters_cached(
    str(FILE_PATH),
    mtime,
    ai_query,
    search_query,
    search_mode,
    phrase_mode,
    tuple(selected_ages),
    tuple(selected_forms),
    irn,
    vnum_concat,
    vnum_concat_core,
    first_range,
    latest_range,
    int(AI_MAX_HITS_TARGET_DEFAULT),
)

DROP_COLS = [
    "FIRST_YR_NUM", "LATEST_YR_NUM", "WEB", "RESPCAT_ID", "VNUM", "VERS_ORIG",
    "__BLOB_NORM", "__QTEXT_NORM", "__CAT_NORM", "__SCALE", "__CAT_SIG",
    "__SUBJ_1_L1", "__SUBJ_1_L2", "__SUBJ_1_L3",
    "__SUBJ_2_L1", "__SUBJ_2_L2", "__SUBJ_2_L3",
    "__SUBJ_3_L1", "__SUBJ_3_L2", "__SUBJ_3_L3",
    "SUBJ_1", "SUBJ_1_TEXT_LEV1", "SUBJ_1_TEXT_LEV2", "SUBJ_1_TEXT_LEV3",
    "SUBJ_2", "SUBJ_2_TEXT_LEV1", "SUBJ_2_TEXT_LEV2", "SUBJ_2_TEXT_LEV3",
    "SUBJ_3", "SUBJ_3_TEXT_LEV1", "SUBJ_3_TEXT_LEV2", "SUBJ_3_TEXT_LEV3",
]

safe_df = filtered.drop(columns=DROP_COLS, errors="ignore")
safe_df = make_arrow_safe(safe_df)

safe_df = safe_df.rename(columns={
    "ITEMREFNO": "irn",
    "QNAME": "variable_label",
    "BRANCH": "branch",
    "FORM": "form",
    "FIRST_YR": "first_yr",
    "LATEST_YR": "latest_yr",
    "ORIGQ": "original_question",
    "CHG_YR": "year_question_changed",
    "CHG_TYPE": "type_of_question_change",
    "QUESTION_TEXT": "question_text",
    "CATEGORY_TEXT": "response_categories",
    "VERSION": "version",
    "VNUM_CONCAT": "vnum_concat",
    "VNUM_CONCAT_CORE": "vnum_concat_core",
})

if "branch" in safe_df.columns:
    safe_df["age"] = safe_df.apply(lambda r: branch_form_to_age(r.get("branch", ""), r.get("form", "")), axis=1)
    safe_df["form"] = safe_df.apply(lambda r: result_form_label(r.get("branch", ""), r.get("form", "")), axis=1)
    safe_df = safe_df.drop(columns=["branch"], errors="ignore")

if "original_question" in safe_df.columns:
    safe_df["original_question"] = safe_df["original_question"].apply(origq_to_yes_no)

preferred_order = [
    "irn",
    "variable_label",
    "age",
    "form",
    "first_yr",
    "latest_yr",
    "original_question",
    "year_question_changed",
    "type_of_question_change",
    "question_text",
    "response_categories",
    "version",
    "vnum_concat",
    "vnum_concat_core",
]

cols = [c for c in preferred_order if c in safe_df.columns]
safe_df = safe_df[cols]

PRETTY_COLS = {
    "irn": "Question\nID",
    "variable_label": "Variable\nlabel",
    "age": "Age",
    "form": "Form",
    "first_yr": "First\nyear",
    "latest_yr": "Latest\nyear",
    "original_question": "Original\nQuestion",
    "year_question_changed": "Year Question\nChanged",
    "type_of_question_change": "Type of\nQuestion Change",
    "question_text": "Question\ntext",
    "response_categories": "Response\nCategories",
    "version": "Version",
    "vnum_concat": "VNUM_CONCAT",
    "vnum_concat_core": "VNUM_CONCAT\nCORE",
}

safe_df_pretty = safe_df.rename(columns=PRETTY_COLS)

has_ai = bool((ai_query or "").strip())
has_lit = bool((search_query or "").strip())
has_id = bool(str(irn or "").strip() or str(vnum_concat or "").strip() or str(vnum_concat_core or "").strip())
age_changed = set(tuple(selected_ages)) != set(("BY", "BX"))
form_changed = set(tuple(selected_forms)) != set(tuple(str(i) for i in range(1, 7)))
year_filter_on = (first_range is not None) or (latest_range is not None)
has_any_filter_intent = bool(has_id or age_changed or form_changed or year_filter_on)
has_search_intent = bool(has_ai or has_lit or has_any_filter_intent)

st.markdown('<div id="results"></div>', unsafe_allow_html=True)
total = len(safe_df_pretty)

results_left, results_right = st.columns([0.85, 0.15])

with results_left:
    st.subheader("Results")

with results_right:
    accessible_view = st.toggle(
        "Accessible view",
        value=False,
        key="ui_accessible_view"
    )

if has_search_intent:
    if ai_debug.get("used_ai"):
        stage_hits_after_ai = ai_debug.get("stage_hits_after_ai") or 0
        if stage_hits_after_ai > total:
            st.write(f"Matches: {total:,}+ (AI-ranked by relevance)")
        else:
            st.write(f"Matches: {total:,} (AI-ranked by relevance)")
    else:
        st.write(f"Matches: {total:,}")

if (search_query or "").strip() and not (ai_query or "").strip():
    st.caption(f"Deterministic search: {search_query}")

if total > 0:
    total_pages = (total - 1) // int(page_size) + 1
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="ui_page")
    start = (page - 1) * int(page_size)
    end = min(start + int(page_size), total)
    page_df = safe_df_pretty.iloc[start:end].copy()
    st.caption(f"Showing {start + 1:,}-{end:,} of {total:,} matches.")
else:
    start = 0
    end = 0
    page_df = safe_df_pretty.copy()

if accessible_view:
    st.caption("Accessible list view: each result is an expandable, structured block.")
    page_df_internal = safe_df.iloc[start:end].copy() if total > 0 else safe_df.copy()
    for i, row in page_df_internal.reset_index(drop=True).iterrows():
        irn_val = row.get("irn", "")
        varlabel_val = row.get("variable_label", "")
        age_val = row.get("age", "")
        form_val = row.get("form", "")
        fy = row.get("first_yr", "")
        ly = row.get("latest_yr", "")
        origq_val = row.get("original_question", "")
        chgyr_val = row.get("year_question_changed", "")
        chgtype_val = row.get("type_of_question_change", "")
        version_val = row.get("version", "")
        vcat_val = row.get("vnum_concat", "")
        vcore_val = row.get("vnum_concat_core", "")
        cattext_val = row.get("response_categories", "")
        key_seed = f"row{i}_irn{irn_val}_vc{vcat_val}_vcc{vcore_val}_a{age_val}_f{form_val}"
        title_bits = []
        if str(irn_val).strip():
            title_bits.append(f"Question ID {irn_val}")
        if str(varlabel_val).strip():
            title_bits.append(f"Variable label {varlabel_val}")
        if str(age_val).strip():
            title_bits.append(f"AGE {age_val}")
        if str(form_val).strip():
            title_bits.append(f"FORM {form_val}")
        if str(vcat_val).strip():
            title_bits.append(f"VNUM_CONCAT {vcat_val}")
        if str(vcore_val).strip():
            title_bits.append(f"VNUM_CONCAT_CORE {vcore_val}")
        header = " - ".join(title_bits) if title_bits else "Result"
        with st.expander(header, expanded=False):
            years_line = ""
            if str(fy).strip() or str(ly).strip():
                years_line = f"{fy}-{ly}".strip("-")
            if years_line:
                st.write(f"Years: {years_line}")
            if str(origq_val).strip():
                st.write(f"Original Question: {origq_val}")
            if str(chgyr_val).strip() and str(chgyr_val).strip() != "--":
                st.write(f"Year Question Changed: {chgyr_val}")
            if str(chgtype_val).strip() and str(chgtype_val).strip() != "--":
                st.write(f"Type of Question Change: {chgtype_val}")
            if str(version_val).strip():
                st.write(f"Version: {version_val}")
            st.text_area("Question text", value=str(row.get("question_text", "")), height=180, key=f"qa_text_{key_seed}")
            st.text_area("Response Categories", value=str(cattext_val), height=110, key=f"qa_cat_{key_seed}")

            other = row.drop(labels=["question_text", "response_categories"], errors="ignore")

            field_map = {
                "irn": "Question ID"
            }

            other_df = pd.DataFrame({
                "Field": [field_map.get(x, x) for x in other.index],
                "Value": other.values
            })

            st.dataframe(other_df, hide_index=True, width="stretch", key=f"qa_meta_{key_seed}")
else:
    st.caption("Tip: click any column heading to sort (click again to reverse; third click resets).")
    render_wrapped_html_table(page_df, height_px=800)


def _env_bool(name: str, default: bool = True) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if val == "":
        return default
    return val in ("1", "true", "t", "yes", "y", "on")


EMBEDDINGS_ENABLED = _env_bool("MTF_USE_EMBEDDINGS", True)

if EMBEDDINGS_ENABLED:
    st.caption("Semantic index: ready")
else:
    st.caption("Semantic index: disabled (MTF_USE_EMBEDDINGS=0)")
