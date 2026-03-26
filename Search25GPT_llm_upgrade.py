# =====================================================
# Search25GPT_llm_upgrade.py
# LLM search planner for Search25GPT
# =====================================================

import json
import os
import re
from typing import Dict, List, Optional

from openai import AzureOpenAI


def _get_azure_client() -> AzureOpenAI:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
    api_key = (
        os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        or os.environ.get("API_KEY", "").strip()
    )
    shortcode = os.environ.get("SHORTCODE", "").strip()

    if not endpoint or not api_version or not api_key:
        raise RuntimeError("Missing Azure OpenAI configuration")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        api_key=api_key,
        organization=shortcode,
    )


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_str_list(xs, maxn: int = 12, allow_phrases: bool = True) -> List[str]:
    out: List[str] = []
    if not isinstance(xs, list):
        return out

    for x in xs:
        if not isinstance(x, str):
            continue
        s = _norm(x)
        if not s:
            continue
        if not allow_phrases and " " in s:
            continue
        out.append(s)

    seen = set()
    deduped = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)

    return deduped[:maxn]


def llm_build_search_plan(user_query: str) -> Dict[str, object]:
    q = (user_query or "").strip()
    if not q:
        return {}

    try:
        client = _get_azure_client()
        chat_dep = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
        if not chat_dep:
            return {}

        system = """
You are building a search plan for a survey codebook search engine.

Return ONLY valid JSON with these keys:
- scale
- entity
- role
- timeframe
- include_terms
- include_phrases
- exclude_terms
- subject_hints

Allowed values:
- scale: RISK_4, DISAPPROVAL, AVAILABILITY_4, FRIENDS_USE_5, EDU_7, INITIATION, or null
- role: MOTHER, FATHER, PARENT, or null
- timeframe: PAST_YEAR, PAST_30D, LIFETIME, or null

Rules:
- entity must be a list of likely topic/substance terms.
- include_terms must be short useful search terms.
- include_phrases must be short useful phrases likely to match survey wording.
- exclude_terms should only contain terms likely to create false positives.
- subject_hints should be short conceptual hints like school, peers, parents, discipline, progression.
- Return JSON only. No explanation.
"""

        user = f"""
User query:
{q}

Return JSON only.
"""

        resp = client.chat.completions.create(
            model=chat_dep,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=350,
        )

        txt = (resp.choices[0].message.content or "").strip()
        txt = re.sub(r"^```json\s*|\s*```$", "", txt, flags=re.I)

        plan = json.loads(txt)
        if not isinstance(plan, dict):
            return {}

        return {
            "scale": plan.get("scale"),
            "entity": _clean_str_list(plan.get("entity", []), maxn=10, allow_phrases=True),
            "role": plan.get("role"),
            "timeframe": plan.get("timeframe"),
            "include_terms": _clean_str_list(plan.get("include_terms", []), maxn=12, allow_phrases=False),
            "include_phrases": _clean_str_list(plan.get("include_phrases", []), maxn=12, allow_phrases=True),
            "exclude_terms": _clean_str_list(plan.get("exclude_terms", []), maxn=12, allow_phrases=True),
            "subject_hints": _clean_str_list(plan.get("subject_hints", []), maxn=8, allow_phrases=True),
        }

    except Exception:
        return {}


def enhanced_parse(
    ai_query: str,
    entity_detector,
    role_parser,
    scale_parser,
    timeframe_parser,
) -> Dict[str, object]:
    plan = llm_build_search_plan(ai_query)

    role = plan.get("role")
    scale = plan.get("scale")
    timeframe = plan.get("timeframe")
    entity = plan.get("entity")

    if not entity:
        entity = entity_detector(ai_query)
    if not role:
        role = role_parser(ai_query)
    if not scale:
        scale = scale_parser(ai_query)
    if not timeframe:
        timeframe = timeframe_parser(ai_query)

    return {
        "role": role,
        "scale": scale,
        "timeframe": timeframe,
        "entity": entity or [],
        "include_terms": plan.get("include_terms", []),
        "include_phrases": plan.get("include_phrases", []),
        "exclude_terms": plan.get("exclude_terms", []),
        "subject_hints": plan.get("subject_hints", []),
    }
