# -*- coding: utf-8 -*-
"""
nlu_router.py

Original lightweight rule-based NLU (intent classification and year extraction)
is preserved exactly. This file appends a safe OpenAI summarizer and a small
helper `route_and_answer(agent, user_text, use_llm=True)` that refines the
rule-based answer with ChatGPT **only if** a valid API key is present.

This change is intentionally non-invasive: if OpenAI is not available or the
call fails, the original text answer is returned unchanged.
"""

from typing import Optional
import re
import os
import logging
from dotenv import load_dotenv

# ---- Original rule-based NLU unchanged ----
INTENT_KEYWORDS = {
    "labour_overview": [
        "labour", "labor", "employment", "unemployment", "workforce",
        "jobs", "workers", "labour market", "labor market",
    ],
    "top_occupations": [
        "top occupation", "most common jobs", "most common occupations",
        "top jobs", "popular jobs", "occupation",
    ],
    "households": [
        "household", "households", "family", "families",
    ],
    "density": [
        "population density", "densely populated", "density",
    ],
    "housing_units": [
        "housing units", "dwellings", "apartments", "houses",
        "residential units",
    ],
    "students": [
        "students", "school enrollment", "pupils", "enrolment",
    ],
    "teachers": [
        "teachers", "teaching staff", "instructors",
    ],
    "higher_education": [
        "higher education", "university", "universities", "college",
        "tertiary education",
    ],
}


def classify_intent(question: str) -> str:
    q = question.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return intent
    if "population" in q:
        return "density"
    return "labour_overview" if "unemployment" in q or "employment" in q else "unknown"


def extract_year(question: str, default_year: Optional[int] = None) -> Optional[int]:
    years = re.findall(r"\b(19[0-9]{2}|20[0-9]{2}|2100)\b", question)
    if years:
        try:
            return int(years[0])
        except ValueError:
            pass
    return default_year

# ---- End original NLU ----

# ---- New: safe OpenAI summarizer layer ----
load_dotenv()  # load .env from project root (where you placed OPENAI_API_KEY)

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change if you prefer

# Try to import the openai package. If missing we will skip LLM calls.
try:
    import openai  # standard package
except Exception:  # ImportError or others
    openai = None
    LOG.debug("openai package not importable. LLM refinement disabled.")

# Try to support both old-style openai.ChatCompletion and new-style OpenAI client
_have_old_chat = False
_have_new_client = False
_new_client_cls = None

if openai:
    # detect old-style ChatCompletion API
    if hasattr(openai, "ChatCompletion"):
        _have_old_chat = True
    # detect new-style client class (openai.OpenAI or importable as class)
    try:
        # new SDK sometimes exposes "OpenAI" class
        from openai import OpenAI as _OpenAI  # type: ignore
        _have_new_client = True
        _new_client_cls = _OpenAI
    except Exception:
        # not available
        _have_new_client = False

# If API key present, attempt to configure clients lazily later
def _call_openai_chat_old(prompt_messages, model, max_tokens=400, temperature=0.2):
    """Call old-style openai.ChatCompletion.create"""
    resp = openai.ChatCompletion.create(
        model=model,
        messages=prompt_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # safe extraction
    return resp

def _call_openai_chat_new(prompt_messages, model, max_tokens=400, temperature=0.2):
    """Call new-style OpenAI client"""
    client = _new_client_cls(api_key=OPENAI_API_KEY)  # create client instance
    resp = client.chat.completions.create(
        model=model,
        messages=prompt_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp

def summarize_with_llm(text: str, system_prompt: str = None, max_tokens: int = 400) -> str:
    """
    Try to call OpenAI to refine the given text.
    - Will not raise; returns original text on any error.
    - Tries both old and new SDK patterns depending on what's available.
    """
    if not OPENAI_API_KEY:
        LOG.debug("OPENAI_API_KEY not set; skipping LLM summarization.")
        return text
    if not openai:
        LOG.debug("openai package not available; skipping LLM summarization.")
        return text

    system_msg = system_prompt or (
        "You are a helpful statistician and data analyst. "
        "Edit the user's answer to be concise, clear and conversational while preserving all factual numbers exactly."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Rewrite this answer to be concise and clear while preserving numerical facts exactly:\n\n{text}"},
    ]

    # Try old-style API first (most installations with 'openai' support this)
    try:
        if _have_old_chat:
            LOG.debug("Attempting old-style openai.ChatCompletion.create()")
            openai.api_key = OPENAI_API_KEY  # configure
            resp = _call_openai_chat_old(messages, model=OPENAI_MODEL, max_tokens=max_tokens)
            if resp and "choices" in resp and len(resp["choices"]) > 0:
                content = resp["choices"][0].get("message", {}).get("content")
                if content:
                    return content.strip()
    except Exception as e_old:
        # log the error for debugging but continue to try new-style if available
        LOG.exception("Old-style OpenAI call failed: %s", e_old)

    # Try new-style client if available
    try:
        if _have_new_client and _new_client_cls is not None:
            LOG.debug("Attempting new-style OpenAI client.chat.completions.create()")
            resp2 = _call_openai_chat_new(messages, model=OPENAI_MODEL, max_tokens=max_tokens)
            # new-style SDK may return nested objects; try to extract text safely
            try:
                # try common keys
                choices = getattr(resp2, "choices", None) or resp2.get("choices", None)
                if choices and len(choices) > 0:
                    # many SDKs use choices[0].message.content or choices[0]["message"]["content"]
                    first = choices[0]
                    # handle object with attributes
                    if hasattr(first, "message") and hasattr(first.message, "content"):
                        return first.message.content.strip()
                    # handle dict
                    if isinstance(first, dict):
                        msg = first.get("message", {})
                        if isinstance(msg, dict) and "content" in msg:
                            return msg["content"].strip()
            except Exception:
                LOG.exception("Failed to parse new-style OpenAI response structure.")
    except Exception as e_new:
        LOG.exception("New-style OpenAI client call failed: %s", e_new)

    # If everything failed, return original text
    LOG.debug("LLM refinement did not produce a transformed answer; returning original text.")
    return text


def route_and_answer(agent, user_text: str, use_llm: bool = True) -> str:
    """
    Produce an answer using the existing agent and optionally refine with LLM.
    This helper makes a single small change to the output path — the agent logic itself is unchanged.
    """
    try:
        raw = agent.answer_question(user_text)
    except Exception as e:
        LOG.exception("Agent.answer_question raised an exception")
        return f"Error producing answer: {e}"

    if not use_llm:
        return raw

    # Run LLM refinement; this will not raise
    refined = summarize_with_llm(raw)
    return refined
