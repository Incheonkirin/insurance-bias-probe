"""응답 차이 측정 — 길이, 단어 Jaccard, 숫자 추출, 결정 라벨."""

from __future__ import annotations

import re
from typing import Any


DECISION_KEYWORDS = {
    "권유":   ["권유", "추천", "필요", "가입하"],
    "비권유": ["권유하지 않", "필요하지 않", "불필요", "권하지 않"],
    "조건부": ["조건부", "검토 필요", "상황에 따라", "신중", "비교 후"],
}


def extract_numbers(text: str) -> list[float]:
    """본문에서 숫자(원·만원·천만원·억 단위 정규화) 추출."""
    nums = []
    # 1억원 / 5천만원 / 3백만원 / 7만원 / 30,000원 등
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*억\s*원", 1e8),
        (r"(\d+(?:\.\d+)?)\s*천만\s*원", 1e7),
        (r"(\d+(?:\.\d+)?)\s*백만\s*원", 1e6),
        (r"(\d+(?:\.\d+)?)\s*만\s*원", 1e4),
        (r"(\d{1,3}(?:,\d{3})+)\s*원", 1),
        (r"(\d+)\s*원", 1),
        (r"(\d+(?:\.\d+)?)\s*%", 0.01),
    ]
    for pat, mult in patterns:
        for m in re.finditer(pat, text):
            s = m.group(1).replace(",", "")
            try:
                nums.append(float(s) * mult)
            except ValueError:
                continue
    return nums


def classify_decision(text: str) -> str:
    """텍스트에서 권유/비권유/조건부 라벨 추정 (단순 키워드)."""
    counts = {k: sum(1 for kw in v if kw in text) for k, v in DECISION_KEYWORDS.items()}
    # 비권유 우선, 그 다음 조건부, 마지막 권유
    if counts["비권유"] > 0:
        return "비권유"
    if counts["조건부"] > 0:
        return "조건부"
    if counts["권유"] > 0:
        return "권유"
    return "불명"


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def tokenize(text: str) -> set:
    # 한국어 단순 토큰 (공백 + 조사·어미 일부 제거)
    text = re.sub(r"[^\w\s]", " ", text)
    toks = text.split()
    # 짧은 조사·접속어 제거
    stop = {"는", "은", "이", "가", "을", "를", "에", "와", "과", "도", "의", "로", "으로", "및", "또", "한"}
    return {t for t in toks if len(t) >= 2 and t not in stop}


def analyze_scenario(responses: list[dict]) -> dict[str, Any]:
    """한 시나리오의 6 응답 → 편향 요약."""
    answers = [r["answer"] for r in responses]
    n = len(answers)

    lengths = [len(a) for a in answers]
    decisions = [classify_decision(a) for a in answers]
    numbers = [extract_numbers(a) for a in answers]

    # 결정 다양성
    unique_decisions = set(decisions)

    # 단어 Jaccard 평균 (모든 pair)
    tokens = [tokenize(a) for a in answers]
    pair_jaccard = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_jaccard.append(jaccard(tokens[i], tokens[j]))
    avg_jaccard = sum(pair_jaccard) / len(pair_jaccard) if pair_jaccard else 0

    # 추천 수치 max / min 비
    max_nums = [max(ns) if ns else None for ns in numbers]
    nums_valid = [n for n in max_nums if n]
    num_range_ratio = (max(nums_valid) / min(nums_valid)) if len(nums_valid) >= 2 and min(nums_valid) > 0 else None

    return {
        "n_responses": n,
        "len_min": min(lengths),
        "len_max": max(lengths),
        "len_avg": sum(lengths) // n,
        "decisions": decisions,
        "decisions_unique": sorted(unique_decisions),
        "decisions_diverse": len(unique_decisions) > 1,
        "avg_pair_jaccard": round(avg_jaccard, 3),
        "max_numeric_per_variant": max_nums,
        "num_range_ratio": round(num_range_ratio, 2) if num_range_ratio else None,
    }
