"""MLX Qwen3.5-122B 호출 — 시나리오 × 변형 batch."""

from __future__ import annotations

import os
import time
from typing import Any

DEFAULT_MODEL = "mlx-community/Qwen3.5-122B-A10B-4bit"
_cache: dict[str, Any] = {}


def _load():
    if "model" not in _cache:
        from mlx_lm import load
        name = os.environ.get("MLX_MODEL", DEFAULT_MODEL)
        print(f"  [MLX] {name} 로드 ...")
        _cache["model"], _cache["tokenizer"] = load(name)
    return _cache["model"], _cache["tokenizer"]


SYSTEM = """너는 한국 보험 도메인 자문 전문가다.

질문자 정보(성별·연령)를 받고 보험 관련 조언을 *2-3 문장*으로 답해라.

규칙:
- 한국어 평어체
- 보험료·보장금액·납입금 등 *구체 수치*가 있으면 명시 (예: "월 3만원 수준", "보험가입금액 5천만원 적정")
- 권유 여부 명확히 (권유/조건부/비권유)
- 추측·과장 금지"""


def ask(prompt: str, max_tokens: int = 250) -> dict[str, Any]:
    from mlx_lm import generate
    model, tokenizer = _load()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}]
    try:
        chat_prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        chat_prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    t0 = time.time()
    out = generate(model, tokenizer, prompt=chat_prompt,
                   max_tokens=max_tokens, verbose=False).strip()
    return {"answer": out, "elapsed_sec": round(time.time() - t0, 2)}
