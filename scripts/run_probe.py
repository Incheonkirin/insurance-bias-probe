"""60 호출 + 분석 + HTML 사이트.

실행:
    uv run python scripts/run_probe.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.probe.analyze import analyze_scenario
from src.probe.run_probe import ask


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config" / "scenarios.yaml").read_text(encoding="utf-8"))
    scenarios = cfg["scenarios"]
    genders = cfg["variations"]["genders"]
    ages = cfg["variations"]["ages"]

    total = len(scenarios) * len(genders) * len(ages)
    print(f"[1] {len(scenarios)} 시나리오 × {len(genders)} 성별 × {len(ages)} 연령 = {total} 호출\n")

    all_results = []
    t_start = time.time()
    counter = 0
    for sc in scenarios:
        responses = []
        for g in genders:
            for a in ages:
                counter += 1
                prompt = sc["prompt"].format(gender=g, age=a)
                res = ask(prompt)
                responses.append({
                    "gender": g, "age": a, "prompt": prompt,
                    "answer": res["answer"], "elapsed_sec": res["elapsed_sec"],
                })
        analysis = analyze_scenario(responses)
        all_results.append({
            "code": sc["code"], "category": sc["category"],
            "responses": responses, "analysis": analysis,
        })
        print(f"  [{counter:>2}/{total}] {sc['code']} {sc['category']:>6}  "
              f"결정 {analysis['decisions']}  jaccard {analysis['avg_pair_jaccard']}")

    elapsed = time.time() - t_start

    # 저장
    out_dir = root / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    json_path = out_dir / f"probe_{today}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(),
                   "n_calls": total, "elapsed_sec": round(elapsed, 1),
                   "scenarios": all_results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[JSON] {json_path}")
    print(f"[총 소요] {elapsed:.1f}초")

    # 요약 — 결정 다양한 시나리오, 추천 수치 차이 큰 시나리오
    diverse_count = sum(1 for r in all_results if r["analysis"]["decisions_diverse"])
    high_num_var = [r for r in all_results
                    if r["analysis"]["num_range_ratio"] and r["analysis"]["num_range_ratio"] > 2.0]
    print(f"\n[요약] 결정 다양한 시나리오: {diverse_count}/{len(scenarios)}")
    print(f"       추천 수치 max/min > 2.0인 시나리오: {len(high_num_var)}")

    # HTML
    html = _render(all_results, elapsed, diverse_count)
    html_path = root / "docs" / "index.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    print(f"[site] {html_path}")


def _render(results, elapsed, diverse_count) -> str:
    cards = "\n".join(_card(r) for r in results)
    return f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="UTF-8"><title>Insurance Bias Probe v0</title>
<style>
  body {{ font-family: -apple-system, "Apple SD Gothic Neo", sans-serif;
          max-width: 1080px; margin: 30px auto; padding: 0 20px; line-height: 1.55; color: #222; }}
  h1 {{ font-size: 1.7rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 28px; padding-bottom: 4px; border-bottom: 2px solid #305496; color: #305496; }}
  .meta {{ color: #777; font-size: 0.88rem; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 10px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
  th {{ background: #305496; color: white; font-size: 0.82rem; }}
  .answer {{ max-width: 280px; }}
  .stats {{ background: #f5f5f5; padding: 8px 12px; border-radius: 4px; font-size: 0.88rem; }}
  .diverse {{ color: #d32f2f; font-weight: 600; }}
  .consistent {{ color: #2e7d32; }}
  .footer {{ color: #888; font-size: 0.82rem; margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd; }}
</style></head><body>
<h1>Insurance Bias Probe v0</h1>
<p class="meta">10 시나리오 × 2 성 × 3 연령 = 60 호출  ·  MLX Qwen3.5-122B-A10B  ·  {elapsed:.0f}초  ·  결정 다양 {diverse_count}/10</p>
{cards}
<div class="footer">
v0 한계 — 단일 LLM, 60 호출(통계적 약함), 법적 허용 차등과 부당 편향 분리 X.<br/>
<a href="https://github.com/Incheonkirin/insurance-bias-probe-v0">github.com/Incheonkirin/insurance-bias-probe-v0</a>
</div>
</body></html>"""


def _card(r):
    rows = "\n".join(
        f"<tr><td>{resp['gender']}</td><td>{resp['age']}</td>"
        f"<td class='answer'>{resp['answer']}</td></tr>"
        for resp in r["responses"]
    )
    a = r["analysis"]
    diverse_cls = "diverse" if a["decisions_diverse"] else "consistent"
    diverse_text = "결정 일관 X" if a["decisions_diverse"] else "결정 일관"
    num_text = (f"max/min {a['num_range_ratio']}x" if a["num_range_ratio"]
                else "수치 비교 불가")
    return f"""<h2>{r['code']}  {r['category']}</h2>
<div class="stats">
  <span class="{diverse_cls}">{diverse_text}</span>  ·  결정 {a['decisions']}  ·  단어 Jaccard {a['avg_pair_jaccard']}  ·  추천 수치 {num_text}
</div>
<table>
<tr><th>성</th><th>연령</th><th>응답</th></tr>
{rows}
</table>"""


if __name__ == "__main__":
    main()
