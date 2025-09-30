import json, argparse, random
from pathlib import Path
from common import read_md, normalize_ws, sys_prompt

random.seed(42)

def emit_sample(q, a, source):
    return {
        "messages": [
            {"role":"system","content": sys_prompt()},
            {"role":"user","content": q},
            {"role":"assistant","content": f"{a}\nSource: {source}"}
        ]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinic", default="Pune")
    ap.add_argument("--currency", default="INR")
    ap.add_argument("--treatments", required=True)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--faq", required=True)
    ap.add_argument("--recent", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train, val, evalq = [], [], []

    # Treatments → Q/A variants
    treatments = json.loads(Path(args.treatments).read_text(encoding="utf-8"))
    for t in treatments:
        q1 = f"How long does {t['name']} take and how many visits?"
        a1 = (f"Typically {t['duration_minutes']} minutes, about {t['visits']} visit(s). "
              f"Price band: ₹{t['price_band_inr'][0]}–₹{t['price_band_inr'][1]}.")
        train.append(emit_sample(q1, a1, "treatments.json#"+t["code"]))

        q2 = f"What are aftercare steps for {t['name']}?"
        a2 = " ; ".join(t["aftercare"])
        train.append(emit_sample(q2, a2, "treatments.json#"+t["code"]))

        q3 = f"Any risks with {t['name']}?"
        a3 = " ; ".join(t["risks"]) if t["risks"] else "Minimal risks when indicated."
        train.append(emit_sample(q3, a3, "treatments.json#"+t["code"]))

    # Policies + FAQ → Q/A
    for p in args.policies:
        text = read_md(Path(p))
        head = Path(p).stem
        if "appointments" in p:
            train.append(emit_sample("What are clinic hours?",
                                     "Mon–Sat 9:30–18:30 IST; call for emergency slots.",
                                     f"{head}.md#hours"))
        if "cancellations" in p:
            train.append(emit_sample("Cancellation policy?",
                                     "24h notice requested; same-day may incur ₹300 fee.",
                                     f"{head}.md#policy"))
        if "emergency" in p:
            train.append(emit_sample("What are dental red flags?",
                                     "Uncontrolled bleeding, facial swelling, high fever, trauma. Call immediately.",
                                     f"{head}.md#red-flags"))

    faq_md = read_md(Path(args.faq))
    train.append(emit_sample("Is scaling painful?",
                             "Mild discomfort; anesthesia for sensitive cases.",
                             "faq.md#scaling-pain"))
    train.append(emit_sample("Do whitening results last?",
                             "6–12 months depending on diet/habits.",
                             "faq.md#whitening-duration"))

    # Recent queries → Eval questions without answers (for retrieval tests)
    with Path(args.recent).open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            evalq.append({"question": obj["q"], "expect_source_hint": "recent_queries.jsonl"})

    # Split: 80/20 train/val
    random.shuffle(train)
    split = int(0.8 * len(train))
    Path(out / "train.jsonl").write_text("\n".join(json.dumps(x) for x in train[:split]), encoding="utf-8")
    Path(out / "val.jsonl").write_text("\n".join(json.dumps(x) for x in train[split:]), encoding="utf-8")
    Path(out / "eval.jsonl").write_text("\n".join(json.dumps(x) for x in evalq), encoding="utf-8")

    print(f"Wrote {len(train[:split])} train, {len(train[split:])} val, {len(evalq)} eval to {out}")

if __name__ == "__main__":
    main()
