import os
import re
import json
import argparse
from datasets import load_dataset

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')

def split_sentences(text: str, max_sents: int = 40):
    text = re.sub(r'\s+', ' ', text).strip()
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    return sents[:max_sents]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="data/raw/cnn_dm_test.json")
    args = ap.parse_args()

    print(f"[*] Downloading CNN/DailyMail 3.0.0 [{args.split}[:{args.n}]] ...")
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"{args.split}[:{args.n}]")

    data = []
    for i, ex in enumerate(ds):
        sents = split_sentences(ex["article"])
        ref = re.sub(r'\s+', ' ', ex["highlights"]).strip()
        if len(sents) < 5 or not ref:
            continue
        data.append({
            "id": ex.get("id", str(i)),
            "sentences": sents,
            "highlights": ref
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved {len(data)} samples -> {args.out}")

if __name__ == "__main__":
    main()
