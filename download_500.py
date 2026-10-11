import re
import json
import os
from datasets import load_dataset

print("[*] Downloading 500 samples from CNN/DailyMail (test split)...")
ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test[:500]")

data = []
for i, ex in enumerate(ds):
    text = re.sub(r'\s+', ' ', ex["article"]).strip()
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text) if s.strip()]
    ref = re.sub(r'\s+', ' ', ex["highlights"]).strip()
    if len(sents) >= 5 and ref:
        data.append({
            "id": ex.get("id", str(i)),
            "sentences": sents[:40],
            "highlights": ref
        })

os.makedirs("data/raw", exist_ok=True)
with open("data/raw/cnn_dm_test.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"[✔] Successfully saved {len(data)} samples to data/raw/cnn_dm_test.json")
