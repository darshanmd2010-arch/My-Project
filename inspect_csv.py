import csv
import glob

path = glob.glob("uploads/*.csv")[0]
with open(path, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print("File:", path, "| Rows:", len(rows))
print()
for col in rows[0].keys():
    vals = [r[col] for r in rows]
    missing = sum(1 for v in vals if v is None or v.strip() == "")
    uniq = sorted(set(v for v in vals if v and v.strip()))
    if len(uniq) <= 12:
        info = uniq
    else:
        try:
            nums = [float(v) for v in vals if v.strip()]
            info = f"numeric min={min(nums)} max={max(nums)}"
        except ValueError:
            info = f"text, {len(uniq)} unique, e.g. {uniq[:3]}"
    print(f"{col} | missing={missing} | {info}")
