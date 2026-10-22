"""数据侦察：结构、缺失、取值分布、时间跨度。只读，不改原始数据。"""

import pandas as pd

CSV = "data/chapter_requests.csv"

df = pd.read_csv(CSV, dtype={"load_status": "string"})

print("shape:", df.shape)
print("\ncolumns / dtypes / nulls:")
print(df.isna().sum().to_string())
print("\n--- head ---")
print(df.head(3).to_string())

req = pd.to_datetime(df["req_time"], utc=True)
syn = pd.to_datetime(df["catalog_synced_at"], utc=True)
lag_min = (req - syn).dt.total_seconds() / 60

print("\n--- 时间 ---")
print("req_time  范围:", req.min(), "→", req.max())
print("synced    范围:", syn.min(), "→", syn.max())
print("同步滞后(分钟) describe:")
print(lag_min.describe().to_string())
print("滞后为负(同步晚于请求)条数:", int((lag_min < 0).sum()))

print("\n--- 分类列 ---")
for col in ["region", "category", "load_status"]:
    print(f"\n{col} (nunique={df[col].nunique(dropna=True)}):")
    print(df[col].value_counts(dropna=False).to_string())

print("\n--- 数值列 ---")
for col in ["cache_ttl_sec", "chapter_words"]:
    print(f"\n{col}:")
    print(df[col].describe().to_string())

print("\n--- load_status 与同步滞后 ---")
tmp = df.assign(lag_min=lag_min)
print(tmp.groupby(tmp["load_status"].fillna("NULL"))["lag_min"]
      .agg(["count", "mean", "median", "min", "max"]).round(1).to_string())
