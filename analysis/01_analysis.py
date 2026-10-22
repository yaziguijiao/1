"""主分析：找出章节加载失败的机制，并量化成本。

口径（题目给定）：
  每次加载失败   ≈ 损失 75 分钟
  每提前触发一次目录同步 ≈ 花费 12 分钟
失败率只在 load_status 有回传（0/1）的样本上计算，NULL 单独报告。
"""

import numpy as np
import pandas as pd

CSV = "data/chapter_requests.csv"
COST_FAIL = 75.0   # 分钟 / 次失败
COST_SYNC = 12.0   # 分钟 / 次提前同步

df = pd.read_csv(CSV, dtype={"load_status": "string"})
df["req"] = pd.to_datetime(df["req_time"], utc=True)
df["syn"] = pd.to_datetime(df["catalog_synced_at"], utc=True)
df["lag_min"] = (df["req"] - df["syn"]).dt.total_seconds() / 60

rep = df[df["load_status"].notna()].copy()
rep["failed"] = (rep["load_status"] == "0").astype(int)

print(f"总请求 {len(df)}，有回传 {len(rep)}，未回传 NULL {len(df) - len(rep)}")
print(f"失败 {int(rep.failed.sum())}，成功率 {1 - rep.failed.mean():.2%}，"
      f"失败率 {rep.failed.mean():.2%}")

print("\n=== 1. 同步滞后 → 失败率 ===")
bins = [0, 10, 20, 30, 40, 50, 60, 71]
rep["lag_bin"] = pd.cut(rep.lag_min, bins=bins, right=False,
                        labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+"])
t = rep.groupby("lag_bin", observed=True).agg(请求数=("failed", "size"), 失败数=("failed", "sum"))
t["失败率"] = (t.失败数 / t.请求数).map("{:.1%}".format)
print(t.to_string())

print("\n=== 2. 缓存剩余有效时长 → 失败率 ===")
rep["ttl_bin"] = pd.cut(rep.cache_ttl_sec, [-1, 5, 10, 20, 30],
                        labels=["0-5s", "6-10s", "11-20s", "21-30s"])
t2 = rep.groupby("ttl_bin", observed=True).agg(请求数=("failed", "size"), 失败数=("failed", "sum"))
t2["失败率"] = (t2.失败数 / t2.请求数).map("{:.1%}".format)
print(t2.to_string())

print("\n=== 3. 滞后 × 缓存 交叉失败率 ===")
pv = rep.pivot_table(index="lag_bin", columns="ttl_bin", values="failed",
                     aggfunc=["size", "mean"], observed=True)
print("请求数:\n", pv["size"].to_string())
print("失败率:\n", (pv["mean"] * 100).round(1).to_string())

print("\n=== 4. 区域 ===")
t4 = rep.groupby("region").agg(请求数=("failed", "size"), 失败数=("failed", "sum"),
                               滞后中位数=("lag_min", "median"), 滞后均值=("lag_min", "mean"))
t4["失败率"] = (t4.失败数 / t4.请求数).map("{:.1%}".format)
print(t4.round(1).to_string())

print("\n=== 5. 品类 ===")
t5 = rep.groupby("category").agg(请求数=("failed", "size"), 失败数=("failed", "sum"))
t5["失败率"] = (t5.失败数 / t5.请求数).map("{:.1%}".format)
print(t5.sort_values("失败率", ascending=False).to_string())

print("\n=== 6. 滞后是否随日期累积 ===")
rep["date"] = rep["req"].dt.tz_convert("Asia/Shanghai").dt.date
t6 = rep.groupby("date").agg(请求数=("failed", "size"), 失败数=("failed", "sum"),
                             滞后中位数=("lag_min", "median"))
t6["失败率"] = (t6.失败数 / t6.请求数).map("{:.1%}".format)
print(t6.round(1).to_string())

print("\n=== 7. 成本量化（当前 vs 策略） ===")
n_fail = int(rep.failed.sum())
print(f"当前：失败 {n_fail} 次 → {n_fail * COST_FAIL:.0f} 分钟损失")

# 若把「请求时滞后」压到阈值 T 以内，能避免多少失败？代价是按需同步次数。
print("\n阈值T | 落在T外的请求 | 其中失败 | 可避免失败 | 避免损失(min) | 需同步次数 | 同步成本(min) | 净收益(min)")
for T in [10, 15, 20, 25, 30, 40, 50]:
    outside = rep[rep.lag_min > T]
    # 每个落在 T 外的请求，视为需要在该时刻前补一次同步（保守假设）
    n_sync = len(outside)
    avoided = int(outside.failed.sum())
    save = avoided * COST_FAIL
    cost = n_sync * COST_SYNC
    print(f"{T:>5} | {len(outside):>13} | {avoided:>8} | {avoided:>10} | {save:>13.0f} | {n_sync:>10} | {cost:>12.0f} | {save - cost:>+11.0f}")
