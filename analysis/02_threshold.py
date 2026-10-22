"""阈值定位与同步节拍：确认 20 分钟是否是真实拐点，并看同步本身的频率结构。"""

import numpy as np
import pandas as pd

df = pd.read_csv("data/chapter_requests.csv", dtype={"load_status": "string"})
df["req"] = pd.to_datetime(df["req_time"], utc=True)
df["syn"] = pd.to_datetime(df["catalog_synced_at"], utc=True)
df["lag_min"] = (df["req"] - df["syn"]).dt.total_seconds() / 60
rep = df[df["load_status"].notna()].copy()
rep["failed"] = (rep["load_status"] == "0").astype(int)

print("=== A. 细粒度拐点（每 2 分钟一档） ===")
for lo in range(0, 70, 2):
    s = rep[(rep.lag_min >= lo) & (rep.lag_min < lo + 2)]
    if len(s) >= 5:
        print(f"{lo:>2}-{lo + 2:>2} min   n={len(s):>4}  失败率={s.failed.mean():6.1%}")

print("\n=== B. 阈值候选：滞后>T 判为高风险，与真实失败的一致性 ===")
print("阈值T | 高风险n | 命中失败(TP) | FP | FN | 精确率 | 召回率 | 提升度")
base = rep.failed.mean()
for T in [12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 30]:
    hi = rep.lag_min > T
    tp = int((hi & (rep.failed == 1)).sum())
    fp = int((hi & (rep.failed == 0)).sum())
    fn = int((~hi & (rep.failed == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    print(f"{T:>5} | {tp + fp:>7} | {tp:>12} | {fp:>2} | {fn:>2} | {prec:>6.1%} | {rec:>6.1%} | {prec / base:>5.1f}x")

print("\n=== C. 目录同步节拍：每个区域的不同同步时间点 ===")
for r, g in df.groupby("region"):
    syn = sorted(g.syn.unique())
    gaps = np.diff(np.array(syn, dtype="datetime64[m]")).astype(float)
    print(f"{r}: 请求{len(g)}，不同同步时间点{len(syn)}个，"
          f"相邻同步间隔 中位数={np.median(gaps):.0f}min 最小={gaps.min():.0f} 最大={gaps.max():.0f}")

print("\n=== D. 滞后是否“整块”出现（同一同步版本被反复使用）===")
print("重复度：每个 catalog_synced_at 平均承载请求数 =",
      round(len(df) / df.catalog_synced_at.nunique(), 2))

print("\n=== E. TTL 在低滞后/高滞后下的分布（看是否独立） ===")
for label, s in [("滞后<=20", rep[rep.lag_min <= 20]), ("滞后>20", rep[rep.lag_min > 20])]:
    print(f"{label}: n={len(s)}  失败率={s.failed.mean():.1%}  "
          f"TTL中位数={s.cache_ttl_sec.median():.0f}s  TTL均值={s.cache_ttl_sec.mean():.1f}s")

print("\n=== F. 双重条件：高滞后 且 缓存将过期 ===")
q = rep[(rep.lag_min > 20) & (rep.cache_ttl_sec <= 5)]
r1 = rep[(rep.lag_min <= 20) & (rep.cache_ttl_sec > 5)]
print(f"高滞后+低TTL: n={len(q)} 失败率={q.failed.mean():.1%}")
print(f"低滞后+高TTL: n={len(r1)} 失败率={r1.failed.mean():.1%}")

print("\n=== G. NULL 未回传的性质 ===")
nul = df[df.load_status.isna()]
print(f"n={len(nul)}，滞后中位数={nul.lag_min.median():.0f}，滞后>20占比={(nul.lag_min > 20).mean():.1%}")
print(nul.region.value_counts().to_string())
