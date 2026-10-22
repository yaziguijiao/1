"""稳健性检验：控制滞后后，区域/品类/缓存是否还有解释力；以及策略净收益。

结论要能扛住质疑，所以把“滞后是不是真因”做成分层对照：
在同一个滞后层内比较各区域、各品类、各 TTL 档，看差异是否消失。
"""

import numpy as np
import pandas as pd

COST_FAIL, COST_SYNC = 75.0, 12.0

df = pd.read_csv("data/chapter_requests.csv", dtype={"load_status": "string"})
df["req"] = pd.to_datetime(df["req_time"], utc=True)
df["syn"] = pd.to_datetime(df["catalog_synced_at"], utc=True)
df["lag_min"] = (df["req"] - df["syn"]).dt.total_seconds() / 60
rep = df[df["load_status"].notna()].copy()
rep["failed"] = (rep["load_status"] == "0").astype(int)

print("=== A. 滞后分布的空洞（是否存在 20~22 分钟样本）===")
for lo, hi in [(18, 20), (20, 21), (21, 22), (22, 24)]:
    s = rep[(rep.lag_min >= lo) & (rep.lag_min < hi)]
    print(f"[{lo},{hi})  n={len(s)}  失败率={s.failed.mean() if len(s) else float('nan'):.1%}")

print("\n=== B. 整体对照 ===")
lo = rep[rep.lag_min <= 20]
hi = rep[rep.lag_min > 20]
print(f"滞后<=20: n={len(lo)} 失败{int(lo.failed.sum())} 失败率={lo.failed.mean():.2%}")
print(f"滞后 >20: n={len(hi)} 失败{int(hi.failed.sum())} 失败率={hi.failed.mean():.2%}")
print(f"失败请求中滞后>20的比例: {hi.failed.sum() / rep.failed.sum():.1%}")
print(f"风险比 RR = {hi.failed.mean() / lo.failed.mean():.1f}")

print("\n=== C. 控制滞后后看区域（各层内失败率）===")
for name, sub in [("滞后<=20", lo), ("滞后>20", hi)]:
    g = sub.groupby("region").agg(n=("failed", "size"), 失败=("failed", "sum"))
    g["失败率"] = (g.失败 / g.n * 100).round(1)
    print(f"\n{name}:\n{g.to_string()}")

print("\n=== D. 控制滞后后看品类 ===")
for name, sub in [("滞后<=20", lo), ("滞后>20", hi)]:
    g = sub.groupby("category").agg(n=("failed", "size"), 失败=("failed", "sum"))
    g["失败率"] = (g.失败 / g.n * 100).round(1)
    print(f"\n{name}:\n{g.sort_values('失败率', ascending=False).to_string()}")

print("\n=== E. 控制滞后后看缓存 TTL（各层内按 TTL 二分）===")
for name, sub in [("滞后<=20", lo), ("滞后>20", hi)]:
    sub = sub.assign(ttl_hi=sub.cache_ttl_sec > sub.cache_ttl_sec.median())
    g = sub.groupby("ttl_hi").agg(n=("failed", "size"), 失败=("failed", "sum"))
    g["失败率"] = (g.失败 / g.n * 100).round(1)
    g.index = ["TTL低半区", "TTL高半区"]
    print(f"\n{name}:\n{g.to_string()}")

print("\n=== F. NULL 敏感性：把未回传计为失败 ===")
n_all = len(df)
n_fail_all = int((df.load_status == "0").sum()) + int(df.load_status.isna().sum())
print(f"仅计已回传: {int(rep.failed.sum())}/{len(rep)} = {rep.failed.mean():.1%}")
print(f"NULL全计失败: {n_fail_all}/{n_all} = {n_fail_all / n_all:.1%}")
print(f"失败请求数上限 {n_fail_all} → 损失 {n_fail_all * COST_FAIL:.0f} 分钟")

print("\n=== G. 策略净收益（悲观：每个高风险请求各算一次同步，不考虑合并）===")
n_sync = len(hi)
avoid = int(hi.failed.sum())
save = avoid * COST_FAIL
cost = n_sync * COST_SYNC
print(f"高风险请求(滞后>20) {n_sync} 个，其中失败 {avoid} 个")
print(f"避免损失 {save:.0f} 分钟 - 同步成本 {cost:.0f} 分钟 = 净收益 {save - cost:+.0f} 分钟")
print(f"投入产出比 {save / cost:.1f} : 1")

print("\n=== H. 策略净收益（乐观：按“滞后>20 的同步版本”合并计费）===")
stale_versions = hi.groupby(["region", "catalog_synced_at"]).ngroups
save2, cost2 = avoid * COST_FAIL, stale_versions * COST_SYNC
print(f"需提前同步的不同版本 {stale_versions} 个 → 成本 {cost2:.0f}，净收益 {save2 - cost2:+.0f} 分钟")

print("\n=== I. 按剩余风险的失败率上界（滞后<=20 仍失败的部分）===")
print(f"压到<=20 后仍会失败: {int(lo.failed.sum())} 次 → 残余损失 {lo.failed.sum() * COST_FAIL:.0f} 分钟")
print(f"相比当前 {rep.failed.sum() * COST_FAIL:.0f} 分钟，下降 "
      f"{1 - lo.failed.sum() / rep.failed.sum():.1%}")
