"""定稿计算：先去重，再给出两条失效路径的最终数字与成本。

去重依据：request_id 是「全局唯一业务编号」，重复行内容完全一致，属导出侧重复。
主口径用去重后数据；原始口径保留为敏感性对照。
"""

import numpy as np
import pandas as pd

COST_FAIL, COST_SYNC = 75.0, 12.0

raw = pd.read_csv("data/chapter_requests.csv", dtype={"load_status": "string"})
df = raw.drop_duplicates(subset=["request_id"], keep="first").copy()

df["req"] = pd.to_datetime(df["req_time"], utc=True)
df["syn"] = pd.to_datetime(df["catalog_synced_at"], utc=True)
df["lag_min"] = (df["req"] - df["syn"]).dt.total_seconds() / 60

print("=== 0. 去重 ===")
print(f"原始行 {len(raw)}，唯一 request_id {df.request_id.nunique()}，"
      f"删除重复 {len(raw) - len(df)} 行")

rep = df[df["load_status"].notna()].copy()
rep["failed"] = (rep["load_status"] == "0").astype(int)
rep["stale"] = rep.lag_min > 20
rep["ttl_low"] = rep.cache_ttl_sec <= 5

print(f"有回传 {len(rep)}，失败 {int(rep.failed.sum())}，失败率 {rep.failed.mean():.2%}，"
      f"NULL {len(df) - len(rep)}")

print("\n=== 1. 主效应（去重后）===")
print(f"滞后<=20: n={int((~rep.stale).sum())} 失败率={rep.loc[~rep.stale, 'failed'].mean():.2%}")
print(f"滞后 >20: n={int(rep.stale.sum())} 失败率={rep.loc[rep.stale, 'failed'].mean():.2%}")
print(f"失败来自高滞后占比: {rep.loc[rep.stale, 'failed'].sum() / rep.failed.sum():.1%}")

print("\n=== 2. 区域 × 机制（关键表）===")
rows = []
for r, g in rep.groupby("region"):
    a = g[g.stale & ~g.ttl_low]
    b = g[~g.stale & g.ttl_low]
    c = g[~g.stale & ~g.ttl_low]
    d = g[g.stale & g.ttl_low]
    rows.append({
        "区域": r,
        "滞后>20,TTL>5": f"{a.failed.mean():.1%} (n={len(a)})" if len(a) else "-",
        "滞后>20,TTL<=5": f"{d.failed.mean():.1%} (n={len(d)})" if len(d) else "-",
        "滞后<=20,TTL<=5": f"{b.failed.mean():.1%} (n={len(b)})" if len(b) else "-",
        "滞后<=20,TTL>5": f"{c.failed.mean():.1%} (n={len(c)})" if len(c) else "-",
    })
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 3. 两条路径的独立检验 ===")
hd = rep[rep.region == "华东节点"]
oth = rep[rep.region != "华东节点"]
print(f"华东  TTL<=5: {int(hd[hd.ttl_low].failed.sum())}/{len(hd[hd.ttl_low])} "
      f"= {hd[hd.ttl_low].failed.mean():.1%}")
print(f"华东  TTL >5: {int(hd[~hd.ttl_low].failed.sum())}/{len(hd[~hd.ttl_low])} "
      f"= {hd[~hd.ttl_low].failed.mean():.1%}")
print(f"华东  滞后<=20: {hd[~hd.stale].failed.mean():.1%}  |  滞后>20: {hd[hd.stale].failed.mean():.1%}"
      f"  ← 滞后对华东几乎无区分度")
print(f"其他  TTL<=5 & 滞后<=20: {oth[oth.ttl_low & ~oth.stale].failed.mean():.1%}"
      f"  ← 同样条件下其他节点几乎不失败")
print(f"其他  滞后>20: {oth[oth.stale].failed.mean():.1%}  ← 主力失效路径")

print("\n=== 4. 成本（去重后）===")
n_fail = int(rep.failed.sum())
print(f"当前失败 {n_fail} 次 → {n_fail * COST_FAIL:.0f} 分钟")

# 路径A：四区目录滞后兜底
a_risk = oth[oth.stale]
print(f"\n策略A（四区滞后<=20）: 高风险 {len(a_risk)}，可避免失败 {int(a_risk.failed.sum())}，"
      f"省 {a_risk.failed.sum() * COST_FAIL:.0f}，成本上界 {len(a_risk) * COST_SYNC:.0f}，"
      f"净收益 {a_risk.failed.sum() * COST_FAIL - len(a_risk) * COST_SYNC:+.0f} 分钟")

# 路径B：华东缓存 TTL 兜底（成本口径未给定，只修收益与残余）
b_risk = hd[hd.ttl_low]
print(f"策略B（华东 TTL<=5 预热）: 高风险 {len(b_risk)}，可避免失败 {int(b_risk.failed.sum())}，"
      f"省 {b_risk.failed.sum() * COST_FAIL:.0f} 分钟（预热成本口径未给定，不做净收益断言）")

print(f"\n两条都做后残余失败: {int(rep[(~rep.stale) & (~(rep.ttl_low & (rep.region == '华东节点')))].failed.sum())}")
resid = rep[(~rep.stale) & ~((rep.ttl_low) & (rep.region == "华东节点"))]
print(f"残余失败 {int(resid.failed.sum())} 次 → {resid.failed.sum() * COST_FAIL:.0f} 分钟，"
      f"较当前下降 {1 - resid.failed.sum() / n_fail:.1%}")

print("\n=== 5. 敏感性：不去重 / NULL 计失败 ===")
rep_raw = raw[raw.load_status.notna()].copy()
rep_raw["failed"] = (rep_raw["load_status"] == "0").astype(int)
print(f"不去重: 失败率 {rep_raw.failed.mean():.2%}，失败 {int(rep_raw.failed.sum())} 次 → "
      f"{rep_raw.failed.sum() * COST_FAIL:.0f} 分钟")
n_all = int((df.load_status == "0").sum() + df.load_status.isna().sum())
print(f"NULL 计失败: {n_all}/{len(df)} = {n_all / len(df):.2%} → {n_all * COST_FAIL:.0f} 分钟")
