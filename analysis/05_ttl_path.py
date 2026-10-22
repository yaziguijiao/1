"""第二条故障路径：低 TTL 是否独立于区域；以及数据里的重复行。

前一轮发现华东低滞后层 TTL<=5 失败率 44%，需要判断：
  (a) 是华东特有的第二故障路径，还是
  (b) 全平台通用的「缓存即将过期 + 目录新鲜」失效模式，只是华东样本更集中。
"""

import pandas as pd

df = pd.read_csv("data/chapter_requests.csv", dtype={"load_status": "string"})
df["req"] = pd.to_datetime(df["req_time"], utc=True)
df["syn"] = pd.to_datetime(df["catalog_synced_at"], utc=True)
df["lag_min"] = (df["req"] - df["syn"]).dt.total_seconds() / 60
rep = df[df["load_status"].notna()].copy()
rep["failed"] = (rep["load_status"] == "0").astype(int)

print("=== 0. 数据质量：重复 ===")
dup_id = df[df.duplicated("request_id", keep=False)]
print(f"request_id 重复行数: {len(dup_id)}")
if len(dup_id):
    print(dup_id.sort_values("request_id").to_string(index=False))
print(f"整行完全重复（除 request_id 外全同）: "
      f"{df.drop(columns=['request_id']).duplicated().sum()}")
print(f"唯一 request_id 数: {df.request_id.nunique()} / {len(df)}")

print("\n=== 1. 低滞后层内：TTL<=5 与区域 ===")
lo = rep[rep.lag_min <= 20]
g = lo.groupby("region").apply(
    lambda s: pd.Series({
        "TTL<=5 的n": int((s.cache_ttl_sec <= 5).sum()),
        "TTL<=5 失败": int(((s.cache_ttl_sec <= 5) & (s.failed == 1)).sum()),
        "TTL<=5 失败率": (s.loc[s.cache_ttl_sec <= 5, "failed"].mean() if (s.cache_ttl_sec <= 5).any() else float("nan")),
        "TTL>5 的n": int((s.cache_ttl_sec > 5).sum()),
        "TTL>5 失败率": (s.loc[s.cache_ttl_sec > 5, "failed"].mean() if (s.cache_ttl_sec > 5).any() else float("nan")),
    }),
    include_groups=False,
)
g["TTL<=5 失败率"] = (g["TTL<=5 失败率"] * 100).round(1)
g["TTL>5 失败率"] = (g["TTL>5 失败率"] * 100).round(1)
print(g.to_string())

print("\n=== 2. 全样本：TTL<=5 分区域失败率（不控制滞后）===")
g2 = rep[rep.cache_ttl_sec <= 5].groupby("region").agg(
    n=("failed", "size"), 失败=("failed", "sum"))
g2["失败率"] = (g2.失败 / g2.n * 100).round(1)
print(g2.to_string())

print("\n=== 3. 二因素交叉：TTL 是否只在低滞后时才有解释力 ===")
rep["ttl_low"] = rep.cache_ttl_sec <= 5
rep["lag_hi"] = rep.lag_min > 20
pv = rep.pivot_table(index="lag_hi", columns="ttl_low", values="failed",
                     aggfunc=["size", "mean"], observed=True)
print("请求数:\n", pv["size"].to_string())
print("失败率(%):\n", (pv["mean"] * 100).round(1).to_string())

print("\n=== 4. 剔除华东后，低滞后 + TTL<=5 是否仍然高失败 ===")
sub = lo[(lo.region != "华东节点") & (lo.cache_ttl_sec <= 5)]
print(f"其他四区 低滞后&TTL<=5: n={len(sub)} 失败={int(sub.failed.sum())} "
      f"失败率={sub.failed.mean():.1%}")
sub2 = lo[(lo.region == "华东节点") & (lo.cache_ttl_sec <= 5)]
print(f"华东     低滞后&TTL<=5: n={len(sub2)} 失败={int(sub2.failed.sum())} "
      f"失败率={sub2.failed.mean():.1%}")

print("\n=== 5. 若两条规则并存，覆盖率如何 ===")
r1 = rep.lag_min > 20                      # 规则A：目录陈旧
r2 = rep.cache_ttl_sec <= 5                # 规则B：缓存将过期
for name, mask in [("仅 A", r1 & ~r2), ("仅 B", r2 & ~r1), ("A 且 B", r1 & r2), ("都不满足", ~r1 & ~r2)]:
    s = rep[mask]
    print(f"{name:>8}: n={len(s):>4} 失败={int(s.failed.sum()):>3} 失败率={s.failed.mean():.1%}")
