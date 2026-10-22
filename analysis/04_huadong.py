"""追查华东节点异常：低滞后却失败的 13 条样本，特征是否与主流失败不同。

思路：如果是“另一条故障路径”，这 13 条应该在某个维度上抱团
（同一时段、同一 TTL 区间、同一品类、同一字数区间）；
如果散落各处，更可能是随机噪声或未观测变量。
"""

import pandas as pd

df = pd.read_csv("data/chapter_requests.csv", dtype={"load_status": "string"})
df["req"] = pd.to_datetime(df["req_time"], utc=True)
df["syn"] = pd.to_datetime(df["catalog_synced_at"], utc=True)
df["lag_min"] = (df["req"] - df["syn"]).dt.total_seconds() / 60
df["local"] = df["req"].dt.tz_convert("Asia/Shanghai")
df["date"] = df["local"].dt.date
df["hour"] = df["local"].dt.hour
rep = df[df["load_status"].notna()].copy()
rep["failed"] = (rep["load_status"] == "0").astype(int)

hd = rep[rep.region == "华东节点"]
print("=== 华东节点：低滞后失败样本明细 ===")
sus = hd[(hd.lag_min <= 20) & (hd.failed == 1)]
print(f"共 {len(sus)} 条 / 低滞后总数 {len(hd[hd.lag_min <= 20])}")
cols = ["request_id", "local", "lag_min", "cache_ttl_sec", "chapter_words", "category"]
print(sus[cols].sort_values("local").to_string(index=False))

print("\n=== 这些失败在时间上是否聚集 ===")
print("按日期:", sus.groupby("date").size().to_dict())
print("按小时:", sus.groupby("hour").size().to_dict())

print("\n=== 华东 vs 其他：低滞后层的 TTL / 字数分布 ===")
lo = rep[rep.lag_min <= 20]
for name, g in [("华东", lo[lo.region == "华东节点"]), ("其他四区", lo[lo.region != "华东节点"])]:
    f = g["failed"] == 1
    print(f"{name}: n={len(g)} 失败={int(f.sum())} | "
          f"失败者TTL均值={g.loc[f, 'cache_ttl_sec'].mean():.1f} 成功者TTL均值={g.loc[~f, 'cache_ttl_sec'].mean():.1f} | "
          f"失败者字数均值={g.loc[f, 'chapter_words'].mean():.0f} 成功者={g.loc[~f, 'chapter_words'].mean():.0f}")

print("\n=== 华东低滞后：TTL 分档失败率（找第二条阈值）===")
hlo = hd[hd.lag_min <= 20].copy()
hlo["ttl_bin"] = pd.cut(hlo.cache_ttl_sec, [-1, 5, 10, 15, 20, 30],
                        labels=["0-5", "6-10", "11-15", "16-20", "21-30"])
t = hlo.groupby("ttl_bin", observed=True).agg(n=("failed", "size"), 失败=("failed", "sum"))
t["失败率"] = (t.失败 / t.n * 100).round(1)
print(t.to_string())

print("\n=== 全量：小时 × 失败率（看是否有时间性故障窗口）===")
t2 = rep.groupby("hour").agg(n=("failed", "size"), 失败=("failed", "sum"))
t2["失败率"] = (t2.失败 / t2.n * 100).round(1)
print(t2.to_string())

print("\n=== 华东失败是否与“同日其他区域也失败”同步发生（全局事件）===")
sus_dates = sus["date"].unique()
print("华东低滞后失败出现的日期:", sorted(map(str, sus_dates)))
for d in sorted(sus_dates):
    day = rep[rep.date == d]
    print(f"  {d}: 全区失败 {int(day.failed.sum())}/{len(day)}，"
          f"其中华东低滞后失败 {int(((day.region == '华东节点') & (day.lag_min <= 20) & (day.failed == 1)).sum())}")
