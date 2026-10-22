"""峰值稳健性：阈值在负载升高时是否漂移，以及负载是否本身在推高滞后。

数据里没有 QPS 字段，只能用请求量做负载代理。
节点×小时粒度下中位数仅 2 条请求，噪声过大，因此改用「一天中的小时」聚合
（7 天 × 5 节点合并，每小时 48–78 条），作为忙闲代理。
"""

import pandas as pd

df = pd.read_csv("data/chapter_requests.csv", dtype={"load_status": "string"})
df = df.drop_duplicates("request_id").copy()
df["req"] = pd.to_datetime(df["req_time"], utc=True)
df["syn"] = pd.to_datetime(df["catalog_synced_at"], utc=True)
df["lag_min"] = (df["req"] - df["syn"]).dt.total_seconds() / 60
df["hour"] = df["req"].dt.tz_convert("Asia/Shanghai").dt.hour
df["date"] = df["req"].dt.tz_convert("Asia/Shanghai").dt.date

rep = df[df["load_status"].notna()].copy()
rep["failed"] = (rep["load_status"] == "0").astype(int)
rep["stale"] = rep.lag_min > 20
rep["ttl_low"] = rep.cache_ttl_sec <= 5

print("=== 1. 忙闲与滞后是否相关（按小时聚合）===")
byh = rep.groupby("hour").agg(n=("lag_min", "size"), 平均滞后=("lag_min", "mean"),
                              陈旧占比=("stale", "mean"), 失败率=("failed", "mean"))
byh["陈旧占比"] = (byh.陈旧占比 * 100).round(1)
byh["失败率"] = (byh.失败率 * 100).round(1)
byh["平均滞后"] = byh.平均滞后.round(1)
print(byh.to_string())
print(f"\n请求量 vs 平均滞后 相关: {byh.n.corr(byh.平均滞后):+.2f}")
print(f"请求量 vs 陈旧占比 相关: {byh.n.corr(byh.陈旧占比):+.2f}")

print("\n=== 2. 高低负载小时分开看阈值 ===")
hour_vol = byh["n"]
rep["high_load"] = rep["hour"].map(hour_vol) >= hour_vol.median()
print(f"高负载请求 {int(rep.high_load.sum())}，低负载 {int((~rep.high_load).sum())}")
for name, s in [("高负载小时", rep[rep.high_load]), ("低负载小时", rep[~rep.high_load])]:
    lo, hi = s[~s.stale], s[s.stale]
    print(f"{name}: 滞后<=20 失败率={lo.failed.mean():.1%} (n={len(lo)}) | "
          f"滞后>20 失败率={hi.failed.mean():.1%} (n={len(hi)})")

print("\n=== 3. 华东 TTL 阈值是否漂移 ===")
hd = rep[rep.region == "华东节点"]
for name, s in [("高负载", hd[hd.high_load]), ("低负载", hd[~hd.high_load])]:
    a, b = s[s.ttl_low], s[~s.ttl_low]
    print(f"{name}: TTL<=5 失败率={a.failed.mean():.1%} (n={len(a)}) | "
          f"TTL>5 失败率={b.failed.mean():.1%} (n={len(b)})")

print("\n=== 4. 边界是否随负载移动 ===")
for name, s in [("高负载", rep[rep.high_load]), ("低负载", rep[~rep.high_load])]:
    print(f"\n{name}:")
    for lo, hi in [(0, 10), (10, 18), (18, 21), (21, 30)]:
        g = s[(s.lag_min >= lo) & (s.lag_min < hi)]
        if len(g):
            print(f"  滞后 {lo:>2}-{hi:>2}: n={len(g):>3} 失败率={g.failed.mean():6.1%}")

print("\n=== 5. 日级波动（是否有单日异常）===")
byd = rep.groupby("date").agg(n=("failed", "size"), 失败=("failed", "sum"),
                              平均滞后=("lag_min", "mean"), 陈旧占比=("stale", "mean"))
byd["失败率"] = (byd.失败 / byd.n * 100).round(1)
byd["陈旧占比"] = (byd.陈旧占比 * 100).round(1)
byd["平均滞后"] = byd.平均滞后.round(1)
print(byd.to_string())
