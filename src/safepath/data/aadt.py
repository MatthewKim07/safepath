"""Match City traffic counts to the graph and expand them to AADT.

TMCs cover 8 or 14 hours of one day; SVCs cover whole days, mostly Tuesday-Thursday. Both
are expanded to Annual Average Daily Traffic with factors derived from the City's own
15-minute SVC data (hour-of-day, day-of-week) and from repeated SVC counts at the same
locations (month of year):

    TMC:  AADT = entering_in_window * R[era, road group, day type, window]
                 / D[dow] * W / M[month]
    SVC:  AADT = weekday_daily * W / M[month]

R  = 24-hour volume / volume inside the TMC window (ratio of sums over complete count-days)
D  = a day-of-week's volume relative to the average weekday (from >= 7-day counts)
W  = average day of the week / average weekday (mean of D over the 7 days)
M  = a month's weekday volume relative to the annual average weekday (location fixed effects)

Outputs in data/processed/: tmc_aadt.parquet and svc_aadt.parquet (every count, with its
matched unit and AADT), counted_intersections.parquet and counted_segments.parquet (one row
per counted unit: most recent count year and AADT, plus history summary), and
aadt_factors.json. Report: reports/volumes.md.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

from safepath.data.volumes import SVC_15MIN, SVC_COUNTS, TMC_COUNTS, TMC_WINDOWS, window_bins
from safepath.graph import INTERSECTIONS, SEGMENTS

log = logging.getLogger(__name__)

TMC_AADT = "tmc_aadt.parquet"
SVC_AADT = "svc_aadt.parquet"
COUNTED_INTERSECTIONS = "counted_intersections.parquet"
COUNTED_SEGMENTS = "counted_segments.parquet"
FACTORS = "aadt_factors.json"

MAJOR_CLASSES = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link",
}  # fmt: skip
COVID_YEARS = {2020, 2021}
BINS_PER_DAY = 96
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Street-name normalisation so City abbreviations ("Ave", "Blvd") match OSM full names.
ABBREV = {
    "av": "avenue", "ave": "avenue", "rd": "road", "st": "street", "blvd": "boulevard",
    "bv": "boulevard", "dr": "drive", "cres": "crescent", "crt": "court", "ct": "court",
    "pkwy": "parkway", "hwy": "highway", "pl": "place", "ln": "lane", "trl": "trail",
    "gt": "gate", "sq": "square", "ter": "terrace", "terr": "terrace", "cir": "circle",
    "gdns": "gardens", "pk": "park", "expy": "expressway", "e": "east", "w": "west",
    "n": "north", "s": "south", "mt": "mount", "collectors": "collector",
}  # fmt: skip
GENERIC = {
    "avenue", "road", "street", "boulevard", "drive", "crescent", "court", "parkway",
    "place", "lane", "trail", "gate", "square", "terrace", "circle", "gardens", "way",
    "east", "west", "north", "south", "the",
}  # fmt: skip


def street_key(name: str) -> str:
    """Comparable core of a street name: 'Steeles Ave E' and 'Steeles Avenue East' -> 'steeles'."""
    name = re.sub(r"\(.*?\)", " ", str(name).lower())  # drop "(id: 13458569)", "(West)"
    tokens = re.sub(r"[^a-z0-9 ]", " ", name).split()
    tokens = [ABBREV.get(t, t) for t in tokens]
    core = [t for t in tokens if t not in GENERIC]
    return " ".join(core or tokens)


def street_keys(names: str, sep: str) -> set[str]:
    return {street_key(n) for n in str(names).split(sep) if n.strip()}


def project_points(df: pd.DataFrame, crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    ).to_crs(crs)


# --------------------------------------------------------------------------- matching


def match_tmc(
    tmc: pd.DataFrame, intersections: gpd.GeoDataFrame, max_m: float, k: int = 5
) -> pd.DataFrame:
    """Each intersection TMC -> consolidated intersection within `max_m`. Among the `k`
    nearest, prefer the one sharing the most street names with the count, then the nearest."""
    pts = project_points(tmc, intersections.crs)
    tree = cKDTree(np.column_stack([intersections["x"], intersections["y"]]))
    dist, idx = tree.query(np.column_stack([pts.geometry.x, pts.geometry.y]), k=k)
    inter_keys = intersections["street_names"].map(lambda s: street_keys(s, " / ")).to_numpy()
    ids = intersections["intersection_id"].to_numpy()
    rows = []
    for i, name in enumerate(tmc["location_name"]):
        keys = street_keys(name, "/")
        best = None
        for d, j in zip(dist[i], idx[i], strict=True):
            if d > max_m:
                break
            shared = len(keys & inter_keys[j])
            if best is None or shared > best[2]:
                best = (ids[j], d, shared)
        nearest_d = float(dist[i][0])
        if best is None:
            rows.append((np.nan, np.nan, 0, nearest_d))
        else:
            rows.append((best[0], float(best[1]), best[2], nearest_d))
    out = pd.DataFrame(
        rows, columns=["intersection_id", "match_m", "names_shared", "nearest_m"], index=tmc.index
    )
    return tmc.join(out)


def counted_street(location_name: str) -> str:
    """'Danforth Ave: Moberly Ave - Woodbine Ave' -> 'Danforth Ave'; also 'X to Y' -> 'X'."""
    name = re.sub(r"\(.*?\)", " ", str(location_name)).strip()  # "(id: 123)" has a colon
    if ":" in name:
        return name.split(":")[0]
    return re.split(r"\s+(?:to|-)\s+", name)[0]


def match_svc(
    svc: pd.DataFrame, segments: gpd.GeoDataFrame, max_m: float, unnamed_max_m: float
) -> pd.DataFrame:
    """Each SVC -> nearest undirected segment within `max_m` whose name matches the counted
    street; failing that, the nearest segment if it is within `unnamed_max_m`. Expressway
    mainline (`highway=motorway`) is only matched by name and never to a ramp count: ramps
    merge into the mainline, so a ramp tube is often within metres of it."""
    pts = project_points(svc, segments.crs).reset_index(names="row")
    seg = segments[["segment_id", "name", "highway", "oneway", "geometry"]].reset_index(drop=True)
    pi, si = seg.sindex.query(pts.geometry.buffer(max_m), predicate="intersects")
    cand = pd.DataFrame({"row": pts["row"].to_numpy()[pi], "seg_pos": si})
    cand["dist"] = pts.geometry.values[pi].distance(seg.geometry.values[si])
    cand = cand[cand["dist"] <= max_m]
    want = svc["location_name"].map(lambda n: street_key(counted_street(n)))
    seg_key = seg["name"].map(lambda n: street_key(n) if isinstance(n, str) else "")
    cand["name_ok"] = want.reindex(cand["row"]).to_numpy() == seg_key.to_numpy()[cand["seg_pos"]]
    mainline = seg["highway"].to_numpy()[cand["seg_pos"]] == "motorway"
    is_ramp = want.str.contains(r"\bramp\b").reindex(cand["row"]).to_numpy()
    cand = cand[~mainline | (cand["name_ok"].to_numpy() & ~is_ramp)]
    cand = cand.sort_values(["row", "name_ok", "dist"], ascending=[True, False, True])
    best = cand.drop_duplicates("row").set_index("row")
    best = best[best["name_ok"] | (best["dist"] <= unnamed_max_m)]
    out = svc.copy()
    pos = best["seg_pos"].reindex(out.index)
    out["segment_id"] = seg["segment_id"].reindex(pos.to_numpy()).to_numpy()
    out["match_m"] = best["dist"].reindex(out.index)
    out["name_match"] = best["name_ok"].reindex(out.index).astype("boolean")
    out["highway"] = seg["highway"].reindex(pos.to_numpy()).to_numpy()
    out["seg_oneway"] = seg["oneway"].reindex(pos.to_numpy()).astype(str).eq("True").to_numpy()
    return out


# --------------------------------------------------------------------------- factors


def era(year: pd.Series | np.ndarray) -> np.ndarray:
    """Pre-pandemic vs pandemic-and-after travel patterns."""
    return np.where(np.asarray(year) <= 2019, "pre2020", "2020on")


def complete_days(raw15: pd.DataFrame, directions: pd.Series) -> pd.DataFrame:
    """Count-days where every direction the count recorded has all 96 bins, as a
    (count_id, date) x minute-of-day volume matrix summed over directions."""
    df = raw15.copy()
    df["date"] = df["time_start"].dt.normalize()
    df["minute"] = df["time_start"].dt.hour * 60 + df["time_start"].dt.minute
    n = df.groupby(["count_id", "date", "direction"])["minute"].transform("nunique")
    df = df[n == BINS_PER_DAY]
    n_dir = df.groupby(["count_id", "date"])["direction"].transform("nunique")
    df = df[n_dir.to_numpy() == df["count_id"].map(directions).to_numpy()]
    mat = df.pivot_table(
        index=["count_id", "date"], columns="minute", values="volume", aggfunc="sum"
    )
    return mat[mat.sum(axis=1) > 0]


def day_table(days: pd.DataFrame, svc: pd.DataFrame) -> pd.DataFrame:
    """One row per complete count-day: daily total, window volumes, and grouping keys."""
    meta = svc.set_index("count_id")
    idx = days.index.to_frame(index=False)
    out = pd.DataFrame(
        {
            "count_id": idx["count_id"].to_numpy(),
            "date": idx["date"].to_numpy(),
            "daily": days.sum(axis=1).to_numpy(),
        }
    )
    for code in TMC_WINDOWS:
        out[f"win_{code}"] = days[window_bins(code)].sum(axis=1).to_numpy()
    minutes = np.asarray(days.columns)
    for h in range(24):
        in_hour = (minutes >= h * 60) & (minutes < (h + 1) * 60)
        out[f"h{h}"] = days.loc[:, in_hour].sum(axis=1).to_numpy()
    dates = pd.to_datetime(out["date"])
    out["dow"] = dates.dt.day_of_week
    out["daytype"] = np.where(out["dow"] >= 5, "weekend", "weekday")
    out["era"] = era(dates.dt.year)
    hw = out["count_id"].map(meta["highway"])
    out["group"] = np.where(hw.isin(MAJOR_CLASSES), "major", "minor")
    out.loc[hw.isna().to_numpy(), "group"] = np.nan
    return out


def window_ratios(days: pd.DataFrame) -> pd.DataFrame:
    """R = sum(daily) / sum(window) per (era, group, daytype, window code)."""
    rows = []
    for (e, g, t), d in days.dropna(subset=["group"]).groupby(["era", "group", "daytype"]):
        for code in TMC_WINDOWS:
            ratio = d["daily"] / d[f"win_{code}"].where(d[f"win_{code}"] > 0)
            rows.append(
                {
                    "era": e,
                    "group": g,
                    "daytype": t,
                    "window": code,
                    "R": d["daily"].sum() / d[f"win_{code}"].sum(),
                    "R_p10": ratio.quantile(0.1),
                    "R_p90": ratio.quantile(0.9),
                    "count_days": len(d),
                    "counts": d["count_id"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def dow_factors(days: pd.DataFrame) -> pd.DataFrame:
    """D[dow] = day volume / mean weekday volume, averaged over counts covering all 7 days."""
    per = days.groupby(["count_id", "dow"])["daily"].mean().unstack()
    per = per.dropna()
    rel = per.div(per[[0, 1, 2, 3, 4]].mean(axis=1), axis=0)
    return pd.DataFrame({"D": rel.mean(), "D_sd": rel.std(), "counts": len(rel)})


def month_factors(svc: pd.DataFrame, years: tuple[int, int]) -> tuple[pd.DataFrame, int]:
    """Location fixed-effects regression of log weekday daily volume on month and year:
    log v = a_location + b_month + c_year. Only locations counted in at least two months
    identify b. Returns M[month] normalised so the day-weighted annual mean is 1."""
    d = svc[
        svc["avg_weekday_daily"].gt(0)
        & svc["date_start"].dt.year.between(*years)
        & ~svc["date_start"].dt.year.isin(COVID_YEARS)
    ].copy()
    d["month"] = d["date_start"].dt.month
    d["year"] = d["date_start"].dt.year
    months_per_loc = d.groupby("centreline_id")["month"].transform("nunique")
    d = d[months_per_loc >= 2].reset_index(drop=True)
    n = len(d)
    loc = d["centreline_id"].astype("category").cat.codes.to_numpy()
    month = d["month"].to_numpy() - 2  # January is the reference
    year_codes = d["year"].astype("category")
    yr = year_codes.cat.codes.to_numpy() - 1
    rows = np.arange(n)
    X_loc = coo_matrix((np.ones(n), (rows, loc)))
    m_rows = rows[month >= 0]
    X_month = coo_matrix((np.ones(len(m_rows)), (m_rows, month[month >= 0])), shape=(n, 11))
    y_rows = rows[yr >= 0]
    n_years = len(year_codes.cat.categories) - 1
    X_year = coo_matrix((np.ones(len(y_rows)), (y_rows, yr[yr >= 0])), shape=(n, n_years))
    X = hstack([X_loc, X_month, X_year]).tocsr()
    beta = lsqr(X, np.log(d["avg_weekday_daily"].to_numpy()), atol=1e-10, btol=1e-10)[0]
    b = np.concatenate([[0.0], beta[X_loc.shape[1] : X_loc.shape[1] + 11]])
    days_in_month = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    M = np.exp(b) / np.average(np.exp(b), weights=days_in_month)
    out = pd.DataFrame(
        {
            "month": range(1, 13),
            "M": M,
            "counts": d.groupby("month").size().reindex(range(1, 13), fill_value=0).to_numpy(),
        }
    )
    return out, d["centreline_id"].nunique()


def month_factor_ci(
    svc: pd.DataFrame, years: tuple[int, int], n_boot: int = 100, seed: int = 7
) -> pd.DataFrame:
    """2.5/97.5 percentiles of M from resampling locations with replacement."""
    rng = np.random.default_rng(seed)
    svc = svc.reset_index(drop=True)
    groups = list(svc.groupby("centreline_id").indices.values())
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[k] for k in pick])
        # a location drawn twice becomes two locations, each with its own fixed effect
        labels = np.repeat(np.arange(len(pick)), [len(groups[k]) for k in pick])
        sample = svc.iloc[rows].assign(centreline_id=labels)
        draws.append(month_factors(sample, years)[0]["M"].to_numpy())
    q = np.quantile(np.vstack(draws), [0.025, 0.975], axis=0)
    return pd.DataFrame({"month": range(1, 13), "M_lo": q[0], "M_hi": q[1]})


def holdout_window_error(days: pd.DataFrame) -> pd.DataFrame:
    """Error of the 24-hour expansion on held-out counts: ratios from even count ids, applied
    to odd ones' simulated TMC windows and compared with their true daily volume."""
    d = days.dropna(subset=["group"])
    d = d[d["daytype"] == "weekday"]
    fit, test = d[d["count_id"] % 2 == 0], d[d["count_id"] % 2 == 1]
    R = window_ratios(fit).set_index(["era", "group", "daytype", "window"])["R"]
    rows = []
    for code in TMC_WINDOWS:
        key = list(
            zip(test["era"], test["group"], test["daytype"], [code] * len(test), strict=True)
        )
        est = test[f"win_{code}"].to_numpy() * R.reindex(key).to_numpy()
        err = est / test["daily"].to_numpy() - 1
        rows.append(
            {
                "window": code,
                "test count-days": len(test),
                "median abs % error": 100 * np.nanmedian(np.abs(err)),
                "p90 abs % error": 100 * np.nanquantile(np.abs(err), 0.9),
                "median % bias": 100 * np.nanmedian(err),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- AADT


def combine_svc_siblings(svc: pd.DataFrame) -> pd.DataFrame:
    """Same location and start date with several count ids: sum them when the raw files show
    they cover different directions, otherwise keep the largest (a duplicate study)."""
    key = ["centreline_id", "date_start"]
    multi = svc.duplicated(key, keep=False)
    single, groups = svc[~multi], svc[multi]
    merged = []
    for _, g in groups.groupby(key):
        dirs = [set(s.split(",")) if s else set() for s in g["directions"]]
        disjoint = all(dirs) and sum(len(s) for s in dirs) == len(set().union(*dirs))
        if disjoint:
            row = g.iloc[0].copy()
            for col in ("avg_daily", "avg_weekday_daily", "avg_weekend_daily"):
                row[col] = g[col].sum(min_count=1)
            row["directions"] = ",".join(sorted(set().union(*dirs)))
            row["n_directions"] = len(set().union(*dirs))
            row["combined"] = "summed directions"
        else:
            row = g.loc[g["avg_daily"].idxmax()].copy()
            row["combined"] = "kept largest"
        merged.append(row)
    out = pd.concat([single.assign(combined=""), pd.DataFrame(merged)], ignore_index=True)
    return out.sort_values("count_id").reset_index(drop=True)


def tmc_aadt(
    tmc: pd.DataFrame,
    group: pd.Series,
    R: pd.DataFrame,
    D: pd.DataFrame,
    W: float,
    M: pd.DataFrame,
) -> pd.DataFrame:
    out = tmc.copy()
    out["year"] = out["date"].dt.year
    out["group"] = group.reindex(out["intersection_id"]).to_numpy()
    out["era"] = era(out["year"])
    dow = out["date"].dt.day_of_week
    out["daytype"] = np.where(dow >= 5, "weekend", "weekday")
    r = R.set_index(["era", "group", "daytype", "window"])["R"]
    key = list(zip(out["era"], out["group"], out["daytype"], out["duration"], strict=True))
    out["R"] = r.reindex(key).to_numpy()
    expected = out["duration"].map(lambda c: len(window_bins(c)))
    # a few raw counts miss bins; scale their window volume up to the full window
    out["bin_scale"] = np.where(out["raw_bins"] < expected, expected / out["raw_bins"], 1.0)
    out["daily"] = out["entering"] * out["bin_scale"] * out["R"]
    out["D"] = D["D"].reindex(dow).to_numpy()
    out["M"] = M.set_index("month")["M"].reindex(out["date"].dt.month).to_numpy()
    out["aadt"] = out["daily"] / out["D"] * W / out["M"]
    return out


def svc_aadt(svc: pd.DataFrame, D: pd.DataFrame, W: float, M: pd.DataFrame) -> pd.DataFrame:
    out = svc.copy()
    out["year"] = out["date_start"].dt.year
    weekend_rel = D.loc[[5, 6], "D"].mean()
    weekday = out["avg_weekday_daily"].where(
        out["avg_weekday_daily"] > 0, out["avg_weekend_daily"] / weekend_rel
    )
    out["M"] = M.set_index("month")["M"].reindex(out["date_start"].dt.month).to_numpy()
    out["aadt"] = weekday * W / out["M"]
    # a one-direction count on a two-way street misses the other direction's traffic
    out["one_direction_on_two_way"] = (out["n_directions"] == 1) & ~out["seg_oneway"]
    # City tubes on expressway mainline give one direction or one lane as often as the full
    # cross-section (Gardiner counts range 4k-118k); the 400-series use MTO AADT instead
    out["expressway_mainline"] = out["highway"].eq("motorway")
    out["usable"] = out["segment_id"].notna() & out["aadt"].gt(0)
    out["usable"] &= ~out["one_direction_on_two_way"] & ~out["expressway_mainline"]
    return out


def latest_by_unit(counts: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Per unit: most recent count year and its mean AADT, the most recent non-2020/21 one,
    and the history (number of counts, first year, mean AADT over all counts)."""
    c = counts.dropna(subset=[unit, "aadt"])
    g = c.groupby(unit)
    out = pd.DataFrame(
        {
            "n_counts": g.size(),
            "first_year": g["year"].min(),
            "latest_year": g["year"].max(),
            "aadt_all_mean": g["aadt"].mean(),
        }
    )
    last = c[c["year"] == c[unit].map(out["latest_year"])]
    out["latest_aadt"] = last.groupby(unit)["aadt"].mean()
    nc = c[~c["year"].isin(COVID_YEARS)]
    ncy = nc.groupby(unit)["year"].max()
    out["latest_noncovid_year"] = ncy
    nc_last = nc[nc["year"] == nc[unit].map(ncy)]
    out["latest_noncovid_aadt"] = nc_last.groupby(unit)["aadt"].mean()
    out.index = out.index.astype(int)
    return out.rename_axis(unit).reset_index()


def intersection_groups(static: pd.DataFrame) -> pd.Series:
    """'major' if any approach is an arterial or higher (primary/secondary/motorway/trunk)."""
    rank = static["max_class_rank"]
    return pd.Series(np.where(rank >= 3, "major", "minor"), index=static.index)


def build_aadt(cfg: dict[str, Any]) -> dict[str, Any]:
    processed = Path(cfg["paths"]["processed_dir"])
    v = cfg["volumes"]
    intersections = gpd.read_parquet(processed / INTERSECTIONS)
    segments = gpd.read_parquet(processed / SEGMENTS)
    static = pd.read_parquet(processed / "static_features.parquet")
    tmc = pd.read_parquet(processed / TMC_COUNTS)
    svc = pd.read_parquet(processed / SVC_COUNTS)
    raw15 = pd.read_parquet(processed / SVC_15MIN)

    tmc = tmc[tmc["centreline_type"] == 2].reset_index(drop=True)  # intersection counts only
    tmc = match_tmc(tmc, intersections, v["match"]["tmc_intersection_m"])
    svc = match_svc(svc, segments, v["match"]["svc_segment_m"], v["match"]["svc_unnamed_m"])

    days = day_table(complete_days(raw15, svc.set_index("count_id")["n_directions"]), svc)
    R = window_ratios(days)
    D = dow_factors(days)
    W = float(D["D"].mean())
    M, month_locations = month_factors(svc, tuple(v["month_factor_years"]))

    t = tmc_aadt(tmc, intersection_groups(static), R, D, W, M)
    s = svc_aadt(combine_svc_siblings(svc), D, W, M)
    t.to_parquet(processed / TMC_AADT, index=False)
    s.to_parquet(processed / SVC_AADT, index=False)
    ci = latest_by_unit(t, "intersection_id")
    cs = latest_by_unit(s[s["usable"]], "segment_id")
    ci.to_parquet(processed / COUNTED_INTERSECTIONS, index=False)
    cs.to_parquet(processed / COUNTED_SEGMENTS, index=False)

    factors = {
        "window_ratios": R.to_dict(orient="records"),
        "dow": D.reset_index(names="dow").to_dict(orient="records"),
        "W": W,
        "month": M.to_dict(orient="records"),
        "month_locations": month_locations,
    }
    (processed / FACTORS).write_text(json.dumps(factors, indent=2, default=float))
    log.info(
        "AADT: %d TMCs -> %d intersections, %d SVCs -> %d segments",
        t["intersection_id"].notna().sum(),
        len(ci),
        s["usable"].sum(),
        len(cs),
    )
    return {
        "tmc": t,
        "svc": s,
        "days": days,
        "R": R,
        "D": D,
        "W": W,
        "M": M,
        "month_locations": month_locations,
        "counted_intersections": ci,
        "counted_segments": cs,
        "intersections": intersections,
        "segments": segments,
        "static": static,
    }
