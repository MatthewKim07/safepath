"""MTO provincial highway volumes and collision rates for the 400-series in Toronto.

The 401, 400, 404, 409, 427 and QEW are patrolled by the OPP, so TPS has almost no crashes on
them. Their exposure and collision rates come from the Ministry of Transportation's
Technical Publications library (Open Government Licence - Ontario):

- `Traffic_Volumes_1988-2024.csv`: AADT per highway section and year (LHRS sections).
- `Provincial_Highways_Traffic_Volumes_1988-2021.pdf`: the same sections with total reportable
  collisions and collision rate (collisions per million vehicle-km, ramps excluded) for
  1988-2019 and 2021.

Each section is named after one interchange and runs between consecutive interchanges. The
interchanges are located on the OSM graph by their crossing road, OSM motorway segments are
assigned to the section whose interchange interval they fall in, and whether a section's
name marks its start or its end is decided per highway by which convention reproduces the
MTO section lengths (reports/mto.md).

Outputs in data/processed/: mto_sections.parquet (per section: AADT history, collisions and
rates), mto_segments.parquet (OSM motorway segment -> section).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shapely
from pypdf import PdfReader
from shapely.ops import nearest_points

from safepath.data.aadt import street_key
from safepath.data.report import GRID, SURFACE, TEXT
from safepath.fetch import FetchError, with_retries
from safepath.graph import SEGMENTS
from safepath.locations import markdown_table

log = logging.getLogger(__name__)

MTO_SECTIONS = "mto_sections.parquet"
MTO_SEGMENTS = "mto_segments.parquet"
MTO_PDF_ROWS = "mto_pdf_rows.parquet"

# OSM name of each highway's mainline carriageways (collector/express lanes included).
OSM_NAME = {
    "401": r"^Highway 401\b",
    "400": r"^Highway 400\b",
    "404": r"^Highway 404\b",
    "409": r"^Highway 409\b",
    "427": r"^Highway 427\b",
    "QEW": r"^Queen Elizabeth Way\b",
}
# MTO codes the QEW as highway 1.
MTO_NUMBER = {"QEW": 1}

# Crossing road at each section's named interchange, as an OSM name (or a highway regex),
# for every section that can touch Toronto. Keyed by (highway, LHRS). Sections whose
# crossing is outside the graph (Mississauga, Vaughan, Pickering) are listed with None and
# serve only as the far end of the outermost Toronto sections.
CROSSINGS: dict[tuple[str, int], str | None] = {
    ("401", 47675): None,  # Hwys 410 & 403
    ("401", 47673): None,  # Dixie Rd
    ("401", 47672): "Renforth Drive",
    ("401", 47671): r"^Highway 427\b",
    ("401", 47667): "Dixon Road",
    ("401", 47663): "Islington Avenue",
    ("401", 47661): "Weston Road",
    ("401", 47660): r"^Highway 400\b",
    ("401", 47656): "Keele Street",
    ("401", 47654): "Dufferin Street",
    ("401", 47652): "Allen Road",
    ("401", 47650): "Bathurst Street",
    ("401", 47647): "Avenue Road",
    ("401", 47643): "Yonge Street",
    ("401", 47642): "Bayview Avenue",
    ("401", 47640): "Leslie Street",
    ("401", 47635): r"^(?:Highway 404|Don Valley Parkway)\b",
    ("401", 47633): "Victoria Park Avenue",
    ("401", 47631): "Warden Avenue",
    ("401", 47629): "Kennedy Road",
    ("401", 47628): "Brimley Road",
    ("401", 47627): "McCowan Road",
    ("401", 47626): "Markham Road",
    ("401", 47625): "Neilson Road",
    ("401", 47624): "Morningside Avenue",
    ("401", 47622): "Meadowvale Road",
    ("401", 47614): "Kingston Road",
    ("401", 47612): None,  # Whites Rd
    ("400", 46795): "Black Creek Drive",  # Maple Leaf Dr: the 400 begins at Black Creek Dr
    ("400", 46800): "Jane Street",
    ("400", 46810): r"^Highway 401\b",
    ("400", 46815): "Finch Avenue West",
    ("400", 46817): "Steeles Avenue West",
    ("400", 46818): None,  # Hwy 407
    ("404", 48450): r"^Highway 401\b",
    ("404", 48459): "Sheppard Avenue East",
    ("404", 48469): "Finch Avenue East",
    ("404", 48490): "Steeles Avenue East",
    ("404", 48492): None,  # Hwy 407
    ("409", 48980): r"^Highway 401\b",
    ("409", 48983): "Kipling Avenue",
    ("409", 48986): "Martin Grove Road",
    ("409", 48990): "Attwell Drive",
    ("427", 50205): "Evans Avenue",
    ("427", 50210): r"^(?:Queen Elizabeth Way|Gardiner Expressway)\b",
    ("427", 50219): "Dundas Street West",
    ("427", 50260): "Burnhamthorpe Road",
    ("427", 50280): "Rathburn Road",
    ("427", 50300): r"^Highway 401\b",
    ("427", 50310): "Dixon Road",
    ("427", 50320): r"^Highway 409\b",
    ("427", 50350): "Rexdale Boulevard",
    ("427", 50355): "Finch Avenue West",
    ("427", 50365): None,  # Hwy 407
    ("QEW", 10151): None,  # Cawthra Rd
    ("QEW", 10153): None,  # Dixie Rd
    ("QEW", 10155): "Evans Avenue",
}
LOCATE_MAX_M = 400  # a crossing road farther than this from the highway is not its interchange
RATE_YEARS = (2015, 2019)  # pooled collision-rate window: recent, pre-pandemic, five years


def norm_desc(desc: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(desc).upper())


def download(cfg: dict[str, Any]) -> dict[str, Path]:
    m = cfg["mto"]
    out = Path(cfg["paths"]["raw_dir"]) / "mto"
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, url in m["files"].items():
        dest = out / m["filenames"][key]
        paths[key] = dest
        if dest.exists() and dest.stat().st_size > 0:
            continue

        def fetch(url: str = url, dest: Path = dest) -> None:
            # the library's file server rejects requests without a browser user agent
            resp = requests.get(url, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "html" in ctype:
                raise FetchError(f"{url} returned an HTML page, not the file")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)

        log.info("downloading %s", dest.name)
        with_retries(
            fetch,
            what=f"MTO download {dest.name}",
            attempts=m["fetch_attempts"],
            backoff_seconds=m["backoff_seconds"],
        )
    return paths


def load_aadt_csv(path: Path, highways: list[str]) -> pd.DataFrame:
    """Section-year AADT for the requested highways (year 9999 marks non-assumed sections)."""
    raw = pd.read_csv(path, thousands=",", low_memory=False, encoding="cp1252")
    numbers = {MTO_NUMBER[h] if h in MTO_NUMBER else int(h): h for h in highways}
    raw = raw[raw["Hwy No"].isin(numbers) & (raw["Year"] != 9999)]
    out = pd.DataFrame(
        {
            "hwy": raw["Hwy No"].map(numbers),
            "lhrs": raw["LHRS"].astype(int),
            "year": raw["Year"].astype(int),
            "description": raw["Description"].str.strip(),
            "region": raw["Region"],
            "length_km": raw["Length"].astype(float),
            "aadt": pd.to_numeric(raw["AADT"], errors="coerce"),
            "alpha_code": raw["Alpha Code"],
            "order": raw.index,  # file order runs along the highway
        }
    )
    return out.reset_index(drop=True)


PDF_ROW = re.compile(r"^(\d{4}) (QEW|\d{1,4}) (.+)$")
NUM = r"(?:[\d,]+(?:\.\d+)?|N/A)"


def parse_pdf_lines(lines: list[str], highways: set[str]) -> pd.DataFrame:
    """Rows of the section listing: the first row of a section carries its description and
    length; later rows only the year. The last 9 tokens are AADT, SADT, SAWDT, WADT,
    truck AADT, collisions, collision rate, truck collisions, truck collision rate."""
    rows, section = [], None
    for line in lines:
        m = PDF_ROW.match(line.strip())
        if not m:
            continue
        year, hwy, rest = int(m.group(1)), m.group(2), m.group(3).split()
        if hwy not in highways or len(rest) < 10:
            continue
        tail = rest[-9:]
        if not all(re.fullmatch(NUM, t) for t in tail):
            continue
        head = rest[:-10]
        if head:
            section = (hwy, " ".join(head[:-1]), float(head[-1]))
        if section is None or section[0] != hwy:
            continue

        def num(t: str) -> float:
            return np.nan if t == "N/A" else float(t.replace(",", ""))

        rows.append(
            {
                "hwy": hwy,
                "description": section[1],
                "dist_km": section[2],
                "year": year,
                "pattern": rest[-10],
                "aadt": num(tail[0]),
                "collisions": num(tail[5]),
                "cr": num(tail[6]),
            }
        )
    return pd.DataFrame(rows)


def load_pdf(path: Path, highways: list[str], pages: list[tuple[int, int]]) -> pd.DataFrame:
    """Parse the printed-page ranges `pages` (from the table of contents) of the MTO PDF."""
    reader = PdfReader(str(path))
    offset = None
    for i in range(len(reader.pages)):  # printed page numbers start after the front matter
        m = re.search(r"Page (\d+)\s*$", reader.pages[i].extract_text() or "")
        if m:
            offset = i + 1 - int(m.group(1))
            break
    if offset is None:
        raise ValueError("could not find printed page numbers in the MTO PDF")
    lines: list[str] = []
    for lo, hi in pages:
        for p in range(lo + offset, hi + offset):
            lines.extend((reader.pages[p - 1].extract_text() or "").splitlines())
    names = {("QEW" if h == "QEW" else h) for h in highways}
    out = parse_pdf_lines(lines, names)
    if out.empty:
        raise ValueError("no section rows parsed from the MTO PDF")
    return out


def attach_collisions(sections: pd.DataFrame, pdf: pd.DataFrame) -> pd.DataFrame:
    """Join PDF section rows (collisions, rates) to CSV sections by highway, normalised
    description, and length to 0.1 km."""
    pdf = pdf.assign(key=pdf["description"].map(norm_desc))
    sec = sections.assign(key=sections["description"].map(norm_desc))
    joined = sec.merge(
        pdf[["hwy", "key", "dist_km", "year", "collisions", "cr", "aadt"]].rename(
            columns={"aadt": "pdf_aadt"}
        ),
        on=["hwy", "key", "year"],
        how="left",
    )
    # the PDF rounds lengths to 0.1 km; a description reused elsewhere on the highway with
    # a different length is a different section
    same = (joined["dist_km"] - joined["length_km"]).abs() <= 0.051
    joined.loc[~same, ["collisions", "cr", "pdf_aadt"]] = np.nan
    joined = joined.sort_values("collisions", na_position="last")
    joined = joined.drop_duplicates(["hwy", "lhrs", "year"]).sort_values(["order", "year"])
    return joined.drop(columns=["key", "dist_km"]).reset_index(drop=True)


# --------------------------------------------------------------------------- mapping


def highway_segments(segments: gpd.GeoDataFrame, hwy: str) -> gpd.GeoDataFrame:
    named = segments["name"].fillna("").str.contains(OSM_NAME[hwy], regex=True)
    return segments[named & (segments["highway"] == "motorway")]


def locate_crossing(
    mainline: shapely.Geometry, segments: gpd.GeoDataFrame, crossing: str
) -> tuple[shapely.Point | None, float]:
    """Point on the highway closest to the crossing road (by OSM name or highway regex)."""
    names = segments["name"].fillna("")
    if crossing.startswith("^"):
        cand = segments[names.str.contains(crossing, regex=True)]
    else:  # any class: the Allen and Black Creek crossings are themselves expressways
        cand = segments[names.map(street_key) == street_key(crossing)]
    cand = cand[cand.distance(mainline) <= LOCATE_MAX_M]
    if cand.empty:
        return None, np.nan
    road = shapely.union_all(cand.geometry.values)
    on_hwy, on_road = nearest_points(mainline, road)
    return on_hwy, on_hwy.distance(on_road)


def chainage(points: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Position of each point along the polyline through `boundary` (n x 2), in units of
    boundary intervals: 0 at the first boundary, 1 at the second, ... Points before the
    first or after the last boundary extrapolate along the end intervals."""
    seg_start, seg_end = boundary[:-1], boundary[1:]
    d = seg_end - seg_start
    L2 = (d**2).sum(axis=1)
    rel = points[:, None, :] - seg_start[None, :, :]
    t = (rel * d[None]).sum(axis=2) / L2[None]
    clipped = np.clip(t, 0, 1)
    proj = seg_start[None] + clipped[..., None] * d[None]
    dist = np.linalg.norm(points[:, None, :] - proj, axis=2)
    best = dist.argmin(axis=1)
    pos = best + clipped[np.arange(len(points)), best]
    first, last = best == 0, best == len(d) - 1
    pos[first & (t[:, 0] < 0)] = t[first & (t[:, 0] < 0), 0]
    pos[last & (t[:, -1] > 1)] = len(d) - 1 + t[last & (t[:, -1] > 1), -1]
    return pos


def map_highway(
    hwy: str, sections: pd.DataFrame, segments: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Locate this highway's interchanges, choose the naming convention, and assign each
    OSM mainline segment to a section. Returns (located sections, segment map, rule)."""
    hw = highway_segments(segments, hwy)
    if hw.empty:
        return pd.DataFrame(), pd.DataFrame(), ""
    mainline = shapely.union_all(hw.geometry.values)
    secs = sections.sort_values("order").reset_index(drop=True)
    pts, gaps = [], []
    for row in secs.itertuples():
        crossing = CROSSINGS.get((hwy, row.lhrs))
        point, gap = (
            (None, np.nan) if crossing is None else locate_crossing(mainline, segments, crossing)
        )
        pts.append(point)
        gaps.append(gap)
    secs["crossing_gap_m"] = gaps
    secs["located"] = [p is not None for p in pts]
    loc = secs[secs["located"]]
    xy = np.array([[p.x, p.y] for p in pts if p is not None])
    if len(xy) == 0:
        raise ValueError(f"highway {hwy}: no interchange located on the OSM graph")
    if len(xy) == 1:
        # only one interchange inside the graph (the QEW's Toronto stub): the whole stub
        # belongs to that section
        secs["osm_spacing_km"] = np.nan
        secs["rule_err_end"] = secs["rule_err_start"] = np.nan
        segmap = pd.DataFrame(
            {
                "segment_id": hw["segment_id"].to_numpy(),
                "hwy": hwy,
                "lhrs": loc["lhrs"].iloc[0],
                "length_m": hw["length"].to_numpy(),
                "osm_name": hw["name"].to_numpy(),
            }
        )
        return secs, segmap, "single"
    spacing = np.linalg.norm(np.diff(xy, axis=0), axis=1) / 1000
    idx = loc.index.to_numpy()
    # convention 'end': section i spans (boundary i-1, boundary i]; 'start': [i, i+1)
    contiguous = np.diff(idx) == 1
    err_end = np.abs(np.log(spacing / secs.loc[idx[1:], "length_km"].to_numpy()))[contiguous]
    err_start = np.abs(np.log(spacing / secs.loc[idx[:-1], "length_km"].to_numpy()))[contiguous]
    rule = "end" if np.median(err_end) <= np.median(err_start) else "start"
    secs["osm_spacing_km"] = np.nan
    if rule == "end":
        secs.loc[idx[1:][contiguous], "osm_spacing_km"] = spacing[contiguous]
    else:
        secs.loc[idx[:-1][contiguous], "osm_spacing_km"] = spacing[contiguous]
    secs["rule_err_end"] = np.median(err_end)
    secs["rule_err_start"] = np.median(err_start)

    mid = shapely.line_interpolate_point(hw.geometry.values, 0.5, normalized=True)
    pos = chainage(np.column_stack([shapely.get_x(mid), shapely.get_y(mid)]), xy)
    interval = np.floor(pos).astype(int)  # between located boundaries i and i + 1
    n = len(idx)
    inner = np.clip(interval, 0, n - 2)
    if rule == "end":  # the section named at the far boundary of the interval
        sec_row = idx[inner + 1]
        sec_row = np.where(interval < 0, idx[0], sec_row)
        sec_row = np.where(interval >= n - 1, idx[-1] + 1, sec_row)
    else:  # the section named at the near boundary
        sec_row = idx[inner]
        sec_row = np.where(interval < 0, idx[0] - 1, sec_row)
        sec_row = np.where(interval >= n - 1, idx[-1], sec_row)
    sec_row = np.clip(sec_row, 0, len(secs) - 1)
    segmap = pd.DataFrame(
        {
            "segment_id": hw["segment_id"].to_numpy(),
            "hwy": hwy,
            "lhrs": secs.loc[sec_row, "lhrs"].to_numpy(),
            "length_m": hw["length"].to_numpy(),
            "osm_name": hw["name"].to_numpy(),
        }
    )
    return secs, segmap, rule


def section_summary(history: pd.DataFrame, latest_year: int) -> pd.DataFrame:
    """Per section: latest AADT, pooled 2015-2019 collision rate (collisions / MVKM), and
    the 2021 rate."""
    h = history.copy()
    h["mvkm"] = h["aadt"] * 365 * h["length_km"] / 1e6
    win = h[h["year"].between(*RATE_YEARS)]
    g = win.groupby(["hwy", "lhrs"])
    pooled = pd.DataFrame(
        {
            "collisions_per_year": g["collisions"].sum(min_count=1) / g["year"].nunique(),
            "mvkm_per_year": g["mvkm"].mean(),
            "rate_years": g["collisions"].count(),
        }
    )
    pooled["cr"] = pooled["collisions_per_year"] / pooled["mvkm_per_year"]
    latest = h[h["year"] == latest_year].set_index(["hwy", "lhrs"])
    y21 = h[h["year"] == 2021].set_index(["hwy", "lhrs"])
    out = latest[["description", "length_km", "aadt", "alpha_code", "order"]].join(pooled)
    out["cr_2021"] = y21["cr"]
    out = out.rename(columns={"aadt": f"aadt_{latest_year}"})
    return out.reset_index()


def fill_rates(sections: pd.DataFrame) -> pd.DataFrame:
    """Sections without a published rate (resplit or renamed since 2010) take their
    highway's pooled rate over the mapped sections."""
    out = sections.copy()
    mapped = out[(out["mapped_km"] > 0) & out["collisions_per_year"].notna()]
    pooled = mapped.groupby("hwy")[["collisions_per_year", "mvkm_per_year"]].sum()
    rate = pooled["collisions_per_year"] / pooled["mvkm_per_year"]
    out["cr_source"] = np.where(out["cr"].notna(), "section", "highway pooled")
    out["cr"] = out["cr"].fillna(out["hwy"].map(rate))
    return out


def plot_sections(
    segments: gpd.GeoDataFrame, segmap: pd.DataFrame, sections: pd.DataFrame, path: Path
) -> None:
    seg = segments.merge(segmap[["segment_id", "hwy", "lhrs"]], on="segment_id")
    seg = seg.merge(sections[["hwy", "lhrs", "cr"]], on=["hwy", "lhrs"], how="left")
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    segments[segments["highway"].isin(["primary", "secondary", "motorway"])].plot(
        ax=ax, color=GRID, linewidth=0.4
    )
    seg.plot(
        ax=ax, column="cr", cmap="viridis", linewidth=2.2, legend=True,
        legend_kwds={"label": "collisions per million vehicle-km (2015-2019)", "shrink": 0.6},
    )  # fmt: skip
    ax.set_title(
        "400-series sections mapped to OSM (colour = MTO collision rate)",
        color=TEXT, fontsize=10, loc="left",
    )  # fmt: skip
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_report(
    cfg: dict[str, Any],
    sections: pd.DataFrame,
    located: pd.DataFrame,
    segmap: pd.DataFrame,
    rules: dict[str, str],
    pdf_rows: int,
    path: Path,
) -> None:
    latest = cfg["mto"]["latest_year"]
    mapped = sections[sections["mapped_km"] > 0].copy()
    per_hwy = []
    for hwy, g in mapped.groupby("hwy"):
        w = g["length_km"]
        per_hwy.append(
            {
                "highway": hwy,
                "sections": len(g),
                "MTO km": w.sum(),
                "OSM carriageway km": g["mapped_km"].sum(),
                f"AADT {latest} (length-wtd)": np.average(g[f"aadt_{latest}"], weights=w),
                "AADT min": g[f"aadt_{latest}"].min(),
                "AADT max": g[f"aadt_{latest}"].max(),
                "collisions/yr 2015-19": g["collisions_per_year"].sum(),
                "CR 2015-19": g["collisions_per_year"].sum()
                / g.loc[g["collisions_per_year"].notna(), "mvkm_per_year"].sum(),
                "CR 2021": np.nan
                if g["cr_2021"].isna().all()
                else np.average(g["cr_2021"].fillna(0), weights=g["mvkm_per_year"]),
                "naming rule": rules.get(hwy, ""),
            }
        )
    per_hwy = pd.DataFrame(per_hwy)
    sec_table = mapped.sort_values(["hwy", "order"])[
        ["hwy", "lhrs", "description", "length_km", "mapped_km", f"aadt_{latest}",
         "collisions_per_year", "cr", "cr_2021", "cr_source"]
    ]  # fmt: skip
    loc = located[located["located"]]
    unmapped_motorway = (
        cfg["_motorway_named_km"] - segmap.loc[segmap["hwy"].notna(), "length_m"].sum() / 1000
    )
    lines = [
        "# MTO provincial highway volumes and collision rates (CP5)",
        "",
        "## Sources",
        "",
        "Ontario Ministry of Transportation, Technical Publications library "
        f"({cfg['mto']['portal']}), Open Government Licence - Ontario:",
        "",
        f"- `{cfg['mto']['filenames']['aadt_csv']}`: AADT by highway section, 1988-{latest}.",
        f"- `{cfg['mto']['filenames']['volumes_pdf']}`: the same sections with total reportable "
        "collisions and collision rate (CR, collisions per million vehicle-km; ramp collisions "
        f"excluded) for 1988-2019 and 2021. {pdf_rows:,} section-year rows parsed for these "
        "highways. MTO did not publish 2020 and has not published rates after 2021.",
        "- The Ontario Data Catalogue `traffic-volume` dataset (1371_traffic_volumes.csv) stops "
        "in 2010 and has no collision data, so it is not used.",
        "",
        "## Method",
        "",
        "- Each section is named after one interchange. The interchange is located on the OSM "
        "graph as the point of the highway closest to the crossing road (crosswalk in "
        f"`src/safepath/data/mto.py`; crossings farther than {LOCATE_MAX_M} m are rejected).",
        "- Whether a section runs from its named interchange or to it differs between "
        "highways, so each highway uses the convention under which the straight-line spacing "
        "of located interchanges best matches the MTO section lengths (median |log ratio| "
        "below).",
        "- OSM mainline motorway segments (`highway=motorway`, named after the highway, "
        "collector and express lanes included) are assigned to the section whose interchange "
        "interval contains their midpoint. Ramps (`motorway_link`) are not assigned: MTO "
        "rates exclude ramps.",
        f"- Collision rate = collisions / MVKM pooled over {RATE_YEARS[0]}-{RATE_YEARS[1]} "
        "(MVKM = AADT x 365 x section km / 1e6, as MTO defines it); 2021 shown for "
        "comparison. AADT is the latest year.",
        "",
        "## Per highway (sections that touch Toronto)",
        "",
        markdown_table(per_hwy, floats="{:,.2f}"),
        "",
        "Province-wide freeway CR from the same publication: 0.6 (2015-2019), 0.4 (2021). "
        f"Above it in Toronto: {', '.join(per_hwy.loc[per_hwy['CR 2015-19'] > 0.6, 'highway'])}; "
        f"below: {', '.join(per_hwy.loc[per_hwy['CR 2015-19'] <= 0.6, 'highway'])}. "
        "For comparison, the TPS-derived "
        "motorway rate used in CP4 was 1.15 crashes/km/yr, i.e. roughly 1.15 / (0.365 x "
        "AADT/1000) per MVKM; at a 400-series AADT of 300,000 that is ~0.01 per MVKM, two "
        "orders of magnitude below MTO's figure, which is the undercount Phase 2 fixes.",
        "",
        "## Interchange location and naming convention",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "highway": h,
                        "rule": rules[h],
                        "median |log(OSM spacing / MTO km)| if 'end'": g["rule_err_end"].iloc[0],
                        "... if 'start'": g["rule_err_start"].iloc[0],
                        "interchanges located": int(g["located"].sum()),
                        "median crossing gap m": g["crossing_gap_m"].median(),
                    }
                    for h, g in located.groupby("hwy")
                ]
            ),
            floats="{:.2f}",
        ),
        "",
        f"{len(loc)} interchanges located. OSM mainline km on the six highways not assigned "
        f"to any section: {unmapped_motorway:.1f}. `mapped_km` counts every carriageway "
        "(the 401 has four through Toronto: collector and express each way) and is lumpy "
        "because a long OSM motorway segment goes wholly to the section holding its midpoint. "
        "The 401 west of the 427 and the 400/404/427 north of Steeles lie outside the City "
        "graph. Sections with no published rate (the 427 Finch Ave Ext section, resplit after "
        "2010) take their highway's pooled rate (`cr_source`).",
        "",
        "## Sections",
        "",
        markdown_table(sec_table, floats="{:,.2f}"),
        "",
        "![MTO sections on the OSM graph](figures/mto_sections.png)",
    ]
    path.write_text("\n".join(lines) + "\n")


def run_mto(cfg: dict[str, Any]) -> pd.DataFrame:
    m = cfg["mto"]
    processed = Path(cfg["paths"]["processed_dir"])
    paths = download(cfg)
    highways = list(OSM_NAME)
    history = load_aadt_csv(paths["aadt_csv"], highways)
    pdf_cache = processed / MTO_PDF_ROWS
    if pdf_cache.exists():
        pdf = pd.read_parquet(pdf_cache)
    else:
        pdf = load_pdf(paths["volumes_pdf"], highways, [tuple(p) for p in m["pdf_pages"]])
        pdf.to_parquet(pdf_cache, index=False)
    history = attach_collisions(history, pdf)
    sections = section_summary(history, m["latest_year"])

    segments = gpd.read_parquet(processed / SEGMENTS)
    located, segmaps, rules = [], [], {}
    for hwy in highways:
        secs = sections[sections["hwy"] == hwy]
        loc, segmap, rule = map_highway(hwy, secs, segments)
        if loc.empty:
            continue
        located.append(loc)
        segmaps.append(segmap)
        rules[hwy] = rule
    located_df = pd.concat(located, ignore_index=True)
    segmap = pd.concat(segmaps, ignore_index=True)
    km = segmap.groupby(["hwy", "lhrs"])["length_m"].sum() / 1000
    sections = sections.merge(km.rename("mapped_km").reset_index(), on=["hwy", "lhrs"], how="left")
    sections["mapped_km"] = sections["mapped_km"].fillna(0)
    sections = fill_rates(sections)
    segmap = segmap.merge(
        sections[["hwy", "lhrs", f"aadt_{m['latest_year']}", "cr", "cr_source"]],
        on=["hwy", "lhrs"],
    )
    sections.to_parquet(processed / MTO_SECTIONS, index=False)
    segmap.to_parquet(processed / MTO_SEGMENTS, index=False)

    named = segments["highway"].eq("motorway") & segments["name"].fillna("").str.contains(
        "|".join(OSM_NAME.values()), regex=True
    )
    cfg = {**cfg, "_motorway_named_km": segments.loc[named, "length"].sum() / 1000}
    plot_sections(
        segments, segmap, sections, Path(cfg["paths"]["figures_dir"]) / "mto_sections.png"
    )
    write_report(
        cfg,
        sections,
        located_df,
        segmap,
        rules,
        len(pdf),
        Path(cfg["paths"]["reports_dir"]) / "mto.md",
    )
    log.info(
        "MTO: %d sections mapped to %d OSM segments", (sections["mapped_km"] > 0).sum(), len(segmap)
    )
    return sections
