"""SafePath: fastest vs safest routes across Toronto. Run with `python -m safepath app`."""

from __future__ import annotations

import folium
import streamlit as st
from streamlit_folium import st_folium

from safepath.app.data import (
    LAYER_MEASURES,
    LAYER_MIN_AADT,
    RISK_BANDS,
    build_router,
    risk_layer,
    route_latlon,
    to_wgs84,
)
from safepath.routing import PRESETS, fastest_route, route_stats, safest_route

MAP_CLICK = "Click on the map"
FAST_COLOR, SAFE_COLOR = "#8e8d86", "#008300"
TORONTO = (43.70, -79.40)

st.set_page_config(page_title="SafePath", layout="wide")


@st.cache_resource(show_spinner="Loading the Toronto road graph...")
def get_router():
    router = build_router()
    return router, to_wgs84(router.G.graph["crs"])


@st.cache_data(show_spinner=False)
def get_risk_layer(measure: str):
    return risk_layer(measure=measure)


def add_risk_layer(fmap: folium.Map, measure: str) -> None:
    label = LAYER_MEASURES[measure][1]
    layer = folium.FeatureGroup(name=f"Top 5% by {label}")
    for r in get_risk_layer(measure).itertuples():
        note = "<br>proxy: SPF estimate, own counts misattributed" if r.is_proxy_fallback else ""
        folium.CircleMarker(
            (r.lat, r.lon),
            radius=2.5 + 4 * r.size**0.5,
            color=r.color,
            weight=1,
            fill=True,
            fill_color=r.color,
            fill_opacity=0.6,
            tooltip=folium.Tooltip(
                f"<b>{r.street_names}</b> ({r.band} by {label})"
                f"<br>{r.risk:.1f} expected crashes/yr"
                f"<br>{r.risk_per_mev:.2f} crashes per million entering vehicles"
                f"<br>{r.entering_aadt:,.0f} vehicles enter per day{note}"
            ),
        ).add_to(layer)
    layer.add_to(fmap)


def stats_card(fast, safe, lam: float) -> None:
    s = route_stats(safe, fast)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fastest route", f"{fast.minutes:.1f} min")
    c1.caption(f"{fast.risk:.1f} expected crashes per million trips · {fast.km:.1f} km")
    c2.metric(f"Safest route (λ = {lam:g})", f"{safe.minutes:.1f} min")
    c2.caption(f"{safe.risk:.1f} expected crashes per million trips · {safe.km:.1f} km")
    c3.metric("Extra time", f"+{s['extra_minutes']:.1f} min")
    c3.caption(f"{safe.km - fast.km:+.1f} km vs the fastest route")
    cut = s["risk_reduction_pct"]
    c4.metric("Your crash risk on this trip", f"-{cut:.0f}%" if round(cut) > 0 else "no change")
    c4.caption(f"{safe.risk - fast.risk:+.1f} expected crashes per million trips")


HOW_IT_WORKS = """
**Risk is per vehicle.** Each route's risk is the expected number of crashes for one driver
making the trip, shown per million trips. At an intersection that is its Empirical Bayes
expected crashes per year (a negative binomial Safety Performance Function on traffic
volume and road features, blended with its last five years of Toronto Police crashes)
divided by the vehicles entering it each year. On a road segment it is a crash rate per
vehicle-km times the length. Traffic volumes come from City of Toronto counts plus a model
for uncounted roads.

**Highways.** The 401, 400, 404, 409, 427 and QEW are patrolled by the OPP, so Toronto
Police data barely sees their crashes; they use the Ministry of Transportation's own
collision rates per vehicle-km. Toronto Police records many Gardiner and DVP crashes at the
surface intersections beside or under them; crashes there above what the model expects are
moved back to the expressway.

**Routing.** Each road edge costs `travel time + λ · k · risk`. `k` is set so that at
λ = 1 a typical route's risk weighs as much as its travel time; λ = 0 is the fastest route.
U-turns cost an extra minute.

**Limitations.** Per-vehicle routing minimises your own risk, not everyone's: it can still
send traffic through local streets when they are safer per vehicle, which moves risk onto
the people who live and walk there. Highway rates rest on the assumption that MTO and
Toronto Police crash records are comparable (routing behaviour holds when they are halved
or doubled). Travel times are free-flow, which flatters long highway detours. Volumes on
uncounted roads are modelled (typical error 20-45%). Locations are police-reported and
intersection-snapped, and road features describe today's OpenStreetMap.
"""


def _param(name: str, default: str) -> str:
    return st.query_params.get(name, default)


def pick_endpoints() -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    options = [*PRESETS, MAP_CLICK]
    o_default = _param("origin", "Union Station")
    d_default = _param("destination", "Yorkdale")
    origin = st.sidebar.selectbox(
        "Origin", options, index=options.index(o_default) if o_default in options else 0
    )
    destination = st.sidebar.selectbox(
        "Destination", options, index=options.index(d_default) if d_default in options else 1
    )
    clicks = st.session_state.setdefault("clicks", [])
    slots = [name for name in (origin, destination) if name == MAP_CLICK]
    st.session_state["click_slots"] = len(slots)
    if slots:
        st.sidebar.caption("Map clicks fill the origin first, then the destination.")
        if st.sidebar.button("Reset map clicks"):
            clicks.clear()
    pending = iter(clicks)
    o = next(pending, None) if origin == MAP_CLICK else PRESETS[origin]
    d = next(pending, None) if destination == MAP_CLICK else PRESETS[destination]
    return o, d


def main() -> None:
    router, transformer = get_router()
    st.sidebar.title("SafePath")
    st.sidebar.caption("Toronto routes weighted by learned crash risk.")
    o, d = pick_endpoints()
    lam = st.sidebar.slider(
        "Fastest ↔ Safest (λ)",
        min_value=0.0,
        max_value=5.0,
        value=float(_param("lam", "1.0")),
        step=0.25,
        help="0 = fastest route. At 1, a typical route's crash risk is weighted as much as "
        "its travel time.",
    )

    show_risk = st.sidebar.toggle(
        "Show top 5% riskiest intersections", value=_param("risk", "1") == "1"
    )
    measure = "crashes"
    if show_risk:
        measure = st.sidebar.radio(
            "Rank intersections by",
            list(LAYER_MEASURES),
            index=list(LAYER_MEASURES).index(_param("layer", "crashes")),
            format_func=lambda m: {
                "crashes": "Expected crashes per year",
                "rate": f"Crashes per vehicle (≥ {LAYER_MIN_AADT:,} vehicles/day)",
            }[m],
        )
        st.sidebar.markdown(
            " ".join(f"<span style='color:{c}'>●</span> {label}" for _, label, c in RISK_BANDS),
            unsafe_allow_html=True,
        )

    fmap = folium.Map(location=TORONTO, zoom_start=11, tiles="OpenStreetMap")
    if show_risk:
        add_risk_layer(fmap, measure)
    routes = None
    if o and d:
        on, dn = router.nearest_node(*o), router.nearest_node(*d)
        fast = fastest_route(router, on, dn)
        safe = safest_route(router, on, dn, lam)
        routes = (fast, safe)
        fast_pts = route_latlon(router, fast, transformer)
        safe_pts = route_latlon(router, safe, transformer)
        folium.PolyLine(
            fast_pts,
            color=FAST_COLOR,
            weight=6,
            opacity=0.8,
            tooltip=f"Fastest: {fast.minutes:.1f} min",
        ).add_to(fmap)
        folium.PolyLine(
            safe_pts,
            color=SAFE_COLOR,
            weight=4,
            opacity=0.9,
            tooltip=f"Safest (λ={lam:g}): {safe.minutes:.1f} min",
        ).add_to(fmap)
        folium.Marker(o, tooltip="Origin", icon=folium.Icon(color="blue")).add_to(fmap)
        folium.Marker(d, tooltip="Destination", icon=folium.Icon(color="red")).add_to(fmap)
        lats = [p[0] for p in fast_pts + safe_pts]
        lons = [p[1] for p in fast_pts + safe_pts]
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))

    out = st_folium(
        fmap, height=640, use_container_width=True, returned_objects=["last_clicked"], key="map"
    )
    click = (out or {}).get("last_clicked")
    if click:
        point = (click["lat"], click["lng"])
        clicks = st.session_state["clicks"]
        slots = st.session_state.get("click_slots", 0)
        if point != st.session_state.get("last_click") and len(clicks) < slots:
            st.session_state["last_click"] = point
            clicks.append(point)
            st.rerun()

    if routes:
        stats_card(*routes, lam)
    else:
        st.info("Pick an origin and destination, or click the map twice.")
    with st.expander("How it works / limitations"):
        st.markdown(HOW_IT_WORKS)


main()
