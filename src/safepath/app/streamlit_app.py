"""SafePath: fastest vs safest routes across Toronto. Run with `python -m safepath app`."""

from __future__ import annotations

import folium
import streamlit as st
from streamlit_folium import st_folium

from safepath.app.data import build_router, route_latlon, to_wgs84
from safepath.routing import PRESETS, fastest_route, route_stats, safest_route

MAP_CLICK = "Click on the map"
FAST_COLOR, SAFE_COLOR = "#8e8d86", "#008300"
TORONTO = (43.70, -79.40)

st.set_page_config(page_title="SafePath", layout="wide")


@st.cache_resource(show_spinner="Loading the Toronto road graph...")
def get_router():
    router = build_router()
    return router, to_wgs84(router.G.graph["crs"])


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

    fmap = folium.Map(location=TORONTO, zoom_start=11, tiles="OpenStreetMap")
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
        fast, safe = routes
        s = route_stats(safe, fast)
        st.write(
            f"Fastest route: {fast.minutes:.1f} min, {fast.risk:.0f} expected crashes/yr on route. "
            f"Safest: {safe.minutes:.1f} min, {safe.risk:.0f}. "
            f"+{s['extra_minutes']:.1f} min, -{s['risk_reduction_pct']:.0f}% expected crashes."
        )
    else:
        st.info("Pick an origin and destination, or click the map twice.")


main()
