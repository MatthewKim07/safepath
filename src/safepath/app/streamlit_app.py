"""SafePath: fastest vs safest routes in Toronto and London, Ontario.

Runs only from the precomputed app_bundle/ (safepath.app.store): numpy, pandas, scipy,
pyarrow, Streamlit and folium. Locally: `python -m safepath app`; deployed: streamlit_app.py
at the repository root.
"""

from __future__ import annotations

import folium
import streamlit as st
from streamlit_folium import st_folium

from safepath.app.store import (
    LAYER_MEASURES,
    LAYER_MIN_AADT,
    RISK_BANDS,
    City,
    available_cities,
    load_city,
    risk_layer,
)
from safepath.router import NoRoute, fastest_route, route_stats, safest_route

MAP_CLICK = "Click on the map"
FAST_COLOR, SAFE_COLOR = "#8e8d86", "#008300"
RELATIVE_BADGE = (
    "No local crash data: relative risk from a Toronto-trained safety model and London "
    "traffic volumes."
)


@st.cache_resource(show_spinner="Loading the road graph...")
def get_city(key: str) -> City:
    return load_city(key)


@st.cache_data(show_spinner=False)
def get_risk_layer(key: str, measure: str):
    return risk_layer(get_city(key), measure)


def risk_unit(city: City) -> str:
    return "relative risk (uncalibrated)" if city.relative else "expected crashes per million trips"


def add_risk_layer(fmap: folium.Map, city: City, measure: str) -> None:
    label = LAYER_MEASURES[measure][1]
    layer = folium.FeatureGroup(name=f"Top 5% by {label}")
    for r in get_risk_layer(city.key, measure).itertuples():
        note = "<br>proxy: SPF estimate, own counts misattributed" if r.is_proxy_fallback else ""
        if city.relative:
            note += "<br><i>relative: Toronto model, not London crash data</i>"
        factors = "".join(f"<li>{f}</li>" for f in (r.factor_1, r.factor_2, r.factor_3) if f)
        html = (
            f"<b>{r.street_names}</b> ({r.band} by {label})"
            f"<br>{r.risk:.1f} expected crashes/yr"
            f"<br>{r.risk_per_mev:.2f} crashes per million entering vehicles"
            f"<br>{r.entering_aadt:,.0f} vehicles enter per day{note}"
            f"<br><i>Main factors (expected crashes/yr):</i><ul style='margin:2px 0 0 16px;"
            f"padding:0'>{factors}</ul>"
        )
        folium.CircleMarker(
            (r.lat, r.lon),
            radius=2.5 + 4 * r.size**0.5,
            color=r.color,
            weight=1,
            fill=True,
            fill_color=r.color,
            fill_opacity=0.6,
            bubbling_mouse_events=False,  # a click opens the popup, not a map-click route point
            tooltip=folium.Tooltip(html),
            popup=folium.Popup(html, max_width=340),
        ).add_to(layer)
    layer.add_to(fmap)


def stats_card(city: City, fast, safe, lam: float) -> None:
    s = route_stats(safe, fast)
    unit = risk_unit(city)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fastest route", f"{fast.minutes:.1f} min")
    c1.caption(f"{fast.risk:.1f} {unit} · {fast.km:.1f} km")
    c2.metric(f"Safest route (λ = {lam:g})", f"{safe.minutes:.1f} min")
    c2.caption(f"{safe.risk:.1f} {unit} · {safe.km:.1f} km")
    c3.metric("Extra time", f"+{s['extra_minutes']:.1f} min")
    c3.caption(f"{safe.km - fast.km:+.1f} km vs the fastest route")
    cut = s["risk_reduction_pct"]
    c4.metric(
        "Relative crash risk on this trip" if city.relative else "Your crash risk on this trip",
        f"-{cut:.0f}%" if round(cut) > 0 else "no change",
    )
    c4.caption(f"{safe.risk - fast.risk:+.1f} {unit}")


HOW_TORONTO = """
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

**Why an intersection is risky.** The factors shown for each intersection split its
expected crashes into the model's terms (intersection type, traffic volume, legs, leg
angle, streetcar tracks, road classes) and its own crash history relative to what those
predict, with how much each multiplies expected crashes per year.

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

HOW_LONDON = """
**No local crash data.** London publishes no collision records. Intersection risk is
Toronto's Safety Performance Function (traffic volume, signal, legs, leg angle, road
classes; no crash history) applied to London's roads and the City of London's traffic
volumes, divided by the vehicles entering each intersection. Road segments use Toronto's
crash rates per vehicle-km by road class; the 401 and 402 use the Ministry of
Transportation's collision rates. Tested on Ottawa, this no-local-data setting found about
twice as many next-year crashes in its top 5% as a road-class lookup, but less than a
city's own crash history would.

**So read it as relative risk.** The numbers keep Toronto's reporting level and are not
calibrated for London. Use them to compare routes and intersections within London, not as
the chance of a crash. London Police's published top-10 collision intersections (2014-2016)
all sit in the model's top 1%, but so do they by traffic volume alone.

**Routing.** Each road edge costs `travel time + λ · k · risk`, with `k` set for London so
that at λ = 1 a typical route's risk weighs as much as its travel time. U-turns cost an
extra minute.

**Limitations.** The City's volume layer has no count dates and rounds to 500 vehicles; most
local streets have no count and use a class median. Travel times are free-flow and road
features describe today's OpenStreetMap.

**Data.** Traffic volumes: City of London open data, used under the
[Terms of Use – City of London Open Data](https://london.maps.arcgis.com/sharing/rest/content/items/e31458fd0c7e41dd9f93144a9550781d/data).
Highway rates: Ontario Ministry of Transportation (Open Government Licence – Ontario). Roads:
© OpenStreetMap contributors (ODbL).
"""


def _param(name: str, default: str) -> str:
    return st.query_params.get(name, default)


def pick_endpoints(city: City):
    presets = {n: tuple(p) for n, p in city.meta["presets"].items()}
    options = [*presets, MAP_CLICK]
    o_default, d_default = city.meta["default_trip"]
    o_default = _param("origin", o_default)
    d_default = _param("destination", d_default)
    origin = st.sidebar.selectbox(
        "Origin", options, index=options.index(o_default) if o_default in options else 0,
        key=f"origin_{city.key}",
    )  # fmt: skip
    destination = st.sidebar.selectbox(
        "Destination", options,
        index=options.index(d_default) if d_default in options else 1,
        key=f"destination_{city.key}",
    )  # fmt: skip
    clicks = st.session_state.setdefault(f"clicks_{city.key}", [])
    slots = [name for name in (origin, destination) if name == MAP_CLICK]
    st.session_state["click_slots"] = len(slots)
    if slots:
        st.sidebar.caption("Map clicks fill the origin first, then the destination.")
        if st.sidebar.button("Reset map clicks"):
            clicks.clear()
    pending = iter(clicks)
    o = next(pending, None) if origin == MAP_CLICK else presets[origin]
    d = next(pending, None) if destination == MAP_CLICK else presets[destination]
    return o, d


def main() -> None:
    st.set_page_config(page_title="SafePath", layout="wide")
    st.sidebar.title("SafePath")
    cities = available_cities()
    labels = {c: {"toronto": "Toronto", "london": "London, ON"}.get(c, c) for c in cities}
    default_city = _param("city", "toronto")
    key = st.sidebar.selectbox(
        "City", cities, index=cities.index(default_city) if default_city in cities else 0,
        format_func=lambda c: labels[c],
    )  # fmt: skip
    city = get_city(key)
    st.sidebar.caption(f"{city.meta['label']} routes weighted by crash risk.")
    if city.relative:
        st.warning(RELATIVE_BADGE, icon="ℹ️")
    o, d = pick_endpoints(city)
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

    fmap = folium.Map(location=city.meta["center"], zoom_start=city.meta["zoom"],
                      tiles="OpenStreetMap")  # fmt: skip
    if show_risk:
        add_risk_layer(fmap, city, measure)
    routes = None
    router = city.router
    if o and d:
        on, dn = router.nearest_node(*o), router.nearest_node(*d)
        try:
            fast = fastest_route(router, on, dn)
            safe = safest_route(router, on, dn, lam)
        except NoRoute:
            st.error("No drivable route between those points.")
        else:
            routes = (fast, safe)
            fast_pts, safe_pts = router.latlon(fast), router.latlon(safe)
            folium.PolyLine(fast_pts, color=FAST_COLOR, weight=6, opacity=0.8,
                            tooltip=f"Fastest: {fast.minutes:.1f} min").add_to(fmap)  # fmt: skip
            folium.PolyLine(safe_pts, color=SAFE_COLOR, weight=4, opacity=0.9,
                            tooltip=f"Safest (λ={lam:g}): {safe.minutes:.1f} min",
                            ).add_to(fmap)  # fmt: skip
            folium.Marker(o, tooltip="Origin", icon=folium.Icon(color="blue")).add_to(fmap)
            folium.Marker(d, tooltip="Destination", icon=folium.Icon(color="red")).add_to(fmap)
            pts = fast_pts + safe_pts
            if pts:
                lats, lons = [p[0] for p in pts], [p[1] for p in pts]
                fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]],
                                padding=(20, 20))  # fmt: skip

    out = st_folium(
        fmap, height=640, use_container_width=True, returned_objects=["last_clicked"],
        key=f"map_{city.key}",
    )  # fmt: skip
    click = (out or {}).get("last_clicked")
    if click:
        point = (click["lat"], click["lng"])
        clicks = st.session_state[f"clicks_{city.key}"]
        slots = st.session_state.get("click_slots", 0)
        if point != st.session_state.get("last_click") and len(clicks) < slots:
            st.session_state["last_click"] = point
            clicks.append(point)
            st.rerun()

    if routes:
        stats_card(city, *routes, lam)
    else:
        st.info("Pick an origin and destination, or click the map twice.")
    with st.expander("How it works / limitations"):
        st.markdown(HOW_LONDON if city.relative else HOW_TORONTO)


if __name__ == "__main__":  # `streamlit run` executes this file as __main__
    main()
