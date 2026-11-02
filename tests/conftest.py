import networkx as nx
import osmnx as ox
import pytest

CRS = "EPSG:32617"


def _graph() -> nx.MultiDiGraph:
    """A main street (west-east) crossed by a jogged side street.

    n2 (200, 0) and n6 (212, 0) are 12 m apart: one real-world intersection drawn as two
    OSM nodes. n1, n3, n4, n5 are dead ends.
        n4 (200, 200)
          |
    n1 -- n2 - n6 -- n3        y = 0
                |
              n5 (212, -200)
    """
    G = nx.MultiDiGraph(crs=CRS)
    coords = {1: (0, 0), 2: (200, 0), 6: (212, 0), 3: (400, 0), 4: (200, 200), 5: (212, -200)}
    for n, (x, y) in coords.items():
        G.add_node(n, x=float(x), y=float(y))
    streets = [
        (1, 2, "Main Street"),
        (2, 6, "Main Street"),
        (6, 3, "Main Street"),
        (4, 2, "Cross Avenue"),
        (6, 5, "Cross Avenue"),
    ]
    for i, (u, v, name) in enumerate(streets):
        (x1, y1), (x2, y2) = coords[u], coords[v]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        for a, b in ((u, v), (v, u)):
            G.add_edge(
                a, b, osmid=100 + i, length=length, name=name, highway="secondary", oneway=False
            )
    nx.set_node_attributes(G, ox.stats.count_streets_per_node(G), name="street_count")
    return G


@pytest.fixture
def toy_graph() -> nx.MultiDiGraph:
    return _graph()


@pytest.fixture
def class_rates():
    """Build the per-edge rate table attach_risk expects from a rate per highway class."""

    def build(G: nx.MultiDiGraph, edge_map, by_class: dict[str, float]):
        e = edge_map.copy()
        e["rate_group"] = [
            G.edges[u, v, k]["highway"] for u, v, k in e[["u", "v", "key"]].itertuples(index=False)
        ]
        e["rate_per_mvkm"] = e["rate_group"].map(by_class).fillna(0.0)
        return e

    return build
