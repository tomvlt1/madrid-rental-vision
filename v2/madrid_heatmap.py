"""
Madrid neighborhood heatmap with real OpenStreetMap basemap underneath.

Each of Madrid's 21 official districts is colored by the median rent of the
zone it belongs to (we use 9 zones in our model — see ZONE_MAP below).

Output: v2/figures/eda/06_madrid_heatmap.png
"""

import json
import math
import sys
from pathlib import Path

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.patches import Polygon as MPLPolygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_FILE = PROJECT_ROOT / "v2" / "data" / "listings_clean_v2.csv"
GEOJSON_FILE = Path("/tmp/madrid_districts.geojson")
OUT = PROJECT_ROOT / "v2" / "figures" / "eda" / "06_madrid_heatmap.png"
OUT_PER_M2 = PROJECT_ROOT / "v2" / "figures" / "eda" / "07_madrid_heatmap_per_m2.png"

# Best-effort mapping from geojson district names to our 9 model zones.
# Derived from src/data/neighborhoods.py (the v1 keyword map): each district
# is assigned the zone its core neighbourhoods are mapped to.
DISTRICT_TO_ZONE = {
    "Centro": "Centro",
    "Arganzuela": "Arganzuela",
    "Salamanca": "Salamanca-Retiro",
    "Retiro": "Salamanca-Retiro",
    "Chamberi": "Chamberí",
    "Chamartin": "Norte",
    "Tetuan": "Norte",
    "Fuencarral-El Pardo": "Norte",
    "Moncloa-Aravaca": "Oeste",
    "Latina": "Sur-Sureste",
    "Carabanchel": "Sur-Sureste",
    "Usera": "Sur-Sureste",
    "Puente de Vallecas": "Sur-Sureste",
    "Moratalaz": "Sur-Sureste",
    "Villaverde": "Sur-Sureste",
    "Villa de Vallecas": "Sur-Sureste",
    "Vicalvaro": "Sur-Sureste",
    "Ciudad Lineal": "Periferia Norte",
    "Hortaleza": "Periferia Norte",
    "San Blas": "Periferia Norte",
    "Barajas": "Periferia Norte",
}

R_MERC = 6378137.0


def lonlat_to_mercator(lon, lat):
    """WGS84 -> Web Mercator (EPSG:3857), needed to align with contextily basemaps."""
    x = lon * math.pi / 180.0 * R_MERC
    y = math.log(math.tan(math.pi / 4 + lat * math.pi / 360.0)) * R_MERC
    return x, y


def transform_ring(ring):
    return [lonlat_to_mercator(p[0], p[1]) for p in ring]


def feature_polygons(feature):
    """Yield the outer ring of every polygon in a feature, in mercator coords."""
    geom = feature["geometry"]
    if geom["type"] == "Polygon":
        yield transform_ring(geom["coordinates"][0])
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield transform_ring(poly[0])


def feature_centroid(feature):
    """Centroid of the largest polygon in the feature, in mercator coords."""
    geom = feature["geometry"]
    if geom["type"] == "Polygon":
        outer = transform_ring(geom["coordinates"][0])
    else:
        # Pick the polygon with the most points (proxy for largest)
        outer = max(
            (transform_ring(p[0]) for p in geom["coordinates"]),
            key=len,
        )
    cx_pt = float(np.mean([p[0] for p in outer]))
    cy_pt = float(np.mean([p[1] for p in outer]))
    return cx_pt, cy_pt


def build_choropleth(value_per_zone, title, out_file, units, fmt="€{:.0f}"):
    with open(GEOJSON_FILE) as f:
        geo = json.load(f)

    # Resolve district -> value via zone mapping
    district_value = {}
    for d, z in DISTRICT_TO_ZONE.items():
        district_value[d] = value_per_zone.get(z)

    vals = [v for v in district_value.values() if v is not None]
    vmin, vmax = float(min(vals)), float(max(vals))
    cmap = cm.YlOrRd

    fig, ax = plt.subplots(figsize=(11, 10))

    label_xy = []  # (x, y, name, value)
    all_x, all_y = [], []

    for feature in geo["features"]:
        name = feature["properties"]["name"]
        v = district_value.get(name)
        face = cmap((v - vmin) / (vmax - vmin)) if v is not None else (0.85, 0.85, 0.85, 0.6)

        for ring in feature_polygons(feature):
            patch = MPLPolygon(
                ring,
                facecolor=face,
                edgecolor="white",
                linewidth=1.0,
                alpha=0.78,
                zorder=2,
            )
            ax.add_patch(patch)
            all_x.extend([p[0] for p in ring])
            all_y.extend([p[1] for p in ring])

        cx_pt, cy_pt = feature_centroid(feature)
        label_xy.append((cx_pt, cy_pt, name, v))

    # Labels: district name + value
    for x, y, name, v in label_xy:
        if v is not None:
            txt = f"{name}\n{fmt.format(v)}"
        else:
            txt = name
        ax.annotate(
            txt,
            xy=(x, y),
            ha="center",
            va="center",
            fontsize=7.5,
            color="black",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    # Set extent
    pad = 800
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # OSM basemap underneath — light grey style for choropleth legibility
    cx.add_basemap(
        ax,
        crs="EPSG:3857",
        source=cx.providers.CartoDB.Positron,
        zorder=1,
    )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label(units, fontsize=10)

    fig.suptitle(title, fontsize=13, y=0.97)
    fig.tight_layout()
    fig.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_file.name}")


def main():
    print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["sqft_m2"] = pd.to_numeric(df["sqft_m2"], errors="coerce")
    df = df.dropna(subset=["price", "sqft_m2", "zone"]).reset_index(drop=True)

    median_by_zone = df.groupby("zone")["price"].median().to_dict()
    print("Median rent per zone:")
    for z, v in sorted(median_by_zone.items(), key=lambda kv: kv[1]):
        print(f"  {z:20s} €{v:>5.0f}")

    df["price_per_m2"] = df["price"] / df["sqft_m2"]
    median_per_m2_by_zone = df.groupby("zone")["price_per_m2"].median().to_dict()

    print("\nGenerating heatmaps...")
    build_choropleth(
        median_by_zone,
        "Madrid rent by district (median monthly rent of the zone)",
        OUT,
        units="Median monthly rent (€)",
        fmt="€{:.0f}",
    )
    build_choropleth(
        median_per_m2_by_zone,
        "Madrid rent intensity by district (median €/m² of the zone)",
        OUT_PER_M2,
        units="Median rent per m² (€)",
        fmt="€{:.1f}/m²",
    )

    print(f"\nSaved to: {OUT.parent}")


if __name__ == "__main__":
    main()
