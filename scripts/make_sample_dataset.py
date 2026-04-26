"""
Generate a small synthetic listings_clean_sample.csv that matches the
real schema. Lets the instructor (and teammates without the real data)
run the pipeline end-to-end without needing the unshipped Idealista
dataset.

Output:
    data/processed/listings_clean_sample.csv     (50 listings, full schema)

The synthetic price is generated as a noisy linear function of size,
zone effect, and rooms so that a model trained on it actually learns
something — the demo metrics on this set won't match the real-data
numbers, but the pipeline executes correctly.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N = 50
OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "listings_clean_sample.csv"

ZONES = [
    "Centro", "Salamanca-Retiro", "Chamberí", "Norte",
    "Periferia Norte", "Sur-Sureste", "Arganzuela", "Oeste",
]
ZONE_PRICE_OFFSET = {  # rough mean rent shift per zone (in EUR)
    "Centro": 200,
    "Salamanca-Retiro": 700,
    "Chamberí": 400,
    "Norte": 100,
    "Periferia Norte": -150,
    "Sur-Sureste": -300,
    "Arganzuela": 50,
    "Oeste": 250,
}
LOCATIONS = {
    "Centro": ["Lavapiés-Embajadores, Madrid", "Sol, Madrid", "Malasaña-Universidad, Madrid"],
    "Salamanca-Retiro": ["Castellana, Madrid", "Recoletos, Madrid", "Goya, Madrid"],
    "Chamberí": ["Trafalgar, Madrid", "Almagro, Madrid"],
    "Norte": ["Conde Orgaz-Piovera, Madrid", "Hispanoamérica, Madrid"],
    "Periferia Norte": ["Sanchinarro, Madrid", "Las Tablas, Madrid"],
    "Sur-Sureste": ["Vallecas, Madrid", "Vicálvaro, Madrid"],
    "Arganzuela": ["Imperial, Madrid", "Acacias, Madrid"],
    "Oeste": ["Aravaca, Madrid", "Pozuelo, Madrid"],
}
SAMPLE_DESCRIPTIONS = [
    "Piso reformado y luminoso en el centro de Madrid. Cocina equipada, salón amplio, dormitorio con armario empotrado.",
    "Apartamento moderno con terraza, vistas despejadas y excelente iluminación natural. Edificio con ascensor.",
    "Vivienda completamente amueblada en zona prime, ideal para profesionales. Calefacción central y aire acondicionado.",
    "Estudio céntrico cerca del metro, perfecto para una persona. Recientemente reformado con calidades modernas.",
    "Ático con terraza panorámica, dos dormitorios, dos baños. Plaza de garaje y trastero incluidos.",
    "Piso clásico en finca señorial, techos altos, suelos de tarima. Necesita pequeñas actualizaciones.",
    "Dúplex luminoso con tres dormitorios, dos baños y aseo de cortesía. Zona tranquila y bien comunicada.",
]


def synthesize(rng):
    rows = []
    for i in range(N):
        zone = rng.choice(ZONES)
        sqft = int(rng.normal(85, 35))
        sqft = max(20, min(sqft, 300))
        rooms = max(1, min(int(round(sqft / 30 + rng.normal(0, 0.6))), 5))
        bathrooms = max(1, min(int(round(rooms * 0.7 + rng.normal(0, 0.4))), 3))
        floor_num = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, None], p=[0.05, 0.15, 0.2, 0.2, 0.15, 0.1, 0.08, 0.04, 0.03])

        # synthetic rent: ~22 EUR/m^2 + zone shift + room premium + noise
        base = 22 * sqft + ZONE_PRICE_OFFSET[zone] + (rooms - 2) * 80 + (bathrooms - 1) * 60
        if floor_num is not None and floor_num >= 5:
            base += 100  # high-floor bonus
        price = float(round(max(500, base + rng.normal(0, 220))))

        elevator = bool(rng.choice([True, False], p=[0.7, 0.3]))
        ac = bool(rng.choice([True, False], p=[0.55, 0.45]))
        terrace = bool(rng.choice([True, False], p=[0.3, 0.7]))
        furnished = bool(rng.choice([True, False], p=[0.65, 0.35]))
        heating = bool(rng.choice([True, False], p=[0.85, 0.15]))
        exterior = bool(rng.choice([True, False], p=[0.7, 0.3]))
        parking = bool(rng.choice([True, False], p=[0.25, 0.75]))
        storage = bool(rng.choice([True, False], p=[0.3, 0.7]))

        location = rng.choice(LOCATIONS[zone])
        description = rng.choice(SAMPLE_DESCRIPTIONS)
        num_images = int(rng.integers(8, 30))
        image_urls = json.dumps([f"https://example.invalid/sample/{i}/{j}.jpg" for j in range(num_images)])
        floor_str = "" if floor_num is None else f"planta {floor_num}ª"

        rows.append({
            "url": f"https://www.idealista.com/inmueble/9{i:08d}/",
            "price": price,
            "title": f"Sample listing #{i + 1}",
            "sqft_m2": sqft,
            "rooms": float(rooms),
            "bathrooms": float(bathrooms),
            "floor": floor_str,
            "elevator": elevator, "ac": ac, "terrace": terrace,
            "furnished": furnished, "heating": heating, "exterior": exterior,
            "parking": parking, "storage": storage,
            "location": location,
            "description": description,
            "image_urls": image_urls,
            "num_images": num_images,
            "zone": zone,
            "price_per_m2": round(price / sqft, 2),
            "floor_num": floor_num,
        })
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    df = synthesize(rng)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(df)} rows)")
    print(df[["price", "sqft_m2", "rooms", "zone"]].describe(include="all").to_string())


if __name__ == "__main__":
    main()
