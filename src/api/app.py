# FastAPI backend for the real estate agent suite.
# Three views: /scout (find under-marketed listings), /listings/{id} (diagnose),
# /simulate (test new photos + description).

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from torchvision import transforms

from src.config import IMAGES_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.vision.finetune import PriceResNet

MODELS_DIR = PROJECT_ROOT / "models"
ENRICHED_FILE = PROCESSED_DIR / "listings_enriched.parquet"

# Test-set MAE from models/results.json (gb_tabular_text_finetuned_image).
# Post-leakage-fix (split manifest + feature-tuple dedup): 457 on 1,425 listings.
# Surfaced on every prediction so the UI can draw an honest ±confidence band.
MODEL_MAE_EUR = 457.0

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

NUMERIC_FEATURES = ["sqft_m2", "rooms", "bathrooms", "floor_num", "num_images"]
BOOL_FEATURES = ["elevator", "ac", "terrace", "furnished", "heating",
                 "exterior", "parking", "storage"]

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading enriched listings...")
    state["df"] = pd.read_parquet(ENRICHED_FILE)

    print("Loading sklearn pipeline...")
    state["gb_tab"] = joblib.load(MODELS_DIR / "gb_tabular.joblib")
    state["gb_tabular_text"] = joblib.load(MODELS_DIR / "gb_tabular_text.joblib")
    state["gb_tabular_photos"] = joblib.load(MODELS_DIR / "gb_tabular_finetuned_image.joblib")
    state["gb_all"] = joblib.load(MODELS_DIR / "gb_tabular_text_finetuned_image.joblib")
    state["tab_scaler"] = joblib.load(MODELS_DIR / "tab_scaler.joblib")
    state["img_scaler_ft"] = joblib.load(MODELS_DIR / "img_scaler_finetuned.joblib")
    state["pca_ft"] = joblib.load(MODELS_DIR / "pca_finetuned.joblib")
    state["text_scaler"] = joblib.load(MODELS_DIR / "text_scaler.joblib")
    state["pca_text"] = joblib.load(MODELS_DIR / "pca_text.joblib")
    with open(MODELS_DIR / "feature_names.json") as f:
        state["feature_names"] = json.load(f)

    print("Loading fine-tuned ResNet...")
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    state["device"] = device
    resnet = PriceResNet()
    resnet.load_state_dict(torch.load(MODELS_DIR / "resnet_finetuned.pt",
                                      map_location="cpu", weights_only=True))
    resnet = resnet.to(device)
    resnet.train(False)
    state["resnet"] = resnet

    print("Loading sentence transformer...")
    state["text_model"] = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # text embeddings + premium centroid + cached pca vectors
    text_emb = np.load(PROCESSED_DIR / "text_embeddings.npy")
    text_idx = pd.read_csv(PROCESSED_DIR / "text_embeddings_index.csv")
    text_idx["listing_id"] = text_idx["listing_id"].astype(str)
    text_lookup = dict(zip(text_idx["listing_id"], range(len(text_idx))))
    state["text_emb"] = text_emb
    state["text_lookup"] = text_lookup
    text_scaled_all = state["text_scaler"].transform(text_emb)
    state["text_pca_all"] = state["pca_text"].transform(text_scaled_all)

    # fine-tuned image embeddings (per-listing mean-pooled): needed for
    # feature-by-feature decomposition on cached listings.
    ft_emb = np.load(PROCESSED_DIR / "embeddings_finetuned.npy")
    ft_idx = pd.read_csv(PROCESSED_DIR / "embeddings_finetuned_index.csv")
    ft_idx["listing_id"] = ft_idx["listing_id"].astype(str)
    state["ft_lookup"] = dict(zip(ft_idx["listing_id"], range(len(ft_idx))))
    ft_scaled_all = state["img_scaler_ft"].transform(ft_emb)
    state["ft_pca_all"] = state["pca_ft"].transform(ft_scaled_all)

    df = state["df"]
    q75 = df["rent_eur"].quantile(0.75)
    premium_ids = df[df["rent_eur"] >= q75]["listing_id"].tolist()
    rows = [text_lookup[lid] for lid in premium_ids if lid in text_lookup]
    centroid = text_emb[rows].mean(axis=0)
    state["premium_centroid"] = centroid / (np.linalg.norm(centroid) + 1e-9)

    # per-listing tabular feature vectors (scaled) for /intake
    from src.api.precompute import build_tab_matrix
    clean = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
    clean["listing_id"] = clean["url"].apply(
        lambda u: u.rstrip("/").split("/")[-1]
    ).astype(str)
    tab_raw = build_tab_matrix(clean, state["feature_names"])
    tab_scaled_all = state["tab_scaler"].transform(tab_raw)
    state["tab_scaled_by_id"] = {
        lid: tab_scaled_all[i] for i, lid in enumerate(clean["listing_id"])
    }

    print(f"Ready. {len(state['df'])} listings loaded on {device}.")
    yield
    state.clear()


app = FastAPI(title="Rental Intelligence API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ---------- response models ----------

class ScoutItem(BaseModel):
    listing_id: str
    url: str
    title: Optional[str] = None
    zone: str
    rent_eur: float
    sqft_m2: float
    rooms: Optional[float] = None
    predicted_rent_tabular_eur: float
    image_score_eur: Optional[float] = None
    image_score_percentile: Optional[float] = None
    rent_gap_pct: float
    under_marketing_score: float
    thumbnail_url: Optional[str] = None


class PhotoScore(BaseModel):
    image_url: str
    score_eur: float
    rank_in_listing: int


class Diagnosis(BaseModel):
    weakest_photo: Optional[PhotoScore] = None
    strongest_photo: Optional[PhotoScore] = None
    is_under_marketed: bool
    peer_rent_gap_eur: Optional[float] = None  # rent below peer-expected, if any
    verdict: str


class ListingDetail(BaseModel):
    listing_id: str
    url: str
    title: Optional[str] = None
    location: Optional[str] = None
    zone: str
    rent_eur: float
    sqft_m2: float
    rooms: Optional[float] = None
    bathrooms: Optional[float] = None
    description: Optional[str] = None
    predicted_rent_tabular_eur: float
    predicted_rent_full_eur: Optional[float] = None
    zone_median_rent_eur: float
    image_score_eur: Optional[float] = None
    image_score_percentile: Optional[float] = None
    text_distance_premium: Optional[float] = None
    photos: list[PhotoScore]
    diagnosis: Diagnosis
    breakdown: Optional["FeatureBreakdown"] = None
    mae_eur: float = 443.0


class SimulateResponse(BaseModel):
    predicted_rent_eur: float
    per_photo_scores: list[PhotoScore]
    description_distance_premium: Optional[float] = None
    delta_vs_baseline_eur: Optional[float] = None
    baseline_rent_eur: Optional[float] = None
    suggestions: list[str]


class IntakeBaseline(BaseModel):
    listing_id: str
    title: Optional[str] = None
    zone: str
    sqft_m2: float
    rooms: Optional[float] = None
    current_rent_eur: float
    peer_expected_rent_full_eur: Optional[float] = None  # tabular + photos + text
    peer_expected_rent_tabular_eur: float  # tabular only
    image_score_eur: Optional[float] = None
    image_score_percentile: Optional[float] = None
    num_existing_photos: int
    thumbnail_url: Optional[str] = None


class IntakeWithExtras(BaseModel):
    predicted_rent_eur: float
    predicted_rent_mae_eur: float  # ±band for honest UX
    delta_vs_current_rent_eur: float
    delta_vs_previous_model_eur: Optional[float] = None
    per_extra_scores: list[PhotoScore]
    replaced_photo_urls: list[str]  # existing photos we swapped out
    kept_photo_count: int  # existing photos retained
    total_photos_considered: int  # kept + extras
    suggestions: list[str]


class IntakeResponse(BaseModel):
    baseline: IntakeBaseline
    with_extras: Optional[IntakeWithExtras] = None


class PredictLiveRequest(BaseModel):
    # Minimum required fields
    sqft: float
    zone: Optional[str] = None   # one of the 8 canonical zones
    location: Optional[str] = None  # free text: inferred to zone if zone is missing
    # Optional identifiers / comparison
    listing_id: Optional[str] = None
    current_rent_eur: Optional[float] = None
    # Optional tabular features (default false/0)
    rooms: Optional[float] = None
    bathrooms: Optional[float] = None
    floor_num: Optional[int] = None
    num_images: int = 0
    elevator: bool = False
    ac: bool = False
    terrace: bool = False
    furnished: bool = False
    heating: bool = False
    exterior: bool = False
    parking: bool = False
    storage: bool = False
    # Optional richer features
    description: Optional[str] = None
    image_urls: Optional[list[str]] = None
    # "tabular" (fast, instant) or "full" (downloads images, slower)
    mode: str = "tabular"


class FeatureBreakdown(BaseModel):
    """Per-feature-block decomposition of a prediction.
    All deltas are relative to the tabular-only baseline.
    Contributions don't have to sum exactly to `full_eur`: tree ensembles
    have interaction terms, captured in `interaction_eur`.
    """
    tabular_eur: float  # base prediction from size, rooms, zone, amenities
    with_text_eur: Optional[float] = None  # tabular + description only
    with_photos_eur: Optional[float] = None  # tabular + photos only
    full_eur: Optional[float] = None  # tabular + text + photos
    text_delta_eur: Optional[float] = None  # with_text_eur - tabular_eur
    photos_delta_eur: Optional[float] = None  # with_photos_eur - tabular_eur
    interaction_eur: Optional[float] = None  # nonlinear interaction of text × photos
    tabular_note: Optional[str] = None  # e.g. "80m² · 2 rooms · Centro · elevator"
    text_note: Optional[str] = None     # e.g. "description 72% similar to premium"
    photos_note: Optional[str] = None   # e.g. "12 photos, avg strength 44/100"


class PhotoImpact(BaseModel):
    image_url: str  # URL the extension can match against DOM thumbnails
    score_eur: float  # fine-tuned ResNet head output (per-image proxy)
    rank_in_listing: int  # 1 = strongest photo in the listing
    delta_vs_listing_mean_eur: float  # this photo vs the listing's average
    tone: str  # "helps", "hurts", "neutral" (bucketed by delta)


class PredictLiveResponse(BaseModel):
    predicted_rent_eur: float
    mae_eur: float
    features_used: list[str]
    cached: bool
    zone_median_rent_eur: Optional[float] = None
    diagnosis: str  # "below_peer", "on_peer", "above_peer", "unknown"
    current_rent_eur: Optional[float] = None
    delta_vs_current_eur: Optional[float] = None
    breakdown: Optional[FeatureBreakdown] = None
    per_photo_impact: Optional[list[PhotoImpact]] = None


# ---------- helpers ----------

def _thumbnail_url(listing_id: str) -> Optional[str]:
    listing_dir = IMAGES_DIR / listing_id
    if not listing_dir.exists():
        return None
    files = sorted(listing_dir.glob("*.jpg")) + sorted(listing_dir.glob("*.webp"))
    if not files:
        return None
    return f"/images/{listing_id}/{files[0].name}"


def _photos_from_row(row) -> list[PhotoScore]:
    files = row["photo_filenames"] if row["photo_filenames"] is not None else []
    scores = row["photo_scores_eur"] if row["photo_scores_eur"] is not None else []
    if len(files) == 0:
        return []
    ranked = sorted(zip(files, scores), key=lambda x: -x[1])
    rank_map = {f: i + 1 for i, (f, _) in enumerate(ranked)}
    out = []
    for f, s in zip(files, scores):
        out.append(PhotoScore(
            image_url=f"/images/{row['listing_id']}/{f}",
            score_eur=float(s),
            rank_in_listing=rank_map[f],
        ))
    return out


def _build_diagnosis(row, photos) -> Diagnosis:
    weakest = min(photos, key=lambda p: p.score_eur) if photos else None
    strongest = max(photos, key=lambda p: p.score_eur) if photos else None

    rent = row["rent_eur"]
    pred_tab = row["predicted_rent_tabular_eur"]
    pred_full = row.get("predicted_rent_full_eur")
    img_pct = row.get("image_score_percentile")

    is_under = False
    verdict_bits = []
    if pd.notna(pred_tab) and rent < pred_tab * 0.92:
        is_under = True
        verdict_bits.append(f"rent is {(1 - rent/pred_tab)*100:.0f}% below tabular prediction")
    if pd.notna(img_pct) and img_pct < 35:
        is_under = True
        verdict_bits.append(f"photos rank in the {img_pct:.0f}th percentile")

    gap_eur = None
    if pd.notna(pred_full) and pred_full > rent:
        gap_eur = float(pred_full - rent)

    if is_under:
        verdict = "Below peer-expected: " + "; ".join(verdict_bits) + "."
    elif photos and weakest and strongest and (strongest.score_eur - weakest.score_eur) > 300:
        verdict = "Photo scores vary widely within the listing: swapping the weakest may reduce days-on-market."
    else:
        verdict = "On peer average: no triage action recommended."

    return Diagnosis(
        weakest_photo=weakest,
        strongest_photo=strongest,
        is_under_marketed=is_under,
        peer_rent_gap_eur=gap_eur,
        verdict=verdict,
    )


def _score_photos(pil_images: list[Image.Image]) -> list[float]:
    device = state["device"]
    model = state["resnet"]
    tensors = [TRANSFORM(img.convert("RGB")) for img in pil_images]
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        log_preds = model(batch).cpu().numpy()
    return [float(np.expm1(x)) for x in log_preds]


def _embed_photos(pil_images: list[Image.Image]) -> np.ndarray:
    device = state["device"]
    model = state["resnet"]
    tensors = [TRANSFORM(img.convert("RGB")) for img in pil_images]
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        emb = model.extract_embedding(batch).cpu().numpy()
    return emb.mean(axis=0)  # mean-pool across listing


def _build_tab_row(sqft, rooms, bathrooms, zone, num_images,
                   bathrooms_default=1, floor_num=0,
                   elevator=False, ac=False, terrace=False,
                   furnished=False, heating=False, exterior=False,
                   parking=False, storage=False) -> np.ndarray:
    feature_names = state["feature_names"]
    row = {
        "sqft_m2": sqft, "rooms": rooms,
        "bathrooms": bathrooms if bathrooms is not None else bathrooms_default,
        "floor_num": floor_num, "num_images": num_images,
        "elevator": float(elevator), "ac": float(ac), "terrace": float(terrace),
        "furnished": float(furnished), "heating": float(heating),
        "exterior": float(exterior), "parking": float(parking),
        "storage": float(storage),
    }
    zones = ["Arganzuela", "Centro", "Chamberí", "Norte", "Oeste",
             "Periferia Norte", "Salamanca-Retiro", "Sur-Sureste"]
    for z in zones:
        row[f"zone_{z}"] = 1.0 if z == zone else 0.0
    return np.array([[row.get(f, 0.0) for f in feature_names]], dtype=np.float32)


# ---------- endpoints ----------

@app.get("/")
def root():
    return {"status": "ok", "listings": len(state.get("df", [])),
            "endpoints": ["/scout", "/listings/{id}", "/simulate", "/intake", "/predict-live"]}


def _tabular_note_from_row(r) -> str:
    parts = [f"{int(round(r['sqft_m2']))}m²"]
    if pd.notna(r.get("rooms")):
        parts.append(f"{int(r['rooms'])} rm")
    parts.append(str(r["zone"]))
    return " · ".join(parts)


def _tabular_note_from_req(req: "PredictLiveRequest") -> str:
    parts = [f"{int(round(req.sqft))}m²"]
    if req.rooms is not None:
        parts.append(f"{int(req.rooms)} rm")
    if req.zone:
        parts.append(req.zone)
    return " · ".join(parts)


def _make_text_note(delta_eur: Optional[float], sim_pct: Optional[float]) -> Optional[str]:
    """Build a description annotation grounded in the model's actual delta.
    sim_pct (0–100) is surface cosine similarity to the premium-text centroid,
    shown as raw context, not as an inference direction, because the GB
    model can disagree with it.
    """
    if delta_eur is None:
        return None
    if abs(delta_eur) < 50:
        verdict = "model reads as neutral"
    elif delta_eur > 0:
        verdict = "model reads phrasings as premium-leaning"
    else:
        verdict = "model reads specific phrasings as below-market"
    if sim_pct is not None:
        verdict += f" (surface overlap with premium vocabulary: {sim_pct:.0f}%)"
    return verdict


def _make_photos_note(
    delta_eur: Optional[float],
    n_photos: Optional[int],
    pct_global: Optional[float],
) -> Optional[str]:
    """Photos annotation uses the model's delta as ground truth (zone-aware
    by construction) and reports the global percentile as secondary context.
    """
    if not n_photos:
        return None
    parts = [f"{int(n_photos)} photos"]
    if delta_eur is not None:
        if abs(delta_eur) < 50:
            parts.append("score neutrally for the model")
        elif delta_eur > 0:
            parts.append("score above peers in this zone")
        else:
            parts.append("score below peers in this zone")
    if pct_global is not None:
        parts.append(f"{int(pct_global)}th pct vs all Madrid")
    return ", ".join(parts)


def _compute_breakdown(
    tab_scaled: np.ndarray,
    txt_pca: Optional[np.ndarray],
    img_pca: Optional[np.ndarray],
    tabular_note: Optional[str] = None,
    text_sim_pct: Optional[float] = None,
    photos_count: Optional[int] = None,
    photos_pct_global: Optional[float] = None,
) -> FeatureBreakdown:
    """Run each ablation model and package the contributions.
    Notes are built AFTER deltas are known, so the description/photo copy
    reflects the model's actual direction instead of raw surface metrics.
    """
    # Tabular-only baseline
    tab_log = float(state["gb_tab"].predict(tab_scaled)[0])
    tab_eur = float(np.expm1(tab_log))

    has_text = txt_pca is not None
    has_img = img_pca is not None

    with_text_eur = None
    with_photos_eur = None
    full_eur = None
    text_delta = None
    photos_delta = None
    interaction = None

    if has_text:
        combined_t = np.hstack([tab_scaled, txt_pca])
        with_text_eur = float(np.expm1(state["gb_tabular_text"].predict(combined_t)[0]))
        text_delta = with_text_eur - tab_eur

    if has_img:
        combined_i = np.hstack([tab_scaled, img_pca])
        with_photos_eur = float(
            np.expm1(state["gb_tabular_photos"].predict(combined_i)[0])
        )
        photos_delta = with_photos_eur - tab_eur

    if has_text and has_img:
        combined_all = np.hstack([tab_scaled, txt_pca, img_pca])
        full_eur = float(np.expm1(state["gb_all"].predict(combined_all)[0]))
        interaction = full_eur - tab_eur - (text_delta or 0.0) - (photos_delta or 0.0)

    text_note = _make_text_note(text_delta, text_sim_pct) if has_text else None
    photos_note = _make_photos_note(
        photos_delta,
        photos_count,
        photos_pct_global,
    ) if has_img else None

    return FeatureBreakdown(
        tabular_eur=tab_eur,
        with_text_eur=with_text_eur,
        with_photos_eur=with_photos_eur,
        full_eur=full_eur,
        text_delta_eur=text_delta,
        photos_delta_eur=photos_delta,
        interaction_eur=interaction,
        tabular_note=tabular_note,
        text_note=text_note,
        photos_note=photos_note,
    )


def _diagnose(pred_eur: float, current_eur: Optional[float]) -> str:
    if current_eur is None:
        return "unknown"
    delta = current_eur - pred_eur
    if abs(delta) < MODEL_MAE_EUR:
        return "fair"
    return "overpriced" if delta > 0 else "underpriced"


def _download_image(url: str, timeout: float = 5.0) -> Optional[Image.Image]:
    import requests as _req
    try:
        r = _req.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.idealista.com/",
            },
        )
        if r.status_code == 200 and r.content:
            return Image.open(io.BytesIO(r.content))
    except Exception:
        return None
    return None


@app.post("/predict-live", response_model=PredictLiveResponse)
def predict_live(req: PredictLiveRequest):
    """Predict peer-expected rent from raw features.
    Fast cached path for listings we know; live inference otherwise.
    Used by the Idealista browser extension.
    """
    from src.data.neighborhoods import get_zone

    df = state["df"]
    # Resolve zone: accept canonical zone or infer from free-text location.
    canonical_zones = {
        "Arganzuela", "Centro", "Chamberí", "Norte", "Oeste",
        "Periferia Norte", "Salamanca-Retiro", "Sur-Sureste",
    }
    zone = req.zone if req.zone in canonical_zones else None
    if zone is None and req.location:
        inferred = get_zone(req.location)
        if inferred in canonical_zones:
            zone = inferred
    if zone is None:
        raise HTTPException(
            status_code=400,
            detail="Could not determine zone. Pass `zone` (one of 8 canonical zones) or a `location` string.",
        )
    # Mutate a local copy for downstream use without touching the request object
    req = req.model_copy(update={"zone": zone})
    zone_medians = df.groupby("zone")["rent_eur"].median().to_dict()
    zone_median = zone_medians.get(zone)

    # ---- fast path: cached lookup -----------------------------------
    if req.listing_id:
        match = df[df["listing_id"] == req.listing_id]
        if not match.empty:
            r = match.iloc[0]
            if req.mode == "full" and pd.notna(r["predicted_rent_full_eur"]):
                pred = float(r["predicted_rent_full_eur"])
                features_used = ["cached:full"]
            else:
                pred = float(r["predicted_rent_tabular_eur"])
                features_used = ["cached:tabular"]
            current = req.current_rent_eur if req.current_rent_eur is not None else float(r["rent_eur"])

            # Build breakdown from cached embeddings for this listing
            lid = r["listing_id"]
            breakdown = None
            tab_vec = state["tab_scaled_by_id"].get(lid)
            if tab_vec is not None:
                tab_vec = tab_vec.reshape(1, -1)
                txt_pca = None
                img_pca = None
                if lid in state["text_lookup"]:
                    txt_pca = state["text_pca_all"][state["text_lookup"][lid]].reshape(1, -1)
                if lid in state["ft_lookup"]:
                    img_pca = state["ft_pca_all"][state["ft_lookup"][lid]].reshape(1, -1)
                text_sim_pct = None
                if r.get("text_distance_premium") is not None and not pd.isna(r.get("text_distance_premium")):
                    text_sim_pct = max(0.0, 1.0 - float(r["text_distance_premium"])) * 100
                photos_count = int(r["num_images"]) if pd.notna(r.get("num_images")) else None
                photos_pct_global = float(r["image_score_percentile"]) if pd.notna(r.get("image_score_percentile")) else None
                breakdown = _compute_breakdown(
                    tab_vec, txt_pca, img_pca,
                    tabular_note=_tabular_note_from_row(r),
                    text_sim_pct=text_sim_pct,
                    photos_count=photos_count,
                    photos_pct_global=photos_pct_global,
                )

            return PredictLiveResponse(
                predicted_rent_eur=pred,
                mae_eur=MODEL_MAE_EUR,
                features_used=features_used,
                cached=True,
                zone_median_rent_eur=zone_median,
                diagnosis=_diagnose(pred, current),
                current_rent_eur=current,
                delta_vs_current_eur=(pred - current) if current is not None else None,
                breakdown=breakdown,
            )

    # ---- live path --------------------------------------------------
    rooms = req.rooms if req.rooms is not None else max(1.0, round(req.sqft / 35.0))
    tab_raw = _build_tab_row(
        sqft=req.sqft,
        rooms=rooms,
        bathrooms=req.bathrooms,
        zone=req.zone,
        num_images=req.num_images,
        floor_num=req.floor_num or 0,
        elevator=req.elevator, ac=req.ac, terrace=req.terrace,
        furnished=req.furnished, heating=req.heating, exterior=req.exterior,
        parking=req.parking, storage=req.storage,
    )
    tab_scaled = state["tab_scaler"].transform(tab_raw)
    features_used = ["tabular"]

    has_text = bool(req.description and req.description.strip())
    has_images = bool(req.image_urls) and req.mode == "full"

    txt_pca_vec: Optional[np.ndarray] = None
    img_pca_vec: Optional[np.ndarray] = None
    n_photos_used = 0

    if has_text:
        txt_emb = state["text_model"].encode([req.description])
        txt_scaled = state["text_scaler"].transform(txt_emb)
        txt_pca_vec = state["pca_text"].transform(txt_scaled)
        features_used.append("text")

    per_photo_impact: Optional[list[PhotoImpact]] = None
    if has_images:
        # Parallel download: no photo cap. Concurrent I/O keeps wall-time
        # bounded even for 30+ photo listings.
        from concurrent.futures import ThreadPoolExecutor
        urls = req.image_urls or []
        kept_urls: list[str] = []
        pil_images: list[Image.Image] = []
        if urls:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for url, img in zip(urls, pool.map(_download_image, urls)):
                    if img is not None:
                        pil_images.append(img)
                        kept_urls.append(url)
        if pil_images:
            emb = _embed_photos(pil_images).reshape(1, -1)
            emb_scaled = state["img_scaler_ft"].transform(emb)
            img_pca_vec = state["pca_ft"].transform(emb_scaled)
            n_photos_used = len(pil_images)
            features_used.append(f"images:{n_photos_used}")

            # Score each image individually so we can flag helps-vs-hurts
            # per-photo. Same ResNet head as the precompute uses.
            scores = _score_photos(pil_images)
            if scores:
                listing_mean = float(np.mean(scores))
                sorted_idx = sorted(range(len(scores)), key=lambda i: -scores[i])
                rank_map = {i: r + 1 for r, i in enumerate(sorted_idx)}
                # Help threshold: ~5% of the listing mean, with a floor of
                # €40 so we don't chatter on tiny deltas.
                threshold = max(40.0, 0.05 * listing_mean)
                per_photo_impact = []
                for i, (url, s) in enumerate(zip(kept_urls, scores)):
                    delta = float(s) - listing_mean
                    if delta > threshold:
                        tone = "helps"
                    elif delta < -threshold:
                        tone = "hurts"
                    else:
                        tone = "neutral"
                    per_photo_impact.append(
                        PhotoImpact(
                            image_url=url,
                            score_eur=float(s),
                            rank_in_listing=rank_map[i],
                            delta_vs_listing_mean_eur=delta,
                            tone=tone,
                        )
                    )

    # Choose the model to report as "the" predicted rent
    if txt_pca_vec is not None and img_pca_vec is not None:
        combined = np.hstack([tab_scaled, txt_pca_vec, img_pca_vec])
        pred_log = float(state["gb_all"].predict(combined)[0])
    elif txt_pca_vec is not None:
        combined = np.hstack([tab_scaled, txt_pca_vec])
        pred_log = float(state["gb_tabular_text"].predict(combined)[0])
    elif img_pca_vec is not None:
        combined = np.hstack([tab_scaled, img_pca_vec])
        pred_log = float(state["gb_tabular_photos"].predict(combined)[0])
    else:
        pred_log = float(state["gb_tab"].predict(tab_scaled)[0])

    pred = float(np.expm1(pred_log))

    # Build breakdown (runs every available ablation model for display).
    # We pass surface-level context (similarity %, photo count) and let the
    # breakdown helper attach notes AFTER it sees the actual deltas: so the
    # copy always agrees with the direction of the model's contribution.
    text_sim_pct = None
    if has_text and txt_pca_vec is not None:
        txt_raw_vec = state["text_model"].encode([req.description])[0]
        norm = np.linalg.norm(txt_raw_vec) + 1e-9
        sim = float(np.dot(txt_raw_vec / norm, state["premium_centroid"]))
        text_sim_pct = max(0.0, sim) * 100

    breakdown = _compute_breakdown(
        tab_scaled,
        txt_pca_vec,
        img_pca_vec,
        tabular_note=_tabular_note_from_req(req),
        text_sim_pct=text_sim_pct,
        photos_count=n_photos_used if n_photos_used > 0 else None,
        photos_pct_global=None,  # not applicable to live listings outside our dataset
    )

    return PredictLiveResponse(
        predicted_rent_eur=pred,
        mae_eur=MODEL_MAE_EUR,
        features_used=features_used,
        cached=False,
        zone_median_rent_eur=zone_median,
        diagnosis=_diagnose(pred, req.current_rent_eur),
        current_rent_eur=req.current_rent_eur,
        delta_vs_current_eur=(pred - req.current_rent_eur) if req.current_rent_eur is not None else None,
        breakdown=breakdown,
        per_photo_impact=per_photo_impact,
    )


def _parse_listing_id(raw: str) -> str:
    s = raw.strip().rstrip("/")
    if "/" in s:
        s = s.split("/")[-1]
    return s


@app.post("/intake", response_model=IntakeResponse)
async def intake(
    listing_id: str = Form(...),
    images: Optional[list[UploadFile]] = File(None),
):
    """Take an existing listing ID and optional extra photos.
    Returns baseline state; if extras given, returns a what-if prediction
    pooled with existing photos.
    """
    lid = _parse_listing_id(listing_id)
    df = state["df"]
    match = df[df["listing_id"] == lid]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Listing {lid} not found")
    row = match.iloc[0]

    listing_dir = IMAGES_DIR / lid
    existing_files = []
    if listing_dir.exists():
        existing_files = sorted(listing_dir.glob("*.jpg")) + sorted(listing_dir.glob("*.webp"))

    baseline = IntakeBaseline(
        listing_id=lid,
        title=row.get("title"),
        zone=row["zone"],
        sqft_m2=float(row["sqft_m2"]),
        rooms=float(row["rooms"]) if pd.notna(row["rooms"]) else None,
        current_rent_eur=float(row["rent_eur"]),
        peer_expected_rent_full_eur=float(row["predicted_rent_full_eur"]) if pd.notna(row["predicted_rent_full_eur"]) else None,
        peer_expected_rent_tabular_eur=float(row["predicted_rent_tabular_eur"]),
        image_score_eur=float(row["image_score_eur"]) if pd.notna(row["image_score_eur"]) else None,
        image_score_percentile=float(row["image_score_percentile"]) if pd.notna(row["image_score_percentile"]) else None,
        num_existing_photos=len(existing_files),
        thumbnail_url=_thumbnail_url(lid),
    )

    pil_extras: list[Image.Image] = []
    extra_filenames: list[str] = []
    for up in images or []:
        try:
            data = await up.read()
            if not data:
                continue
            pil_extras.append(Image.open(io.BytesIO(data)))
            extra_filenames.append(up.filename or f"upload_{len(extra_filenames)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad image {up.filename}: {e}")

    if not pil_extras:
        return IntakeResponse(baseline=baseline, with_extras=None)

    # score extras individually (rank display; NOT presented as € to the user)
    extra_scores = _score_photos(pil_extras)

    # --- REPLACE-WORST-N semantics ---
    # Use precomputed per-photo scores from the parquet to identify weakest originals.
    # Swap out N weakest (where N = number of uploads), then mean-pool (kept + extras).
    stored_filenames = list(row["photo_filenames"]) if row["photo_filenames"] is not None else []
    stored_scores = list(row["photo_scores_eur"]) if row["photo_scores_eur"] is not None else []

    # fall back to filesystem scan if parquet missing scores
    if not stored_filenames and existing_files:
        stored_filenames = [f.name for f in existing_files]
        stored_scores = [0.0] * len(stored_filenames)

    n_uploads = len(pil_extras)
    n_replace = min(n_uploads, len(stored_filenames))

    if n_replace > 0:
        ranked = sorted(
            zip(stored_filenames, stored_scores),
            key=lambda x: x[1],  # ascending: weakest first
        )
        replaced = ranked[:n_replace]  # weakest N to drop
        kept = ranked[n_replace:]
        replaced_filenames = [f for f, _ in replaced]
        kept_filenames = [f for f, _ in kept]
    else:
        replaced_filenames = []
        kept_filenames = stored_filenames

    # Load tensors for kept originals + uploaded extras
    kept_tensors: list[torch.Tensor] = []
    for fname in kept_filenames:
        path = listing_dir / fname
        try:
            img = Image.open(path).convert("RGB")
            kept_tensors.append(TRANSFORM(img))
        except Exception:
            continue
    extra_tensors = [TRANSFORM(img.convert("RGB")) for img in pil_extras]
    all_tensors = kept_tensors + extra_tensors

    if not all_tensors:
        raise HTTPException(status_code=500, detail="No usable images")

    device = state["device"]
    chunk = 32
    embs: list[np.ndarray] = []
    for i in range(0, len(all_tensors), chunk):
        batch = torch.stack(all_tensors[i:i + chunk]).to(device)
        with torch.no_grad():
            e = state["resnet"].extract_embedding(batch).cpu().numpy()
        embs.append(e)
    all_embs = np.concatenate(embs, axis=0)
    mean_emb = all_embs.mean(axis=0).reshape(1, -1)

    ft_scaled = state["img_scaler_ft"].transform(mean_emb)
    ft_pca_vec = state["pca_ft"].transform(ft_scaled)

    tab_vec = state["tab_scaled_by_id"].get(lid)
    if tab_vec is None:
        raise HTTPException(status_code=500, detail="Tabular features missing for this listing")
    tab_vec = tab_vec.reshape(1, -1)

    if lid in state["text_lookup"]:
        txt_pca_vec = state["text_pca_all"][state["text_lookup"][lid]].reshape(1, -1)
    else:
        txt_pca_vec = np.zeros((1, state["text_pca_all"].shape[1]), dtype=np.float32)

    combined = np.hstack([tab_vec, txt_pca_vec, ft_pca_vec])
    pred_log = float(state["gb_all"].predict(combined)[0])
    pred_eur = float(np.expm1(pred_log))

    # per-extra photo score objects (ranked among uploads; UI converts to rank label)
    ranked_idx = sorted(range(len(extra_scores)), key=lambda i: -extra_scores[i])
    rank_of = {i: r + 1 for r, i in enumerate(ranked_idx)}
    per_extra = [
        PhotoScore(
            image_url=f"upload/{extra_filenames[i]}",
            score_eur=float(extra_scores[i]),
            rank_in_listing=rank_of[i],
        )
        for i in range(len(extra_scores))
    ]

    replaced_urls = [f"/images/{lid}/{f}" for f in replaced_filenames]

    delta_vs_current = pred_eur - baseline.current_rent_eur
    delta_vs_prev = (
        pred_eur - baseline.peer_expected_rent_full_eur
        if baseline.peer_expected_rent_full_eur is not None
        else None
    )

    # suggestions: framed around days-on-market, not direct rent lift
    suggestions: list[str] = []
    if n_replace > 0:
        avg_replaced = float(np.mean([s for _, s in ranked[:n_replace]])) if ranked else 0.0
        avg_new = float(np.mean(extra_scores)) if extra_scores else 0.0
        if avg_new > avg_replaced + 100:
            suggestions.append(
                f"New photos score above the weakest existing set: swapping them in is the stronger choice."
            )
        elif avg_new < avg_replaced - 100:
            suggestions.append(
                "New photos score below the originals being replaced: consider keeping current set."
            )
        else:
            suggestions.append(
                "New photos score near the replaced originals: substitution looks neutral."
            )
        suggestions.append(
            f"Swapping the {n_replace} weakest original photo{'s' if n_replace != 1 else ''} "
            "could reduce days-on-market and help defend the asking price."
        )
    if delta_vs_prev is not None and abs(delta_vs_prev) > MODEL_MAE_EUR:
        direction = "lifts" if delta_vs_prev > 0 else "lowers"
        suggestions.append(
            f"This substitution {direction} the model prediction by "
            f"{formatEur_str(abs(delta_vs_prev))}/mo: outside the model's ±{int(MODEL_MAE_EUR)} MAE band."
        )
    else:
        suggestions.append(
            f"Predicted change is inside the model's ±{int(MODEL_MAE_EUR)} MAE band: "
            "treat as noise, not a reliable signal."
        )

    with_extras = IntakeWithExtras(
        predicted_rent_eur=pred_eur,
        predicted_rent_mae_eur=MODEL_MAE_EUR,
        delta_vs_current_rent_eur=delta_vs_current,
        delta_vs_previous_model_eur=delta_vs_prev,
        per_extra_scores=per_extra,
        replaced_photo_urls=replaced_urls,
        kept_photo_count=len(kept_filenames),
        total_photos_considered=len(all_tensors),
        suggestions=suggestions,
    )
    return IntakeResponse(baseline=baseline, with_extras=with_extras)


def formatEur_str(v: float) -> str:
    return f"EUR {int(round(v)):,}"


@app.get("/listings", response_model=list[ScoutItem])
def listings_batch(ids: str):
    """Batch fetch: ?ids=a,b,c -> list of ScoutItem in request order."""
    wanted = [i.strip() for i in ids.split(",") if i.strip()]
    if not wanted:
        return []
    df = state["df"]
    lookup = {r["listing_id"]: r for _, r in df[df["listing_id"].isin(wanted)].iterrows()}
    out = []
    for lid in wanted:
        r = lookup.get(lid)
        if r is None:
            continue
        out.append(ScoutItem(
            listing_id=r["listing_id"],
            url=r["url"],
            title=r.get("title"),
            zone=r["zone"],
            rent_eur=float(r["rent_eur"]),
            sqft_m2=float(r["sqft_m2"]),
            rooms=float(r["rooms"]) if pd.notna(r["rooms"]) else None,
            predicted_rent_tabular_eur=float(r["predicted_rent_tabular_eur"]),
            image_score_eur=float(r["image_score_eur"]) if pd.notna(r["image_score_eur"]) else None,
            image_score_percentile=float(r["image_score_percentile"]) if pd.notna(r["image_score_percentile"]) else None,
            rent_gap_pct=float(r["rent_gap_pct"]),
            under_marketing_score=float(r["under_marketing_score"]),
            thumbnail_url=_thumbnail_url(r["listing_id"]),
        ))
    return out


@app.get("/scout", response_model=list[ScoutItem])
def scout(
    zone: Optional[str] = None,
    min_sqft: Optional[float] = None,
    max_sqft: Optional[float] = None,
    min_rent: Optional[float] = None,
    max_rent: Optional[float] = None,
    limit: int = 30,
    sort: str = "under_marketing",
):
    df = state["df"]
    mask = pd.Series(True, index=df.index)
    if zone:
        mask &= df["zone"] == zone
    if min_sqft is not None:
        mask &= df["sqft_m2"] >= min_sqft
    if max_sqft is not None:
        mask &= df["sqft_m2"] <= max_sqft
    if min_rent is not None:
        mask &= df["rent_eur"] >= min_rent
    if max_rent is not None:
        mask &= df["rent_eur"] <= max_rent
    sub = df[mask]

    if sort == "under_marketing":
        sub = sub.sort_values("under_marketing_score", ascending=False)
    elif sort == "rent_gap":
        sub = sub.sort_values("rent_gap_pct", ascending=False)
    elif sort == "rent_asc":
        sub = sub.sort_values("rent_eur", ascending=True)
    elif sort == "rent_desc":
        sub = sub.sort_values("rent_eur", ascending=False)

    sub = sub.head(limit)
    out = []
    for _, r in sub.iterrows():
        out.append(ScoutItem(
            listing_id=r["listing_id"],
            url=r["url"],
            title=r.get("title"),
            zone=r["zone"],
            rent_eur=float(r["rent_eur"]),
            sqft_m2=float(r["sqft_m2"]),
            rooms=float(r["rooms"]) if pd.notna(r["rooms"]) else None,
            predicted_rent_tabular_eur=float(r["predicted_rent_tabular_eur"]),
            image_score_eur=float(r["image_score_eur"]) if pd.notna(r["image_score_eur"]) else None,
            image_score_percentile=float(r["image_score_percentile"]) if pd.notna(r["image_score_percentile"]) else None,
            rent_gap_pct=float(r["rent_gap_pct"]),
            under_marketing_score=float(r["under_marketing_score"]),
            thumbnail_url=_thumbnail_url(r["listing_id"]),
        ))
    return out


@app.get("/listings/{listing_id}", response_model=ListingDetail)
def listing_detail(listing_id: str):
    df = state["df"]
    row = df[df["listing_id"] == listing_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Listing not found")
    row = row.iloc[0]
    photos = _photos_from_row(row)

    # Feature-by-feature breakdown using cached per-listing embeddings.
    breakdown = None
    lid = row["listing_id"]
    tab_vec = state["tab_scaled_by_id"].get(lid)
    if tab_vec is not None:
        tab_vec = tab_vec.reshape(1, -1)
        txt_pca = (
            state["text_pca_all"][state["text_lookup"][lid]].reshape(1, -1)
            if lid in state["text_lookup"]
            else None
        )
        img_pca = (
            state["ft_pca_all"][state["ft_lookup"][lid]].reshape(1, -1)
            if lid in state["ft_lookup"]
            else None
        )
        text_sim_pct = None
        if row.get("text_distance_premium") is not None and not pd.isna(row.get("text_distance_premium")):
            text_sim_pct = max(0.0, 1.0 - float(row["text_distance_premium"])) * 100
        photos_count = int(row["num_images"]) if pd.notna(row.get("num_images")) else None
        photos_pct_global = (
            float(row["image_score_percentile"])
            if pd.notna(row.get("image_score_percentile"))
            else None
        )
        breakdown = _compute_breakdown(
            tab_vec, txt_pca, img_pca,
            tabular_note=_tabular_note_from_row(row),
            text_sim_pct=text_sim_pct,
            photos_count=photos_count,
            photos_pct_global=photos_pct_global,
        )

    return ListingDetail(
        listing_id=row["listing_id"],
        url=row["url"],
        title=row.get("title"),
        location=row.get("location"),
        zone=row["zone"],
        rent_eur=float(row["rent_eur"]),
        sqft_m2=float(row["sqft_m2"]),
        rooms=float(row["rooms"]) if pd.notna(row["rooms"]) else None,
        bathrooms=float(row["bathrooms"]) if pd.notna(row["bathrooms"]) else None,
        description=row.get("description"),
        predicted_rent_tabular_eur=float(row["predicted_rent_tabular_eur"]),
        predicted_rent_full_eur=float(row["predicted_rent_full_eur"]) if pd.notna(row["predicted_rent_full_eur"]) else None,
        zone_median_rent_eur=float(row["zone_median_rent_eur"]),
        image_score_eur=float(row["image_score_eur"]) if pd.notna(row["image_score_eur"]) else None,
        image_score_percentile=float(row["image_score_percentile"]) if pd.notna(row["image_score_percentile"]) else None,
        text_distance_premium=float(row["text_distance_premium"]) if pd.notna(row["text_distance_premium"]) else None,
        photos=photos,
        diagnosis=_build_diagnosis(row, photos),
        breakdown=breakdown,
        mae_eur=MODEL_MAE_EUR,
    )


@app.post("/simulate", response_model=SimulateResponse)
async def simulate(
    images: list[UploadFile] = File(...),
    sqft: float = Form(...),
    rooms: float = Form(...),
    zone: str = Form(...),
    bathrooms: Optional[float] = Form(None),
    description: Optional[str] = Form(None),
    baseline_listing_id: Optional[str] = Form(None),
):
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")

    pil_images = []
    for up in images:
        try:
            pil_images.append(Image.open(io.BytesIO(await up.read())))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad image {up.filename}: {e}")

    photo_prices = _score_photos(pil_images)
    ranked = sorted(range(len(photo_prices)), key=lambda i: -photo_prices[i])
    rank_map = {i: r + 1 for r, i in enumerate(ranked)}
    photo_scores = [
        PhotoScore(image_url=f"upload/{images[i].filename}", score_eur=photo_prices[i],
                   rank_in_listing=rank_map[i])
        for i in range(len(photo_prices))
    ]

    # mean-pooled fine-tuned embedding across uploads
    img_mean_emb = _embed_photos(pil_images).reshape(1, -1)
    ft_scaled = state["img_scaler_ft"].transform(img_mean_emb)
    ft_pca_vec = state["pca_ft"].transform(ft_scaled)

    # description handling
    desc_dist = None
    if description and description.strip():
        txt_emb = state["text_model"].encode([description])
        v = txt_emb[0] / (np.linalg.norm(txt_emb[0]) + 1e-9)
        desc_dist = float(1.0 - np.dot(v, state["premium_centroid"]))
        txt_scaled = state["text_scaler"].transform(txt_emb)
        txt_pca_vec = state["pca_text"].transform(txt_scaled)
    else:
        # fallback: use premium centroid as neutral prior-ish: predict using empty text
        txt_scaled = state["text_scaler"].transform(np.zeros((1, 384), dtype=np.float32))
        txt_pca_vec = state["pca_text"].transform(txt_scaled)

    # tabular
    tab_raw = _build_tab_row(
        sqft=sqft, rooms=rooms, bathrooms=bathrooms, zone=zone,
        num_images=len(pil_images),
    )
    tab_scaled = state["tab_scaler"].transform(tab_raw)

    combined = np.hstack([tab_scaled, txt_pca_vec, ft_pca_vec])
    pred_log = float(state["gb_all"].predict(combined)[0])
    pred_eur = float(np.expm1(pred_log))

    # baseline comparison
    delta = None
    baseline_rent = None
    if baseline_listing_id:
        df = state["df"]
        base_row = df[df["listing_id"] == baseline_listing_id]
        if not base_row.empty:
            baseline_rent = float(base_row.iloc[0]["rent_eur"])
            delta = pred_eur - baseline_rent

    # suggestions
    suggestions = []
    if len(photo_prices) >= 2:
        weak_idx = ranked[-1]
        strong_idx = ranked[0]
        spread = photo_prices[strong_idx] - photo_prices[weak_idx]
        if spread > 300:
            suggestions.append(
                f"Photo '{images[weak_idx].filename}' scores EUR {photo_prices[weak_idx]:.0f} "
                f"vs your best photo at EUR {photo_prices[strong_idx]:.0f}. Consider removing or re-shooting."
            )
    if desc_dist is not None and desc_dist > 0.35:
        suggestions.append(
            "Your description reads differently from premium listings. "
            "Mention renovations, views, natural light, or finishes."
        )
    if not description or len(description.strip()) < 50:
        suggestions.append("Add a longer description: most high-rent listings have 200+ word descriptions.")
    if not suggestions:
        suggestions.append("Photos and description look solid: this listing is well-presented.")

    return SimulateResponse(
        predicted_rent_eur=pred_eur,
        per_photo_scores=photo_scores,
        description_distance_premium=desc_dist,
        delta_vs_baseline_eur=delta,
        baseline_rent_eur=baseline_rent,
        suggestions=suggestions,
    )
