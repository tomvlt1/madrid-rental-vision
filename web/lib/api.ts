export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type ScoutItem = {
  listing_id: string;
  url: string;
  title?: string | null;
  zone: string;
  rent_eur: number;
  sqft_m2: number;
  rooms?: number | null;
  predicted_rent_tabular_eur: number;
  image_score_eur?: number | null;
  image_score_percentile?: number | null;
  rent_gap_pct: number;
  under_marketing_score: number;
  thumbnail_url?: string | null;
};

export type PhotoScore = {
  image_url: string;
  score_eur: number;
  rank_in_listing: number;
};

export type Diagnosis = {
  weakest_photo?: PhotoScore | null;
  strongest_photo?: PhotoScore | null;
  is_under_marketed: boolean;
  peer_rent_gap_eur?: number | null;
  verdict: string;
};

export type FeatureBreakdown = {
  tabular_eur: number;
  with_text_eur?: number | null;
  with_photos_eur?: number | null;
  full_eur?: number | null;
  text_delta_eur?: number | null;
  photos_delta_eur?: number | null;
  interaction_eur?: number | null;
  tabular_note?: string | null;
  text_note?: string | null;
  photos_note?: string | null;
};

export type ListingDetail = {
  listing_id: string;
  url: string;
  title?: string | null;
  location?: string | null;
  zone: string;
  rent_eur: number;
  sqft_m2: number;
  rooms?: number | null;
  bathrooms?: number | null;
  description?: string | null;
  predicted_rent_tabular_eur: number;
  predicted_rent_full_eur?: number | null;
  zone_median_rent_eur: number;
  image_score_eur?: number | null;
  image_score_percentile?: number | null;
  text_distance_premium?: number | null;
  photos: PhotoScore[];
  diagnosis: Diagnosis;
  breakdown?: FeatureBreakdown | null;
  mae_eur?: number;
};

export type SimulateResponse = {
  predicted_rent_eur: number;
  per_photo_scores: PhotoScore[];
  description_distance_premium?: number | null;
  delta_vs_baseline_eur?: number | null;
  baseline_rent_eur?: number | null;
  suggestions: string[];
};

export type IntakeBaseline = {
  listing_id: string;
  title?: string | null;
  zone: string;
  sqft_m2: number;
  rooms?: number | null;
  current_rent_eur: number;
  peer_expected_rent_full_eur?: number | null;
  peer_expected_rent_tabular_eur: number;
  image_score_eur?: number | null;
  image_score_percentile?: number | null;
  num_existing_photos: number;
  thumbnail_url?: string | null;
};

export type IntakeWithExtras = {
  predicted_rent_eur: number;
  predicted_rent_mae_eur: number;
  delta_vs_current_rent_eur: number;
  delta_vs_previous_model_eur?: number | null;
  per_extra_scores: PhotoScore[];
  replaced_photo_urls: string[];
  kept_photo_count: number;
  total_photos_considered: number;
  suggestions: string[];
};

export const MODEL_MAE_EUR = 457;

export type IntakeResponse = {
  baseline: IntakeBaseline;
  with_extras?: IntakeWithExtras | null;
};

export const ZONES = [
  "Arganzuela",
  "Centro",
  "Chamberí",
  "Norte",
  "Oeste",
  "Periferia Norte",
  "Salamanca-Retiro",
  "Sur-Sureste",
];

export function imageUrl(path: string | null | undefined): string | undefined {
  if (!path) return undefined;
  if (path.startsWith("http")) return path;
  return `${API_URL}${path}`;
}

export function formatEur(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  const abs = Math.abs(value);
  const formatted = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
  }).format(abs);
  return value < 0 ? "−" + formatted : formatted;
}

export type ScoutFilters = {
  zone?: string;
  min_sqft?: number;
  max_sqft?: number;
  min_rent?: number;
  max_rent?: number;
  limit?: number;
  sort?: string;
};

export async function fetchListingsByIds(ids: string[]): Promise<ScoutItem[]> {
  if (ids.length === 0) return [];
  const params = new URLSearchParams({ ids: ids.join(",") });
  const res = await fetch(`${API_URL}/listings?${params}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`listings ${res.status}`);
  return res.json();
}

export async function fetchScout(filters: ScoutFilters): Promise<ScoutItem[]> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([k, v]) => {
    if (v !== undefined && v !== "" && v !== null) params.set(k, String(v));
  });
  const res = await fetch(`${API_URL}/scout?${params}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`scout ${res.status}`);
  return res.json();
}

export async function fetchListing(id: string): Promise<ListingDetail> {
  const res = await fetch(`${API_URL}/listings/${id}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`listing ${res.status}`);
  return res.json();
}

export async function simulate(form: FormData): Promise<SimulateResponse> {
  const res = await fetch(`${API_URL}/simulate`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `simulate ${res.status}`);
  }
  return res.json();
}

export async function intake(form: FormData): Promise<IntakeResponse> {
  const res = await fetch(`${API_URL}/intake`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `intake ${res.status}`);
  }
  return res.json();
}

export function parseListingId(raw: string): string {
  const trimmed = raw.trim().replace(/\/$/, "");
  if (!trimmed) return "";
  if (trimmed.includes("/")) return trimmed.split("/").pop() ?? trimmed;
  return trimmed;
}
