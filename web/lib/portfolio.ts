"use client";

import { useCallback, useEffect, useState } from "react";
import { fetchScout, ScoutItem } from "@/lib/api";

const KEY_IDS = "casaintel.portfolio";
const KEY_REALIZATION = "casaintel.realization";
const COMMISSION_RATE = 0.06;
const DEFAULT_REALIZATION = 0.25; // agency captures ~25% of modeled gap, per reality check

export const COMMISSION_PCT = COMMISSION_RATE;
export const DEFAULT_REALIZATION_RATE = DEFAULT_REALIZATION;

function readIds(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(KEY_IDS);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}

function writeIds(ids: string[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(KEY_IDS, JSON.stringify(ids));
}

function readRealization(): number {
  if (typeof window === "undefined") return DEFAULT_REALIZATION;
  const raw = window.localStorage.getItem(KEY_REALIZATION);
  if (!raw) return DEFAULT_REALIZATION;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed >= 0 && parsed <= 1 ? parsed : DEFAULT_REALIZATION;
}

function writeRealization(v: number) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(KEY_REALIZATION, String(v));
}

async function seedMix(): Promise<string[]> {
  // Fetch healthy + flagged separately so the demo portfolio looks like a realistic
  // agency book (mostly healthy, a handful of re-shoot candidates).
  const [flagged, healthy] = await Promise.all([
    fetchScout({ sort: "under_marketing", limit: 6 }),
    // "rent_desc" pulls bigger/pricier listings which tend to be fairly priced
    fetchScout({ sort: "rent_desc", limit: 12 }),
  ]);
  const flaggedIds = new Set(flagged.map((i) => i.listing_id));
  const healthyPick = healthy
    .filter((i) => !flaggedIds.has(i.listing_id) && i.rent_gap_pct < 0.05)
    .slice(0, 6);
  return [...flagged.map((i) => i.listing_id), ...healthyPick.map((i) => i.listing_id)];
}

export function usePortfolio() {
  const [ids, setIds] = useState<string[]>([]);
  const [ready, setReady] = useState(false);
  const [realization, setRealizationState] = useState<number>(DEFAULT_REALIZATION);

  useEffect(() => {
    setRealizationState(readRealization());
    const existing = readIds();
    if (existing.length > 0) {
      setIds(existing);
      setReady(true);
      return;
    }
    let cancelled = false;
    seedMix()
      .then((seeded) => {
        if (cancelled) return;
        setIds(seeded);
        writeIds(seeded);
      })
      .catch(() => {
        if (!cancelled) setIds([]);
      })
      .finally(() => {
        if (!cancelled) setReady(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const add = useCallback((id: string) => {
    setIds((cur) => {
      if (cur.includes(id)) return cur;
      const next = [...cur, id];
      writeIds(next);
      return next;
    });
  }, []);

  const remove = useCallback((id: string) => {
    setIds((cur) => {
      const next = cur.filter((x) => x !== id);
      writeIds(next);
      return next;
    });
  }, []);

  const seed = useCallback(async () => {
    try {
      const seeded = await seedMix();
      setIds(seeded);
      writeIds(seeded);
    } catch {
      // swallow
    }
  }, []);

  const reset = useCallback(() => {
    writeIds([]);
    setIds([]);
  }, []);

  const setRealization = useCallback((v: number) => {
    const clamped = Math.min(1, Math.max(0, v));
    setRealizationState(clamped);
    writeRealization(clamped);
  }, []);

  const contains = useCallback((id: string) => ids.includes(id), [ids]);

  return {
    ids,
    ready,
    add,
    remove,
    reset,
    seed,
    contains,
    realization,
    setRealization,
  };
}

export type PortfolioStats = {
  managedCount: number;
  managedRent: number;
  commissionMonthly: number;
  commissionYearly: number;
  gapMonthly: number; // raw model-vs-actual gap (not realization-adjusted)
  commissionUpsideYearly: number; // gap × commission × 12 × realization
  flaggedCount: number;
};

export function computeStats(items: ScoutItem[], realization: number): PortfolioStats {
  const managedCount = items.length;
  const managedRent = items.reduce((s, i) => s + i.rent_eur, 0);
  const commissionMonthly = managedRent * COMMISSION_RATE;
  const commissionYearly = commissionMonthly * 12;

  const gapMonthly = items.reduce((s, i) => {
    const delta = i.predicted_rent_tabular_eur - i.rent_eur;
    return s + Math.max(0, delta);
  }, 0);
  const commissionUpsideYearly = gapMonthly * COMMISSION_RATE * 12 * realization;

  const flaggedCount = items.filter((i) => {
    const gap = i.rent_gap_pct;
    const photosLow = (i.image_score_percentile ?? 100) < 40;
    return gap > 0.08 || photosLow;
  }).length;

  return {
    managedCount,
    managedRent,
    commissionMonthly,
    commissionYearly,
    gapMonthly,
    commissionUpsideYearly,
    flaggedCount,
  };
}

export function listingStatus(item: ScoutItem): {
  label: string;
  tone: "good" | "warn" | "bad" | "neutral";
} {
  const gap = item.rent_gap_pct;
  const photosPct = item.image_score_percentile ?? 50;
  if (gap > 0.15 && photosPct < 35)
    return { label: "Triage — photos + price", tone: "bad" };
  if (gap > 0.1) return { label: "Below peer average", tone: "warn" };
  if (photosPct < 30) return { label: "Photos weak", tone: "warn" };
  if (gap < -0.05) return { label: "Above peer average", tone: "neutral" };
  return { label: "On peer average", tone: "good" };
}
