"use client";

import Link from "next/link";
import { useEffect, useState, use } from "react";
import {
  fetchListing,
  FeatureBreakdown,
  formatEur,
  imageUrl,
  ListingDetail,
  MODEL_MAE_EUR,
  PhotoScore,
} from "@/lib/api";
import { COMMISSION_PCT, usePortfolio } from "@/lib/portfolio";

export default function ListingPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const portfolio = usePortfolio();
  const [data, setData] = useState<ListingDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchListing(id)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <p className="text-slate-500">Loading…</p>;
  if (error) return <p className="text-red-700">Failed: {error}</p>;
  if (!data) return null;

  const { diagnosis } = data;
  const gap = data.predicted_rent_tabular_eur - data.rent_eur;
  const weakPhotos = [...data.photos]
    .sort((a, b) => a.score_eur - b.score_eur)
    .slice(0, 3);
  const inPortfolio = portfolio.contains(id);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-4 text-sm">
          <Link href="/" className="text-teal-700 hover:underline">
            ← Portfolio
          </Link>
          <a
            href={data.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-500 hover:text-slate-700"
          >
            View source ↗
          </a>
        </div>
        {portfolio.ready && (
          <button
            onClick={() => (inPortfolio ? portfolio.remove(id) : portfolio.add(id))}
            className={`rounded-md border px-3 py-1.5 text-sm font-medium ${
              inPortfolio
                ? "border-red-200 bg-white text-red-700 hover:bg-red-50"
                : "border-teal-700 bg-teal-700 text-white hover:bg-teal-800"
            }`}
          >
            {inPortfolio ? "Remove from portfolio" : "+ Add to portfolio"}
          </button>
        )}
      </div>

      <header className="flex items-start justify-between gap-6 flex-wrap">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
            {data.title ?? data.listing_id}
          </h1>
          <p className="mt-1 text-slate-600">
            {data.location ?? data.zone} · {data.sqft_m2.toFixed(0)}m²
            {data.rooms ? ` · ${data.rooms.toFixed(0)} rooms` : ""}
            {data.bathrooms ? ` · ${data.bathrooms.toFixed(0)} bath` : ""}
          </p>
        </div>
        <div className="text-right">
          <div className="text-3xl font-semibold text-slate-900">
            {formatEur(data.rent_eur)}
          </div>
          <div className="text-sm text-slate-500">current monthly rent</div>
        </div>
      </header>

      <ActionPlan
        data={data}
        gap={gap}
        realization={portfolio.realization}
        weakPhotos={weakPhotos}
      />

      {data.breakdown && <BreakdownCard breakdown={data.breakdown} mae={data.mae_eur ?? MODEL_MAE_EUR} />}

      <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Stat
          label="Peer expected rent"
          value={formatEur(data.predicted_rent_tabular_eur)}
          hint={`tabular model · ±${formatEur(MODEL_MAE_EUR)} MAE`}
        />
        <Stat
          label="Peer expected (with photos + text)"
          value={formatEur(data.predicted_rent_full_eur)}
          hint="full model — not a price ceiling"
        />
        <Stat
          label={`Zone median · ${data.zone}`}
          value={formatEur(data.zone_median_rent_eur)}
          hint="across all zone listings"
        />
      </section>

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900">
            Photo scorecard ({data.photos.length})
          </h2>
          <p className="text-xs text-slate-500">
            Photos ranked within this listing — the model scores each image and we
            show its relative strength. Not individual rent values.
          </p>
        </div>
        <PhotoGrid
          photos={data.photos}
          weakestUrl={diagnosis.weakest_photo?.image_url}
          strongestUrl={diagnosis.strongest_photo?.image_url}
        />
      </section>

      {data.description && (
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-lg font-semibold text-slate-900">Listing copy</h2>
          {data.text_distance_premium != null && (
            <p className="mt-1 text-sm text-slate-500">
              Lexical similarity to top-quartile rent listings:{" "}
              <span className="font-medium text-slate-900">
                {(100 * (1 - Math.min(1, data.text_distance_premium))).toFixed(0)}%
              </span>{" "}
              {data.text_distance_premium > 0.35
                ? "— distinct vocabulary from premium listings"
                : "— shares vocabulary with premium listings"}
              {" "}
              <span className="text-xs text-slate-400">
                (measures word-overlap, not apartment quality)
              </span>
            </p>
          )}
          <p className="mt-3 text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">
            {data.description}
          </p>
        </section>
      )}
    </div>
  );
}

function ActionPlan({
  data,
  gap,
  realization,
  weakPhotos,
}: {
  data: ListingDetail;
  gap: number;
  realization: number;
  weakPhotos: PhotoScore[];
}) {
  const actions: string[] = [];
  const significant = gap > MODEL_MAE_EUR;
  if (gap > 150) {
    actions.push(
      significant
        ? `Rent sits ${formatEur(gap)}/mo below peer-expected — pricing review is justified (outside ±${formatEur(MODEL_MAE_EUR)} MAE).`
        : `Rent sits ${formatEur(gap)}/mo below peer-expected, but within model noise — treat as weak signal.`,
    );
  }
  // Photo re-shoot recommendation needs TWO signals to fire:
  //  (a) listing's overall photos rank low vs peers
  //  (b) intra-listing spread is wide enough that swapping the weakest would
  //      meaningfully shift the mean embedding. If all photos are uniformly
  //      mediocre, a re-shoot won't help.
  const photoSpread =
    data.photos.length >= 2
      ? Math.max(...data.photos.map((p) => p.score_eur)) -
        Math.min(...data.photos.map((p) => p.score_eur))
      : 0;
  const PHOTO_SPREAD_MIN = 250; // ~half of MAE — below this, all photos are similar
  const photoRankLow = (data.image_score_percentile ?? 100) < 35;
  if (photoRankLow && weakPhotos.length > 0 && photoSpread > PHOTO_SPREAD_MIN) {
    actions.push(
      `Re-shoot the ${weakPhotos.length} weakest photos to reduce days-on-market and defend the asking price.`,
    );
  } else if (photoRankLow && photoSpread <= PHOTO_SPREAD_MIN) {
    actions.push(
      "Listing photos rank low vs peers but are uniformly so — a single re-shoot won't shift the mean. Consider a full re-shoot.",
    );
  }
  if (data.text_distance_premium != null && data.text_distance_premium > 0.4) {
    actions.push(
      "Description is lexically distinct from top-quartile rent listings — vocabulary, not necessarily quality.",
    );
  }
  if (actions.length === 0) {
    actions.push("Listing is on peer average — no triage action recommended.");
  }

  const commissionUpside = gap > 0 ? gap * COMMISSION_PCT * 12 * realization : 0;
  const healthy = actions[0].startsWith("Listing is on peer average");

  return (
    <div
      className={
        healthy
          ? "rounded-lg border border-slate-200 bg-white p-5"
          : "rounded-lg border-2 border-teal-600 bg-teal-50 p-5"
      }
    >
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <div className="text-xs font-medium uppercase tracking-wide text-teal-800">
            Action plan
          </div>
          <h2 className="mt-1 text-lg font-semibold text-slate-900">
            {healthy
              ? "No fixes recommended."
              : `Peer-expected rent sits ${formatEur(Math.max(0, gap))}/mo above current — triage candidate.`}
          </h2>
          <ul className="mt-3 space-y-1.5 text-sm text-slate-700">
            {actions.map((a, i) => (
              <li key={i} className="flex gap-2">
                <span
                  aria-hidden
                  className="mt-1.5 h-1.5 w-1.5 rounded-full bg-teal-600 shrink-0"
                />
                <span>{a}</span>
              </li>
            ))}
          </ul>
        </div>
        {commissionUpside > 0 && (
          <div className="rounded-md bg-white border border-teal-200 px-4 py-2 text-right">
            <div className="text-xs font-medium text-teal-700 uppercase tracking-wide">
              Potential commission upside
            </div>
            <div className="text-2xl font-semibold text-teal-800">
              +{formatEur(commissionUpside)}
            </div>
            <div className="text-xs text-slate-500">
              / yr @ {Math.round(COMMISSION_PCT * 100)}% ·{" "}
              {Math.round(realization * 100)}% realization
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function BreakdownCard({
  breakdown,
  mae,
}: {
  breakdown: FeatureBreakdown;
  mae: number;
}) {
  const total = breakdown.full_eur ?? breakdown.tabular_eur;
  const interactionMeaningful =
    breakdown.full_eur != null &&
    breakdown.interaction_eur != null &&
    Math.abs(breakdown.interaction_eur) > 20;
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Why this number</h2>
          <p className="mt-0.5 text-sm text-slate-600">
            Each feature block&apos;s individual contribution, from four trained
            ablation models.
          </p>
        </div>
        <div className="text-xs text-slate-500">
          Model error: ± {formatEur(mae)}
        </div>
      </div>
      <div className="mt-4 divide-y divide-slate-100">
        <BreakdownRow
          label="Tabular baseline"
          sub={breakdown.tabular_note ?? "size, rooms, zone, amenities"}
          value={formatEur(breakdown.tabular_eur)}
        />
        {breakdown.photos_delta_eur != null && (
          <BreakdownRow
            label="+ photos"
            sub={breakdown.photos_note ?? "fine-tuned ResNet embedding"}
            delta={breakdown.photos_delta_eur}
          />
        )}
        {breakdown.text_delta_eur != null && (
          <BreakdownRow
            label="+ description"
            sub={breakdown.text_note ?? "multilingual sentence-transformer"}
            delta={breakdown.text_delta_eur}
          />
        )}
        {interactionMeaningful && (
          <BreakdownRow
            label="Combined effect"
            sub={
              (breakdown.interaction_eur ?? 0) > 0
                ? "photos + description together are less negative than the sum of parts"
                : "photos + description together are more negative than the sum of parts"
            }
            delta={breakdown.interaction_eur!}
          />
        )}
        <BreakdownRow
          label="Full model"
          sub="tabular + text + photos"
          value={formatEur(total) + " ± " + formatEur(mae)}
          bold
        />
      </div>
    </section>
  );
}

function BreakdownRow({
  label,
  sub,
  value,
  delta,
  bold,
}: {
  label: string;
  sub?: string;
  value?: string;
  delta?: number;
  bold?: boolean;
}) {
  const signedDelta =
    delta != null
      ? (delta > 0 ? "+" : "") + formatEur(delta)
      : value;
  const deltaCls =
    delta == null
      ? ""
      : delta > 0
      ? "text-emerald-700"
      : delta < 0
      ? "text-amber-700"
      : "text-slate-500";
  return (
    <div className={`flex items-start justify-between gap-4 py-2.5 ${bold ? "font-semibold" : ""}`}>
      <div className="min-w-0">
        <div className={`text-sm ${bold ? "text-slate-900" : "text-slate-800"}`}>
          {label}
        </div>
        {sub && <div className="text-xs text-slate-500">{sub}</div>}
      </div>
      <div
        className={`tabular-nums text-sm whitespace-nowrap ${bold ? "text-slate-900" : deltaCls || "text-slate-900"}`}
      >
        {signedDelta}
      </div>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="text-xs font-medium text-slate-500 uppercase tracking-wide">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold text-slate-900">{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

function PhotoGrid({
  photos,
  weakestUrl,
  strongestUrl,
}: {
  photos: PhotoScore[];
  weakestUrl?: string;
  strongestUrl?: string;
}) {
  if (photos.length === 0) {
    return <p className="text-sm text-slate-500">No photos for this listing.</p>;
  }
  const scores = photos.map((p) => p.score_eur);
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const range = max > min ? max - min : 1;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
      {photos.map((p) => {
        const isWeakest = p.image_url === weakestUrl;
        const isStrongest = p.image_url === strongestUrl;
        const ringClass = isStrongest
          ? "ring-2 ring-emerald-500"
          : isWeakest
          ? "ring-2 ring-amber-500"
          : "ring-1 ring-slate-200";
        const normalized = Math.round(((p.score_eur - min) / range) * 100);

        return (
          <div key={p.image_url} className="space-y-1.5">
            <div
              className={`relative aspect-[4/3] overflow-hidden rounded-md bg-slate-100 ${ringClass}`}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imageUrl(p.image_url)}
                alt={`Photo ${p.rank_in_listing}`}
                className="h-full w-full object-cover"
                loading="lazy"
              />
              <span className="absolute top-1.5 left-1.5 rounded bg-black/60 text-white text-xs px-1.5 py-0.5">
                #{p.rank_in_listing} of {photos.length}
              </span>
              {isStrongest && (
                <span className="absolute top-1.5 right-1.5 rounded bg-emerald-600 text-white text-[10px] font-medium px-1.5 py-0.5">
                  best
                </span>
              )}
              {isWeakest && (
                <span className="absolute top-1.5 right-1.5 rounded bg-amber-600 text-white text-[10px] font-medium px-1.5 py-0.5">
                  re-shoot
                </span>
              )}
            </div>
            <div className="flex items-center justify-between text-xs text-slate-600">
              <span>Strength {normalized}/100</span>
            </div>
            <div className="h-1 rounded-full bg-slate-100 overflow-hidden">
              <div
                className="h-full bg-teal-600"
                style={{ width: `${Math.max(3, normalized)}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
