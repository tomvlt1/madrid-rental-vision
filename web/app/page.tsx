"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  fetchListingsByIds,
  formatEur,
  imageUrl,
  ScoutItem,
  ZONES,
} from "@/lib/api";
import {
  COMMISSION_PCT,
  computeStats,
  listingStatus,
  usePortfolio,
} from "@/lib/portfolio";

type SortKey = "gap" | "rent" | "photos" | "zone";
type SortDir = "asc" | "desc";

export default function PortfolioPage() {
  const portfolio = usePortfolio();
  const [items, setItems] = useState<ScoutItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("gap");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [zoneFilter, setZoneFilter] = useState<string>("");

  useEffect(() => {
    if (!portfolio.ready) return;
    if (portfolio.ids.length === 0) {
      setItems([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    fetchListingsByIds(portfolio.ids)
      .then(setItems)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [portfolio.ready, portfolio.ids]);

  const filtered = useMemo(
    () => (zoneFilter ? items.filter((i) => i.zone === zoneFilter) : items),
    [items, zoneFilter],
  );

  const sorted = useMemo(() => {
    const list = [...filtered];
    const dir = sortDir === "asc" ? 1 : -1;
    list.sort((a, b) => {
      switch (sortKey) {
        case "gap":
          return (
            dir *
            ((a.predicted_rent_tabular_eur - a.rent_eur) -
              (b.predicted_rent_tabular_eur - b.rent_eur))
          );
        case "rent":
          return dir * (a.rent_eur - b.rent_eur);
        case "photos":
          return dir * ((a.image_score_percentile ?? 0) - (b.image_score_percentile ?? 0));
        case "zone":
          return dir * a.zone.localeCompare(b.zone);
      }
    });
    return list;
  }, [filtered, sortKey, sortDir]);

  const stats = useMemo(
    () => computeStats(items, portfolio.realization),
    [items, portfolio.realization],
  );

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  return (
    <div className="space-y-6">
      <header className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
            Portfolio
          </h1>
          <p className="mt-1 text-slate-600 max-w-2xl">
            Your managed listings, benchmarked against the model. Flagged rows are
            triage priorities — where rent sits below peer average or photos rank in
            the bottom tier.
          </p>
        </div>
        <Link
          href="/intake"
          className="shrink-0 inline-flex items-center gap-2 rounded-md bg-teal-700 px-4 py-2 text-sm font-medium text-white hover:bg-teal-800"
        >
          + New listing intake
        </Link>
      </header>

      <StatsRow
        stats={stats}
        realization={portfolio.realization}
        onRealizationChange={portfolio.setRealization}
      />

      <HowItWorks />

      {error && (
        <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          Failed to load: {error}. Is the API running on {process.env.NEXT_PUBLIC_API_URL}?
        </div>
      )}

      {!portfolio.ready || loading ? (
        <TableSkeleton />
      ) : items.length === 0 ? (
        <EmptyState onSeed={portfolio.seed} />
      ) : (
        <>
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div className="flex items-center gap-3">
              <label className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                Zone
              </label>
              <select
                value={zoneFilter}
                onChange={(e) => setZoneFilter(e.target.value)}
                className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm"
              >
                <option value="">All</option>
                {ZONES.map((z) => (
                  <option key={z} value={z}>
                    {z}
                  </option>
                ))}
              </select>
              <span className="text-xs text-slate-500">
                {sorted.length} of {items.length} listings
              </span>
            </div>
            <button
              onClick={portfolio.reset}
              className="text-xs text-slate-500 hover:text-slate-800 underline"
            >
              Reset demo portfolio
            </button>
          </div>

          <PortfolioTable
            items={sorted}
            onRemove={portfolio.remove}
            sortKey={sortKey}
            sortDir={sortDir}
            onSort={toggleSort}
          />
        </>
      )}
    </div>
  );
}

function StatsRow({
  stats,
  realization,
  onRealizationChange,
}: {
  stats: ReturnType<typeof computeStats>;
  realization: number;
  onRealizationChange: (v: number) => void;
}) {
  return (
    <section className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <StatCard
          label="Listings managed"
          value={stats.managedCount.toString()}
          hint={`${stats.flaggedCount} flagged for triage`}
          tone={stats.flaggedCount > 0 ? "warn" : "neutral"}
        />
        <StatCard label="Managed rent / mo" value={formatEur(stats.managedRent)} />
        <StatCard
          label={`Commission / mo @ ${Math.round(COMMISSION_PCT * 100)}%`}
          value={formatEur(stats.commissionMonthly)}
          hint={formatEur(stats.commissionYearly) + " / yr"}
        />
        <StatCard
          label="Peer rent gap / mo"
          value={formatEur(stats.gapMonthly)}
          hint="sum of listings below peer average"
          tone="accent"
        />
        <StatCard
          label="Potential comm. upside / yr"
          value={formatEur(stats.commissionUpsideYearly)}
          hint={`at ${Math.round(realization * 100)}% realization over 12mo`}
          tone="accent"
        />
      </div>
      <RealizationSlider value={realization} onChange={onRealizationChange} />
    </section>
  );
}

function RealizationSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 flex items-center gap-4 flex-wrap">
      <div className="min-w-[220px]">
        <div className="text-xs font-medium text-slate-500 uppercase tracking-wide">
          Realization rate
        </div>
        <div className="text-sm text-slate-700">
          What fraction of the peer rent gap the agency actually closes within 12
          months. Industry-realistic range is 20–40%.
        </div>
      </div>
      <div className="flex-1 flex items-center gap-3 min-w-[260px]">
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={Math.round(value * 100)}
          onChange={(e) => onChange(Number(e.target.value) / 100)}
          className="flex-1 accent-teal-700"
        />
        <span className="font-mono text-sm tabular-nums w-12 text-right text-slate-800">
          {Math.round(value * 100)}%
        </span>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "accent" | "warn" | "neutral";
}) {
  const cls =
    tone === "accent"
      ? "border-teal-300 bg-teal-50"
      : tone === "warn"
      ? "border-amber-200 bg-amber-50"
      : "border-slate-200 bg-white";
  const valCls = tone === "accent" ? "text-teal-800" : "text-slate-900";
  return (
    <div className={`rounded-lg border p-4 ${cls}`}>
      <div className="text-xs font-medium text-slate-500 uppercase tracking-wide">
        {label}
      </div>
      <div className={`mt-1 text-2xl font-semibold ${valCls}`}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

function HowItWorks() {
  return (
    <details className="group rounded-lg border border-slate-200 bg-white">
      <summary className="cursor-pointer list-none p-4 flex items-center justify-between gap-2">
        <span className="text-sm font-semibold text-slate-900">
          How the flag works (and what it doesn&apos;t mean)
        </span>
        <span className="text-xs text-slate-500 group-open:rotate-180 transition-transform">
          ▾
        </span>
      </summary>
      <div className="px-4 pb-4 text-sm text-slate-600 space-y-2">
        <p>A listing is flagged as triage priority on two signals:</p>
        <ol className="list-decimal pl-5 space-y-1">
          <li>
            <strong>Rent below peer-expected</strong> — a gradient-boosting regressor
            fit on 1,470 Madrid rentals estimates the expected rent for apartments of
            similar size / zone / amenities. Actual rent more than 8% below that
            estimate triggers a flag.
          </li>
          <li>
            <strong>Photos in the bottom ~40%</strong> — a fine-tuned ResNet-50 head
            predicts per-image log-price; we average per-listing and rank against the
            dataset.
          </li>
        </ol>
        <p className="text-xs">
          <strong>Caveat:</strong> both signals are correlational. A listing below
          peer-expected rent may simply have an unobserved negative (noisy street,
          dark, awkward layout). Use this list as a triage shortlist, not a
          prescription. The per-listing detail view shows why each one was flagged.
        </p>
        <p className="text-xs">
          <strong>Commission upside</strong> = (peer rent gap) × 6% × 12 months ×
          realization rate. The default 25% reflects that agencies rarely close the
          full gap in one cycle — adjust if your book behaves differently.
        </p>
      </div>
    </details>
  );
}

function PortfolioTable({
  items,
  onRemove,
  sortKey,
  sortDir,
  onSort,
}: {
  items: ScoutItem[];
  onRemove: (id: string) => void;
  sortKey: SortKey;
  sortDir: SortDir;
  onSort: (key: SortKey) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
      <table className="w-full text-sm">
        <thead className="bg-slate-50 text-slate-600">
          <tr>
            <Th>Listing</Th>
            <Th sortable active={sortKey === "zone"} dir={sortDir} onClick={() => onSort("zone")}>
              Zone / Size
            </Th>
            <Th align="right" sortable active={sortKey === "rent"} dir={sortDir} onClick={() => onSort("rent")}>
              Rent
            </Th>
            <Th align="right">Peer expected</Th>
            <Th align="right" sortable active={sortKey === "gap"} dir={sortDir} onClick={() => onSort("gap")}>
              Gap €/mo
            </Th>
            <Th align="center" sortable active={sortKey === "photos"} dir={sortDir} onClick={() => onSort("photos")}>
              Photos
            </Th>
            <Th>Status</Th>
            <Th></Th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <Row key={item.listing_id} item={item} onRemove={onRemove} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Th({
  children,
  align = "left",
  sortable = false,
  active = false,
  dir,
  onClick,
}: {
  children?: React.ReactNode;
  align?: "left" | "right" | "center";
  sortable?: boolean;
  active?: boolean;
  dir?: SortDir;
  onClick?: () => void;
}) {
  const alignCls =
    align === "right" ? "text-right" : align === "center" ? "text-center" : "text-left";
  const base = `px-4 py-2.5 text-xs font-semibold uppercase tracking-wide ${alignCls}`;
  if (!sortable) return <th scope="col" className={base}>{children}</th>;
  return (
    <th scope="col" className={base}>
      <button
        onClick={onClick}
        className={`inline-flex items-center gap-1 ${active ? "text-slate-900" : "text-slate-600"} hover:text-slate-900`}
      >
        {children}
        <span className="text-[10px]">
          {active ? (dir === "desc" ? "▼" : "▲") : "⇅"}
        </span>
      </button>
    </th>
  );
}

function Row({ item, onRemove }: { item: ScoutItem; onRemove: (id: string) => void }) {
  const gap = item.predicted_rent_tabular_eur - item.rent_eur;
  const status = listingStatus(item);
  const statusCls =
    status.tone === "bad"
      ? "bg-red-50 text-red-800 border-red-200"
      : status.tone === "warn"
      ? "bg-amber-50 text-amber-800 border-amber-200"
      : status.tone === "good"
      ? "bg-emerald-50 text-emerald-800 border-emerald-200"
      : "bg-slate-50 text-slate-700 border-slate-200";
  const photosPct = item.image_score_percentile;
  return (
    <tr className="border-t border-slate-100 hover:bg-slate-50/70">
      <td className="px-4 py-3">
        <Link href={`/listings/${item.listing_id}`} className="flex items-center gap-3 group">
          <div className="h-14 w-20 overflow-hidden rounded bg-slate-100 shrink-0">
            {item.thumbnail_url ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={imageUrl(item.thumbnail_url)}
                alt=""
                className="h-full w-full object-cover"
                loading="lazy"
              />
            ) : null}
          </div>
          <div className="min-w-0">
            <div className="font-medium text-slate-900 truncate max-w-[22ch] group-hover:text-teal-700">
              {item.title ?? item.listing_id}
            </div>
            <div className="text-xs text-slate-500 truncate max-w-[22ch]">
              ID {item.listing_id}
            </div>
          </div>
        </Link>
      </td>
      <td className="px-4 py-3 text-slate-700">
        {item.zone}
        <div className="text-xs text-slate-500">
          {item.sqft_m2.toFixed(0)}m²
          {item.rooms ? ` · ${item.rooms.toFixed(0)} rm` : ""}
        </div>
      </td>
      <td className="px-4 py-3 text-right font-medium text-slate-900">
        {formatEur(item.rent_eur)}
      </td>
      <td className="px-4 py-3 text-right text-slate-500">
        {formatEur(item.predicted_rent_tabular_eur)}
      </td>
      <td
        className={`px-4 py-3 text-right font-semibold ${
          gap > 150 ? "text-teal-700" : gap < -150 ? "text-slate-500" : "text-slate-400"
        }`}
      >
        {gap > 0 ? "+" : ""}
        {formatEur(gap)}
      </td>
      <td className="px-4 py-3 text-center">
        {photosPct != null ? (
          <PhotoPill pct={photosPct} />
        ) : (
          <span className="text-xs text-slate-400">—</span>
        )}
      </td>
      <td className="px-4 py-3">
        <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${statusCls}`}>
          {status.label}
        </span>
      </td>
      <td className="px-4 py-3 text-right">
        <button
          onClick={() => onRemove(item.listing_id)}
          className="text-xs text-slate-400 hover:text-red-600"
          title="Remove from portfolio"
          aria-label={`Remove ${item.listing_id}`}
        >
          ✕
        </button>
      </td>
    </tr>
  );
}

function PhotoPill({ pct }: { pct: number }) {
  const tone = pct < 33 ? "bad" : pct > 66 ? "good" : "neutral";
  const cls =
    tone === "bad"
      ? "bg-red-50 text-red-700 border-red-200"
      : tone === "good"
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : "bg-slate-50 text-slate-700 border-slate-200";
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs ${cls}`}>
      {pct.toFixed(0)}th pct
    </span>
  );
}

function EmptyState({ onSeed }: { onSeed: () => void }) {
  return (
    <div className="rounded-lg border-2 border-dashed border-slate-200 bg-white p-12 text-center">
      <h2 className="text-lg font-semibold text-slate-900">No listings in your portfolio</h2>
      <p className="mt-2 text-slate-600 max-w-md mx-auto text-sm">
        Reseed with a mixed demo portfolio (6 flagged triage candidates + 6 healthy
        listings) to see a realistic agency book.
      </p>
      <button
        onClick={onSeed}
        className="mt-4 rounded-md bg-teal-700 px-4 py-2 text-sm font-medium text-white hover:bg-teal-800"
      >
        Load demo portfolio
      </button>
    </div>
  );
}

function TableSkeleton() {
  return (
    <div className="rounded-lg border border-slate-200 bg-white overflow-hidden">
      <div className="bg-slate-50 h-10" />
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="h-16 border-t border-slate-100 animate-pulse bg-white" />
      ))}
    </div>
  );
}
