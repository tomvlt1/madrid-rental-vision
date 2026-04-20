"use client";

import Link from "next/link";
import { useMemo, useRef, useState } from "react";
import {
  formatEur,
  imageUrl,
  intake,
  IntakeBaseline,
  IntakeResponse,
  MODEL_MAE_EUR,
  parseListingId,
  PhotoScore,
} from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio";

type Preview = { file: File; url: string };

export default function IntakePage() {
  const portfolio = usePortfolio();
  const [idInput, setIdInput] = useState("");
  const [listingId, setListingId] = useState<string | null>(null);
  const [baseline, setBaseline] = useState<IntakeBaseline | null>(null);
  const [previews, setPreviews] = useState<Preview[]>([]);
  const [result, setResult] = useState<IntakeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetchingBaseline, setFetchingBaseline] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadBaseline = async (rawId: string) => {
    const lid = parseListingId(rawId);
    if (!lid) {
      setError("Enter a listing ID or URL.");
      return;
    }
    setError(null);
    setBaseline(null);
    setResult(null);
    clearPreviews();
    setFetchingBaseline(true);
    const form = new FormData();
    form.append("listing_id", lid);
    try {
      const resp = await intake(form);
      setListingId(resp.baseline.listing_id);
      setBaseline(resp.baseline);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setFetchingBaseline(false);
    }
  };

  const addFiles = (files: FileList | File[]) => {
    const arr = Array.from(files).filter((f) => f.type.startsWith("image/"));
    const next = arr.map((f) => ({ file: f, url: URL.createObjectURL(f) }));
    setPreviews((cur) => [...cur, ...next]);
  };

  const removeAt = (idx: number) => {
    setPreviews((cur) => {
      URL.revokeObjectURL(cur[idx].url);
      return cur.filter((_, i) => i !== idx);
    });
  };

  const clearPreviews = () => {
    previews.forEach((p) => URL.revokeObjectURL(p.url));
    setPreviews([]);
  };

  const runTest = async () => {
    if (!listingId) return;
    if (previews.length === 0) {
      setError("Drop at least one new photo to test.");
      return;
    }
    setError(null);
    setLoading(true);
    const form = new FormData();
    form.append("listing_id", listingId);
    previews.forEach((p) => form.append("images", p.file));
    try {
      const resp = await intake(form);
      setResult(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setIdInput("");
    setListingId(null);
    setBaseline(null);
    setResult(null);
    clearPreviews();
    setError(null);
  };

  const inPortfolio = listingId ? portfolio.contains(listingId) : false;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
          Listing intake
        </h1>
        <p className="mt-1 text-slate-600 max-w-2xl">
          Paste a listing ID or URL. Upload photos you&apos;re considering
          — they&apos;ll <strong>replace the weakest existing photos</strong> and we
          re-predict peer-expected rent. This is a ranked substitution hypothesis,
          not a guaranteed counterfactual.
        </p>
      </header>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            loadBaseline(idInput);
          }}
          className="flex flex-wrap gap-3 items-end"
        >
          <label className="flex-1 min-w-[260px]">
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
              Listing ID or URL
            </span>
            <input
              value={idInput}
              onChange={(e) => setIdInput(e.target.value)}
              placeholder="111149213  or  https://www.idealista.com/inmueble/111149213/"
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-mono"
            />
          </label>
          <button
            type="submit"
            disabled={fetchingBaseline || !idInput.trim()}
            className="rounded-md bg-teal-700 px-4 py-2 text-sm font-medium text-white hover:bg-teal-800 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {fetchingBaseline ? "Loading…" : "Load listing"}
          </button>
          {portfolio.ready && portfolio.ids.length > 0 && (
            <QuickPick
              ids={portfolio.ids}
              onPick={(lid) => {
                setIdInput(lid);
                loadBaseline(lid);
              }}
            />
          )}
          {listingId && (
            <button
              type="button"
              onClick={reset}
              className="text-sm text-slate-500 hover:text-slate-800 underline"
            >
              Start over
            </button>
          )}
        </form>
      </div>

      {error && (
        <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
          {error}
        </div>
      )}

      {baseline && (
        <BaselineCard
          baseline={baseline}
          inPortfolio={inPortfolio}
          onToggle={() =>
            inPortfolio ? portfolio.remove(baseline.listing_id) : portfolio.add(baseline.listing_id)
          }
        />
      )}

      {baseline && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          <section className="lg:col-span-2 space-y-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">
                Test new photos
              </h2>
              <p className="mt-1 text-sm text-slate-600">
                Your uploads will swap out the weakest existing photos (1-for-1), then
                the model re-predicts on the new set. Drop N photos → N weakest
                originals get replaced.
              </p>
            </div>

            <Dropzone
              onFiles={addFiles}
              onClick={() => fileInputRef.current?.click()}
              count={previews.length}
            />
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={(e) => {
                if (e.target.files) addFiles(e.target.files);
                e.target.value = "";
              }}
            />

            {previews.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600">
                    {previews.length} new photo{previews.length === 1 ? "" : "s"} staged
                  </span>
                  <button
                    type="button"
                    onClick={clearPreviews}
                    className="text-xs text-slate-500 hover:text-slate-800 underline"
                  >
                    Clear
                  </button>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  {previews.map((p, i) => (
                    <div
                      key={p.url}
                      className="relative group aspect-square overflow-hidden rounded-md border border-slate-200"
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={p.url} alt="" className="h-full w-full object-cover" />
                      <button
                        type="button"
                        onClick={() => removeAt(i)}
                        className="absolute top-1 right-1 rounded bg-black/60 text-white text-xs px-1.5 py-0.5 opacity-0 group-hover:opacity-100 transition"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <button
              onClick={runTest}
              disabled={loading || previews.length === 0}
              className="w-full rounded-md bg-teal-700 px-5 py-2.5 text-white font-medium hover:bg-teal-800 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? "Scoring…" : "Run substitution hypothesis"}
            </button>

            <p className="text-xs text-slate-500">
              Model MAE on test set is ±{formatEur(MODEL_MAE_EUR)}. Any predicted
              delta inside that band is noise, not signal.
            </p>
          </section>

          <aside className="lg:col-span-3">
            {!result && !loading && previews.length === 0 && (
              <div className="rounded-lg border-2 border-dashed border-slate-200 bg-white p-8 text-center text-slate-500">
                Drop new photos on the left. We&apos;ll swap them in for the weakest
                existing photos and re-predict.
              </div>
            )}
            {!result && !loading && previews.length > 0 && (
              <div className="rounded-lg border-2 border-dashed border-slate-300 bg-amber-50 p-6 text-center text-amber-800 text-sm">
                {previews.length} photo{previews.length === 1 ? "" : "s"} staged — click{" "}
                <em>Run substitution hypothesis</em> to re-score.
              </div>
            )}
            {loading && (
              <div className="rounded-lg border border-slate-200 bg-white p-8 text-center text-slate-500">
                Re-pooling embeddings and running the regressor…
              </div>
            )}
            {result?.with_extras && (
              <ResultPanel
                baseline={baseline}
                extras={result.with_extras}
                previews={previews}
              />
            )}
          </aside>
        </div>
      )}
    </div>
  );
}

function QuickPick({
  ids,
  onPick,
}: {
  ids: string[];
  onPick: (id: string) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
        From portfolio
      </span>
      <select
        onChange={(e) => {
          if (e.target.value) onPick(e.target.value);
        }}
        defaultValue=""
        className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm"
      >
        <option value="" disabled>
          Pick a listing…
        </option>
        {ids.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>
    </label>
  );
}

function BaselineCard({
  baseline,
  inPortfolio,
  onToggle,
}: {
  baseline: IntakeBaseline;
  inPortfolio: boolean;
  onToggle: () => void;
}) {
  const pct = baseline.image_score_percentile;
  const photoTone =
    pct == null ? "neutral" : pct < 33 ? "bad" : pct > 66 ? "good" : "neutral";
  return (
    <section className="rounded-lg border border-slate-200 bg-white overflow-hidden">
      <div className="flex flex-col md:flex-row">
        {baseline.thumbnail_url && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={imageUrl(baseline.thumbnail_url)}
            alt=""
            className="h-40 md:h-auto md:w-64 object-cover bg-slate-100 shrink-0"
          />
        )}
        <div className="p-5 flex-1 space-y-3">
          <div className="flex items-start justify-between gap-3 flex-wrap">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">
                {baseline.title ?? `Listing ${baseline.listing_id}`}
              </h2>
              <p className="text-sm text-slate-500">
                {baseline.zone} · {baseline.sqft_m2.toFixed(0)}m²
                {baseline.rooms ? ` · ${baseline.rooms.toFixed(0)} rooms` : ""} ·{" "}
                {baseline.num_existing_photos} photos
              </p>
            </div>
            <div className="flex gap-2">
              <Link
                href={`/listings/${baseline.listing_id}`}
                className="text-xs text-teal-700 hover:underline px-2 py-1"
              >
                View full detail →
              </Link>
              <button
                onClick={onToggle}
                className={`rounded-md border px-3 py-1 text-xs font-medium ${
                  inPortfolio
                    ? "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    : "border-teal-700 bg-teal-700 text-white hover:bg-teal-800"
                }`}
              >
                {inPortfolio ? "In portfolio ✓" : "+ Add to portfolio"}
              </button>
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MiniStat label="Current rent" value={formatEur(baseline.current_rent_eur)} />
            <MiniStat
              label="Peer expected (tabular)"
              value={formatEur(baseline.peer_expected_rent_tabular_eur)}
              tone={
                baseline.peer_expected_rent_tabular_eur > baseline.current_rent_eur + MODEL_MAE_EUR
                  ? "good"
                  : undefined
              }
            />
            <MiniStat
              label="Peer expected (full model)"
              value={formatEur(baseline.peer_expected_rent_full_eur ?? baseline.peer_expected_rent_tabular_eur)}
              hint="tabular + photos + text"
            />
            <MiniStat
              label="Photo quality"
              value={pct != null ? `${pct.toFixed(0)}th pct` : "—"}
              tone={photoTone === "bad" ? "warn" : photoTone === "good" ? "good" : undefined}
              hint="vs all listings"
            />
          </div>
        </div>
      </div>
    </section>
  );
}

function MiniStat({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "good" | "warn";
}) {
  const cls =
    tone === "good"
      ? "text-emerald-700"
      : tone === "warn"
      ? "text-amber-700"
      : "text-slate-900";
  return (
    <div className="rounded-md bg-slate-50 px-3 py-2">
      <div className="text-[10px] font-medium text-slate-500 uppercase tracking-wide">
        {label}
      </div>
      <div className={`text-sm font-semibold ${cls}`}>{value}</div>
      {hint && <div className="text-[10px] text-slate-500">{hint}</div>}
    </div>
  );
}

function Dropzone({
  onFiles,
  onClick,
  count,
}: {
  onFiles: (files: FileList | File[]) => void;
  onClick: () => void;
  count: number;
}) {
  const [hover, setHover] = useState(false);
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setHover(true);
      }}
      onDragLeave={() => setHover(false)}
      onDrop={(e) => {
        e.preventDefault();
        setHover(false);
        if (e.dataTransfer.files) onFiles(e.dataTransfer.files);
      }}
      onClick={onClick}
      className={`rounded-lg border-2 border-dashed p-6 text-center cursor-pointer transition ${
        hover
          ? "border-teal-500 bg-teal-50"
          : "border-slate-300 bg-white hover:border-slate-400"
      }`}
    >
      <div className="text-sm text-slate-700">
        <span className="font-medium">Drop new photos here</span> or click to browse
      </div>
      <div className="mt-1 text-xs text-slate-500">
        {count === 0 ? "Nothing staged yet" : `${count} staged`}
      </div>
    </div>
  );
}

function ResultPanel({
  baseline,
  extras,
  previews,
}: {
  baseline: IntakeBaseline;
  extras: NonNullable<IntakeResponse["with_extras"]>;
  previews: Preview[];
}) {
  const mae = extras.predicted_rent_mae_eur;
  const deltaPrev = extras.delta_vs_previous_model_eur ?? 0;
  const insideNoise = Math.abs(deltaPrev) < mae;

  // normalized ranking for the new photos (within this upload batch)
  const scores = extras.per_extra_scores.map((p) => p.score_eur);
  const min = scores.length ? Math.min(...scores) : 0;
  const max = scores.length ? Math.max(...scores) : 1;
  const range = max > min ? max - min : 1;

  return (
    <div className="space-y-4">
      <div
        className={`rounded-lg border-2 p-5 ${
          insideNoise
            ? "border-slate-300 bg-white"
            : deltaPrev > 0
            ? "border-teal-600 bg-teal-50"
            : "border-amber-400 bg-amber-50"
        }`}
      >
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-slate-600">
              With substitution, peer-expected rent
            </div>
            <div className="text-4xl font-semibold text-slate-900 mt-1 tabular-nums">
              {formatEur(extras.predicted_rent_eur)}
              <span className="ml-2 text-base font-normal text-slate-500">
                ± {formatEur(mae)}
              </span>
            </div>
            <div className="mt-2 text-sm text-slate-700 space-y-0.5">
              <DeltaLine
                label="vs current list price"
                base={baseline.current_rent_eur}
                delta={extras.delta_vs_current_rent_eur}
                mae={mae}
              />
              {extras.delta_vs_previous_model_eur != null && baseline.peer_expected_rent_full_eur != null && (
                <DeltaLine
                  label="vs peer-expected (existing photos)"
                  base={baseline.peer_expected_rent_full_eur}
                  delta={extras.delta_vs_previous_model_eur}
                  mae={mae}
                />
              )}
            </div>
            {insideNoise && (
              <p className="mt-3 text-xs text-slate-500 italic">
                Predicted change is inside the ±{formatEur(mae)} MAE band — treat as
                noise, not a reliable signal.
              </p>
            )}
          </div>
          <div className="text-xs text-slate-500 text-right">
            Pooled over<br />
            <span className="text-slate-900 font-semibold text-lg">
              {extras.total_photos_considered}
            </span>{" "}
            photos<br />
            <span className="text-slate-600">
              {extras.kept_photo_count} kept + {extras.per_extra_scores.length} new
            </span>
          </div>
        </div>
      </div>

      {extras.replaced_photo_urls.length > 0 && (
        <ReplacedList urls={extras.replaced_photo_urls} previews={previews} />
      )}

      {extras.suggestions.length > 0 && (
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <h3 className="text-sm font-semibold text-slate-900">Read-out</h3>
          <ul className="mt-2 space-y-1.5">
            {extras.suggestions.map((s, i) => (
              <li key={i} className="flex gap-2 text-sm text-slate-700">
                <span
                  aria-hidden
                  className="mt-1.5 h-1.5 w-1.5 rounded-full bg-teal-600 shrink-0"
                />
                <span>{s}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h3 className="text-sm font-semibold text-slate-900">
          Your new photos (ranked within upload batch)
        </h3>
        <p className="text-xs text-slate-500 mt-0.5">
          Strength is a normalized model score across your uploads. Not an absolute
          rent value.
        </p>
        <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 gap-3">
          {extras.per_extra_scores.map((p, idx) => (
            <NewPhotoCard
              key={p.image_url + idx}
              photo={p}
              preview={previews[idx]}
              normalized={Math.round(((p.score_eur - min) / range) * 100)}
              total={extras.per_extra_scores.length}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function ReplacedList({
  urls,
  previews,
}: {
  urls: string[];
  previews: Preview[];
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h3 className="text-sm font-semibold text-slate-900">
        Photos swapped out ({urls.length})
      </h3>
      <p className="text-xs text-slate-500 mt-0.5">
        The weakest-ranked existing photos, replaced 1-for-1 by your uploads.
      </p>
      <div className="mt-3 grid grid-cols-3 sm:grid-cols-6 gap-2">
        {urls.map((url, i) => (
          <div
            key={url}
            className="relative aspect-[4/3] overflow-hidden rounded-md bg-slate-100 ring-1 ring-amber-300"
            title="Replaced"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageUrl(url)}
              alt=""
              className="h-full w-full object-cover opacity-60"
              loading="lazy"
            />
            <span className="absolute inset-0 grid place-items-center bg-amber-100/50">
              <span className="rounded bg-amber-600 text-white text-[10px] font-medium px-1.5 py-0.5">
                replaced
              </span>
            </span>
            {previews[i] && (
              <span className="absolute bottom-1 right-1 text-[10px] text-slate-700 bg-white/80 rounded px-1">
                ↔ #{i + 1}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function DeltaLine({
  label,
  base,
  delta,
  mae,
}: {
  label: string;
  base: number;
  delta: number;
  mae: number;
}) {
  const positive = delta > 0;
  const meaningful = Math.abs(delta) > mae;
  const deltaCls = !meaningful
    ? "text-slate-500"
    : positive
    ? "text-emerald-700"
    : "text-amber-700";
  return (
    <div>
      <span className="text-slate-600">{label}: </span>
      <span className="text-slate-900">{formatEur(base)}</span>
      <span className={`ml-2 font-medium ${deltaCls}`}>
        {positive ? "+" : ""}
        {formatEur(delta)}
      </span>
      {!meaningful && <span className="ml-1 text-[11px] text-slate-400">(in noise)</span>}
    </div>
  );
}

function NewPhotoCard({
  photo,
  preview,
  normalized,
  total,
}: {
  photo: PhotoScore;
  preview?: Preview;
  normalized: number;
  total: number;
}) {
  return (
    <div className="space-y-1.5">
      <div className="relative aspect-[4/3] overflow-hidden rounded-md bg-slate-100 ring-1 ring-slate-200">
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={preview.url} alt="" className="h-full w-full object-cover" />
        ) : null}
        <span className="absolute top-1.5 left-1.5 rounded bg-black/60 text-white text-xs px-1.5 py-0.5">
          #{photo.rank_in_listing} of {total}
        </span>
      </div>
      <div className="flex items-center justify-between text-xs text-slate-600">
        <span>Strength {normalized}/100</span>
      </div>
      <div className="h-1 rounded-full bg-slate-100 overflow-hidden">
        <div className="h-full bg-teal-600" style={{ width: `${Math.max(3, normalized)}%` }} />
      </div>
    </div>
  );
}
