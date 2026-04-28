/*
 * CasaIntel content script: runs on Idealista Madrid pages.
 *
 * Search results: parses each listing card, calls the backend in tabular
 * mode, and injects a small peer-expected-rent badge.
 *
 * Detail page: parses the listing, calls the backend in full mode (with
 * photos + description), and injects a panel near the asking price.
 */
(function () {
  "use strict";

  const API_URL = "http://127.0.0.1:8000";
  const STORAGE_KEY_LISTINGS = "casaintel_saved_listings_v1";
  const HISTORY_CAP = 10;
  const AMENITY_KEYS = [
    "ac",
    "terrace",
    "furnished",
    "parking",
    "elevator",
    "exterior",
    "heating",
    "storage",
  ];
  const AMENITY_LABELS = {
    ac: "AC",
    terrace: "Terrace",
    furnished: "Furnished",
    parking: "Parking",
    elevator: "Elevator",
    exterior: "Exterior",
    heating: "Heating",
    storage: "Storage",
  };

  // ------------------------- utilities ---------------------------------

  function parseEuros(str) {
    if (!str) return null;
    const clean = str.replace(/[^0-9]/g, "");
    const n = parseInt(clean, 10);
    return Number.isFinite(n) && n > 0 ? n : null;
  }

  function parseSqft(str) {
    if (!str) return null;
    const m = str.match(/([0-9]+(?:[.,][0-9]+)?)\s*m(?:²|2)?/i);
    if (!m) return null;
    return Math.round(parseFloat(m[1].replace(",", ".")));
  }

  function parseRooms(str) {
    if (!str) return null;
    if (/estudio/i.test(str)) return 0;
    const m = str.match(/([0-9]+)\s*hab/i);
    return m ? parseInt(m[1], 10) : null;
  }

  function textContent(el) {
    return el ? (el.textContent || "").trim() : "";
  }

  function fmtEur(n) {
    if (n == null || !Number.isFinite(n)) return "--";
    const rounded = Math.round(n);
    const abs = Math.abs(rounded);
    const formatted = "€" + abs.toLocaleString("en-US");
    // ASCII hyphen, not U+2212. jsPDF's default Helvetica doesn't have
    // the Unicode minus glyph, so it renders as a placeholder in PDFs.
    // The visual difference in HTML is negligible.
    return rounded < 0 ? "-" + formatted : formatted;
  }

  // Roll a euro number from `from` to `to` over `durationMs` and write the
  // intermediate values into `node.textContent`. Uses ease-out-cubic so the
  // number decelerates as it lands. Caps at 60fps via requestAnimationFrame.
  function animateEur(node, from, to, durationMs) {
    if (from == null || !Number.isFinite(from)) {
      node.textContent = fmtEur(to);
      return;
    }
    const dur = durationMs || 450;
    const start = performance.now();
    function step(now) {
      const t = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - t, 3);
      const v = from + (to - from) * eased;
      node.textContent = fmtEur(v);
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // Build an honest asymmetric prediction interval. Prefers backend-supplied
  // bounds (predicted_lower_eur / predicted_upper_eur from the stratified
  // residual quantiles in v2/models/intervals.json). Falls back to a
  // log-space-derived percentage approximation if the backend doesn't serve
  // them yet. Either way, the rendering replaces the old "± MAE" symmetric
  // band — predictions in this domain are not symmetric around the point
  // estimate (luxury tail makes the upper bound wider).
  function intervalBounds(result) {
    if (
      result &&
      result.predicted_lower_eur != null &&
      result.predicted_upper_eur != null
    ) {
      return {
        lower: result.predicted_lower_eur,
        upper: result.predicted_upper_eur,
        source: "model",
      };
    }
    const p = result && result.predicted_rent_eur;
    if (p == null || !Number.isFinite(p)) return null;
    // Heuristic: 80% asymmetric range derived from v2 5-fold OOF residuals
    // in log-space (rmse_log ≈ 0.18). expm1-asymmetry pushes the upper
    // bound slightly wider. These factors roughly match the empirical
    // 10/90 quantiles and stay honest about the luxury tail.
    return {
      lower: Math.round(p * 0.82),
      upper: Math.round(p * 1.22),
      source: "approx",
    };
  }

  function fmtRange(bounds) {
    if (!bounds) return "";
    return fmtEur(bounds.lower) + " – " + fmtEur(bounds.upper);
  }

  function el(tag, className, text) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    if (text != null) e.textContent = text;
    return e;
  }

  // --------------------- page-type detection ---------------------------

  function getPageType() {
    const p = location.pathname;
    if (/^\/inmueble\/\d+/.test(p)) return "detail";
    if (/^\/alquiler-viviendas\/madrid/.test(p)) return "list";
    return null;
  }

  // --------------------- list (search results) -------------------------

  function parseListingCard(card) {
    const link = card.querySelector("a.item-link, a[href*='/inmueble/']");
    if (!link) return null;
    const href = link.getAttribute("href") || "";
    const idMatch = href.match(/\/inmueble\/(\d+)/);
    if (!idMatch) return null;
    const listing_id = idMatch[1];

    const priceEl = card.querySelector(".item-price, .price-row, [class*='price']");
    const current_rent_eur = parseEuros(textContent(priceEl));

    const detailEl = card.querySelector(".item-detail-char, .item-detail, [class*='detail']");
    const detailText = textContent(detailEl);
    const sqft = parseSqft(detailText);
    const rooms = parseRooms(detailText);

    const locEl = card.querySelector(".item-link") || link;
    const locationText = textContent(locEl);

    if (!sqft) return null;

    return {
      card,
      listing_id,
      payload: {
        listing_id,
        sqft,
        rooms: rooms != null ? rooms : undefined,
        location: locationText,
        current_rent_eur,
        mode: "tabular",
      },
    };
  }

  function toneLabel(tone) {
    return (
      {
        overpriced: "Overpriced",
        underpriced: "Under peer",
        fair: "Fair price",
      }[tone] || "Peer estimate"
    );
  }

  function deltaRow(wrap, label, note, deltaEur) {
    if (deltaEur == null || !Number.isFinite(deltaEur)) return;
    const row = el("div", "casa-intel-details-row");
    const keyWrap = el("span", "casa-intel-details-key");
    keyWrap.appendChild(el("span", null, label));
    if (note) keyWrap.appendChild(el("span", "casa-intel-details-subkey", note));
    row.appendChild(keyWrap);
    const sign = deltaEur > 0 ? "+" : "";
    const valCls =
      "casa-intel-details-val " +
      (deltaEur > 0 ? "casa-intel-delta-pos" : deltaEur < 0 ? "casa-intel-delta-neg" : "");
    row.appendChild(el("span", valCls, sign + fmtEur(deltaEur)));
    wrap.appendChild(row);
  }

  function buildDetailsSection(result) {
    const wrap = el("div", "casa-intel-details");
    const b = result.breakdown;

    if (b) {
      // Header: feature-by-feature decomposition
      wrap.appendChild(
        el("div", "casa-intel-details-header", "Why this number"),
      );

      // Tabular baseline
      const tabRow = el("div", "casa-intel-details-row");
      const tabKey = el("span", "casa-intel-details-key");
      tabKey.appendChild(el("span", null, "Tabular baseline"));
      if (b.tabular_note) {
        tabKey.appendChild(
          el("span", "casa-intel-details-subkey", b.tabular_note),
        );
      }
      tabRow.appendChild(tabKey);
      tabRow.appendChild(
        el("span", "casa-intel-details-val", fmtEur(b.tabular_eur)),
      );
      wrap.appendChild(tabRow);

      // Photo delta
      if (b.photos_delta_eur != null) {
        deltaRow(wrap, "+ photos", b.photos_note, b.photos_delta_eur);
      }
      // Text delta
      if (b.text_delta_eur != null) {
        deltaRow(wrap, "+ description", b.text_note, b.text_delta_eur);
      }
      // Combined effect (only if meaningful and full_eur available)
      if (b.full_eur != null && b.interaction_eur != null && Math.abs(b.interaction_eur) > 20) {
        const sub =
          b.interaction_eur > 0
            ? "photos + description together are less negative than the sum of parts"
            : "photos + description together are more negative than the sum of parts";
        deltaRow(wrap, "Combined effect", sub, b.interaction_eur);
      }

      // Total row + asymmetric range subline
      const totalRow = el("div", "casa-intel-details-row casa-intel-details-total");
      totalRow.appendChild(el("span", "casa-intel-details-key", "Full model"));
      totalRow.appendChild(
        el(
          "span",
          "casa-intel-details-val",
          fmtEur(b.full_eur ?? result.predicted_rent_eur),
        ),
      );
      wrap.appendChild(totalRow);
      const bounds = intervalBounds(result);
      if (bounds) {
        const rangeRow = el("div", "casa-intel-details-row casa-intel-details-range");
        rangeRow.appendChild(
          el("span", "casa-intel-details-key", "80% range"),
        );
        rangeRow.appendChild(
          el("span", "casa-intel-details-val", fmtRange(bounds)),
        );
        wrap.appendChild(rangeRow);
      }
    }

    // Context rows
    if (result.current_rent_eur != null || result.zone_median_rent_eur != null) {
      const contextHdr = el("div", "casa-intel-details-header", "Context");
      wrap.appendChild(contextHdr);
      if (result.current_rent_eur != null) {
        const row = el("div", "casa-intel-details-row");
        row.appendChild(el("span", "casa-intel-details-key", "Listed at"));
        row.appendChild(
          el("span", "casa-intel-details-val", fmtEur(result.current_rent_eur)),
        );
        wrap.appendChild(row);
      }
      if (result.zone_median_rent_eur != null) {
        const row = el("div", "casa-intel-details-row");
        row.appendChild(el("span", "casa-intel-details-key", "Zone median"));
        row.appendChild(
          el("span", "casa-intel-details-val", fmtEur(result.zone_median_rent_eur)),
        );
        wrap.appendChild(row);
      }
    }

    // Bottom-line action derived from which feature block is dragging the
    // prediction down the hardest. Threshold of €100 so we don't suggest
    // action for deltas inside model noise.
    if (b) {
      const drags = [];
      if (b.photos_delta_eur != null && b.photos_delta_eur < -100) {
        drags.push({ source: "photos", delta: b.photos_delta_eur });
      }
      if (b.text_delta_eur != null && b.text_delta_eur < -100) {
        drags.push({ source: "description", delta: b.text_delta_eur });
      }
      if (drags.length > 0) {
        const parts = drags.map(
          (d) =>
            d.source + " (" + fmtEur(Math.abs(d.delta)) + " drag)"
        );
        const action = el(
          "div",
          "casa-intel-details-action",
          "To lift the model's prediction, improve " + parts.join(" and ") + ".",
        );
        wrap.appendChild(action);
      }
    }

    // Noise guidance — uses the asymmetric range to decide whether the gap
    // is inside the model's plausible spread for this listing.
    const delta = result.delta_vs_current_eur;
    if (delta != null && Number.isFinite(delta)) {
      const bounds = intervalBounds(result);
      const asking = result.current_rent_eur;
      let inRange = null;
      if (bounds && asking != null) {
        inRange = asking >= bounds.lower && asking <= bounds.upper;
      }
      const noteText =
        inRange === true
          ? "Asking sits inside the 80% range " +
            (bounds ? "(" + fmtRange(bounds) + ")" : "") +
            ". Consistent with peers — too close to call as over- or under-priced."
          : inRange === false
          ? "Asking sits outside the 80% range " +
            (bounds ? "(" + fmtRange(bounds) + ")" : "") +
            ". A real pricing difference, not measurement noise."
          : "The " + fmtEur(Math.abs(delta)) + " gap is the difference between asking and the point estimate.";
      const note = el("div", "casa-intel-details-note", noteText);
      wrap.appendChild(note);
    }
    return wrap;
  }

  function attachExpander(wrapper, result) {
    const toggle = el("button", "casa-intel-toggle", "▾ details");
    toggle.setAttribute("type", "button");
    toggle.setAttribute("aria-expanded", "false");
    const details = buildDetailsSection(result);
    details.style.display = "none";
    toggle.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const open = details.style.display === "none";
      details.style.display = open ? "block" : "none";
      toggle.textContent = (open ? "▴" : "▾") + " details";
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    wrapper.appendChild(toggle);
    wrapper.appendChild(details);
  }

  function buildListBadge(result) {
    // Search cards run on tabular-only, which is ~5x noisier on outliers
    // than the full model. We intentionally drop the fair/over/under
    // colored diagnosis here: it would mis-label premium outliers that
    // tabular features can't see (penthouses, big-terrace apartments,
    // luxury finishes). The full verdict lives on the detail page.
    const wrapper = el("div", "casa-intel-badge-soft");
    const peerLine = el("div", "casa-intel-soft-line");
    peerLine.appendChild(el("span", "casa-intel-soft-label", "peer est."));
    peerLine.appendChild(
      el("span", "casa-intel-soft-value", fmtEur(result.predicted_rent_eur)),
    );
    wrapper.appendChild(peerLine);
    const delta = result.delta_vs_current_eur;
    if (delta != null && Number.isFinite(delta)) {
      const sign = delta > 0 ? "+" : "";
      wrapper.appendChild(
        el("div", "casa-intel-soft-delta", sign + fmtEur(delta) + " vs asking"),
      );
    }
    wrapper.appendChild(
      el(
        "div",
        "casa-intel-soft-tag",
        "tabular estimate · open for full analysis",
      ),
    );
    return wrapper;
  }

  function injectListBadge(card, result) {
    if (card.querySelector(".casa-intel-badge")) return;
    const badge = buildListBadge(result);
    const priceHost = card.querySelector(".item-price, [class*='price']");
    if (priceHost && priceHost.parentNode) {
      priceHost.parentNode.insertBefore(badge, priceHost.nextSibling);
    } else {
      card.appendChild(badge);
    }
  }

  async function annotateCard(card) {
    const parsed = parseListingCard(card);
    if (!parsed) return;
    card.dataset.casaIntelPending = "1";
    try {
      const resp = await fetch(API_URL + "/predict-live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed.payload),
      });
      if (!resp.ok) return;
      const data = await resp.json();
      injectListBadge(parsed.card, data);
    } catch (e) {
      // fail silently
    } finally {
      delete card.dataset.casaIntelPending;
    }
  }

  function processListPage() {
    const cards = document.querySelectorAll(
      "article.item, .item-multimedia-container, [data-element-id*='listing']",
    );
    cards.forEach((c) => {
      if (c.dataset.casaIntelDone || c.dataset.casaIntelPending) return;
      c.dataset.casaIntelDone = "1";
      annotateCard(c);
    });
  }

  // --------------------- detail page -----------------------------------

  /**
   * Pull all Idealista image URLs off the page: scans both instantiated
   * DOM <img> tags and the raw HTML (many gallery images are lazy-loaded
   * and only appear as strings in embedded JSON until the user advances
   * the carousel). Dedupes by the numeric master ID so size/format
   * variants collapse to one URL per photo.
   *
   * Typical URL shape:
   *   https://img4.idealista.com/blur/WEB_DETAIL/0/id.pro.es.image.master/
   *     63/a1/d4/1077596597.jpg
   * Master ID = 1077596597 (file basename).
   */
  function extractIdealistaImages() {
    const raw = [];

    // 1. Any attribute on any img-adjacent element.
    const domEls = document.querySelectorAll(
      "img, source, [data-src], [data-lazy], [data-service], [data-original], [style*='background']",
    );
    const attrs = [
      "src", "data-src", "data-lazy", "data-service",
      "data-original", "data-bg", "srcset",
    ];
    domEls.forEach((node) => {
      for (const a of attrs) {
        const v = node.getAttribute(a);
        if (!v) continue;
        if (a === "srcset") {
          v.split(",").forEach((s) => raw.push(s.trim().split(/\s+/)[0]));
        } else {
          raw.push(v);
        }
      }
      const style = node.getAttribute("style");
      if (style) {
        const m = style.match(/url\(['"]?([^'")]+)/);
        if (m) raw.push(m[1]);
      }
    });

    // 2. Regex over raw HTML to catch lazy-loaded URLs embedded in JS state.
    const html = document.documentElement.outerHTML;
    const pattern = /https:\/\/img[0-9]+\.idealista\.com\/[^"'\s\\)]+?\.(?:jpg|jpeg|webp|png)/gi;
    const matches = html.match(pattern) || [];
    raw.push(...matches);

    // Keep only Idealista CDN hits
    const idealistaOnly = raw.filter(
      (u) => typeof u === "string" && /img[0-9]+\.idealista\.com/i.test(u),
    );

    // Group by numeric master ID (file basename without extension).
    const byMaster = new Map();
    for (const u of idealistaOnly) {
      const m = u.match(/([0-9]{5,})\.(?:jpg|jpeg|webp|png)(?:$|\?|#)/i);
      if (!m) continue;
      const id = m[1];
      if (byMaster.has(id)) continue;
      byMaster.set(id, u);
    }

    // Prefer largest-format variants when we have the choice. If any URL for
    // this master is a WEB_DETAIL/ (full-size) we swap to that.
    for (const [id, url] of byMaster.entries()) {
      if (/WEB_DETAIL\//.test(url)) continue;
      const better = idealistaOnly.find(
        (u) => u.includes(id) && /WEB_DETAIL\//.test(u) && /\.jpg($|\?)/i.test(u),
      );
      if (better) byMaster.set(id, better);
    }

    return Array.from(byMaster.values());
  }

  function parseDetailPage() {
    const idMatch = location.pathname.match(/\/inmueble\/(\d+)/);
    if (!idMatch) return null;
    const listing_id = idMatch[1];

    const priceEl = document.querySelector(
      ".info-data-price, .info-data .price, [class*='info-data-price']",
    );
    const current_rent_eur = parseEuros(textContent(priceEl));

    const details = document.querySelectorAll(
      ".details-property-feature-one li, .info-features li, [class*='property-features'] li, .info-features span",
    );
    let sqft = null;
    let rooms = null;
    let bathrooms = null;
    details.forEach((node) => {
      const t = textContent(node);
      if (sqft == null) {
        const s = parseSqft(t);
        if (s) sqft = s;
      }
      if (rooms == null) {
        const r = parseRooms(t);
        if (r != null) rooms = r;
      }
      const bm = t.match(/([0-9]+)\s*ba(?:ñ|n)o/i);
      if (bm) bathrooms = parseInt(bm[1], 10);
    });

    const bodyText = document.body.innerText.toLowerCase();
    const flag = (...kws) => kws.some((k) => bodyText.includes(k));
    const amenities = {
      elevator: flag("ascensor"),
      ac: flag("aire acondicionado", "climatización"),
      terrace: flag("terraza"),
      furnished: flag("amueblado"),
      heating: flag("calefacción", "calefaccion"),
      exterior: flag("exterior"),
      parking: flag("plaza de garaje", "garaje"),
      storage: flag("trastero"),
    };

    const titleEl = document.querySelector("h1, .main-info__title-main");
    const subEl = document.querySelector(".main-info__title-minor, .location");
    const location_text = (textContent(titleEl) + " " + textContent(subEl)).trim();

    const descEl = document.querySelector(
      ".comment, .adCommentsLanguage, [class*='comment']",
    );
    const description = textContent(descEl);

    const image_urls = extractIdealistaImages();

    if (!sqft) return null;

    return {
      listing_id,
      sqft,
      rooms: rooms != null ? rooms : undefined,
      bathrooms: bathrooms != null ? bathrooms : undefined,
      location: location_text,
      current_rent_eur,
      description: description || undefined,
      image_urls: image_urls.length ? image_urls : undefined,
      mode: "full",
      ...amenities,
    };
  }

  function buildDetailPanel(result, payload) {
    const panel = el("div", "casa-intel-detail-panel casa-intel-" + result.diagnosis);
    panel.id = "casa-intel-panel";

    // Header row: title + pin button
    const headerRow = el("div", "casa-intel-header-row");
    headerRow.appendChild(
      el("div", "casa-intel-header", "CasaIntel · " + (result.cached ? "cached" : "live")),
    );
    if (payload) {
      headerRow.appendChild(buildPinButton(payload, result));
    }
    panel.appendChild(headerRow);

    const detailToneLabel =
      {
        overpriced: "Asking above peer",
        underpriced: "Asking below peer",
        fair: "Priced on peer",
        unknown: "Peer estimate",
      }[result.diagnosis] || "Peer estimate";
    panel.appendChild(el("div", "casa-intel-tone", detailToneLabel));

    const big = el("div", "casa-intel-big", fmtEur(result.predicted_rent_eur));
    panel.appendChild(big);

    // Asymmetric 80% interval, replacing the old ± MAE band.
    const bounds = intervalBounds(result);
    if (bounds) {
      const rangeRow = el("div", "casa-intel-range");
      rangeRow.appendChild(el("span", "casa-intel-range-label", "80% range"));
      rangeRow.appendChild(el("span", "casa-intel-range-val", fmtRange(bounds)));
      if (bounds.source === "approx") {
        const tip = el("span", "casa-intel-range-source", "approx.");
        tip.title =
          "Approximate interval derived from 5-fold OOF residuals " +
          "(rmse_log ≈ 0.18). Replace with stratified quantiles from " +
          "v2/models/intervals.json once the backend serves them.";
        rangeRow.appendChild(tip);
      }
      panel.appendChild(rangeRow);
    }

    const delta = result.delta_vs_current_eur;
    if (delta != null && Number.isFinite(delta)) {
      const sign = delta > 0 ? "+" : "";
      panel.appendChild(
        el("div", "casa-intel-delta-big", sign + fmtEur(delta) + " vs current"),
      );
    }

    attachExpander(panel, result);

    // What-if simulator: appears below the breakdown so users see the
    // baseline first, then can experiment.
    if (payload) {
      panel.appendChild(buildWhatIfPanel(result, payload));
    }

    // Negotiation message generator: only renders when asking is meaningfully
    // above peer (5%+) and not absurdly above (which usually means a typo).
    if (payload) {
      const negotiation = buildNegotiationSection(payload, result);
      if (negotiation) panel.appendChild(negotiation);
    }

    // Export PDF: bottom of the panel so it's the last action a user takes
    // after reviewing everything above.
    if (payload) {
      const actions = el("div", "casa-intel-panel-actions");
      actions.appendChild(exportButton(payload, result));
      panel.appendChild(actions);
    }

    return panel;
  }

  function injectDetailPanel(result, anchor, payload) {
    const existing = document.querySelector("#casa-intel-panel");
    if (existing) existing.remove();
    const panel = buildDetailPanel(result, payload);
    insertBelow(anchor, panel);
  }

  function showSkeleton(anchor, text) {
    const existing = document.querySelector("#casa-intel-panel");
    if (existing) existing.remove();
    const skel = el("div", "casa-intel-detail-panel casa-intel-loading");
    skel.id = "casa-intel-panel";
    skel.appendChild(el("div", "casa-intel-header", "CasaIntel"));
    skel.appendChild(el("div", "casa-intel-tone", text));
    insertBelow(anchor, skel);
  }

  // ---------------- per-photo gallery overlay ----------------------

  function masterIdFromUrl(u) {
    if (!u) return null;
    const m = u.match(/([0-9]{5,})\.(?:jpg|jpeg|webp|png)(?:$|\?|#)/i);
    return m ? m[1] : null;
  }

  function resolveImgSources(img) {
    const sources = [];
    const attrs = ["src", "data-src", "data-lazy", "data-service", "data-original"];
    for (const a of attrs) {
      const v = img.getAttribute(a);
      if (v) sources.push(v);
    }
    const srcset = img.getAttribute("srcset");
    if (srcset) {
      srcset.split(",").forEach((s) => {
        const u = s.trim().split(/\s+/)[0];
        if (u) sources.push(u);
      });
    }
    return sources;
  }

  function injectPhotoOverlay(imgEl, impact, total) {
    // Avoid stacking overlays if we re-run
    const parent = imgEl.parentElement;
    if (!parent) return;
    if (parent.querySelector(".casa-intel-photo-overlay[data-ci-master='" + masterIdFromUrl(impact.image_url) + "']")) {
      return;
    }
    // Ensure the parent can host an absolutely-positioned child without
    // breaking Idealista's existing layout.
    const cs = getComputedStyle(parent);
    if (cs.position === "static") parent.style.position = "relative";

    const badge = el(
      "div",
      "casa-intel-photo-overlay casa-intel-photo-" + (impact.tone || "neutral"),
    );
    badge.setAttribute("data-ci-master", masterIdFromUrl(impact.image_url) || "");
    const label =
      impact.tone === "helps"
        ? "Strong"
        : impact.tone === "hurts"
        ? "Weak"
        : "Neutral";
    badge.appendChild(el("span", "casa-intel-photo-tone", label));
    // Room label (zero-shot SigLIP classification) — only renders if the
    // backend provides it. Falls back gracefully if absent.
    if (impact.room_label) {
      badge.appendChild(
        el("span", "casa-intel-photo-room", impact.room_label),
      );
    }
    // Rank within the listing's photos. Intentionally NOT showing a €
    // delta: per-photo score is a model activation in the rent dimension,
    // not an actual rent contribution. Showing "+€418" would imply the
    // photo lifts the listing by that amount, which it doesn't.
    if (impact.rank_in_listing != null && total) {
      badge.appendChild(
        el(
          "span",
          "casa-intel-photo-delta",
          "#" + impact.rank_in_listing + " / " + total,
        ),
      );
    }
    // For weak photos, add a "↗ before/after" link that opens an
    // auto-enhance preview modal. This shows the user what a brightness/
    // contrast/saturation bump could look like.
    if (impact.tone === "hurts" && impact.image_url) {
      const enhanceLink = el(
        "button",
        "casa-intel-photo-enhance-link",
        "↗ see fix",
      );
      enhanceLink.setAttribute("type", "button");
      enhanceLink.title = "Preview an auto-brightness/contrast/saturation fix";
      // Capture-phase + immediate-propagation-stop so Idealista's gallery
      // lightbox handlers (attached higher up the DOM tree) don't fire on
      // the same click and try to open the lightbox over our modal.
      const swallow = (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
      };
      ["mousedown", "pointerdown", "touchstart"].forEach((evt) => {
        enhanceLink.addEventListener(evt, swallow, { capture: true });
      });
      enhanceLink.addEventListener(
        "click",
        (e) => {
          swallow(e);
          showPhotoEnhanceModal(impact);
        },
        { capture: true },
      );
      badge.appendChild(enhanceLink);
    }
    parent.appendChild(badge);
  }

  // ---- photo auto-enhance modal: side-by-side before / after ----------

  function showPhotoEnhanceModal(impact) {
    closePhotoEnhanceModal();

    const overlay = el("div", "casa-intel-enhance-overlay");
    overlay.id = "casa-intel-enhance-overlay";
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) closePhotoEnhanceModal();
    });

    const modal = el("div", "casa-intel-enhance-modal");
    overlay.appendChild(modal);

    // Header
    const header = el("div", "casa-intel-enhance-header");
    const title = el("div", "casa-intel-enhance-title", "Photo auto-fix preview");
    const subtitle = el(
      "div",
      "casa-intel-enhance-sub",
      "Brightness +20%, contrast +15%, saturation +15%. Pure CSS preview, " +
        "not a re-edit — this is what an enhanced version would roughly look like.",
    );
    header.appendChild(title);
    header.appendChild(subtitle);

    const closeBtn = el("button", "casa-intel-enhance-close", "×");
    closeBtn.setAttribute("type", "button");
    closeBtn.addEventListener("click", closePhotoEnhanceModal);
    header.appendChild(closeBtn);
    modal.appendChild(header);

    // Image grid
    const grid = el("div", "casa-intel-enhance-grid");

    function imagePane(label, filterValue) {
      const pane = el("div", "casa-intel-enhance-pane");
      pane.appendChild(el("div", "casa-intel-enhance-label", label));
      const wrap = el("div", "casa-intel-enhance-imgwrap");
      const img = document.createElement("img");
      img.className = "casa-intel-enhance-img";
      img.src = impact.image_url;
      img.alt = label;
      img.referrerPolicy = "no-referrer";
      if (filterValue) img.style.filter = filterValue;
      wrap.appendChild(img);
      pane.appendChild(wrap);
      return pane;
    }

    grid.appendChild(imagePane("Original", null));
    grid.appendChild(
      imagePane("Auto-enhanced", "brightness(1.2) contrast(1.15) saturate(1.15)"),
    );
    modal.appendChild(grid);

    // Strength slider — lets the user dial the enhancement up or down.
    const sliderRow = el("div", "casa-intel-enhance-slider-row");
    sliderRow.appendChild(
      el("span", "casa-intel-enhance-slider-label", "Strength:"),
    );
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "100";
    slider.value = "50";
    slider.className = "casa-intel-enhance-slider";
    sliderRow.appendChild(slider);
    const sliderVal = el("span", "casa-intel-enhance-slider-val", "50%");
    sliderRow.appendChild(sliderVal);
    modal.appendChild(sliderRow);

    const enhancedImg = grid.querySelectorAll(".casa-intel-enhance-img")[1];
    slider.addEventListener("input", () => {
      const pct = parseInt(slider.value, 10);
      sliderVal.textContent = pct + "%";
      // Scale 0..100 → 1.0..1.4 brightness / 1.0..1.3 contrast / 1.0..1.3 sat
      const b = 1 + (0.4 * pct) / 100;
      const c = 1 + (0.3 * pct) / 100;
      const s = 1 + (0.3 * pct) / 100;
      enhancedImg.style.filter = `brightness(${b.toFixed(2)}) contrast(${c.toFixed(2)}) saturate(${s.toFixed(2)})`;
    });

    // Hint
    modal.appendChild(
      el(
        "div",
        "casa-intel-enhance-hint",
        "Tip: in production photography, low-light interiors gain the most " +
          "from brightness + contrast. Reshoot in daylight if possible.",
      ),
    );

    document.body.appendChild(overlay);
    // Esc to close
    document.addEventListener("keydown", onEnhanceKeydown, { once: true });
  }

  function onEnhanceKeydown(e) {
    if (e.key === "Escape") closePhotoEnhanceModal();
  }

  function closePhotoEnhanceModal() {
    const existing = document.getElementById("casa-intel-enhance-overlay");
    if (existing) existing.remove();
  }

  function annotateGalleryPhotos(impacts) {
    const byMaster = new Map();
    for (const imp of impacts) {
      const id = masterIdFromUrl(imp.image_url);
      if (id) byMaster.set(id, imp);
    }
    if (byMaster.size === 0) return;
    const total = impacts.length;

    function pass() {
      const imgs = document.querySelectorAll("img");
      imgs.forEach((i) => {
        const sources = resolveImgSources(i);
        let matched = null;
        for (const s of sources) {
          const id = masterIdFromUrl(s);
          if (id && byMaster.has(id)) {
            matched = byMaster.get(id);
            break;
          }
        }
        if (matched) injectPhotoOverlay(i, matched, total);
      });
    }
    pass();
    // Re-run as new thumbnails lazy-load (Idealista advances the carousel)
    const obs = new MutationObserver(() => pass());
    obs.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["src", "data-src"] });
    // Auto-disconnect after 30s to avoid long-term work
    setTimeout(() => obs.disconnect(), 30000);
  }

  function findPriceBlock() {
    // Strategy 1: known selectors (most reliable when they exist)
    const selectors = [
      ".info-data-price",
      ".price-features__price",
      "[class*='info-data-price']",
      "span[class*='PriceStyled'], div[class*='PriceStyled']",
    ];
    for (const s of selectors) {
      const el = document.querySelector(s);
      if (el) return el;
    }
    // Strategy 2: walk the DOM looking for a leaf element that matches the
    // Spanish rent-price pattern (e.g. "1.800 €/mes"). Returns the closest
    // block-level ancestor so the panel docks naturally below it.
    const priceRegex = /\d[\d.]{2,}\s*€\s*\/\s*(?:mes|mo|mth)/i;
    const candidates = document.querySelectorAll(
      "h1, h2, h3, h4, span, p, strong, b, div",
    );
    for (const node of candidates) {
      if (node.children.length > 2) continue;
      const t = (node.textContent || "").trim();
      if (t.length < 6 || t.length > 80) continue;
      if (priceRegex.test(t)) {
        return node.closest("section, article, header, div[class*='main']") || node;
      }
    }
    return null;
  }

  function insertBelow(anchor, node) {
    if (!anchor) {
      document.body.appendChild(node);
      return;
    }
    // Ascend to a block-level parent so we don't land inline between spans.
    let block = anchor;
    while (
      block.parentElement &&
      getComputedStyle(block).display !== "block" &&
      getComputedStyle(block).display !== "flex"
    ) {
      block = block.parentElement;
    }
    if (block.parentNode) {
      block.parentNode.insertBefore(node, block.nextSibling);
    } else {
      document.body.appendChild(node);
    }
  }

  async function processDetailPage() {
    const payload = parseDetailPage();
    if (!payload) return;
    const anchor = findPriceBlock();

    // Rotating loading messages: each line names a real step the multimodal
    // backend is doing, so the wait quietly communicates the value prop
    // (we read photos AND text AND tabular). Cycle every 1.2s.
    const loadingMessages = [
      "Reading the listing…",
      "Encoding photos through SigLIP…",
      "Comparing to peer rents in the same zone…",
    ];
    let msgIdx = 0;
    showSkeleton(anchor, loadingMessages[0]);
    const rotateTimer = setInterval(() => {
      msgIdx = (msgIdx + 1) % loadingMessages.length;
      const toneEl = document.querySelector(
        "#casa-intel-panel.casa-intel-loading .casa-intel-tone",
      );
      if (toneEl) toneEl.textContent = loadingMessages[msgIdx];
    }, 1200);

    try {
      const resp = await fetch(API_URL + "/predict-live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (resp.ok) {
        injectDetailPanel(data, anchor, payload);
        if (data.per_photo_impact && data.per_photo_impact.length) {
          annotateGalleryPhotos(data.per_photo_impact);
        }
        // If this listing is already pinned, auto-append a fresh history
        // entry — every visit grows the history, which is what makes
        // "saved listings" useful over time.
        await autoAppendHistoryIfPinned(payload, data);
      } else {
        showSkeleton(anchor, "Backend returned " + resp.status);
      }
    } catch (e) {
      showSkeleton(
        anchor,
        "Backend unreachable: is uvicorn running on " + API_URL + "?",
      );
    } finally {
      clearInterval(rotateTimer);
    }
  }

  // ============== SAVED LISTINGS (chrome.storage) =====================

  function chromeStorageAvailable() {
    return typeof chrome !== "undefined" && chrome.storage && chrome.storage.local;
  }

  function getSavedListings() {
    return new Promise((resolve) => {
      if (!chromeStorageAvailable()) return resolve([]);
      chrome.storage.local.get([STORAGE_KEY_LISTINGS], (out) => {
        resolve(out[STORAGE_KEY_LISTINGS] || []);
      });
    });
  }

  function setSavedListings(arr) {
    return new Promise((resolve) => {
      if (!chromeStorageAvailable()) return resolve();
      chrome.storage.local.set({ [STORAGE_KEY_LISTINGS]: arr }, resolve);
    });
  }

  function snapshotFromResult(payload, result) {
    return {
      ts: Date.now(),
      asking: result.current_rent_eur != null ? result.current_rent_eur : null,
      predicted: result.predicted_rent_eur,
      diagnosis: result.diagnosis || null,
      mae: result.mae_eur != null ? result.mae_eur : null,
    };
  }

  async function pinListing(payload, result) {
    const all = await getSavedListings();
    const existing = all.findIndex((r) => r.listing_id === payload.listing_id);
    const snapshot = snapshotFromResult(payload, result);
    if (existing === -1) {
      all.push({
        listing_id: payload.listing_id,
        url: location.href,
        title: payload.location || ("Listing " + payload.listing_id),
        sqft: payload.sqft,
        rooms: payload.rooms,
        bathrooms: payload.bathrooms,
        history: [snapshot],
      });
    } else {
      all[existing].history.push(snapshot);
      all[existing].history = all[existing].history.slice(-HISTORY_CAP);
      all[existing].url = location.href;
      all[existing].title = payload.location || all[existing].title;
    }
    await setSavedListings(all);
    refreshLauncherBadge();
    return all;
  }

  async function unpinListing(listing_id) {
    const all = await getSavedListings();
    const filtered = all.filter((r) => r.listing_id !== listing_id);
    await setSavedListings(filtered);
    refreshLauncherBadge();
    return filtered;
  }

  async function isPinned(listing_id) {
    const all = await getSavedListings();
    return all.some((r) => r.listing_id === listing_id);
  }

  async function autoAppendHistoryIfPinned(payload, result) {
    if (!payload || !payload.listing_id) return;
    const all = await getSavedListings();
    const idx = all.findIndex((r) => r.listing_id === payload.listing_id);
    if (idx === -1) return;
    // Skip if the most recent snapshot is less than 60s old (page reload spam)
    const last = all[idx].history[all[idx].history.length - 1];
    if (last && Date.now() - last.ts < 60 * 1000) return;
    all[idx].history.push(snapshotFromResult(payload, result));
    all[idx].history = all[idx].history.slice(-HISTORY_CAP);
    await setSavedListings(all);
  }

  function buildPinButton(payload, result) {
    const btn = el("button", "casa-intel-pin-btn");
    btn.setAttribute("type", "button");
    btn.textContent = "📌 Pin";
    isPinned(payload.listing_id).then((pinned) => {
      if (pinned) {
        btn.textContent = "✓ Pinned";
        btn.classList.add("pinned");
      }
    });
    btn.addEventListener("click", async (e) => {
      e.preventDefault();
      e.stopPropagation();
      const wasPinned = await isPinned(payload.listing_id);
      if (wasPinned) {
        await unpinListing(payload.listing_id);
        btn.textContent = "📌 Pin";
        btn.classList.remove("pinned");
      } else {
        await pinListing(payload, result);
        btn.textContent = "✓ Pinned";
        btn.classList.add("pinned");
        // Subtle scale-pop to acknowledge the action
        btn.classList.remove("casa-intel-pin-pop");
        // Re-trigger by forcing a reflow before re-adding
        void btn.offsetWidth;
        btn.classList.add("casa-intel-pin-pop");
      }
    });
    return btn;
  }

  // ----- launcher (always-visible button bottom-right) + drawer ----------

  function refreshLauncherBadge() {
    const launcher = document.getElementById("casa-intel-saved-launcher");
    if (!launcher) return;
    getSavedListings().then((all) => {
      const newText = all.length > 0 ? `📌 Saved (${all.length})` : "📌 Saved";
      // Only animate when the count actually changes (avoid spurious pops on
      // every refresh).
      if (launcher.textContent !== newText && launcher.textContent !== "") {
        launcher.classList.remove("casa-intel-launcher-pop");
        void launcher.offsetWidth;
        launcher.classList.add("casa-intel-launcher-pop");
      }
      launcher.textContent = newText;
    });
  }

  function attachSavedListingsLauncher() {
    if (document.getElementById("casa-intel-saved-launcher")) return;
    const launcher = el("button", "casa-intel-saved-launcher");
    launcher.id = "casa-intel-saved-launcher";
    launcher.setAttribute("type", "button");
    launcher.textContent = "📌 Saved";
    launcher.addEventListener("click", async () => {
      const existing = document.getElementById("casa-intel-saved-drawer");
      if (existing) {
        existing.remove();
        return;
      }
      const drawer = await buildSavedListingsDrawer();
      document.body.appendChild(drawer);
    });
    document.body.appendChild(launcher);
    refreshLauncherBadge();
  }

  async function buildSavedListingsDrawer() {
    const all = await getSavedListings();
    const drawer = el("div", "casa-intel-saved-drawer");
    drawer.id = "casa-intel-saved-drawer";

    const header = el("div", "casa-intel-saved-header");
    const titleWrap = el("div", "casa-intel-saved-title-wrap");
    titleWrap.appendChild(el("div", "casa-intel-saved-title-main", "Saved listings"));
    titleWrap.appendChild(
      el(
        "div",
        "casa-intel-saved-title-sub",
        all.length === 0 ? "Pin listings from any detail page" : `${all.length} pinned`,
      ),
    );
    header.appendChild(titleWrap);

    const closeBtn = el("button", "casa-intel-saved-close", "×");
    closeBtn.setAttribute("type", "button");
    closeBtn.addEventListener("click", () => drawer.remove());
    header.appendChild(closeBtn);
    drawer.appendChild(header);

    if (all.length === 0) {
      const empty = el("div", "casa-intel-saved-empty");
      empty.appendChild(
        el(
          "div",
          "casa-intel-saved-empty-title",
          "Nothing saved yet.",
        ),
      );
      empty.appendChild(
        el(
          "div",
          "casa-intel-saved-empty-body",
          "Pin a listing on its detail page. We'll keep snapshots of how its peer estimate moves between visits, so you can see at a glance whether the model's view of it has shifted.",
        ),
      );
      drawer.appendChild(empty);
      return drawer;
    }

    const list = el("div", "casa-intel-saved-list");
    all.forEach((r) => list.appendChild(buildSavedRow(r, drawer)));
    drawer.appendChild(list);

    return drawer;
  }

  function buildSavedRow(record, drawer) {
    const row = el("div", "casa-intel-saved-row");

    const titleLink = document.createElement("a");
    titleLink.className = "casa-intel-saved-row-title";
    titleLink.textContent = record.title || "Listing " + record.listing_id;
    titleLink.href = record.url;
    titleLink.target = "_blank";
    titleLink.rel = "noopener noreferrer";
    row.appendChild(titleLink);

    const meta = el("div", "casa-intel-saved-row-meta");
    const metaParts = [];
    if (record.sqft) metaParts.push(record.sqft + " m²");
    if (record.rooms != null) metaParts.push(record.rooms + " hab.");
    if (record.bathrooms != null) metaParts.push(record.bathrooms + " baño");
    meta.textContent = metaParts.join(" · ");
    row.appendChild(meta);

    // History strip: dot per snapshot, color by diagnosis
    if (record.history && record.history.length > 0) {
      const histWrap = el("div", "casa-intel-saved-history");
      histWrap.appendChild(el("span", "casa-intel-saved-history-label", "history:"));
      record.history.forEach((h) => {
        const dot = el("span", "casa-intel-saved-history-dot");
        if (h.diagnosis) dot.classList.add("casa-intel-tone-" + h.diagnosis);
        const dateStr = new Date(h.ts).toLocaleDateString();
        const timeStr = new Date(h.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        const askingStr = h.asking != null ? "asking " + fmtEur(h.asking) + ", " : "";
        dot.title = `${dateStr} ${timeStr} — ${askingStr}predicted ${fmtEur(h.predicted)}`;
        histWrap.appendChild(dot);
      });
      row.appendChild(histWrap);

      const last = record.history[record.history.length - 1];
      const snap = el("div", "casa-intel-saved-row-snapshot");
      snap.appendChild(el("span", "casa-intel-saved-snap-label", "Predicted:"));
      snap.appendChild(el("span", "casa-intel-saved-snap-val", fmtEur(last.predicted)));
      if (last.asking != null) {
        snap.appendChild(el("span", "casa-intel-saved-snap-label", "· Asking:"));
        snap.appendChild(el("span", "casa-intel-saved-snap-val", fmtEur(last.asking)));
        const delta = last.asking - last.predicted;
        if (Math.abs(delta) > 50) {
          const sign = delta > 0 ? "+" : "";
          const cls = "casa-intel-saved-snap-delta " + (delta > 0 ? "neg" : "pos");
          snap.appendChild(el("span", cls, sign + fmtEur(delta)));
        }
      }
      row.appendChild(snap);

      // Price change between first and last snapshot, if any
      const first = record.history[0];
      if (record.history.length > 1 && first.predicted && last.predicted) {
        const change = last.predicted - first.predicted;
        if (Math.abs(change) > 20) {
          const sign = change > 0 ? "+" : "";
          row.appendChild(
            el(
              "div",
              "casa-intel-saved-row-change",
              `Model prediction has shifted ${sign}${fmtEur(change)} since you pinned this.`,
            ),
          );
        }
      }
    }

    const actions = el("div", "casa-intel-saved-row-actions");
    const openBtn = el("button", "casa-intel-saved-row-action", "Open listing");
    openBtn.setAttribute("type", "button");
    openBtn.addEventListener("click", () => {
      window.open(record.url, "_blank", "noopener,noreferrer");
    });
    actions.appendChild(openBtn);

    const refreshBtn = el(
      "button",
      "casa-intel-saved-row-action",
      "↻ Re-evaluate (cached)",
    );
    refreshBtn.setAttribute("type", "button");
    refreshBtn.addEventListener("click", async () => {
      refreshBtn.disabled = true;
      refreshBtn.textContent = "Re-evaluating…";
      await reEvaluateCachedListing(record);
      drawer.remove();
      const fresh = await buildSavedListingsDrawer();
      document.body.appendChild(fresh);
    });
    actions.appendChild(refreshBtn);

    const unpinBtn = el(
      "button",
      "casa-intel-saved-row-action casa-intel-saved-row-unpin",
      "Unpin",
    );
    unpinBtn.setAttribute("type", "button");
    unpinBtn.addEventListener("click", async () => {
      await unpinListing(record.listing_id);
      drawer.remove();
      const fresh = await buildSavedListingsDrawer();
      document.body.appendChild(fresh);
    });
    actions.appendChild(unpinBtn);

    row.appendChild(actions);
    return row;
  }

  async function reEvaluateCachedListing(record) {
    // On-demand re-evaluation against the local backend's cache. We don't
    // re-scrape Idealista (that needs the user opening the page), but we do
    // recompute the model's view, which is meaningful because the cached
    // prediction can shift if the model has been retrained or recalibrated.
    try {
      const resp = await fetch(API_URL + "/predict-live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          listing_id: record.listing_id,
          sqft: record.sqft || 0,
          rooms: record.rooms,
          bathrooms: record.bathrooms,
          mode: "tabular",
        }),
      });
      if (!resp.ok) return null;
      const data = await resp.json();
      const all = await getSavedListings();
      const idx = all.findIndex((r) => r.listing_id === record.listing_id);
      if (idx === -1) return null;
      all[idx].history.push({
        ts: Date.now(),
        asking: null,
        predicted: data.predicted_rent_eur,
        diagnosis: data.diagnosis || null,
        mae: data.mae_eur || null,
      });
      all[idx].history = all[idx].history.slice(-HISTORY_CAP);
      await setSavedListings(all);
      return data;
    } catch (e) {
      return null;
    }
  }

  // ============== WHAT-IF SIMULATOR ====================================

  function buildWhatIfPanel(result, payload) {
    const wrap = el("div", "casa-intel-whatif");
    wrap.appendChild(el("div", "casa-intel-details-header", "What if?"));
    wrap.appendChild(
      el(
        "div",
        "casa-intel-whatif-hint",
        "Toggle features and resize to see how peer-expected rent changes",
      ),
    );

    // Working copy of the payload. Recall the API in tabular mode for speed
    // (image extraction takes seconds; toggles need to feel instant).
    const state = { ...payload, mode: "tabular" };

    // Anchor the simulator on the headline number (gb_all, full multimodal).
    // The simulator hits gb_tab on each toggle for speed, so its raw output
    // is naturally lower than the headline by the photo+description
    // contribution. We apply that contribution as a constant offset so the
    // displayed numbers stay in headline space — toggling and untoggling
    // an amenity correctly returns to the headline, while real toggle
    // changes still produce a meaningful delta in the tabular dimension.
    const baseEur = result.predicted_rent_eur;
    const tabBaseEur =
      result.breakdown && result.breakdown.tabular_eur != null
        ? result.breakdown.tabular_eur
        : result.predicted_rent_eur;
    const photoTextOffset = baseEur - tabBaseEur;

    const togglesRow = el("div", "casa-intel-whatif-toggles");
    AMENITY_KEYS.forEach((k) => {
      const btn = el("button", "casa-intel-whatif-toggle");
      btn.setAttribute("type", "button");
      btn.setAttribute("data-amenity", k);
      btn.textContent = AMENITY_LABELS[k];
      if (state[k]) btn.classList.add("active");
      btn.addEventListener("click", () => {
        state[k] = !state[k];
        btn.classList.toggle("active", state[k]);
        debouncedRecall();
      });
      togglesRow.appendChild(btn);
    });
    wrap.appendChild(togglesRow);

    // Sqft slider
    const sliderRow = el("div", "casa-intel-whatif-slider-row");
    sliderRow.appendChild(
      el("span", "casa-intel-whatif-slider-label", "Size:"),
    );
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "20";
    slider.max = "300";
    slider.step = "5";
    slider.value = String(state.sqft || 60);
    slider.className = "casa-intel-whatif-slider";
    const sliderVal = el(
      "span",
      "casa-intel-whatif-slider-val",
      (state.sqft || 60) + " m²",
    );
    slider.addEventListener("input", () => {
      state.sqft = parseInt(slider.value, 10);
      sliderVal.textContent = state.sqft + " m²";
      debouncedRecall();
    });
    sliderRow.appendChild(slider);
    sliderRow.appendChild(sliderVal);
    wrap.appendChild(sliderRow);

    // Result line
    const resultRow = el("div", "casa-intel-whatif-result");
    resultRow.appendChild(
      el("span", "casa-intel-whatif-result-label", "New peer estimate:"),
    );
    const resultVal = el(
      "span",
      "casa-intel-whatif-result-val",
      fmtEur(baseEur),
    );
    resultRow.appendChild(resultVal);
    const resultDelta = el("span", "casa-intel-whatif-result-delta", "(baseline)");
    resultRow.appendChild(resultDelta);
    wrap.appendChild(resultRow);

    // Reset button
    const resetBtn = el("button", "casa-intel-whatif-reset", "Reset to actual");
    resetBtn.setAttribute("type", "button");
    resetBtn.addEventListener("click", () => {
      // Restore from original payload
      AMENITY_KEYS.forEach((k) => {
        state[k] = payload[k];
        const btn = togglesRow.querySelector(`[data-amenity="${k}"]`);
        if (btn) btn.classList.toggle("active", !!state[k]);
      });
      state.sqft = payload.sqft;
      slider.value = String(state.sqft || 60);
      sliderVal.textContent = (state.sqft || 60) + " m²";
      animateEur(resultVal, displayedEur, baseEur, 500);
      displayedEur = baseEur;
      resultDelta.textContent = "(baseline)";
      resultDelta.className = "casa-intel-whatif-result-delta";
    });
    wrap.appendChild(resetBtn);

    // Debounced API recall. Tracks the last-displayed value so the animation
    // rolls from where the user is looking, not from baseline every time.
    let timer = null;
    let displayedEur = baseEur;
    function debouncedRecall() {
      if (timer) clearTimeout(timer);
      resultVal.classList.add("loading");
      timer = setTimeout(async () => {
        try {
          const resp = await fetch(API_URL + "/predict-live", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(state),
          });
          if (!resp.ok) {
            resultVal.classList.remove("loading");
            return;
          }
          const data = await resp.json();
          // The API returned a tabular-only prediction. Shift it back into
          // headline space by adding the constant photo+text contribution.
          // Now no-toggle-change lands exactly on baseEur, and toggle deltas
          // are honest measurements of the tabular feature change.
          const newEur = data.predicted_rent_eur + photoTextOffset;
          const delta = newEur - baseEur;
          resultVal.classList.remove("loading");
          // Animate from what's currently shown to the new value
          animateEur(resultVal, displayedEur, newEur, 500);
          // Brief color flash on meaningful change. Threshold of €30 so we
          // don't flash on rounding noise.
          const stepDelta = newEur - displayedEur;
          if (Math.abs(stepDelta) >= 30) {
            const flashClass =
              stepDelta > 0 ? "casa-intel-flash-up" : "casa-intel-flash-down";
            resultVal.classList.remove("casa-intel-flash-up", "casa-intel-flash-down");
            void resultVal.offsetWidth;
            resultVal.classList.add(flashClass);
          }
          displayedEur = newEur;
          if (Math.abs(delta) < 5) {
            resultDelta.textContent = "(matches your listing)";
            resultDelta.className = "casa-intel-whatif-result-delta";
          } else {
            const sign = delta > 0 ? "+" : "";
            resultDelta.textContent = "(" + sign + fmtEur(delta) + " vs your listing)";
            resultDelta.className =
              "casa-intel-whatif-result-delta " +
              (delta > 0 ? "casa-intel-delta-pos" : "casa-intel-delta-neg");
          }
        } catch (e) {
          resultVal.classList.remove("loading");
        }
      }, 450);
    }

    return wrap;
  }

  // ============== NEGOTIATION MESSAGE GENERATOR ========================

  function pctOver(asking, predicted) {
    if (!asking || !predicted || !Number.isFinite(asking) || !Number.isFinite(predicted)) {
      return null;
    }
    return ((asking - predicted) / predicted) * 100;
  }

  function negotiationTier(pct) {
    if (pct == null) return null;
    if (pct < 5) return null; // not over enough to bother
    if (pct < 10) return "slight";
    if (pct < 25) return "notable";
    return null; // >25% is usually a typo or luxury outlier — show warning, not message
  }

  function approxLowerBound(predicted) {
    return Math.round(predicted * 0.93);
  }
  function approxUpperBound(predicted) {
    return Math.round(predicted * 1.07);
  }

  function negotiationTemplate(tier, payload, result) {
    const sqft = payload.sqft;
    const rooms = payload.rooms;
    const zoneText = (payload.location || "la zona").replace(/, Madrid$/, "");
    const asking = result.current_rent_eur;
    const predicted = Math.round(result.predicted_rent_eur);
    const lower = approxLowerBound(predicted);
    const upper = approxUpperBound(predicted);
    const roomsBlock = rooms != null ? rooms + " habitaciones, " : "";

    if (tier === "slight") {
      return (
        "Hola,\n\n" +
        "Estoy interesado/a en su anuncio (" + roomsBlock + sqft + " m² en " + zoneText + ").\n\n" +
        "He estado comparando viviendas similares en la misma zona y veo que el precio " +
        "podría ajustarse ligeramente. ¿Habría flexibilidad sobre los " + asking + " €/mes actuales? " +
        "Estaría dispuesto/a a firmar rápido si llegamos a un acuerdo razonable.\n\n" +
        "Gracias y un saludo."
      );
    }

    if (tier === "notable") {
      return (
        "Hola,\n\n" +
        "Le escribo en relación a su anuncio (" + roomsBlock + sqft + " m², " + zoneText + ").\n\n" +
        "Tras analizar viviendas comparables en " + zoneText + ", propiedades similares " +
        "se alquilan en torno a €" + lower + "–€" + upper + " al mes. El precio actual de " +
        asking + " €/mes está claramente por encima de este rango.\n\n" +
        "¿Sería posible considerar una rebaja para situarlo dentro del rango de mercado? " +
        "Estoy preparado/a para firmar de inmediato si encontramos un punto de acuerdo.\n\n" +
        "Quedo a la espera de su respuesta. Un saludo."
      );
    }

    return null;
  }

  function buildNegotiationSection(payload, result) {
    const pct = pctOver(result.current_rent_eur, result.predicted_rent_eur);
    const tier = negotiationTier(pct);

    // Outlier warning when asking is way above peer (>25%): probably a typo
    // or a luxury listing the model can't see. Don't generate a message.
    if (pct != null && pct >= 25) {
      const wrap = el("div", "casa-intel-negotiate");
      wrap.appendChild(
        el("div", "casa-intel-details-header", "Outlier warning"),
      );
      wrap.appendChild(
        el(
          "div",
          "casa-intel-negotiate-warning",
          "The asking price is " + Math.round(pct) + "% above the peer estimate. " +
            "This is far enough above market that it's likely a luxury outlier the " +
            "model can't see, or a price typo. We're not generating a negotiation " +
            "message for outliers like this — it would over-promise.",
        ),
      );
      return wrap;
    }

    if (!tier) return null;

    const message = negotiationTemplate(tier, payload, result);
    if (!message) return null;

    const tierLabels = {
      slight: "Slightly above peer (1–10%): polite query about flexibility",
      notable: "Notably above peer (10–25%): direct ask with comparables",
    };

    const wrap = el("div", "casa-intel-negotiate");
    wrap.appendChild(el("div", "casa-intel-details-header", "Negotiation message"));
    wrap.appendChild(el("div", "casa-intel-negotiate-tier", tierLabels[tier]));

    const textarea = document.createElement("textarea");
    textarea.className = "casa-intel-negotiate-textarea";
    textarea.value = message;
    textarea.rows = 9;
    textarea.spellcheck = false;
    wrap.appendChild(textarea);

    const actions = el("div", "casa-intel-negotiate-actions");
    const copyBtn = el("button", "casa-intel-negotiate-copy", "Copy to clipboard");
    copyBtn.setAttribute("type", "button");
    copyBtn.addEventListener("click", async () => {
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(textarea.value);
        } else {
          textarea.select();
          document.execCommand("copy");
        }
        copyBtn.textContent = "Copied ✓";
        copyBtn.classList.add("copied");
        setTimeout(() => {
          copyBtn.textContent = "Copy to clipboard";
          copyBtn.classList.remove("copied");
        }, 1800);
      } catch (e) {
        copyBtn.textContent = "Copy failed";
      }
    });
    actions.appendChild(copyBtn);

    wrap.appendChild(actions);
    wrap.appendChild(
      el(
        "div",
        "casa-intel-negotiate-disclaimer",
        "Edit the message before sending. We don't actually send anything — this is just a starting draft.",
      ),
    );

    return wrap;
  }

  // ============== PDF EXPORT ==========================================

  // Per-room photography tips. Used to turn "weak photo" into "weak kitchen
  // photo, here's how to fix it" — the critique only ships the room-specific
  // tip if the backend served a room_label; otherwise falls back to generic.
  const PHOTO_TIPS = {
    Kitchen:        "Try a daytime shot with the lights on, neutral tones, and the surfaces clear.",
    Bedroom:        "Wide-angle, made bed, natural light. Avoid clutter on the floor.",
    Bathroom:       "Wide-angle to make the space feel bigger, lights on, no toiletries on counters.",
    "Living room":  "Open the curtains, turn on lights, declutter sofa and coffee table.",
    "Dining room":  "A clear table reads cleaner. Daytime light works best.",
    Terrace:        "Catch golden-hour light if possible, declutter plants, show the view.",
    Exterior:       "Frame the building's best angle. Daytime, no cars in the way.",
    Hallway:        "Hallways always feel narrow — wide-angle helps a lot.",
    Storage:        "Clear out personal items first, show the empty volume.",
    "Floor plan":   "Floor plans don't lift rent on their own. Make sure the room photos shine.",
    Garage:         "Wide-angle, lights on, no clutter on the floor.",
    Pool:           "Daytime, clear water, no debris.",
  };
  const FALLBACK_TIP = "Try shooting in daylight with better contrast and a wide-angle lens.";

  function exportButton(payload, result) {
    const btn = el("button", "casa-intel-export-btn");
    btn.setAttribute("type", "button");
    btn.textContent = "📄 Export PDF";
    btn.title = "Download a one-page analysis of this listing";
    btn.addEventListener("click", async (e) => {
      e.preventDefault();
      e.stopPropagation();
      btn.disabled = true;
      btn.textContent = "Generating…";
      try {
        await generateAnalysisPDF(payload, result);
        btn.textContent = "✓ Downloaded";
        setTimeout(() => {
          btn.textContent = "📄 Export PDF";
          btn.disabled = false;
        }, 1800);
      } catch (err) {
        console.error("CasaIntel PDF generation failed:", err);
        btn.textContent = "PDF failed";
        setTimeout(() => {
          btn.textContent = "📄 Export PDF";
          btn.disabled = false;
        }, 2400);
      }
    });
    return btn;
  }

  // ---- PDF layout helpers --------------------------------------------

  // Wrap text to a column width using jsPDF's splitTextToSize, then write line
  // by line. Returns the y-coordinate after the last line.
  function pdfText(doc, text, x, y, maxWidth, lineHeight) {
    const lines = doc.splitTextToSize(text, maxWidth);
    for (const line of lines) {
      doc.text(line, x, y);
      y += lineHeight;
    }
    return y;
  }

  function pdfHr(doc, x1, y, x2, color) {
    doc.setDrawColor(...color);
    doc.setLineWidth(0.4);
    doc.line(x1, y, x2, y);
  }

  function pdfSectionHeader(doc, label, x, y) {
    doc.setFont("helvetica", "bold");
    doc.setTextColor(15, 118, 110); // teal
    doc.setFontSize(9);
    doc.text(label.toUpperCase(), x, y);
    doc.setTextColor(15, 23, 42); // slate
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    return y + 5;
  }

  // Group photos into "what works" and "what needs work" buckets and craft
  // human-readable critiques. If room_label is available, the critique reads
  // "the kitchen photo (#3 of 22)..."; if not, it falls back to "photo #3".
  function buildPhotoCritiques(impacts) {
    if (!impacts || impacts.length === 0) return { strengths: [], weaknesses: [] };

    const strong = impacts
      .filter((p) => p.tone === "helps")
      .sort((a, b) => a.rank_in_listing - b.rank_in_listing);
    const weak = impacts
      .filter((p) => p.tone === "hurts")
      .sort((a, b) => b.rank_in_listing - a.rank_in_listing);
    const total = impacts.length;

    const strengths = strong.slice(0, 3).map((p) => {
      const rankPart = `(#${p.rank_in_listing} of ${total})`;
      if (p.room_label) {
        const noun = p.room_label.toLowerCase() + " photo";
        if (p.rank_in_listing === 1) {
          return `The ${noun} ${rankPart} is the strongest in this listing — lead with it.`;
        }
        return `The ${noun} ${rankPart} is among the strongest. Keep it prominent.`;
      }
      // No room label (listing not in our cached dataset). Cleaner copy
      // that doesn't lean on "the photo".
      if (p.rank_in_listing === 1) {
        return `Photo #${p.rank_in_listing} is the strongest in this listing — lead with it.`;
      }
      return `Photo #${p.rank_in_listing} of ${total} ranks among the strongest. Keep it prominent.`;
    });

    const weaknesses = weak.slice(0, 4).map((p) => {
      const rankPart = `(#${p.rank_in_listing} of ${total})`;
      if (p.room_label) {
        const subject = p.room_label + " photo";
        const tip = PHOTO_TIPS[p.room_label] || FALLBACK_TIP;
        return `${subject} ${rankPart} ranks low in this listing. ${tip}`;
      }
      // No room label fallback
      return `Photo #${p.rank_in_listing} of ${total} ranks low in this listing. ${FALLBACK_TIP}`;
    });

    return { strengths, weaknesses };
  }

  function buildAmenityList(payload) {
    const present = AMENITY_KEYS.filter((k) => payload[k]).map(
      (k) => AMENITY_LABELS[k],
    );
    return present.length > 0 ? present.join(", ") : "None marked";
  }

  function fmtVerdict(result) {
    return (
      {
        overpriced: "Asking is above peer estimate",
        underpriced: "Asking is below peer estimate",
        fair: "Asking is consistent with peers",
        unknown: "Peer estimate available",
      }[result.diagnosis] || "Peer estimate available"
    );
  }

  function todayIso() {
    return new Date().toISOString().split("T")[0];
  }

  // ---- main PDF generation -------------------------------------------

  async function generateAnalysisPDF(payload, result) {
    if (!window.jspdf || !window.jspdf.jsPDF) {
      throw new Error("jsPDF not loaded");
    }
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({ unit: "mm", format: "a4" });

    // A4: 210 x 297mm. Use 18mm margins.
    const M = 18;
    const W = 210 - 2 * M;
    let y = M;

    // ---- Title bar ----
    doc.setFillColor(15, 118, 110); // teal
    doc.rect(0, 0, 210, 14, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.text("CasaIntel", M, 9);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.text("Listing analysis", M + 30, 9);
    doc.text(todayIso(), 210 - M, 9, { align: "right" });

    y = 24;

    // ---- Listing header ----
    doc.setTextColor(15, 23, 42);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    const title = (payload.location || "Listing " + payload.listing_id).replace(
      /, Madrid$/,
      "",
    );
    y = pdfText(doc, title, M, y, W, 6);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(100, 116, 139);
    const facts = [
      payload.sqft ? `${payload.sqft} m²` : null,
      payload.rooms != null ? `${payload.rooms} hab.` : null,
      payload.bathrooms != null ? `${payload.bathrooms} baño` : null,
      result.current_rent_eur != null
        ? `Asking ${fmtEur(result.current_rent_eur)}/month`
        : null,
    ]
      .filter(Boolean)
      .join("  ·  ");
    if (facts) {
      doc.text(facts, M, y);
      y += 4.5;
    }
    if (payload.location) {
      doc.text("URL: " + location.href, M, y);
      y += 4.5;
    }
    y += 2;

    // ---- Peer-expected rent block ----
    pdfHr(doc, M, y, 210 - M, [203, 213, 225]);
    y += 5;
    y = pdfSectionHeader(doc, "Peer-expected rent", M, y);

    const bounds = intervalBounds(result);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(20);
    doc.setTextColor(15, 23, 42);
    doc.text(fmtEur(result.predicted_rent_eur), M, y + 5);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(100, 116, 139);
    if (bounds) {
      doc.text(
        `80% range: ${fmtRange(bounds)}` +
          (bounds.source === "approx" ? " (approx.)" : ""),
        M + 50,
        y + 5,
      );
    }
    y += 10;

    // Verdict line
    doc.setFontSize(10);
    doc.setTextColor(15, 23, 42);
    y = pdfText(doc, fmtVerdict(result) + ".", M, y, W, 5);

    if (result.delta_vs_current_eur != null) {
      const delta = result.delta_vs_current_eur;
      const sign = delta > 0 ? "+" : "";
      doc.setTextColor(100, 116, 139);
      doc.setFontSize(9);
      const note =
        bounds && result.current_rent_eur != null
          ? result.current_rent_eur >= bounds.lower &&
            result.current_rent_eur <= bounds.upper
            ? `Asking ${fmtEur(result.current_rent_eur)} sits inside the 80% range. Consistent with peers.`
            : `Asking ${fmtEur(result.current_rent_eur)} sits outside the 80% range — a real pricing difference, not measurement noise.`
          : `Gap between asking and peer estimate: ${sign}${fmtEur(delta)}.`;
      y = pdfText(doc, note, M, y, W, 4.5);
    }
    y += 3;

    // ---- Feature breakdown ----
    pdfHr(doc, M, y, 210 - M, [203, 213, 225]);
    y += 5;
    y = pdfSectionHeader(doc, "Feature-by-feature breakdown", M, y);

    const b = result.breakdown;
    if (b) {
      doc.setFontSize(9);
      const rows = [];
      rows.push([
        "Tabular baseline",
        fmtEur(b.tabular_eur),
        b.tabular_note || "size, rooms, zone, amenities",
      ]);
      if (b.photos_delta_eur != null) {
        const sign = b.photos_delta_eur > 0 ? "+" : "";
        rows.push([
          "+ photos (SigLIP)",
          sign + fmtEur(b.photos_delta_eur),
          b.photos_note || "",
        ]);
      }
      if (b.text_delta_eur != null) {
        const sign = b.text_delta_eur > 0 ? "+" : "";
        rows.push([
          "+ description (MiniLM)",
          sign + fmtEur(b.text_delta_eur),
          b.text_note || "",
        ]);
      }
      rows.push([
        "Full model",
        fmtEur(b.full_eur ?? result.predicted_rent_eur),
        "",
      ]);

      const labelX = M;
      const valX = M + 56;
      const noteX = M + 82;
      for (const [label, val, note] of rows) {
        doc.setFont("helvetica", "normal");
        doc.setTextColor(71, 85, 105);
        doc.text(label, labelX, y);
        doc.setFont("helvetica", "bold");
        doc.setTextColor(15, 23, 42);
        doc.text(val, valX, y);
        if (note) {
          doc.setFont("helvetica", "italic");
          doc.setTextColor(148, 163, 184);
          doc.text(note, noteX, y, { maxWidth: W - 64 });
        }
        y += 5;
      }
    } else {
      doc.setFontSize(9);
      doc.setTextColor(100, 116, 139);
      y = pdfText(
        doc,
        "Breakdown not available — model returned point estimate only.",
        M,
        y,
        W,
        5,
      );
    }
    y += 3;

    // ---- Amenities + tabular ----
    pdfHr(doc, M, y, 210 - M, [203, 213, 225]);
    y += 5;
    y = pdfSectionHeader(doc, "Listing details", M, y);
    doc.setFontSize(9);
    doc.setTextColor(71, 85, 105);
    doc.text("Amenities:", M, y);
    doc.setTextColor(15, 23, 42);
    y = pdfText(doc, buildAmenityList(payload), M + 22, y, W - 22, 4.5);
    y += 2;

    // ---- Photo critiques ----
    const impacts = result.per_photo_impact || [];
    if (impacts.length > 0) {
      pdfHr(doc, M, y, 210 - M, [203, 213, 225]);
      y += 5;
      y = pdfSectionHeader(
        doc,
        `Photo-by-photo analysis (${impacts.length} photos)`,
        M,
        y,
      );

      const { strengths, weaknesses } = buildPhotoCritiques(impacts);

      doc.setFontSize(9.5);
      if (strengths.length > 0) {
        doc.setFont("helvetica", "bold");
        doc.setTextColor(4, 120, 87);
        doc.text("What works", M, y);
        y += 5;
        doc.setFont("helvetica", "normal");
        doc.setTextColor(15, 23, 42);
        for (const s of strengths) {
          y = pdfText(doc, "•  " + s, M + 2, y, W - 2, 4.6);
          y += 1;
        }
        y += 2;
      }
      if (weaknesses.length > 0) {
        doc.setFont("helvetica", "bold");
        doc.setTextColor(185, 28, 28);
        doc.text("What needs work", M, y);
        y += 5;
        doc.setFont("helvetica", "normal");
        doc.setTextColor(15, 23, 42);
        for (const w of weaknesses) {
          y = pdfText(doc, "•  " + w, M + 2, y, W - 2, 4.6);
          y += 1;
        }
        y += 2;
      }
      if (strengths.length === 0 && weaknesses.length === 0) {
        doc.setFont("helvetica", "italic");
        doc.setTextColor(100, 116, 139);
        y = pdfText(
          doc,
          "All photos rank within the model's neutral band — none stand out as strong or weak.",
          M,
          y,
          W,
          4.6,
        );
      }
    }

    // ---- Footer ----
    const footerY = 297 - 14;
    pdfHr(doc, M, footerY, 210 - M, [226, 232, 240]);
    doc.setFont("helvetica", "italic");
    doc.setFontSize(7.5);
    doc.setTextColor(148, 163, 184);
    pdfText(
      doc,
      "Generated by CasaIntel · multimodal peer-rent for Madrid · " +
        "SigLIP image embeddings + multilingual MiniLM + gradient boosting on log-rent · " +
        "R² 0.88, MAE €274, 5-fold CV on 6,047 listings · The 80% range reflects empirical residual quantiles, not a Bayesian credible interval.",
      M,
      footerY + 4,
      W,
      3.2,
    );

    // ---- Save ----
    const safeId = (payload.listing_id || "listing").replace(/[^0-9a-z_-]/gi, "");
    doc.save(`casaintel-${safeId}-${todayIso()}.pdf`);
  }

  // --------------------- boot ------------------------------------------

  // Brand-appropriate console hello for anyone with devtools open. Dry, factual,
  // ends with a small invitation. No emojis in the body — keeps it grown-up.
  try {
    console.log(
      "%cCasaIntel%c  multimodal peer-rent for Madrid",
      "background:#0f766e;color:#fff;padding:2px 8px;border-radius:4px;font-weight:700;letter-spacing:0.04em;",
      "color:#475569;font-weight:500;",
    );
    console.log(
      "%cReading the source? Nice. SigLIP image embeddings + multilingual MiniLM + gradient boosting on log-rent. R² 0.88, MAE €274, 5-fold CV on 6,047 listings. The lower number is the honest one.",
      "color:#64748b;font-style:italic;",
    );
  } catch (_) {
    /* no-op if console is restricted */
  }

  // Saved listings launcher renders on every page (search and detail), so
  // the user can always reach their pinned set.
  attachSavedListingsLauncher();

  const pageType = getPageType();
  if (pageType === "list") {
    processListPage();
    const obs = new MutationObserver(() => processListPage());
    obs.observe(document.body, { childList: true, subtree: true });
  } else if (pageType === "detail") {
    processDetailPage();
  }
})();
