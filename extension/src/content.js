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
    return rounded < 0 ? "−" + formatted : formatted;
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

      // Total row
      const totalRow = el("div", "casa-intel-details-row casa-intel-details-total");
      totalRow.appendChild(el("span", "casa-intel-details-key", "Full model"));
      totalRow.appendChild(
        el(
          "span",
          "casa-intel-details-val",
          fmtEur(b.full_eur ?? result.predicted_rent_eur) +
            " ± " +
            fmtEur(result.mae_eur),
        ),
      );
      wrap.appendChild(totalRow);
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

    // Noise guidance
    const delta = result.delta_vs_current_eur;
    if (delta != null && Number.isFinite(delta)) {
      const mae = result.mae_eur || 443;
      const inNoise = Math.abs(delta) < mae;
      const note = el(
        "div",
        "casa-intel-details-note",
        inNoise
          ? "The " +
              fmtEur(Math.abs(delta)) +
              " gap is smaller than the model's typical ±" +
              fmtEur(mae) +
              " error. Too close to call. This listing is priced in line with peers."
          : "The " +
              fmtEur(Math.abs(delta)) +
              " gap is larger than the model's typical ±" +
              fmtEur(mae) +
              " error. A real pricing difference, not measurement error.",
      );
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

    const big = el("div", "casa-intel-big", fmtEur(result.predicted_rent_eur) + " ");
    const mae = el("span", "casa-intel-mae", "± " + fmtEur(result.mae_eur));
    big.appendChild(mae);
    panel.appendChild(big);

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
    parent.appendChild(badge);
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
    showSkeleton(anchor, "Scoring photos…");

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
      }
    });
    return btn;
  }

  // ----- launcher (always-visible button bottom-right) + drawer ----------

  function refreshLauncherBadge() {
    const launcher = document.getElementById("casa-intel-saved-launcher");
    if (!launcher) return;
    getSavedListings().then((all) => {
      launcher.textContent = all.length > 0 ? `📌 Saved (${all.length})` : "📌 Saved";
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
      drawer.appendChild(
        el(
          "div",
          "casa-intel-saved-empty",
          "No pinned listings yet. Click the 📌 Pin button on any listing's detail panel.",
        ),
      );
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

    const baseEur = result.predicted_rent_eur;

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
      resultVal.textContent = fmtEur(baseEur);
      resultDelta.textContent = "(baseline)";
      resultDelta.className = "casa-intel-whatif-result-delta";
    });
    wrap.appendChild(resetBtn);

    // Debounced API recall
    let timer = null;
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
          const newEur = data.predicted_rent_eur;
          const delta = newEur - baseEur;
          resultVal.textContent = fmtEur(newEur);
          resultVal.classList.remove("loading");
          if (Math.abs(delta) < 5) {
            resultDelta.textContent = "(no meaningful change)";
            resultDelta.className = "casa-intel-whatif-result-delta";
          } else {
            const sign = delta > 0 ? "+" : "";
            resultDelta.textContent = "(" + sign + fmtEur(delta) + " vs original)";
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

  // --------------------- boot ------------------------------------------

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
