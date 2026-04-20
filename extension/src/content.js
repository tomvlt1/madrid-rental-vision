/*
 * CasaIntel content script — runs on Idealista Madrid pages.
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
    if (n == null || !Number.isFinite(n)) return "—";
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
    // colored diagnosis here — it would mis-label premium outliers that
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
   * Pull all Idealista image URLs off the page — scans both instantiated
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

  function buildDetailPanel(result) {
    const panel = el("div", "casa-intel-detail-panel casa-intel-" + result.diagnosis);
    panel.id = "casa-intel-panel";

    panel.appendChild(
      el("div", "casa-intel-header", "CasaIntel · " + (result.cached ? "cached" : "live")),
    );
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

    return panel;
  }

  function injectDetailPanel(result, anchor) {
    const existing = document.querySelector("#casa-intel-panel");
    if (existing) existing.remove();
    const panel = buildDetailPanel(result);
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
    // delta — per-photo score is a model activation in the rent dimension,
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
        injectDetailPanel(data, anchor);
        if (data.per_photo_impact && data.per_photo_impact.length) {
          annotateGalleryPhotos(data.per_photo_impact);
        }
      } else {
        showSkeleton(anchor, "Backend returned " + resp.status);
      }
    } catch (e) {
      showSkeleton(
        anchor,
        "Backend unreachable — is uvicorn running on " + API_URL + "?",
      );
    }
  }

  // --------------------- boot ------------------------------------------

  const pageType = getPageType();
  if (pageType === "list") {
    processListPage();
    const obs = new MutationObserver(() => processListPage());
    obs.observe(document.body, { childList: true, subtree: true });
  } else if (pageType === "detail") {
    processDetailPage();
  }
})();
