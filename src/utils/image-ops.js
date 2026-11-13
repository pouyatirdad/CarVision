// Pure pixel operations used to prepare rear-badge crops for OCR. They work on plain typed arrays (RGBA from
// canvas getImageData, or 1-channel grayscale), have no DOM dependency and are unit-tested under Node.

/** RGBA (Uint8ClampedArray) -> grayscale Uint8Array, ITU-R BT.601 luma. */
export function toGray(rgba) {
  const n = rgba.length >> 2;
  const g = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) g[i] = (rgba[j] * 299 + rgba[j + 1] * 587 + rgba[j + 2] * 114 + 500) / 1000;
  return g;
}

/**
 * Percentile contrast stretch: maps the `low`..`high` percentile range to 0..255. Robust to a few very bright
 * (chrome reflections) or very dark pixels, which is what makes small chrome badges readable.
 */
export function stretchContrast(gray, low = 0.01, high = 0.99) {
  const hist = new Uint32Array(256);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;
  const lo = percentile(hist, gray.length, low);
  const hi = percentile(hist, gray.length, high);
  const out = new Uint8Array(gray.length);
  if (hi <= lo) {
    out.set(gray);
    return out;
  }
  const k = 255 / (hi - lo);
  for (let i = 0; i < gray.length; i++) {
    const v = (gray[i] - lo) * k;
    out[i] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  return out;
}

function percentile(hist, total, p) {
  const target = Math.max(0, Math.min(total - 1, Math.floor(total * p)));
  let acc = 0;
  for (let v = 0; v < 256; v++) {
    acc += hist[v];
    if (acc > target) return v;
  }
  return 255;
}

/** Unsharp mask with a 3x3 box blur: out = gray + amount * (gray - blur). Edges are clamped. */
export function sharpen(gray, width, height, amount = 1) {
  const out = new Uint8Array(gray.length);
  for (let y = 0; y < height; y++) {
    const y0 = y > 0 ? y - 1 : 0;
    const y1 = y < height - 1 ? y + 1 : height - 1;
    for (let x = 0; x < width; x++) {
      const x0 = x > 0 ? x - 1 : 0;
      const x1 = x < width - 1 ? x + 1 : width - 1;
      const sum =
        gray[y0 * width + x0] + gray[y0 * width + x] + gray[y0 * width + x1] +
        gray[y * width + x0] + gray[y * width + x] + gray[y * width + x1] +
        gray[y1 * width + x0] + gray[y1 * width + x] + gray[y1 * width + x1];
      const c = gray[y * width + x];
      const v = c + amount * (c - sum / 9);
      out[y * width + x] = v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
    }
  }
  return out;
}

/** Grayscale -> RGBA (Uint8ClampedArray), opaque. */
export function grayToRGBA(gray) {
  const out = new Uint8ClampedArray(gray.length * 4);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    out[j] = out[j + 1] = out[j + 2] = gray[i];
    out[j + 3] = 255;
  }
  return out;
}
