import { LIMITS } from '../config.js';
import { validateImageFile } from '../utils/validation.js';

/**
 * Validate and decode a user-selected file.
 * - Honours EXIF orientation (phone photos).
 * - Very large photos are down-scaled once (max side LIMITS.maxDecodeSidePx) to bound memory use.
 * The original dimensions are kept for display.
 *
 * Returns { bitmap, previewUrl, info } or throws an Error with a user-facing message.
 */
export async function loadImage(file) {
  const check = await validateImageFile(file);
  if (!check.ok) throw Object.assign(new Error(check.message), { code: check.code });

  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    throw Object.assign(
      new Error('This image could not be decoded. The file may be corrupted or use an unsupported variant of the format.'),
      { code: 'decode-failed' },
    );
  }

  const originalWidth = bitmap.width;
  const originalHeight = bitmap.height;
  const longest = Math.max(originalWidth, originalHeight);
  if (longest > LIMITS.maxDecodeSidePx) {
    const k = LIMITS.maxDecodeSidePx / longest;
    const scaled = await createImageBitmap(bitmap, {
      resizeWidth: Math.round(originalWidth * k),
      resizeHeight: Math.round(originalHeight * k),
      resizeQuality: 'high',
    });
    bitmap.close();
    bitmap = scaled;
  }

  return {
    bitmap,
    previewUrl: URL.createObjectURL(file),
    info: {
      name: file.name,
      width: originalWidth,
      height: originalHeight,
      format: check.format,
      bytes: file.size,
    },
  };
}

export function releaseImage(loaded) {
  if (!loaded) return;
  try {
    loaded.bitmap?.close();
  } catch {
    /* already closed / transferred */
  }
  if (loaded.previewUrl) URL.revokeObjectURL(loaded.previewUrl);
}
