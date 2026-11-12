# Vehicle database (local JSON)

The only vehicle database is the JSON in this folder. It is fetched from the app's own origin
(`<base>/data/vehicles.json`, works under GitHub Pages sub-paths) and parsed by `src/ai/vehicle-database.js`.
There is no SQL/Mongo/SQLite, no remote lookup and no external vehicle API. **Adding or editing a vehicle never
requires a code change** – edit the JSON, reload.

| File | Purpose |
|---|---|
| `vehicles.json` | One record per vehicle (see schema). The database can later be split into several files. |
| `manufacturers.json` | Optional. Per brand: alternative spellings seen on cars, incl. Persian, and the brand's country. |

> The shipped data is a **starter dataset, not a verified catalogue**. Facts such as engine codes, categories, the
> `manufacturer` of rebadged Chinese models, or which records are really sold under which name have not been verified.
> Never invent `generation`, `yearFrom` or `yearTo`: leave them `null` until you have a source.

## vehicles.json

```json
{
  "version": "1.0",
  "vehicles": [
    {
      "id": "lamari-neo",
      "brand": "Lamari",
      "manufacturer": "Lamari",
      "model": "Neo",
      "aliases": ["LAMARI NEO", "LAMARI", "NEO", "لاماری نئو"],
      "badgeTexts": ["LAMARI", "NEO"],
      "engineBadges": ["1.5"],
      "country": "China",
      "market": ["Iran"],
      "category": "SUV",
      "generation": null,
      "yearFrom": null,
      "yearTo": null
    }
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique, stable identifier (lower-case, `brand-model`). |
| `brand` | yes | Brand shown to users. Must equal the `name` in `manufacturers.json` to pick up its extra spellings. |
| `manufacturer` | no | Maker / parent group (e.g. Stellantis, Chery). Informational. |
| `model` | yes | Model name. |
| `aliases` | no | Names OCR may produce: full names, bare model names, Persian names, common misspellings. |
| `badgeTexts` | no | Text expected on the car body. Weaker than `aliases` (a badge alone is only a hint). |
| `engineBadges` | no | Engine labels (`1.5`, `1.6T`, `EF7`). **Supporting evidence only – they never identify a vehicle.** |
| `country`, `market`, `category` | no | Brand country of origin, markets sold in, body type. |
| `generation`, `yearFrom`, `yearTo` | no | Only if verified. Years are integers or `null`. |

Unknown values: use `null` (or omit the field). The loader validates the file (unique ids, types, `yearFrom <= yearTo`)
and refuses to load a broken database with a readable list of problems.

## Adding a new vehicle

1. Copy an existing record at the end of the `vehicles` array and give it a new unique `id`.
2. Fill `brand`, `model`, `country`, `market`, `category`. Leave anything you cannot verify as `null`.
3. `aliases`: put the strings you expect from OCR, **upper-case Latin is fine** (matching is case-insensitive):
   - the full text `"BRAND MODEL"` – a multi-word alias that exists on only one vehicle is the strongest evidence;
   - the bare model name (`"NEO"`);
   - Persian name(s), e.g. `"لاماری نئو"`; spelling variants (`"SAYNA"` / `"SAINA"`).
4. `badgeTexts`: brand + model badge texts. `engineBadges`: engine labels, for supporting evidence only.
5. New brand or new spellings of a brand (e.g. `"IKCO"`, `"ایران خودرو"`)? Add/extend an entry in `manufacturers.json`:
   `{ "name": "<same as brand>", "aliases": ["BRAND", "برند"], "country": "..." }`. A brand missing from this file still
   works – its own `brand` string is used.
6. Run `npm test` (the shipped file is validated) and reload the app.

Tips for good aliases:

- Spacing, punctuation, case and Persian/Arabic letter variants (ي/ی, ك/ک) and digits (۲۰۶/٢٠٦/206) are normalised
  automatically, so `"RAV 4"`, `"RAV4"` and `"rav-4"` need not all be listed; listing both is harmless.
- Don't use a **generic word** (`"PRO"`, `"LX"`, `"PLUS"`, `"SPORT"`) as a bare alias – it appears on many cars. Combine it with
  the model (`"X55 PRO"`). Bare aliases of 1–2 letters are capped at a weak hint by the matcher.
- An alias that is just a brand name is treated as brand evidence, never as a model.
- Don't put engine labels in `aliases` unless they really are model names (`"320I"`).

## How text is matched (summary)

`db.match(["LAMARI","NEO","1.5"])` normalises the text (upper-case, Persian/Arabic variants, digits, punctuation,
`DENA+` → `DENA PLUS`, context-aware OCR fixes such as `LAMAR1` → `LAMARI`, `RAV 4` → `RAV4`) and ranks vehicles:

1. exact full alias (multi-word, unique to one vehicle) → **identified**;
2. brand + model together → **identified**;
3. model (or bare alias) alone → **possible**; badge alone → weak hint;
4. fuzzy alias (typos, only letter-only words of 5+ characters, never numbers: 206 ≠ 207);
5. partial text → weak hint.

A brand alone (`TOYOTA`) only produces a *brand hint*, never a model. Engine labels alone produce nothing.
OCR fixes are only applied when the corrected word is a known name, so valid names are never "corrected" away.
Matches contained in a longer match of another vehicle (`DENA` inside `DENA PLUS`) are discounted.
