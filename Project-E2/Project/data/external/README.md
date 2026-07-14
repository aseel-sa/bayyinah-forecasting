# Internal External-Feature Datasets

Consumed by `external_features.py`. All files are OPTIONAL — a missing file
degrades gracefully to an `unavailable_features` entry; generic calendar
features are generated in code and need no files at all.

## Platform mode

- **Default mode is GENERIC MANUFACTURING**: `country="generic"`,
  `industry="manufacturing"` when not provided.
- **No regional calendar features are included yet** (no Ramadan/Eid/national
  holidays). Regional calendars may return later as an opt-in registry block.
- **Weather climatology here is internal demo/support data, not live API
  data** — replace with real observations before relying on weather features
  for accuracy claims.
- **HVAC-specific features (AHRI shipments) are enabled only when
  `industry == "HVAC"`.** Construction indicators load only if their internal
  file is explicitly present.

## File formats

| File | Columns | Notes |
|---|---|---|
| `weather_<city>_monthly.csv` | `date, temperature[, humidity, rainfall]` | Monthly rows, `date` = month start. CDD/HDD derived automatically (base 18°C). Missing city falls back to the default file WITH a warning. |
| `ahri_shipments.csv` | `date, ahri_shipments` | HVAC-only indicator. Tagged `historical_context` (published with a lag). |
| `construction_activity.csv` | `date, construction_activity` | Manufacturing/HVAC/construction indicator. Loaded only if present. |

## Current contents

- `weather_riyadh_monthly.csv` — **PLACEHOLDER climatology**: long-term
  monthly temperature normals repeated for 2018–2027, NOT actual historical
  observations. With normals, the feature behaves like a smooth seasonal curve
  (still useful, but identical every year).
