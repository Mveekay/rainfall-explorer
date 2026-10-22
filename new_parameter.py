#!/usr/bin/env python3
"""
build_rainfall_explorer.py  (animated edition)
===============================================
Builds the interactive Kenya forecast explorer (one self-contained HTML file) from seven
daily forecast text files.

What the page contains
  * Parameter dropdown  : Rainfall | Maximum temperature | Minimum temperature
  * Forecast day dropdown (Day 1 .. Day 7)
  * County + Ward dropdowns (ward list is filled from the chosen county)
  * Smoothed raster map, vector borders/coast/lakes, county outlines and names
  * Zoom / pan / pinch, hover read-out, click county -> constituencies -> wards
  * A selected area is shown on its own, with the boundaries and names of the
    areas inside it (constituencies + wards for a county, wards for a constituency)
  * ANIMATION: play/pause button, frame scrubber, speed control, cross-fade transitions,
    date display, and loop progress indicator

Input files (one per day, whitespace separated, no header), 5 columns:
    longitude  latitude  rainfall(mm)  min temperature(C)  max temperature(C)
All seven files must cover the same grid.

Usage
    python build_rainfall_explorer.py --files day1.txt day2.txt day3.txt day4.txt day5.txt day6.txt day7.txt
    python build_rainfall_explorer.py --files day*.txt --start-date 2026-10-16 --out explorer.html

Requirements
    pip install numpy pandas scipy matplotlib cartopy shapely pillow
Network access is needed the first time: county / constituency / ward boundaries are
downloaded from GADM 4.1 and coastline / lakes / borders from Natural Earth (cached in --cache).
"""
import argparse, base64, io, json, re, sys, time, urllib.request, zipfile, zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.io.shapereader as shpreader
from scipy.ndimage import zoom as ndzoom
from PIL import Image
import shapely
from shapely import STRtree
from shapely.geometry import shape, box

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
COLS = ["lon", "lat", "rain", "tmin", "tmax"]
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_KEN_{lvl}.json.zip"

WATER = "#00A0FF"                 # sea and lakes
UPSAMPLE = 10                     # cubic-spline upsampling factor used to smooth the rasters
RAIN_DPI, TEMP_DPI = 400, 300     # raster resolution (figure is W*0.8 x H*0.8 inches)

# Rainfall ramp (mm / 24 hr): 13 bands, below 1 mm is transparent, above 100 mm is dark red
RAIN_LEVELS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
RAIN_COLORS = ["#40f300", "#36bf06", "#2baf2a", "#1b8f19", "#ccebff", "#b4dbff", "#8ec3e6",
               "#47a6e6", "#2582c1", "#ffc800", "#ffa500", "#ef7000", "#f00000"]
RAIN_OVER = "#570000"

# Temperature ramp (deg C), shared by maximum and minimum temperature:
# [below 5, 5-10, 10-15, ..., 40-45, above 45]
TEMP_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45]
TEMP_COLORS = ["#a000c8", "#4a12cf", "#0973e8", "#00c8c8", "#00d55d",
               "#6be321", "#e6dc32", "#e9a02b", "#e34c31", "#ad005d"]

PARAM_COLUMNS = ["rain", "tmax", "tmin"]   # order used everywhere in the page


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------
# 1. Load the seven forecast files
# --------------------------------------------------------------------------------------
def load_days(files):
    days = []
    for f in files:
        d = pd.read_csv(f, sep=r"\s+", header=None, names=COLS)
        days.append(d.sort_values(["lat", "lon"]).reset_index(drop=True))
    ref = days[0]
    for k, d in enumerate(days[1:], start=2):
        if len(d) != len(ref) or not np.allclose(d[["lon", "lat"]].values, ref[["lon", "lat"]].values):
            sys.exit(f"Day {k} file does not use the same grid as day 1.")
    return days


def to_grid(d, col, lons, lats):
    """pivot a point table onto a [lat, lon] array (ascending); gaps become NaN"""
    return d.pivot_table(index="lat", columns="lon", values=col).reindex(index=lats, columns=lons).values


# --------------------------------------------------------------------------------------
# 2. Boundaries (GADM counties / constituencies / wards, Natural Earth lines)
# --------------------------------------------------------------------------------------
def fetch_gadm(cache):
    out = {}
    for lvl in (1, 2, 3):
        js = cache / f"gadm41_KEN_{lvl}.json"
        if not js.exists():
            z = cache / f"gadm41_KEN_{lvl}.json.zip"
            log("downloading", GADM_URL.format(lvl=lvl))
            urllib.request.urlretrieve(GADM_URL.format(lvl=lvl), z)
            with zipfile.ZipFile(z) as zf:
                zf.extract(js.name, cache)
        out[lvl] = json.load(open(js))["features"]
    return out


# --------------------------------------------------------------------------------------
# 3. Rasters (smoothed, one PNG per parameter and day, base64)
# --------------------------------------------------------------------------------------
def render_png(Z, extent, kind, dpi):
    X0, X1, Y0, Y1 = extent
    W, H = X1 - X0, Y1 - Y0
    fig = plt.figure(figsize=(W * 0.8, H * 0.8), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent([X0, X1, Y0, Y1], crs=ccrs.PlateCarree())
    ax.set_aspect("auto"); ax.set_axis_off(); ax.patch.set_visible(False)
    ocean = cf.NaturalEarthFeature("physical", "ocean", "10m")
    lakes = cf.NaturalEarthFeature("physical", "lakes", "10m")
    if kind == "rain":
        cmap = ListedColormap(RAIN_COLORS); cmap.set_under((1, 1, 1, 0)); cmap.set_over(RAIN_OVER)
        norm = BoundaryNorm(RAIN_LEVELS, cmap.N)
        fill = 0.0
        # rain is drawn over the sea; land without rain stays white
        ax.add_feature(ocean, facecolor=WATER, edgecolor="none", zorder=0)
        ax.add_feature(cf.NaturalEarthFeature("physical", "land", "10m"), facecolor="white", edgecolor="none", zorder=0.5)
    else:
        cmap = ListedColormap(TEMP_COLORS[1:-1]); cmap.set_under(TEMP_COLORS[0]); cmap.set_over(TEMP_COLORS[-1])
        norm = BoundaryNorm(TEMP_LEVELS, cmap.N)
        fill = 20.0
    Zf = ndzoom(np.nan_to_num(Z, nan=fill), UPSAMPLE, order=3, mode="nearest")
    if kind == "rain":
        Zf = np.clip(Zf, 0, None)
    ax.imshow(Zf, origin="lower", extent=[X0, X1, Y0, Y1], cmap=cmap, norm=norm,
              interpolation="nearest", zorder=1, transform=ccrs.PlateCarree(), aspect="auto")
    if kind != "rain":
        # temperature is shown on land only: the sea is painted over the raster
        ax.add_feature(ocean, facecolor=WATER, edgecolor="none", zorder=1.5)
    ax.add_feature(lakes, facecolor=WATER, edgecolor="none", zorder=2)
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor="white"); plt.close(fig)
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    im = im.quantize(colors=64, method=Image.MEDIANCUT, dither=Image.NONE)
    out = io.BytesIO(); im.save(out, format="PNG", optimize=True)
    return base64.b64encode(out.getvalue()).decode()


# --------------------------------------------------------------------------------------
# 4. Value grids for the hover read-out (uint16, zlib, base64)
#    rainfall: value*10 ; temperatures: (value+10)*10 ; 65535 = no data
# --------------------------------------------------------------------------------------
def encode_grid(Z, offset):
    u = np.where(np.isnan(Z), 65535, np.clip(np.round((Z + offset) * 10), 0, 65534)).astype("<u2")
    return base64.b64encode(zlib.compress(u.tobytes(), 9)).decode()


# --------------------------------------------------------------------------------------
# 5. Area statistics + simplified outlines for counties > constituencies > wards
# --------------------------------------------------------------------------------------
NICE = lambda n: re.sub(r"(?<=[a-z\.])(?=[A-Z])", " ", n).strip()     # "BaringoCentral" -> "Baringo Central"


def outline(g):
    polys = [g] if g.geom_type == "Polygon" else [p for p in g.geoms if p.geom_type == "Polygon"]
    return "".join("M" + "L".join(f"{x:.3f},{-y:.3f}" for x, y in r.coords) + "Z"
                   for p in polys for r in [p.exterior] + list(p.interiors))


def build_admin(gadm, days):
    tree = STRtree(shapely.points(days[0].lon.values, days[0].lat.values))
    vals = {(p, k): days[k][c].values for p, c in enumerate(PARAM_COLUMNS) for k in range(7)}

    def rec(feature, name, tol_cap):
        g = shape(feature["geometry"])
        idx = tree.query(g, predicate="intersects"); near = 0
        if len(idx) == 0:                                   # smaller than one grid cell
            idx = np.array([tree.nearest(g.representative_point())]); near = 1
        mm = []                                             # [param][day] -> mean, extreme
        for p in range(3):
            for k in range(7):
                v = vals[(p, k)][idx]
                mm += [round(float(v.mean()), 1), round(float(v.min() if p == 2 else v.max()), 1)]
        tol = min(tol_cap, 0.03 * np.sqrt(g.area))
        gs = g.simplify(tol, preserve_topology=True)
        if gs.is_empty: gs = g
        rp = gs.representative_point(); b = gs.bounds
        return dict(n=name, d=outline(gs), bbox=[round(v, 3) for v in b], lx=round(rp.x, 3), ly=round(-rp.y, 3),
                    mm=mm, mean=mm[0], max=mm[1], cells=int(len(idx)), near=near)

    counties, cons = {}, {}
    for f in gadm[1]:
        r = rec(f, NICE(f["properties"]["NAME_1"]), 0.004); r["cons"] = []
        counties[f["properties"]["GID_1"]] = r
    for f in gadm[2]:
        p = f["properties"]
        name = NICE(p["NAME_2"]) if not p["NAME_2"].isdigit() else f"Unnamed (code {p['NAME_2']})"
        r = rec(f, name, 0.004); r["wards"] = []
        cons[p["GID_2"]] = r; counties[p["GID_1"]]["cons"].append(r)
    for f in gadm[3]:
        p = f["properties"]
        cons[p["GID_2"]]["wards"].append(rec(f, NICE(p["NAME_3"]), 0.005))
    out = sorted(counties.values(), key=lambda c: c["n"])
    for c in out:
        c["cons"].sort(key=lambda x: x["n"])
        for s in c["cons"]: s["wards"].sort(key=lambda x: x["n"])
    return out


def vector_lines(extent):
    """coastline, international borders and lake outlines as SVG paths (stay sharp when zoomed)"""
    X0, X1, Y0, Y1 = extent
    clip = box(X0 - 0.2, Y0 - 0.2, X1 + 0.2, Y1 + 0.2)

    def paths(geoms, tol=0.003):
        out = []
        for g in geoms:
            g = g.intersection(clip)
            if g.is_empty: continue
            g = g.simplify(tol)
            parts = [g] if g.geom_type in ("LineString", "Polygon") else list(getattr(g, "geoms", []))
            for p in parts:
                if p.geom_type == "Polygon":
                    for r in [p.exterior] + list(p.interiors):
                        out.append("M" + "L".join(f"{x:.3f},{-y:.3f}" for x, y in r.coords))
                elif p.geom_type == "LineString":
                    out.append("M" + "L".join(f"{x:.3f},{-y:.3f}" for x, y in p.coords))
        return "".join(out)

    rd = lambda cat, name: [r.geometry for r in shpreader.Reader(shpreader.natural_earth("10m", cat, name)).records()]
    return dict(coast=paths(rd("physical", "coastline")),
                border=paths(rd("cultural", "admin_0_boundary_lines_land")),
                lake=paths(rd("physical", "lakes")))


# --------------------------------------------------------------------------------------
# 6. Assemble the page
# --------------------------------------------------------------------------------------
J = lambda o: json.dumps(o, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser(description="Build the Kenya forecast explorer HTML page")
    ap.add_argument("--files", nargs=7, required=True, metavar="DAY", help="the seven daily files, day 1 first")
    ap.add_argument("--out", default="rainfall_county_explorer.html")
    ap.add_argument("--start-date", default="2026-10-16", help="date day 1 starts (09:00 EAT), YYYY-MM-DD")
    ap.add_argument("--cache", default="cache", help="folder for downloaded boundary files")
    a = ap.parse_args()
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)

    log("reading forecast files")
    days = load_days(a.files)
    lons = np.sort(days[0].lon.unique()); lats = np.sort(days[0].lat.unique())
    X0, X1, Y0, Y1 = float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())
    extent = (X0, X1, Y0, Y1); W, H = X1 - X0, Y1 - Y0

    log("boundaries")
    gadm = fetch_gadm(cache)

    log("rendering 21 rasters (this takes a few minutes)")
    imgs = []
    for p, col in enumerate(PARAM_COLUMNS):
        row = []
        for k, d in enumerate(days):
            row.append(render_png(to_grid(d, col, lons, lats), extent, "rain" if col == "rain" else "temp",
                                  RAIN_DPI if col == "rain" else TEMP_DPI))
            log(f"  {col} day {k + 1}: {len(row[-1]) / 1e3:.0f} KB")
        imgs.append(row)

    log("hover grids")
    offsets = [0, 10, 10]
    grids = [[encode_grid(to_grid(d, col, lons, lats), offsets[p]) for d in days] for p, col in enumerate(PARAM_COLUMNS)]
    gmeta = dict(nx=len(lons), ny=len(lats), x0=float(lons[0]), y0=float(lats[0]),
                 dx=float((lons[-1] - lons[0]) / (len(lons) - 1)), dy=float((lats[-1] - lats[0]) / (len(lats) - 1)))

    log("area statistics for counties, constituencies and wards")
    admin = build_admin(gadm, days)
    log("vector coastline / borders / lakes")
    lines = vector_lines(extent)

    y, m, dd = (int(x) for x in a.start_date.split("-"))
    start_ms = int(datetime(y, m, dd, tzinfo=timezone.utc).timestamp() * 1000)
    meta = dict(x0=round(X0, 4), y0=round(-Y1, 4), w=round(W, 4), h=round(H, 4))

    tokens = {
        "ASPECT": f"{W:.4f} / {H:.4f}", "VB": f"{meta['x0']} {meta['y0']} {meta['w']} {meta['h']}",
        "X0": meta["x0"], "Y0": meta["y0"], "W": meta["w"], "H": meta["h"],
        "LAKE": lines["lake"], "COAST": lines["coast"], "BORDER": lines["border"],
        "DATA": J(admin).replace("</", "<\\/"), "META": J(meta), "GRID": J(gmeta), "GRID_P": J(grids),
        "GRID_OFF": J(offsets), "IMGS": J(imgs),
        "RAIN_COLORS": J(RAIN_COLORS), "RAIN_LEVELS": J(RAIN_LEVELS),
        "TEMP_COLORS": J(TEMP_COLORS), "TEMP_LEVELS": J(TEMP_LEVELS), "START_MS": start_ms,
    }
    html = TEMPLATE
    for k, v in tokens.items():
        html = html.replace(f"%%{k}%%", str(v))
    left = re.findall(r"%%[A-Z0-9_]+%%", html)
    if left: sys.exit(f"unreplaced template tokens: {set(left)}")
    Path(a.out).write_text(html, encoding="utf-8")
    log(f"wrote {a.out} ({len(html) / 1e6:.1f} MB)")


# --------------------------------------------------------------------------------------
# Page template (HTML + CSS + JavaScript)
# --------------------------------------------------------------------------------------
TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Kenya 24 hr Forecast</title>
<style>
:root{
  --bg:#f4f6f8; --surface:#ffffff; --ink:#14212b; --muted:#566673; --line:#d5dde3; --accent:#0a5fa8; --sel:#d4261c; --hover:rgba(10,95,168,.10);
  box-sizing:border-box; padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){--bg:#10171d; --surface:#18222b; --ink:#e7eef3; --muted:#9db0bd; --line:#2c3a45; --accent:#58b0ff; --sel:#ff6b5e; --hover:rgba(88,176,255,.14);}
}
:root[data-theme="dark"]{--bg:#10171d; --surface:#18222b; --ink:#e7eef3; --muted:#9db0bd; --line:#2c3a45; --accent:#58b0ff; --sel:#ff6b5e; --hover:rgba(88,176,255,.14);}
*,*::before,*::after{box-sizing:inherit}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Source Sans 3","Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:16px;line-height:1.45}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 32px}
h1{font-size:1.35rem;line-height:1.25;margin:0 0 2px;font-weight:700}
.sub{margin:0 0 16px;color:var(--muted);font-size:.95rem;font-weight:700}
.bar{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:flex-end;margin-bottom:14px}
.f{display:flex;flex-direction:column;gap:3px;min-width:0}
.bar label{font-weight:600;font-size:.85rem}
select{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:9px 12px;min-width:190px;max-width:100%}
select:disabled{opacity:.55}
#ward option{font-weight:700}
select:focus-visible,button:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
button.reset{font:inherit;color:var(--accent);background:transparent;border:1px solid var(--line);border-radius:6px;padding:9px 12px;cursor:pointer}
button.reset[hidden]{display:none}
.grid{display:block;max-width:760px;margin:0 auto}
.mapbox{background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden;position:relative}
.zc{position:absolute;top:10px;left:10px;z-index:3;display:flex;flex-direction:column;gap:4px}
.zc button{width:34px;height:34px;border:1px solid #b9c4cc;border-radius:6px;background:#fff;color:#14212b;font-size:1.2rem;line-height:1;cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.2)}
.zc button:hover{background:#eef3f6}
.zc button:focus-visible{outline:3px solid var(--accent);outline-offset:1px}
.mapbox svg{user-select:none;-webkit-user-select:none;display:block;width:100%;height:auto;aspect-ratio:%%ASPECT%%;touch-action:manipulation;cursor:default}
.ln{vector-effect:non-scaling-stroke;pointer-events:none;stroke-linejoin:round}
.ln.lake{stroke:#00355c;stroke-width:.8px}
.ln.coast{stroke:#000;stroke-width:1.1px}
.ln.border{stroke:#000;stroke-width:1.4px;stroke-dasharray:6 4}
.county{fill:transparent;stroke:#6b6b6b;stroke-width:.7px;vector-effect:non-scaling-stroke;cursor:pointer;transition:fill .12s}
.county:hover{fill:var(--hover);stroke:#555}
.only .county{pointer-events:none;stroke:transparent}
.inner{fill:none;stroke:#4a4a4a;stroke-width:1px;stroke-dasharray:4 3;vector-effect:non-scaling-stroke;pointer-events:none;stroke-linejoin:round}
.inner2{fill:none;stroke:#6f6f6f;stroke-width:.6px;vector-effect:non-scaling-stroke;pointer-events:none;stroke-linejoin:round}
.ctx{fill:none;stroke:#e08a82;stroke-width:1.6px;vector-effect:non-scaling-stroke;pointer-events:none}
.selp{fill:rgba(212,38,28,.10);stroke:var(--sel);stroke-width:2.6px;vector-effect:non-scaling-stroke;pointer-events:none}
.hit{fill:transparent;stroke:none;cursor:pointer}
.hit:hover{fill:rgba(10,95,168,.16)}
.hit.nc{cursor:default}
.lab{font-family:inherit;text-anchor:middle;dominant-baseline:central;fill:#111;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round;pointer-events:none;font-weight:600;display:block}
.lab.wd,#selLabel{font-weight:800}
.only #labels .lab{display:none}
.legend{padding:10px 12px 12px;border-top:1px solid #e3e8ec;background:#fff;color:#14212b}
.legend .ttl{font-size:.82rem;margin:0 0 6px;font-weight:600}
.ramp{display:flex;align-items:stretch;height:16px;border:1px solid #888}
.ramp i{flex:1}
.ticks{position:relative;height:1.1rem;font-size:.72rem;color:#333;margin-top:2px}
.ticks span{position:absolute;top:0;transform:translateX(-50%)}
.ticks span:first-child{transform:none}
.ticks span:last-child{transform:translateX(-100%)}
.hint{color:var(--muted);font-size:.9rem;margin:10px 2px 0}
#tip{position:fixed;pointer-events:none;background:#14212b;color:#fff;font-size:.82rem;line-height:1.35;padding:5px 8px;border-radius:5px;z-index:5;display:none;white-space:nowrap}
@media (prefers-reduced-motion:reduce){.county{transition:none}}

/* ---- Animation controls ---- */
/* The cross-fade animation uses CSS transitions on the raster images inside the SVG. */
#base, #base2 {
  transition: opacity 0.3s ease;
}
/* The animation control bar sits below the mapbox and above the legend. */
.anim-bar {
  background: var(--surface);
  border-top: 1px solid var(--line);
  padding: 10px 12px 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
/* Top row: play/pause + date display + speed selector */
.anim-top {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
/* Play / Pause button */
#playBtn {
  font: inherit;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 6px;
  width: 36px;
  height: 36px;
  font-size: 1rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: background .15s;
}
#playBtn:hover { filter: brightness(1.12); }
#playBtn:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
/* Day / date badge shown during playback */
#animDate {
  font-size: .9rem;
  font-weight: 700;
  color: var(--ink);
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
/* Speed selector group */
.speed-group {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: .8rem;
  color: var(--muted);
  flex-shrink: 0;
}
.speed-group span { font-weight: 600; }
.speed-btn {
  font: inherit;
  font-size: .78rem;
  background: var(--surface);
  color: var(--ink);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 3px 7px;
  cursor: pointer;
  transition: background .1s, color .1s;
}
.speed-btn.active, .speed-btn:hover {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
/* Loop progress bar (thin bar spanning the full width, filling left-to-right) */
.anim-progress-wrap {
  height: 3px;
  background: var(--line);
  border-radius: 2px;
  overflow: hidden;
}
#animProgress {
  height: 100%;
  background: var(--accent);
  width: 0%;
  transition: width 0.25s linear;
}
/* Frame scrubber – 7 day buttons in a row */
.scrubber {
  display: flex;
  gap: 4px;
  align-items: stretch;
}
.scrubber-btn {
  flex: 1;
  font: inherit;
  font-size: .72rem;
  font-weight: 600;
  background: var(--surface);
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 5px;
  padding: 5px 2px;
  cursor: pointer;
  text-align: center;
  line-height: 1.2;
  transition: background .1s, color .1s, border-color .1s;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.scrubber-btn:hover {
  background: var(--hover);
  color: var(--ink);
  border-color: var(--accent);
}
/* Active / current day in the scrubber */
.scrubber-btn.active {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
</style>
</head>
<body>
<div class="wrap">
  <h1 id="h1"></h1>
  <p class="sub" id="sub"></p>

  <div class="bar">
    <div class="f"><label for="param">Parameter</label><select id="param"></select></div>
    <div class="f"><label for="day">Forecast day</label><select id="day"></select></div>
    <div class="f"><label for="county">County</label><select id="county"><option value="">All counties</option></select></div>
    <div class="f"><label for="ward">Ward</label><select id="ward" disabled><option value="">Choose a county first</option></select></div>
    <button class="reset" id="reset" hidden>Show all counties</button>
  </div>

  <div class="grid">
    <div class="mapbox">
      <div class="zc"><button id="zin" aria-label="Zoom in" title="Zoom in">+</button><button id="zout" aria-label="Zoom out" title="Zoom out">−</button><button id="zhome" aria-label="Reset view" title="Reset view">⌂</button></div>
      <svg id="map" viewBox="%%VB%%" role="img" aria-label="Forecast map of Kenya with county boundaries">
        <!--
          Two stacked raster images are used for the cross-fade transition.
          #base2 is the "back" layer (incoming frame, starts transparent).
          #base  is the "front" layer (current frame, fades out to 0 when switching).
          After each fade the roles swap so we always have one visible layer.
        -->
        <g id="rasters">
          <image id="base2" href="" x="%%X0%%" y="%%Y0%%" width="%%W%%" height="%%H%%" preserveAspectRatio="none" style="image-rendering:auto;opacity:0"/>
          <image id="base"  href="" x="%%X0%%" y="%%Y0%%" width="%%W%%" height="%%H%%" preserveAspectRatio="none" style="image-rendering:auto;opacity:1"/>
        </g>
        <g id="lines" fill="none"><path d="%%LAKE%%" class="ln lake"/><path d="%%COAST%%" class="ln coast"/><path d="%%BORDER%%" class="ln border"/></g>
        <g id="counties"></g>
        <path id="mask" d="" fill="#fff" fill-rule="evenodd" style="pointer-events:none;display:none"/>
        <path id="inner2" class="inner2" d=""/><path id="inner" class="inner" d=""/>
        <path id="ctx" class="ctx" d=""/><path id="selp" class="selp" d=""/>
        <g id="hit"></g>
        <g id="labels"></g>
        <g id="ilabels"></g>
        <text id="selLabel" class="lab" x="0" y="0" style="display:none"></text>
      </svg>

      <!-- ===== Animation controls ===== -->
      <div class="anim-bar" id="animBar">
        <!-- Thin progress bar spanning the full width of the control strip -->
        <div class="anim-progress-wrap"><div id="animProgress"></div></div>

        <!-- Play/pause + current date label + speed selector -->
        <div class="anim-top">
          <button id="playBtn" aria-label="Play animation" title="Play / Pause (Space)">&#9654;</button>
          <span id="animDate"></span>
          <div class="speed-group">
            <span>Speed:</span>
            <button class="speed-btn" data-ms="1400">Slow</button>
            <button class="speed-btn active" data-ms="800">Med</button>
            <button class="speed-btn" data-ms="400">Fast</button>
          </div>
        </div>

        <!-- Day scrubber: 7 clickable day buttons -->
        <div class="scrubber" id="scrubber" role="group" aria-label="Forecast day scrubber"></div>
      </div>
      <!-- ===== /Animation controls ===== -->

      <div class="legend">
        <p class="ttl" id="ltitle"></p>
        <div class="ramp" id="ramp"></div>
        <div class="ticks" id="ticks"></div>
      </div>
    </div>
    <p class="hint">Choose a county, then a ward, or click a county on the map.</p>
  </div>
</div>
<div id="tip"></div>
<script>
'use strict';
const DATA=%%DATA%%;
const META=%%META%%;
const GRID=%%GRID%%;
const GRID_P=%%GRID_P%%;      // [parameter][day] -> zlib-compressed uint16 grid, base64
const GRID_OFF=%%GRID_OFF%%;  // value offset per parameter used when the grids were encoded
const IMGS=%%IMGS%%;          // [parameter][day] -> base64 PNG
const RAIN={colors:%%RAIN_COLORS%%,levels:%%RAIN_LEVELS%%};
const TEMP={colors:%%TEMP_COLORS%%,levels:%%TEMP_LEVELS%%};  // colours = [below, bins..., above]
const START=%%START_MS%%;     // 09:00 EAT start of day 1 (date part, UTC midnight)

const NS='http://www.w3.org/2000/svg';
const $=id=>document.getElementById(id);
const svg=$('map'), gC=$('counties'), gL=$('labels'), selp=$('selp'), ctx=$('ctx'), selLabel=$('selLabel');
const maskEl=$('mask'), innerEl=$('inner'), inner2El=$('inner2'), ilabels=$('ilabels'), hitG=$('hit');
const selC=$('county'), selW=$('ward'), resetBtn=$('reset'), tip=$('tip'), daySel=$('day'), parSel=$('param');
const full=[META.x0,META.y0,META.w,META.h], ASP=full[2]/full[3];
let cur=full.slice(), moved=false, DAY=1, PARAM=0, gridArr=null, lastDist=0;
const st={c:'',s:'',w:''};
const ptrs=new Map();
const gridsP=[[],[],[]];
const fmt=v=>v.toFixed(1);
const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
const county=n=>DATA.find(c=>c.n===n);

const PARAMS=[
 {name:'Rainfall',title:'24 hr Cumulative Rainfall Forecast',unit:'mm',here:'Rainfall here',ext:'max',ltitle:'Rainfall (mm / 24 hr)',
  colors:RAIN.colors,labels:RAIN.levels,frac:i=>i/RAIN.colors.length,center:false},
 {name:'Maximum temperature',title:'Maximum Temperature Forecast',unit:'°C',here:'Max temperature here',ext:'max',ltitle:'Maximum temperature (°C)',
  colors:TEMP.colors,labels:TEMP.levels,frac:i=>(i+1)/TEMP.colors.length,center:true},
 {name:'Minimum temperature',title:'Minimum Temperature Forecast',unit:'°C',here:'Min temperature here',ext:'lowest',ltitle:'Minimum temperature (°C)',
  colors:TEMP.colors,labels:TEMP.levels,frac:i=>(i+1)/TEMP.colors.length,center:true}
];
const U=v=>fmt(v)+(PARAMS[PARAM].unit==='mm'?' mm':'°C');
const tipText=o=>`${o.n}: mean ${U(o.mean)}, ${PARAMS[PARAM].ext} ${U(o.max)}`;
const dayRange=k=>[new Date(START+(k-1)*864e5),new Date(START+k*864e5)];
const isoDate=d=>d.toISOString().slice(0,10);

// ---------- view (zoom / pan) ----------
function sizeLabels(){
  const px=svg.getBoundingClientRect().width||600, fs=cur[2]/px*10.5;
  document.querySelectorAll('#map text.lab').forEach(t=>{
    const f=(t.classList.contains('wd')||t.id==='selLabel')?cur[2]/px*10:t.classList.contains('il')?fs*.85:fs;
    t.style.fontSize=f+'px'; t.style.strokeWidth=(f*.28)+'px';
  });
}
function updTouch(){
  const z=cur[2]<full[2]-1e-6;
  svg.style.touchAction=z?'none':'pan-y';
  if(!ptrs.size) svg.style.cursor=z?'grab':'default';
}
function setVB(v){cur=v;svg.setAttribute('viewBox',v.join(' '));sizeLabels();updTouch();}
function animateTo(target){
  const start=cur.slice();
  if(matchMedia('(prefers-reduced-motion: reduce)').matches){setVB(target);return;}
  const t0=performance.now(), dur=450;
  (function step(now){
    let k=Math.min(1,(now-t0)/dur); k=k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2;
    setVB(start.map((s,i)=>s+(target[i]-s)*k));
    if(k<1) requestAnimationFrame(step);
  })(t0);
}
function boxFor(o,minSpan){
  const [x0,y0,x1,y1]=o.bbox, minx=x0,maxx=x1,miny=-y1,maxy=-y0;
  let w=(maxx-minx)*1.35, h=(maxy-miny)*1.35;
  w=Math.max(w,minSpan); h=Math.max(h,minSpan/ASP);
  if(w/h>ASP) h=w/ASP; else w=h*ASP;
  w=Math.min(w,full[2]); h=w/ASP;
  let x=(minx+maxx)/2-w/2, y=(miny+maxy)/2-h/2;
  x=Math.min(Math.max(x,full[0]),full[0]+full[2]-w); y=Math.min(Math.max(y,full[1]),full[1]+full[3]-h);
  return [x,y,w,h];
}
const rectOf=()=>svg.getBoundingClientRect();
const toSvg=(cx,cy)=>{const r=rectOf();return [cur[0]+(cx-r.left)/r.width*cur[2],cur[1]+(cy-r.top)/r.height*cur[3]];};
function clampVB(x,y,w){
  w=Math.min(Math.max(w,0.15),full[2]); const h=w/ASP;
  x=Math.min(Math.max(x,full[0]),full[0]+full[2]-w); y=Math.min(Math.max(y,full[1]),full[1]+full[3]-h);
  return [x,y,w,h];
}
function zoomAt(f,cx,cy){
  const p=toSvg(cx,cy), r=rectOf(), w=Math.min(Math.max(cur[2]/f,0.15),full[2]), h=w/ASP;
  setVB(clampVB(p[0]-(cx-r.left)/r.width*w, p[1]-(cy-r.top)/r.height*h, w));
}

// ---------- counties, labels, dropdown ----------
DATA.forEach(c=>{
  const p=document.createElementNS(NS,'path'); p.setAttribute('d',c.d); p.setAttribute('class','county');
  p.addEventListener('click',()=>{if(!moved)go(c.n,'','')});
  p.addEventListener('mouseleave',()=>tip.style.display='none');
  gC.appendChild(p);
  const t=document.createElementNS(NS,'text'); t.setAttribute('x',c.lx); t.setAttribute('y',c.ly); t.setAttribute('class','lab'); t.textContent=c.n;
  gL.appendChild(t);
  const o=document.createElement('option'); o.value=c.n; o.textContent=c.n; selC.appendChild(o);
});

// ---------- selection ----------
function buildHits(){
  hitG.innerHTML='';
  if(!st.c) return;
  const c=county(st.c), s=st.s?c.cons.find(x=>x.n===st.s):null, w=s&&st.w?s.wards.find(x=>x.n===st.w):null;
  let items,nav;
  if(w){items=[w];nav=null;}
  else if(s){items=s.wards;nav=o=>go(st.c,st.s,o.n);}
  else{items=c.cons;nav=o=>go(st.c,o.n,'');}
  items.forEach(o=>{
    const p=document.createElementNS(NS,'path'); p.setAttribute('d',o.d); p.setAttribute('class','hit'+(nav?'':' nc')); p.dataset.tip=tipText(o);
    if(nav) p.addEventListener('click',()=>{if(!moved)nav(o);});
    hitG.appendChild(p);
  });
}
function go(cn,sn,wn){
  st.c=cn; st.s=sn; st.w=wn;
  selC.value=cn;
  const c=cn?county(cn):null;
  if(c){
    const all=[]; c.cons.forEach(x=>x.wards.forEach(y=>all.push([x.n,y.n]))); all.sort((a,b)=>a[1].localeCompare(b[1]));
    selW.innerHTML='<option value="">All wards</option>'+all.map(([a,b])=>`<option value="${esc(a)}|${esc(b)}">${esc(b)} (${esc(a)})</option>`).join('');
    selW.disabled=false; selW.value=wn?sn+'|'+wn:'';
  } else { selW.innerHTML='<option value="">Choose a county first</option>'; selW.disabled=true; }
  const s=c&&sn?c.cons.find(x=>x.n===sn):null;
  const w=s&&wn?s.wards.find(x=>x.n===wn):null;
  resetBtn.hidden=!c;
  selLabel.style.display='none'; selp.setAttribute('d',''); ctx.setAttribute('d',''); maskEl.style.display='none';
  innerEl.setAttribute('d',''); inner2El.setAttribute('d',''); ilabels.innerHTML=''; svg.classList.toggle('only',!!c);
  if(!c){ animateTo(full.slice()); buildHits(); return; }
  const target=w||s||c;
  if(w){ ctx.setAttribute('d',s.d); selp.setAttribute('d',w.d); }
  else if(s){ ctx.setAttribute('d',c.d); selp.setAttribute('d',s.d); }
  else { selp.setAttribute('d',c.d); }
  // the selected area is shown on its own: everything outside it is covered in white
  selLabel.textContent=target.n; selLabel.setAttribute('x',target.lx); selLabel.setAttribute('y',target.ly); selLabel.style.display=w?'block':'none';
  maskEl.setAttribute('d',`M${full[0]-2},${full[1]-2}h${full[2]+4}v${full[3]+4}h${-(full[2]+4)}Z`+target.d); maskEl.style.display='block';
  // boundaries of the areas inside the selection (constituencies + wards for a county, wards for a constituency)
  innerEl.setAttribute('d', w ? '' : s ? s.wards.map(x=>x.d).join('') : c.cons.map(x=>x.d).join(''));
  inner2El.setAttribute('d', (!w&&!s) ? c.cons.map(x=>x.wards.map(y=>y.d).join('')).join('') : '');
  if(!w){
    (s?s.wards:c.cons).forEach(o=>{
      const t=document.createElementNS(NS,'text'); t.setAttribute('x',o.lx); t.setAttribute('y',o.ly);
      t.setAttribute('class',s?'lab il wd':'lab il'); t.textContent=o.n; ilabels.appendChild(t);
    });
  }
  animateTo(boxFor(target, w?0.12:s?0.3:0.6));
  sizeLabels();
  buildHits();
}

// ---------- parameter / day ----------
PARAMS.forEach((P,i)=>{const o=document.createElement('option');o.value=i;o.textContent=P.name;parSel.appendChild(o);});
for(let k=1;k<=7;k++){
  const [s]=dayRange(k), o=document.createElement('option');
  o.value=k; o.textContent='Day '+k+' – '+s.toLocaleDateString('en-GB',{weekday:'short',day:'numeric',month:'short',timeZone:'UTC'});
  daySel.appendChild(o);
}

/* ============================================================
   Cross-fade raster swap
   Two <image> elements are used as a double-buffer.
   - frontEl / backEl track which is currently visible (opacity 1) vs hidden (opacity 0).
   - switchRaster(href) loads the new image into the back layer, fades out the front
     and fades in the back, then swaps the front/back pointers.
   The 300ms CSS transition on #base and #base2 is declared in <style> above.
   ============================================================ */
let frontEl = $('base');   // currently visible image element
let backEl  = $('base2');  // incoming image element
let fadeTimeout = null;

// Swap raster with a solid cross-fade. Returns immediately; visual fade runs smoothly on top.
function switchRaster(href) {
  const newHref = 'data:image/png;base64,' + href;
  if (!frontEl.getAttribute('href')) {
    // Initial page load: display immediately with no transition
    frontEl.setAttribute('href', newHref);
    frontEl.style.opacity = '1';
    backEl.style.opacity = '0';
    return;
  }
  if (frontEl.getAttribute('href') === newHref) return;

  if (fadeTimeout) {
    clearTimeout(fadeTimeout);
    fadeTimeout = null;
  }

  // Set new image on backEl while hidden, then move to top of #rasters stack
  backEl.style.transition = 'none';
  backEl.style.opacity = '0';
  backEl.setAttribute('href', newHref);
  $('rasters').appendChild(backEl);

  // Trigger reflow then smoothly fade in the incoming top layer over the solid background
  backEl.getBoundingClientRect();
  backEl.style.transition = 'opacity 0.25s ease';
  backEl.style.opacity = '1';

  const outgoing = frontEl;
  frontEl = backEl;
  backEl = outgoing;

  // Once incoming layer is fully opaque, hide the old layer underneath
  fadeTimeout = setTimeout(() => {
    outgoing.style.transition = 'none';
    outgoing.style.opacity = '0';
    fadeTimeout = null;
  }, 260);
}

function updateLegend() {
  const P = PARAMS[PARAM];
  $('ramp').innerHTML = P.colors.map(c => `<i style="background:${c}"></i>`).join('');
  $('ticks').innerHTML = P.labels.map((v, j) => `<span style="left:${P.frac(j)*100}%${P.center?';transform:translateX(-50%)':''}">${v}</span>`).join('');
}

function setDay(k, force = false){
  if (k === DAY && !force) return;
  DAY = k; 
  daySel.value = k; 
  parSel.value = PARAM;

  const P = PARAMS[PARAM], i = (PARAM * 7 + k - 1) * 2;
  DATA.forEach(c => {
    c.mean = c.mm[i]; c.max = c.mm[i + 1];
    c.cons.forEach(s => {
      s.mean = s.mm[i]; s.max = s.mm[i + 1];
      s.wards.forEach(w => { w.mean = w.mm[i]; w.max = w.mm[i + 1]; });
    });
  });

  [...gC.children].forEach((p, j) => { p.dataset.tip = tipText(DATA[j]); });
  // Cross-fade the raster to the new day's image.
  switchRaster(IMGS[PARAM][k - 1]);

  const [s, e] = dayRange(k);
  $('sub').textContent = `Valid from 0900 EAT ${isoDate(s)} to 0900 EAT ${isoDate(e)}`;
  $('h1').textContent = P.title; 
  $('ltitle').textContent = P.ltitle;
  document.title = `${P.title} – Kenya, Day ${k}`;
  gridArr = gridsP[PARAM][k - 1] || null;
  buildHits();
  // Keep animation UI in sync.
  updateAnimUI(k);
}

daySel.addEventListener('change', () => {
  if (animPlaying) pauseAnim();
  setDay(+daySel.value);
});

parSel.addEventListener('change', () => {
  PARAM = +parSel.value;
  updateLegend();
  setDay(DAY, true);
});

/* ============================================================
   Animation loop
   ============================================================ */

// Animation state variables.
let animPlaying  = false;  // true while the auto-play loop is running
let animInterval = null;   // setInterval handle
let animSpeed    = 800;    // milliseconds per frame (default: medium)

// Build the day scrubber buttons once.
(function buildScrubber(){
  const container = $('scrubber');
  for(let k=1;k<=7;k++){
    const [s] = dayRange(k);
    // Short label: "Day 1\nMon 16"
    const dateStr = s.toLocaleDateString('en-GB',{weekday:'short',day:'numeric',month:'short',timeZone:'UTC'});
    const btn = document.createElement('button');
    btn.className = 'scrubber-btn';
    btn.dataset.day = k;
    btn.setAttribute('aria-label','Jump to Day '+k+' ('+dateStr+')');
    btn.innerHTML = `<strong>D${k}</strong><br><span style="font-weight:400">${dateStr}</span>`;
    btn.addEventListener('click',()=>{
      // Jump to this day; pause if playing.
      if(animPlaying) pauseAnim();
      setDay(k);
    });
    container.appendChild(btn);
  }
})();

/**
 * Update all animation-related UI to reflect the current day k.
 * Called by setDay() so it always stays in sync, whether the day
 * changed manually (dropdown / scrubber click) or from the loop.
 */
function updateAnimUI(k) {
  // Progress bar: fills proportionally (0% at day 1, 100% at day 7).
  const pBar = $('animProgress');
  if (k === 1) {
    pBar.style.transition = 'none';
    pBar.style.width = '0%';
    pBar.getBoundingClientRect();
    pBar.style.transition = 'width 0.25s linear';
  } else {
    pBar.style.transition = 'width 0.25s linear';
    pBar.style.width = (((k - 1) / 6) * 100) + '%';
  }

  // Date badge.
  const [s] = dayRange(k);
  const label = s.toLocaleDateString('en-GB',{weekday:'short',day:'numeric',month:'long',timeZone:'UTC'});
  $('animDate').textContent = `Day ${k} · ${label}`;

  // Scrubber: highlight the active day button.
  document.querySelectorAll('.scrubber-btn').forEach(btn => {
    btn.classList.toggle('active', +btn.dataset.day === k);
  });
}

/** Advance to the next day (wraps Day 7 -> Day 1). */
function animStep() {
  const next = (DAY % 7) + 1;  // 1->2, 2->3, ..., 7->1
  setDay(next);
}

/** Start the auto-play loop. */
function playAnim() {
  if(animPlaying) return;
  animPlaying = true;
  $('playBtn').innerHTML = '&#9646;&#9646;';  // pause icon
  $('playBtn').setAttribute('aria-label','Pause animation');
  animInterval = setInterval(animStep, animSpeed);
}

/** Pause the auto-play loop. */
function pauseAnim() {
  if(!animPlaying) return;
  animPlaying = false;
  clearInterval(animInterval);
  animInterval = null;
  $('playBtn').innerHTML = '&#9654;';  // play icon
  $('playBtn').setAttribute('aria-label','Play animation');
}

/** Toggle play / pause. */
function toggleAnim() {
  if(animPlaying) pauseAnim(); else playAnim();
}

// Play/pause button click handler.
$('playBtn').addEventListener('click', toggleAnim);

// Keyboard shortcut: Space bar toggles play/pause (unless focus is on a form control).
addEventListener('keydown', e => {
  const isFormEl = ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName);
  if(e.key === ' ' && !isFormEl) {
    e.preventDefault();
    toggleAnim();
  }
  if(e.key === 'Escape' && st.c) go('','','');
  // Arrow keys: step through days manually when not focused on a form control.
  if(!isFormEl && !animPlaying) {
    if(e.key === 'ArrowRight') { setDay(DAY < 7 ? DAY + 1 : 7); }
    if(e.key === 'ArrowLeft')  { setDay(DAY > 1 ? DAY - 1 : 1); }
  }
});

// Speed selector buttons.
document.querySelectorAll('.speed-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const ms = +btn.dataset.ms;
    animSpeed = ms;
    // Update active state visually.
    document.querySelectorAll('.speed-btn').forEach(b => b.classList.toggle('active', b === btn));
    // If playing, restart the interval at the new speed.
    if(animPlaying){
      clearInterval(animInterval);
      animInterval = setInterval(animStep, animSpeed);
    }
  });
});

// ---------- value under the pointer ----------
async function inflate(b64){
  const bin=atob(b64), u=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i);
  const ab=await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();
  return new Uint16Array(ab);
}
function valueAt(lon,lat){
  if(!gridArr) return null;
  const i=Math.round((lon-GRID.x0)/GRID.dx), j=Math.round((lat-GRID.y0)/GRID.dy);
  if(i<0||j<0||i>=GRID.nx||j>=GRID.ny) return null;
  const v=gridArr[j*GRID.nx+i];
  return v===65535?null:v/10-GRID_OFF[PARAM];
}
Promise.all(GRID_P.map(r=>Promise.all(r.map(inflate)))).then(a=>{
  a.forEach((r,p)=>r.forEach((x,d)=>{gridsP[p][d]=x;}));
  gridArr=gridsP[PARAM][DAY-1];
}).catch(()=>{});

// ---------- interaction ----------
const ctr=()=>{const r=rectOf();return [r.left+r.width/2,r.top+r.height/2];};
$('zin').addEventListener('click',()=>zoomAt(1.6,...ctr()));
$('zout').addEventListener('click',()=>zoomAt(1/1.6,...ctr()));
$('zhome').addEventListener('click',()=>go(st.c,st.s,st.w));
svg.addEventListener('wheel',e=>{
  if(e.deltaY>0&&cur[2]>=full[2]-1e-6) return;
  e.preventDefault(); zoomAt(Math.exp(-e.deltaY*0.0015),e.clientX,e.clientY);
},{passive:false});
const pts=()=>[...ptrs.values()];
svg.addEventListener('pointerdown',e=>{
  if(e.pointerType==='mouse'&&e.button!==0) return;
  ptrs.set(e.pointerId,{x:e.clientX,y:e.clientY}); moved=false;
  if(ptrs.size===2){const [a,b]=pts(); lastDist=Math.hypot(a.x-b.x,a.y-b.y);}
});
addEventListener('pointermove',e=>{
  const p=ptrs.get(e.pointerId); if(!p) return;
  if(ptrs.size===1){
    const dx=e.clientX-p.x, dy=e.clientY-p.y;
    if(!moved&&Math.hypot(dx,dy)<5) return;
    moved=true; svg.style.cursor='grabbing'; tip.style.display='none';
    const r=rectOf(); setVB(clampVB(cur[0]-dx/r.width*cur[2],cur[1]-dy/r.height*cur[3],cur[2]));
    p.x=e.clientX; p.y=e.clientY;
  }else if(ptrs.size===2){
    p.x=e.clientX; p.y=e.clientY;
    const [a,b]=pts(), d=Math.hypot(a.x-b.x,a.y-b.y);
    if(lastDist) zoomAt(d/lastDist,(a.x+b.x)/2,(a.y+b.y)/2);
    lastDist=d; moved=true;
  }
});
const endPtr=e=>{ptrs.delete(e.pointerId);lastDist=0;updTouch();};
addEventListener('pointerup',endPtr); addEventListener('pointercancel',endPtr);
svg.addEventListener('pointermove',e=>{
  if(ptrs.size&&moved){tip.style.display='none';return;}
  const only=svg.classList.contains('only'), tt=e.target.dataset&&e.target.dataset.tip;
  if(only&&!tt){tip.style.display='none';return;}
  const [x,y]=toSvg(e.clientX,e.clientY), v=valueAt(x,-y);
  const html=(tt?esc(tt)+'<br>':'')+(v==null?'':`${PARAMS[PARAM].here}: ${v.toFixed(1)}${PARAMS[PARAM].unit==='mm'?' mm':'°C'}`);
  if(!html){tip.style.display='none';return;}
  tip.innerHTML=html; tip.style.display='block'; tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY+14)+'px';
});
svg.addEventListener('pointerleave',()=>{tip.style.display='none';});
selC.addEventListener('change',()=>go(selC.value,'',''));
selW.addEventListener('change',()=>{
  if(!selW.value){go(st.c,'','');return;}
  const [a,b]=selW.value.split('|'); go(st.c,a,b);
});
resetBtn.addEventListener('click',()=>go('','',''));
addEventListener('resize',sizeLabels);

sizeLabels(); updTouch(); updateLegend(); setDay(1, true);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
