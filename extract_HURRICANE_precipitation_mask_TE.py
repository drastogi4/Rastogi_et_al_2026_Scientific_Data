import numpy as np
import xarray as xr
from datetime import datetime
import pandas as pd
import re
from netCDF4 import Dataset, num2date
from scipy.interpolate import RegularGridInterpolator

EARTH_RADIUS_KM = 6371.0

import re
from datetime import datetime
import pandas as pd

def _to_rule_points(track_obj):
    """
    Convert a single track into a list of {time, longitude, latitude} dicts.
    Supports:
      - {'track_id', 'start_time', 'points': [(ts, lon, lat), ...]}  # parse_trajectories output
      - [(ts, lon, lat), ...]                                       # plain list of tuples
    """
    pts_src = track_obj["points"] if isinstance(track_obj, dict) and "points" in track_obj else track_obj
    out = []
    for ts, lon, lat in pts_src:
        # normalize time to python datetime for safety
        if not isinstance(ts, (pd.Timestamp, np.datetime64, datetime)):
            ts = pd.Timestamp(ts)
        out.append({
            "time": pd.Timestamp(ts).to_pydatetime(),
            "longitude": float(lon),
            "latitude": float(lat),
        })
    return out

def load_land_sea_mask(lsm_path):
    """
    Returns a callable lsm(lat, lon_raw) -> land_fraction in [0,1],
    handling latitude order and longitude-domain normalization automatically.
    """
    ds   = Dataset(lsm_path)
    lsm  = ds.variables["lsm"][:]            # (time, lat, lon) or (lat, lon)
    lats = ds.variables["latitude"][:]       # 1D
    lons = ds.variables["longitude"][:]      # 1D

    # Ensure increasing latitude
    if not np.all(np.diff(lats) > 0):
        lats = lats[::-1]
        lsm  = lsm[..., ::-1, :] if lsm.ndim == 3 else lsm[::-1, :]

    # Ensure increasing longitude and reorder
    lon_sort = np.argsort(lons)
    lons = lons[lon_sort]
    lsm  = lsm[..., :, lon_sort] if lsm.ndim == 3 else lsm[:, lon_sort]

    # Pick first time slice if present
    lsm2d = lsm[0, :, :] if lsm.ndim == 3 else lsm

    # Build interpolator on (lat, lon)
    rgi = RegularGridInterpolator(
        (lats, lons), lsm2d.astype(float),
        bounds_error=False,
        fill_value=np.nan,   # <- IMPORTANT: avoid bogus extrapolation
    )

    # Detect lon domain of file
    lon_max = float(lons.max())
    use_0360 = lon_max > 180.0  # True for 0..360 files (typical ERA5), False for -180..180

    def lsm_eval(lat, lon_raw):
        lon = float(lon_raw)
        if use_0360 and lon < 0:   # convert -180..180 -> 0..360
            lon += 360.0
        if not use_0360 and lon > 180:  # convert 0..360 -> -180..180
            lon -= 360.0
        val = rgi((float(lat), lon))
        return float(val) if val == val else np.nan  # keep NaN if out-of-bounds

    return lsm_eval

def _to_lon180(lon):
    """Normalize longitude(s) to [-180, 180]."""
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0

def is_spurious(track, lsm, ocean_thresh=0.1, land_thresh=0.5, eps=1e-6, debug=False):
    """
    KEEP a track unless it meets any REMOVE rule:

      (1') Came from west of 108°W at some earlier time AND later reached NE of (35°N, 100°W) → REMOVE
      (2)  Starts on land AND never touches ocean anywhere → REMOVE
      (3)  Majority east of 100°W AND (actually enters 100–75°W band) AND never touches ocean within that band → REMOVE
      (4)  EXCEPTION to (3): KEEP if the track starts over ocean east of 82°W
      (5)  Any point with lat ≥ 40°N AND lon ≤ −110° → REMOVE  (NEW)

    Notes:
      - lsm(lat, lon) must return land fraction in [0,1] (sea≈0, land≈1).
      - Longitudes are normalized to [-180, 180] before checks.
      - eps is used for robust comparisons near thresholds.
    """
    if not track:
        if debug: print("DROP: empty track")
        return False  # remove empty

    # Ensure chronological order if 'time' exists
    if 'time' in track[0]:
        track = sorted(track, key=lambda p: p['time'])

    npts = len(track)

    # --- Start point info ---
    start_lat = float(track[0]['latitude'])
    start_lon = _to_lon180(float(track[0]['longitude']))
    start_val = lsm(start_lat, start_lon)

    start_on_land    = (not np.isnan(start_val)) and (start_val >= land_thresh - eps)
    start_on_ocean   = (not np.isnan(start_val)) and (start_val <= ocean_thresh + eps)
    start_ocean_east82 = start_on_ocean and (start_lon > -82.0 + eps)

    # --- Accumulators / flags ---
    touched_ocean_any = start_on_ocean
    east100_count = 1 if start_lon >= -100.0 - eps else 0

    # Rule (1'): west→east progression
    seen_west108 = (start_lon <= -108.0 + eps)
    reached_NE_after_west108 = False

    # Rule (3) bookkeeping
    in_band_start = (-100.0 - eps) <= start_lon <= (-75.0 + eps)
    entered_100W_75W_band = in_band_start
    touched_ocean_in_band = in_band_start and start_on_ocean

    # Rule (5) bookkeeping: north of 40N while west of 110W
    north40_west110 = (start_lat >= 40.0 - eps) and (start_lon <= -110.0 + eps)

    # Iterate remaining points
    for pt in track[1:]:
        lat = float(pt['latitude'])
        lon = _to_lon180(float(pt['longitude']))

        # Rule (5): any point north of 40N and west of 110W → remove
        if (lat >= 40.0 - eps) and (lon <= -110.0 + eps):
            north40_west110 = True

        # Rule (1'): has been west of 108W, then later reaches NE(35N,100W)
        if lon <= -108.0 + eps:
            seen_west108 = True
        if seen_west108 and (lat >= 35.0 - eps) and (lon >= -100.0 - eps):
            reached_NE_after_west108 = True

        # Land-sea mask
        v = lsm(lat, lon)

        # Ocean touch anywhere
        if not np.isnan(v) and v <= ocean_thresh + eps:
            touched_ocean_any = True

        # Majority east of 100W?
        if lon >= -100.0 - eps:
            east100_count += 1

        # 100W–75W band checks
        in_band = (-100.0 - eps) <= lon <= (-75.0 + eps)
        if in_band:
            entered_100W_75W_band = True
            if (not np.isnan(v)) and v <= ocean_thresh + eps:
                touched_ocean_in_band = True

    # --- Apply Rule (5) first (unconditional removal) ---
    if north40_west110:
        if debug: print("REMOVE by Rule(5): north of 40°N while west of 110°W")
        return False

    # --- Apply remaining rules ---
    if reached_NE_after_west108:
        if debug: print("REMOVE by Rule(1'): west→east before reaching NE(35N,100W)")
        return False

    if start_on_land and not touched_ocean_any:
        if debug: print("REMOVE by Rule(2): starts on land and never touches ocean")
        return False

    east_fraction = east100_count / npts
    rule3_hits = (east_fraction > 0.5 + eps) and entered_100W_75W_band and (not touched_ocean_in_band)

    if rule3_hits:
        # Rule (4) exception: keep if starts over ocean east of 82°W
        if start_ocean_east82:
            if debug: print("KEEP by Rule(4): exception to Rule(3) (start ocean east of 82°W)")
            return True
        else:
            if debug: print("REMOVE by Rule(3): inland in 100–75W band with no ocean contact in-band")
            return False

    if debug: print("KEEP: no rules matched")
    return True

def track_overlaps_hurricane_season(track,
                                    start_month=5, start_day=15,
                                    end_month=12, end_day=15):
    """
    Return True if any point in the track occurs during relaxed hurricane season.
    Default: May 15 through December 15.
    """
    pts = _to_rule_points(track)

    for pt in pts:
        dt = pt["time"]
        md = (dt.month, dt.day)

        if (start_month, start_day) <= md <= (end_month, end_day):
            return True

    return False


def filter_tracks_overlapping_hurricane_season(tracks,
                                               start_month=5, start_day=15,
                                               end_month=12, end_day=15):
    """
    Keep the whole track if it overlaps relaxed hurricane season.
    """
    kept = []

    for tr in tracks:
        if track_overlaps_hurricane_season(
            tr,
            start_month=start_month,
            start_day=start_day,
            end_month=end_month,
            end_day=end_day
        ):
            kept.append(tr)

    return kept

def truncate_after_california_coast_crossing(track):
    """
    Keep only the portion of the track before it enters the California-coast box.
    Removes the California point and all later points.
    Preserves parse_trajectories() dict structure.
    """
    if isinstance(track, dict) and "points" in track:
        new_track = dict(track)
        new_points = []

        for ts, lon, lat in track["points"]:
            lon180 = float(_to_lon180(float(lon)))

            in_california_coast_box = (
                31.0 <= float(lat) <= 42.5 and
                -125.0 <= lon180 <= -110.0
            )

            if in_california_coast_box:
                break

            new_points.append((ts, lon180, float(lat)))

        new_track["points"] = new_points
        return new_track

    # fallback for plain list [(ts, lon, lat), ...]
    new_points = []

    for ts, lon, lat in track:
        lon180 = float(_to_lon180(float(lon)))

        in_california_coast_box = (
            31.0 <= float(lat) <= 42.5 and
            -125.0 <= lon180 <= -110.0
        )

        if in_california_coast_box:
            break

        new_points.append((ts, lon180, float(lat)))

    return new_points

def truncate_tracks_after_california_coast_crossing(tracks, min_points=2):
    """
    Apply California-coast truncation to all tracks.
    Drop tracks with fewer than min_points left.
    """
    kept = []

    for tr in tracks:
        tr_clean = truncate_after_california_coast_crossing(tr)

        if isinstance(tr_clean, dict) and "points" in tr_clean:
            if len(tr_clean["points"]) >= min_points:
                kept.append(tr_clean)
        else:
            if len(tr_clean) >= min_points:
                kept.append(tr_clean)

    return kept



def filter_spurious_tracks(tracks, lsm, ocean_thresh=0.1, land_thresh=0.5, eps=1e-6, debug=False):
    """
     1. May 15-Dec 15 relaxed hurricane-season overlap filter
     2. Truncate tracks after California-coast crossing
     3. Keep only tracks that pass is_ocean_and_land(...) rules.
    Preserves the original per-track structure (dict with 'points' or list of tuples).
    """
    # 1. Keep only tracks overlapping relaxed hurricane season
    tracks = filter_tracks_overlapping_hurricane_season(
        tracks,
        start_month=5,
        start_day=15,
        end_month=12,
        end_day=15
    )

    # 2. Truncate tracks once they reach California coast
    tracks = truncate_tracks_after_california_coast_crossing(
        tracks,
        min_points=2
    )

    kept = []
    for tr in tracks:
        pts = _to_rule_points(tr)
        if is_spurious(pts, lsm, ocean_thresh=ocean_thresh, land_thresh=land_thresh, eps=eps, debug=debug):
            kept.append(tr)
    return kept


def parse_trajectories(file_path, convert_lon=True, return_as="tracks"):
    """
    Parse either:
      (A) TRACK text files with lines like "TRACK_ID ... START_TIME ..." and
          data rows "<YYYYMMDDHH> <lon> <lat> ..."
      (B) CSV tables with columns:
          track_id, year, month, day, hour, i, j, lon, lat, slp, wind, zs

    Parameters
    ----------
    file_path : str
    convert_lon : bool
        If True, convert longitudes > 180 to [-180, 180] in the returned points.
        (Note: generate_mask_timewindow_fast() will also harmonize to the grid.)
    return_as : {"tracks","flat"}
        "tracks": list of dicts:
            {"track_id": int,
             "start_time": pandas.Timestamp,
             "points": [(Timestamp, lon, lat), ...]}
        "flat": list of (track_id, Timestamp, lon, lat)
    """
    # Peek at the first non-empty line to detect format
    with open(file_path, "r") as fh:
        first_nonempty = ""
        for line in fh:
            s = line.strip()
            if s:
                first_nonempty = s
                break

    if "," in first_nonempty and "track_id" in first_nonempty.lower():
        # ---- CSV format ----
        return _parse_tracks_csv(file_path, convert_lon, return_as)
    else:
        # ---- Original TRACK text format ----
        return _parse_tracks_text(file_path, convert_lon, return_as)


def _parse_tracks_csv(file_path, convert_lon, return_as):
    # Read CSV; tolerate spaces after commas
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Normalize column names (lowercase, strip spaces)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"track_id", "year", "month", "day", "hour", "lon", "lat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Build timestamp
    # Ensure ints (some CSVs read as floats)
    for col in ["year", "month", "day", "hour", "track_id"]:
        df[col] = df[col].astype(int)

    df["time"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]),
        errors="coerce",
    )
    df = df.dropna(subset=["time", "lon", "lat"]).copy()

    # Clean lon/lat
    df["lon"] = df["lon"].astype(float)
    df["lat"] = df["lat"].astype(float)

    # Filter out obvious missing placeholders, if any
    big = 1e20
    df = df[(np.abs(df["lon"]) < big) & (np.abs(df["lat"]) < big)]

    if convert_lon:
        # Map >180 into [-180, 180]
        over = df["lon"] > 180.0
        df.loc[over, "lon"] = df.loc[over, "lon"] - 360.0

    # Sort within track by time
    df = df.sort_values(["track_id", "time"])

    if return_as == "flat":
        return [(int(r.track_id), pd.Timestamp(r.time), float(r.lon), float(r.lat))
                for r in df.itertuples(index=False)]

    # Build "tracks" structure
    tracks = []
    for tid, g in df.groupby("track_id", sort=True):
        pts = [(pd.Timestamp(t), float(lo), float(la)) for t, lo, la in zip(g["time"], g["lon"], g["lat"])]
        if not pts:
            continue
        tracks.append({
            "track_id": int(tid),
            "start_time": pts[0][0],
            "points": pts
        })
    return tracks


def _parse_tracks_text(file_path, convert_lon, return_as):
    # Original TRACK parser
    re_tid   = re.compile(r'^TRACK_ID\s+(\d+)\s+START_TIME\s+(\d{10})')
    re_point = re.compile(r'^\s*(\d{10})\s+([0-9.+-eE]+)\s+([0-9.+-eE]+)')

    tracks = []
    current = {"track_id": None, "start_time": None, "points": []}

    with open(file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m_tid = re_tid.match(line)
            if m_tid:
                if current["track_id"] is not None and current["points"]:
                    tracks.append(current)
                tid_str, start_str = m_tid.groups()
                current = {
                    "track_id": int(tid_str),
                    "start_time": pd.Timestamp(datetime.strptime(start_str, "%Y%m%d%H")),
                    "points": []
                }
                continue

            m_pt = re_point.match(line)
            if m_pt:
                t_str, lon_str, lat_str = m_pt.groups()
                try:
                    lon = float(lon_str); lat = float(lat_str)
                except ValueError:
                    continue
                if abs(lon) > 1e20 or abs(lat) > 1e20:
                    continue
                if convert_lon and lon > 180.0:
                    lon -= 360.0
                ts = pd.Timestamp(datetime.strptime(t_str, "%Y%m%d%H"))
                current["points"].append((ts, lon, lat))
                continue
            # ignore other lines

    if current["track_id"] is not None and current["points"]:
        tracks.append(current)

    if return_as == "flat":
        flat = []
        for tr in tracks:
            tid = tr["track_id"]
            for ts, lon, lat in tr["points"]:
                flat.append((tid, ts, lon, lat))
        return flat

    return tracks
    """
    Accepts:
      - [(Timestamp, lon, lat), ...]
      - [(track_id, Timestamp, lon, lat), ...]
      - [{'track_id': ..., 'start_time': ..., 'points': [(Timestamp, lon, lat), ...]}, ...]
    Returns a sorted flat list: [(Timestamp, lon, lat), ...]
    """
    out = []
    for p in storm_points:
        if isinstance(p, dict) and "points" in p:
            out.extend(p["points"])
        elif isinstance(p, (list, tuple)):
            if len(p) >= 4 and isinstance(p[1], (pd.Timestamp, np.datetime64, datetime)):
                # (track_id, ts, lon, lat)
                ts, lon, lat = p[1], p[2], p[3]
                out.append((pd.Timestamp(ts), float(lon), float(lat)))
            elif len(p) >= 3:
                # (ts, lon, lat)
                ts, lon, lat = p[0], p[1], p[2]
                out.append((pd.Timestamp(ts), float(lon), float(lat)))
        else:
            raise ValueError(f"Unrecognized storm point entry: {type(p)}")
    # sort by time
    out.sort(key=lambda x: x[0])
    return out

def _flatten_storm_points(storm_points):
    """
    Accepts:
      - [(Timestamp, lon, lat), ...]
      - [(track_id, Timestamp, lon, lat), ...]
      - [{'track_id': ..., 'start_time': ..., 'points': [(Timestamp, lon, lat), ...]}, ...]
    Returns a sorted flat list: [(Timestamp, lon, lat), ...]
    """
    out = []
    for p in storm_points:
        if isinstance(p, dict) and "points" in p:
            out.extend(p["points"])
        elif isinstance(p, (list, tuple)):
            if len(p) >= 4 and isinstance(p[1], (pd.Timestamp, np.datetime64, datetime)):
                # (track_id, ts, lon, lat)
                ts, lon, lat = p[1], p[2], p[3]
                out.append((pd.Timestamp(ts), float(lon), float(lat)))
            elif len(p) >= 3:
                # (ts, lon, lat)
                ts, lon, lat = p[0], p[1], p[2]
                out.append((pd.Timestamp(ts), float(lon), float(lat)))
        else:
            raise ValueError(f"Unrecognized storm point entry: {type(p)}")
    # sort by time
    out.sort(key=lambda x: x[0])
    return out

def _to_grid_lon(lon_vals, grid_lon):
    """Map longitudes to the grid’s domain."""
    lon_vals = np.asarray(lon_vals, dtype=float)
    if np.nanmin(grid_lon) >= 0:        # 0..360 grid
        return np.mod(lon_vals, 360.0)
    else:                                # -180..180 grid
        return ((lon_vals + 180.0) % 360.0) - 180.0
def generate_mask_timewindow_fast(
    nc_path, storm_points, year, radius_km, timestep, half_window_hours=12
):
    ds = xr.open_dataset(nc_path)  # add chunks={'time': 24} if using dask
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    times = pd.to_datetime(ds["time"].values)

    # grid mesh + radians
    latg, longg = np.meshgrid(lat, lon, indexing='ij')
    latg_r  = np.deg2rad(latg)
    longg_r = np.deg2rad(longg)
    cos_latg = np.cos(latg_r)

    # Flatten/normalize storm input
    sp = _flatten_storm_points(storm_points)
    if len(sp) == 0:
        xr.Dataset(
            {'storm_mask': xr.DataArray(
                np.zeros((len(times), len(lat), len(lon)), dtype=np.int16),
                coords={'time': times, 'latitude': lat, 'longitude': lon},
                dims=['time','latitude','longitude']
            )}
        ).to_netcdf(f'./output/TRACK_ETCs_PRECIP_mask_{year}_{timestep}h.nc')
        ds.close()
        return

    # Harmonize to grid domain and to numpy types
    sp_times = np.array([np.datetime64(p[0]) for p in sp])
    sp_lons  = _to_grid_lon([p[1] for p in sp], lon)
    sp_lats  = np.array([p[2] for p in sp], dtype=float)

    clon_r_all = np.deg2rad(sp_lons)
    clat_r_all = np.deg2rad(sp_lats)

    # Haversine 'a' threshold for radius test (no arcsin needed)
    a_thresh = np.sin((radius_km / (2.0 * EARTH_RADIUS_KM)))**2

    mask = np.zeros((len(times), len(lat), len(lon)), dtype=bool)
    win  = np.timedelta64(int(half_window_hours), 'h')

    st = sp_times  # alias
    tvals = times.values.astype('datetime64[ns]')

    for ti, t in enumerate(tvals):
        left  = np.searchsorted(st, t - win, side='left')
        right = np.searchsorted(st, t + win, side='left')
        if right <= left:
            continue

        clat_active = clat_r_all[left:right]
        clon_active = clon_r_all[left:right]
        if clat_active.size == 0:
            continue

        within_any = np.zeros_like(latg, dtype=bool)

        # iterate over active centers; vectorized over grid
        for clatr, clonr in zip(clat_active, clon_active):
            dlat = clatr - latg_r
            dlon = ((clonr - longg_r + np.pi) % (2*np.pi)) - np.pi  # wrap to [-pi, pi]
            sin_dlat2 = np.sin(0.5 * dlat)
            sin_dlon2 = np.sin(0.5 * dlon)
            a = sin_dlat2**2 + cos_latg * np.cos(clatr) * (sin_dlon2**2)
            within_any |= (a <= a_thresh)
            # Optional early exit:
            # if within_any.all(): break

        mask[ti] = within_any

    out = xr.Dataset(
        {'storm_mask': xr.DataArray(mask.astype(np.int16),
                                    coords={'time': times, 'latitude': lat, 'longitude': lon},
                                    dims=['time','latitude','longitude'])}
    )
    out.to_netcdf(f'./output/TE_HUR_PRECIP_mask_{year}_{timestep}h.nc')
    ds.close()


# ---- usage ----
timestep = 6
radius_km = 667.0
half_window_hours = 3

for year in range(1980,2025):
    trajectory_file = f'./Hurricanes/TE/output/ERA5_trajectories_hur_{year}_{timestep}hr_csv.txt'
    nc_file = f'./ERA5/input/ERA5_tp_input_{year}_conus.nc'
    tracks = parse_trajectories(trajectory_file)
    lsm_path = "./era5-land-sea-mask.nc"
    lsm = load_land_sea_mask(lsm_path)
    filtered_tracks = filter_spurious_tracks(
                        tracks,
                        lsm,
                        ocean_thresh=0.1,
                        land_thresh=0.5,
                        eps=1e-6,
                        debug=False,   # set True temporarily to see which rule triggers
                        )

    generate_mask_timewindow_fast(
        nc_path=nc_file,
        storm_points=filtered_tracks,
        year=year,
        radius_km=radius_km,
        timestep=timestep,
        half_window_hours=half_window_hours
    )


