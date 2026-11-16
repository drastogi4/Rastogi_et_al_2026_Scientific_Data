# ==== TempestExtremes ETC tracks -> filtered -> precip mask ===================
# Updated: robust great-circle distance, optional cumulative path for QC,
# safer LSM interpolator (no extrapolation), consistent lon normalization.

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from math import radians, sin, cos, asin, sqrt

# ------------------------------- Config --------------------------------------

EARTH_RADIUS_KM = 6371.0

# Year range NOTE: range(START_YEAR, END_YEAR) is END-exclusive.
START_YEAR, END_YEAR = 1980, 2025   # loops only 1980. Use END_YEAR+1 for inclusive.

LAT_MIN, LAT_MAX = 24.0, 50.0
LON_MIN, LON_MAX = -125.0, -66.0

wnet_dir = './Global_ARs_Hurr_ETCs_MCS_fullres025_V2/ETCs_WNet_final/'
LSM_PATH = "./era5-land-sea-mask.nc"

# Mask build params
TIMESTEP_H = 6
RADIUS_KM = 1000.0
HALF_WINDOW_H = 3  # ±3h

# QC: choose whether "path" means cumulative length or net displacement
QC_PATH_IS_CUMULATIVE = True  # set False to require net end-to-end displacement

# ------------------------------- QC settings ---------------------------------

@dataclass
class QCChecks:
    lifetime: bool = True      # (min_hours & min_points)
    speed: bool = True         # (median translation speed bounds)
    path: bool = True          # (min_path_km)
    wind: bool = False         # (enable only if 'wind' exists)
    retro_run: bool = True     # (max contiguous westward run)
    retro_frac: bool = True    # (max westward fraction)
    net_east: bool = True      # (require net eastward motion)

@dataclass
class QCThresh:
    # core thresholds
    min_hours: float = 24.0
    min_points: int = 5
    speed_min: float = 2.0      # m/s
    speed_max: float = 25.0     # m/s
    min_path_km: float = 1000.0
    wind_percentile: Optional[float] = None  # e.g., 0.6 -> 60th pct

    # retrograde thresholds
    west_sector: Tuple[float, float] = (210.0, 330.0)
    max_retro_run_hours: float = 24.0
    max_retro_total_frac: float = 0.33
    require_net_east: bool = True
    min_net_east_km: float = 100.0

# --------------------------- Lon helpers & geometry --------------------------

def lon_to_180(lon):
    """Map longitude(s) to [-180, 180). Works with scalars or numpy arrays."""
    return ((lon + 180.0) % 360.0) - 180.0

def lon_to_360(lon):
    """Map longitude(s) to [0, 360). Works with scalars or numpy arrays."""
    return lon % 360.0

def in_box(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """Point-in-rectangle with lon in [-180,180). Handles wrapped boxes if needed."""
    lon = lon_to_180(lon)
    lon_min = lon_to_180(lon_min)
    lon_max = lon_to_180(lon_max)
    in_lat = (lat_min <= lat <= lat_max)
    if lon_min <= lon_max:
        in_lon = (lon_min <= lon <= lon_max)
    else:
        in_lon = (lon >= lon_min) or (lon <= lon_max)
    return in_lat and in_lon

def great_circle_km(lat1, lon1, lat2, lon2):
    """
    Robust great-circle distance in km:
    - normalizes longitudes to [-180,180)
    - wraps Δλ to [-π, π] to avoid dateline issues
    """
    phi1, phi2 = radians(lat1), radians(lat2)
    lam1 = radians(lon_to_180(lon1))
    lam2 = radians(lon_to_180(lon2))
    dphi = phi2 - phi1
    dlam = ((lam2 - lam1 + np.pi) % (2*np.pi)) - np.pi
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
    return 2*EARTH_RADIUS_KM*asin(sqrt(a))

def path_length_km(lats, lons):
    """Cumulative path length in km over a polyline."""
    if len(lats) < 2:
        return 0.0
    return float(sum(great_circle_km(lats[i], lons[i], lats[i+1], lons[i+1])
                     for i in range(len(lats)-1)))

def distance_km_point_to_box(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """
    Minimum great-circle distance from a point to an axis-aligned lat/lon box (km).
    Returns 0 if the point is inside the box.
    """
    if in_box(lat, lon, lat_min, lat_max, lon_min, lon_max):
        return 0.0

    clat = min(max(lat, lat_min), lat_max)

    lon = lon_to_180(lon)
    lon_min = lon_to_180(lon_min)
    lon_max = lon_to_180(lon_max)

    if lon_min <= lon_max:
        clon = min(max(lon, lon_min), lon_max)
        return great_circle_km(lat, lon, clat, clon)
    else:
        d1 = great_circle_km(lat, lon, clat, lon_min)
        d2 = great_circle_km(lat, lon, clat, lon_max)
        return min(d1, d2)

# --------------------------- Retrograde & speeds ------------------------------

def _initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point1 to point2 in degrees [0,360)."""
    phi1, phi2 = radians(lat1), radians(lat2)
    dlam = radians(lon_to_180(lon2) - lon_to_180(lon1))
    x = sin(dlam) * cos(phi2)
    y = cos(phi1)*sin(phi2) - sin(phi1)*cos(phi2)*cos(dlam)
    theta = np.degrees(np.arctan2(x, y)) % 360.0
    return float(theta)

def _segment_speeds_ms(track_times, track_lats, track_lons):
    """Per-segment translation speeds in m/s."""
    dists_km = [great_circle_km(track_lats[i], track_lons[i], track_lats[i+1], track_lons[i+1])
                for i in range(len(track_lats)-1)]
    dts = [(track_times[i+1] - track_times[i]).total_seconds() for i in range(len(track_times)-1)]
    v = []
    for dk, dt in zip(dists_km, dts):
        if dt > 0:
            v.append((dk*1000.0)/dt)
    return np.array(v) if v else np.array([])

def _retrograde_stats(track_times, track_lats, track_lons, west_sector=(210.0, 330.0)):
    """
    Returns:
      retro_frac   : fraction of total time spent moving westward (bearing in west_sector)
      retro_longest: longest contiguous westward run (hours)
    """
    n = len(track_times)
    if n < 2:
        return 0.0, 0.0

    seg_hours = []
    bears = []
    for i in range(n - 1):
        dh = (track_times[i+1] - track_times[i]).total_seconds() / 3600.0
        lat1, lon1 = track_lats[i],   track_lons[i]
        lat2, lon2 = track_lats[i+1], track_lons[i+1]
        if not (np.isfinite(dh) and dh > 0 and
                np.isfinite(lat1) and np.isfinite(lon1) and
                np.isfinite(lat2) and np.isfinite(lon2)):
            continue
        seg_hours.append(dh)
        bears.append(_initial_bearing_deg(lat1, lon1, lat2, lon2))

    if not seg_hours:
        return 0.0, 0.0

    seg_hours = np.asarray(seg_hours, dtype=float)
    bears = np.asarray(bears, dtype=float)

    west_lo, west_hi = west_sector
    west_mask = (bears >= west_lo) & (bears <= west_hi)

    retro_total_h = float(seg_hours[west_mask].sum()) if west_mask.any() else 0.0
    total_h = float(seg_hours.sum())
    retro_frac = retro_total_h / total_h if total_h > 0 else 0.0

    retro_longest = 0.0
    run_h = 0.0
    for is_w, dh in zip(west_mask, seg_hours):
        if is_w:
            run_h += dh
        else:
            retro_longest = max(retro_longest, run_h)
            run_h = 0.0
    retro_longest = max(retro_longest, run_h)

    return retro_frac, retro_longest

def _zonal_displacement_km(track_lats, track_lons):
    """Approximate net east-west displacement (east positive) in km."""
    dx_total = 0.0
    for i in range(len(track_lats)-1):
        phi_m = radians(0.5*(track_lats[i] + track_lats[i+1]))
        dlam = radians(lon_to_180(track_lons[i+1]) - lon_to_180(track_lons[i]))
        dx_total += EARTH_RADIUS_KM * cos(phi_m) * dlam
    return dx_total

def filter_quality_tracks(tracks_by_year, checks: QCChecks=None, thr: QCThresh=None, return_stats: bool=False):
    checks = checks or QCChecks()
    thr = thr or QCThresh()

    # global wind threshold (if enabled)
    wind_thr = None
    if checks.wind and (thr.wind_percentile is not None):
        all_w = []
        for tracks in tracks_by_year.values():
            for tr in tracks:
                for p in tr:
                    w = p.get('wind', None)
                    if w is not None and np.isfinite(w):
                        all_w.append(w)
        if all_w:
            wind_thr = float(np.quantile(all_w, thr.wind_percentile))

    out = defaultdict(list)
    stats = defaultdict(lambda: dict(kept=0, short=0, speed=0, path=0, wind=0,
                                     retro_run=0, retro_frac=0, net_east=0, other=0))

    for year, tracks in tracks_by_year.items():
        for tr in tracks:
            try:
                t = [datetime.strptime(p['timestep'], "%Y%m%d%H") for p in tr]
            except Exception:
                stats[year]['other'] += 1
                continue
            if len(t) < 2:
                stats[year]['short'] += 1
                continue

            lats = [p['latitude'] for p in tr]
            lons = [p['longitude'] for p in tr]

            # (1) lifetime & points
            if checks.lifetime:
                if len(t) < thr.min_points:
                    stats[year]['short'] += 1; continue
                dur_h = (t[-1] - t[0]).total_seconds()/3600.0
                if dur_h < thr.min_hours:
                    stats[year]['short'] += 1; continue

            # (2) speed bounds (median)
            if checks.speed:
                v = _segment_speeds_ms(t, lats, lons)
                if v.size == 0:
                    stats[year]['speed'] += 1; continue
                v_med = float(np.nanmedian(v))
                if not (thr.speed_min <= v_med <= thr.speed_max):
                    stats[year]['speed'] += 1; continue

            # (3) path length threshold
            if checks.path:
                if QC_PATH_IS_CUMULATIVE:
                    path_km = path_length_km(lats, lons)  # cumulative distance
                else:
                    path_km = great_circle_km(lats[0], lons[0], lats[-1], lons[-1])  # net displacement
                if path_km < thr.min_path_km:
                    stats[year]['path'] += 1; continue

            # (4) wind percentile threshold
            if checks.wind and (wind_thr is not None):
                wmax = np.nanmax([p.get('wind', np.nan) for p in tr])
                if not (np.isfinite(wmax) and wmax >= wind_thr):
                    stats[year]['wind'] += 1; continue

            # (5-7) retrograde constraints (only if any retro check is on)
            need_retro = checks.retro_run or checks.retro_frac or checks.net_east
            if need_retro:
                retro_frac, retro_longest = _retrograde_stats(t, lats, lons, west_sector=thr.west_sector)
                if checks.retro_run and (retro_longest > thr.max_retro_run_hours):
                    stats[year]['retro_run'] += 1; continue
                if checks.retro_frac and (retro_frac > thr.max_retro_total_frac):
                    stats[year]['retro_frac'] += 1; continue
                if checks.net_east and thr.require_net_east:
                    zonal_km = _zonal_displacement_km(lats, lons)
                    if zonal_km < thr.min_net_east_km:
                        stats[year]['net_east'] += 1; continue

            out[year].append(tr)
            stats[year]['kept'] += 1

    return (out, stats) if return_stats else out



# ---------------------------- LSM + spatial filters --------------------------

def load_land_sea_mask(lsm_path):
    """
    Loads the land-sea mask and returns an interpolator.
    - Accepts LSM with or without time dimension
    - Uses fill_value=0.0 to avoid extrapolating to land outside grid
    """
    ds = Dataset(lsm_path)
    try:
        lsm = ds.variables["lsm"][:]  # (time, lat, lon) or (lat, lon)
        lats = ds.variables["latitude"][:]
        lons = ds.variables["longitude"][:]

        lon_sort_idx = np.argsort(lons)
        lons_sorted = lons[lon_sort_idx]

        # ensure ascending latitude; flip data if needed
        if not np.all(np.diff(lats) > 0):
            lats = lats[::-1]
            if lsm.ndim == 3:
                lsm = lsm[:, ::-1, :]
            else:
                lsm = lsm[::-1, :]

        # reorder longitude axis to match sorted lons
        if lsm.ndim == 3:
            lsm = lsm[:, :, lon_sort_idx]
            lsm2d = lsm[0, :, :]
        else:
            lsm = lsm[:, lon_sort_idx]
            lsm2d = lsm

        interpolator = RegularGridInterpolator(
            (lats, lons_sorted),
            lsm2d,
            bounds_error=False,
            fill_value=0.0  # out-of-bounds => ocean
        )
        return interpolator
    finally:
        ds.close()

def touches_conus_bbox(track, lat_min, lat_max, lon_min, lon_max):
    """Any track point inside the CONUS box."""
    return any(in_box(p['latitude'], p['longitude'], lat_min, lat_max, lon_min, lon_max)
               for p in track)

def near_conus_bbox(track, buffer_km, lat_min, lat_max, lon_min, lon_max):
    """
    True if any point in the track is within `buffer_km` of the box (or inside it).
    """
    for p in track:
        d = distance_km_point_to_box(p['latitude'], p['longitude'], lat_min, lat_max, lon_min, lon_max)
        if d <= buffer_km:
            return True
    return False

def is_land_touching_and_within_bounds(track, lsm_interpolator, land_thresh,
                                       lat_min, lat_max, lon_min, lon_max):
    """
    True if:
      - any point has land-sea mask >= land_thresh (touches land), AND
      - any point is inside the CONUS box.
    LSM lookup uses [0,360); bounds use [-180,180).
    """
    touches_land = False
    touches_bounds = False

    for pt in track:
        lat = pt['latitude']
        lon_raw = pt['longitude']

        # Land (0–360 for LSM)
        val = lsm_interpolator((lat, lon_to_360(lon_raw)))
        if val is not None and val >= land_thresh:
            touches_land = True

        # Bounds (−180..180)
        if in_box(lat, lon_raw, lat_min, lat_max, lon_min, lon_max):
            touches_bounds = True

        if touches_land and touches_bounds:
            return True

    return False

def filter_land_touching_tracks(tracks_by_year, lsm_interpolator,
                                land_thresh=0.2,
                                lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    """Keep tracks that touch land and enter the CONUS box."""
    assert None not in (lat_min, lat_max, lon_min, lon_max), "Provide bounds."
    result = defaultdict(list)
    for year, tracks in tracks_by_year.items():
        for tr in tracks:
            if is_land_touching_and_within_bounds(tr, lsm_interpolator, land_thresh,
                                                  lat_min, lat_max, lon_min, lon_max):
                result[year].append(tr)
    return result

def filter_conus_relevant(tracks_by_year, lsm_interpolator,
                          land_thresh=0.2, buffer_km=400,
                          lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    """
    Keep tracks that are either:
      (A) land-touching AND inside the CONUS box, OR
      (B) within `buffer_km` of the box (even if entirely offshore).
    """
    assert None not in (lat_min, lat_max, lon_min, lon_max), "Provide bounds."
    out = defaultdict(list)
    for year, tracks in tracks_by_year.items():
        for tr in tracks:
            if is_land_touching_and_within_bounds(tr, lsm_interpolator, land_thresh,
                                                  lat_min, lat_max, lon_min, lon_max):
                out[year].append(tr)
            elif near_conus_bbox(tr, buffer_km, lat_min, lat_max, lon_min, lon_max):
                out[year].append(tr)
    return out

# ---------------------------- WNET reading -------------------------------------


def read_monthly_wnet_tracks(monthly_dir):
    """
    Read monthly ETC tracks from files like:
      ETC_tracks_YYYYMM_meteorological.txt

    Data lines look like:
      <TRACK_ID> YYYY-MM-DD_HH:MM:SS  <lon>  <lat>  <intensity>

    Returns:
      dict: {year: [track, track, ...]}
            where each track = [{'timestep': 'YYYYMMDDHH', 'longitude': lon, 'latitude': lat}, ...]
    """
    # allow plain or scientific notation for lon/lat; intensity is optional/ignored
    pat = re.compile(
    r'^\s*(\d+)\s+(\d{4}-\d{2}-\d{2}[_ ]\d{2}:\d{2}:\d{2})\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?:\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))?'
    )


    all_tracks = defaultdict(list)

    for year in range(START_YEAR, END_YEAR):  # end_year is exclusive (like your other code)
        
        fp = os.path.join(monthly_dir, f"ETC_tracks_{year}_meteorological_v2.txt")
        if not os.path.exists(fp):
            continue

        tracks_by_id = defaultdict(list)

        with open(fp, "r") as f:
            for raw in f:
                line = raw.strip()
                # skip headers and section lines
                if (not line or line.startswith("#") or
                    line.startswith("Track") or line.startswith("Format") or
                    line.startswith("Total") or line.startswith("Generated") or
                    line.startswith("Created")):
                    continue



                m = pat.match(line)
                if not m:
                    continue

                tid_str, tstr, lon_str, lat_str, wind_str = (m.groups() + (None,))[:5]
                try:
                    tid = int(tid_str)
                    # accept either "YYYY-MM-DD_HH:MM:SS" or "YYYY-MM-DD HH:MM:SS"
                    tstr = tstr.replace(" ", "_")
                    dt = datetime.strptime(tstr, "%Y-%m-%d_%H:%M:%S")
                    ts = dt.strftime("%Y%m%d%H")

                    lon = lon_to_180(float(lon_str))
                    lat = float(lat_str)
                except Exception:
                    continue
                pt = {'timestep': ts, 'longitude': lon, 'latitude': lat}
                if wind_str is not None:
                    try:
                        pt['wind'] = float(wind_str)
                    except:
                        pass
                tracks_by_id[tid].append(pt)
                #tracks_by_id[tid].append({'timestep': ts, 'longitude': lon, 'latitude': lat})

        # finalize this month’s tracks
        for tid, pts in tracks_by_id.items():
            if not pts:
                continue
            pts.sort(key=lambda p: p['timestep'])
            track_year = int(pts[0]['timestep'][:4])
            all_tracks[track_year].append(pts)

    return all_tracks

# ----------------------- Flatten & cadence helper ----------------------------

def flatten_tracks_to_points(tracks_by_year):
    """
    Convert {year: [track, ...]} => sorted list of (Timestamp, lon, lat)
    for all tracks/years provided.
    """
    pts = []
    for _, tracks in tracks_by_year.items():
        for tr in tracks:
            for p in tr:
                ts = p['timestep']
                if isinstance(ts, str):
                    ts = pd.Timestamp(datetime.strptime(ts, "%Y%m%d%H"))
                elif isinstance(ts, datetime):
                    ts = pd.Timestamp(ts)
                pts.append((ts, float(p['longitude']), float(p['latitude'])))
    pts.sort(key=lambda x: x[0])
    return pts

# ---------------------------- Mask generation --------------------------------

def _to_grid_lon(lon_vals, grid_lon):
    """Map longitudes to the grid’s domain."""
    lon_vals = np.asarray(lon_vals, dtype=float)
    if np.nanmin(grid_lon) >= 0:        # 0..360 grid
        return np.mod(lon_vals, 360.0)
    else:                                # -180..180 grid
        return ((lon_vals + 180.0) % 360.0) - 180.0

def _flatten_storm_points(storm_points):
    """
    Accepts:
      - [(Timestamp, lon, lat), ...]
      - [(track_id, Timestamp, lon, lat), ...]
      - [{'track_id': ..., 'start_time': ..., 'points': [(Timestamp, lon, lat), ...]}, ...]
    Returns sorted flat list: [(Timestamp, lon, lat), ...]
    """
    out = []
    for p in storm_points:
        if isinstance(p, dict) and "points" in p:
            out.extend(p["points"])
        elif isinstance(p, (list, tuple)):
            if len(p) >= 4 and isinstance(p[1], (pd.Timestamp, np.datetime64, datetime)):
                ts, lon, lat = p[1], p[2], p[3]
                out.append((pd.Timestamp(ts), float(lon), float(lat)))
            elif len(p) >= 3:
                ts, lon, lat = p[0], p[1], p[2]
                out.append((pd.Timestamp(ts), float(lon), float(lat)))
        else:
            raise ValueError(f"Unrecognized storm point entry: {type(p)}")
    out.sort(key=lambda x: x[0])
    return out

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
    #print("sp",sp)
    print("len-sp",len(sp))
    if len(sp) == 0:
        xr.Dataset(
            {'storm_mask': xr.DataArray(
                np.zeros((len(times), len(lat), len(lon)), dtype=np.int16),
                coords={'time': times, 'latitude': lat, 'longitude': lon},
                dims=['time','latitude','longitude']
            )}
        ).to_netcdf(f'./output/Wnet_ETC_PRECIP_mask_{year}_{timestep}h.nc')
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
    out.to_netcdf(f'./output/Wnet_ETC_PRECIP_mask_{year}_{timestep}h.nc')
    ds.close()

# ------------------------------- USAGE ---------------------------------------

if __name__ == "__main__":
    wnet_data = read_monthly_wnet_tracks(wnet_dir)
    lsm_interpolator = load_land_sea_mask(LSM_PATH)

    wnet_data_land = filter_land_touching_tracks(
        wnet_data, lsm_interpolator, land_thresh=0.2,
        lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX
    )

    wnet_data_relevant = filter_conus_relevant(
        wnet_data, lsm_interpolator, land_thresh=0.2, buffer_km=400,
        lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX
    )

    checks = QCChecks(lifetime=True, speed=False, path=True,
                      wind=False, retro_run=True, retro_frac=True, net_east=True)

    thr = QCThresh(min_hours=48, min_points=8, min_path_km=1000,
                   max_retro_run_hours=24, max_retro_total_frac=0.33,
                   require_net_east=True, min_net_east_km=100)

    wnet_data_qc, wnet_qc_stats = filter_quality_tracks(
        wnet_data_relevant, checks, thr, return_stats=True
    )

    timestep = TIMESTEP_H
    radius_km = RADIUS_KM
    half_window_hours = HALF_WINDOW_H

    for year in range(START_YEAR, END_YEAR+1):  # END exclusive
        print(year)
        nc_file = f'./ERA5/input/ERA5_tp_input_{year}_conus.nc'
        wnet_kept_this_year = {year: wnet_data_qc.get(year, [])}
        storm_points = flatten_tracks_to_points(wnet_kept_this_year)
        generate_mask_timewindow_fast(
            nc_path=nc_file,
            storm_points=storm_points,
            year=year,
            radius_km=radius_km,
            timestep=timestep,
            half_window_hours=half_window_hours
        )

