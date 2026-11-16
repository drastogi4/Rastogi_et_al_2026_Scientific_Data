# ==== ETC tracks -> filtered -> precip mask (end-to-end) =====================
# Cleaned + unified version (6-hour TRACKs; robust distance + filters)

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

START_YEAR = 1980
END_YEAR_INCLUSIVE = 2024  # inclusive
LAT_MIN, LAT_MAX = 24.0, 50.0
LON_MIN, LON_MAX = -125.0, -66.0

TRACK_DIR = './ETCs/TRACK/'
TRACK_SUBDIR_FMT = "ERA5_VOR850_6hr_{year}_DET/ff_trs_pos_{year}"  # 6h only
LSM_PATH = "./era5-land-sea-mask.nc"

# Mask build params
TIMESTEP_H = 6
RADIUS_KM = 1000.0
HALF_WINDOW_H = 3  # ±3h around each analysis time

# QC: choose whether "path" means cumulative length or net displacement
QC_PATH_IS_CUMULATIVE = True  # set False to use end-to-end displacement

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
        # Wrapped box (unlikely for CONUS)
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

    # Clamp lat to range
    clat = min(max(lat, lat_min), lat_max)

    # Clamp lon to range in [-180,180)
    lon = lon_to_180(lon)
    lon_min = lon_to_180(lon_min)
    lon_max = lon_to_180(lon_max)

    if lon_min <= lon_max:
        clon = min(max(lon, lon_min), lon_max)
        return great_circle_km(lat, lon, clat, clon)
    else:
        # Wrapped: distance to either vertical edge
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
    for i in range(n-1):
        dh = (track_times[i+1] - track_times[i]).total_seconds()/3600.0
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

# --------------------------- QC over tracks ----------------------------------

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

# ------------------------ Land-sea mask & filters ----------------------------

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

        # sort longitudes (defensive)
        lon_sort_idx = np.argsort(lons)
        lons_sorted = lons[lon_sort_idx]

        # ensure ascending latitude
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

# ---------------------------- TRACK reading ----------------------------------

def read_track_txt(file_path):
    """
    Reads TRACK 'ff_trs_pos_{year}' files that contain blocks starting with 'TRACK_ID'.
    Returns a list of tracks; each track is a list of dict points with keys:
      'timestep' (YYYYMMDDHH), 'latitude', 'longitude' (in [-180,180))
    """
    tracks = []
    full_track = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("TRACK_ID"):
                if full_track:
                    tracks.append(full_track)
                full_track = []
            elif re.match(r"^\d{10}", line):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        timestep = parts[0]
                        lon = lon_to_180(float(parts[1]))
                        lat = float(parts[2])
                        full_track.append({
                            'timestep': timestep,
                            'latitude': lat,
                            'longitude': lon
                        })
                    except ValueError:
                        continue
    if full_track:
        tracks.append(full_track)
    return tracks

def read_all_track_files(track_dir, start_year, end_year_inclusive):
    """Reads 6-hour TRACK files into {year: [track, ...]}."""
    all_tracks = defaultdict(list)
    for year in range(start_year, end_year_inclusive + 1):
        file_path = os.path.join(track_dir, TRACK_SUBDIR_FMT.format(year=year))
        print(f'Reading TRACK file: {file_path}')
        if os.path.exists(file_path):
            all_tracks[year] = read_track_txt(file_path)
        else:
            print(f'  (missing) {file_path}')
    return all_tracks

# ----------------------- Flatten + cadence sanity checks ---------------------

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
                # else assume pd.Timestamp already
                pts.append((ts, float(p['longitude']), float(p['latitude'])))
    pts.sort(key=lambda x: x[0])
    return pts

def infer_cadence_hours_from_points(storm_points):
    """Infer most common time step (hours) from point timestamps; returns int or None."""
    if not storm_points or len(storm_points) < 2:
        return None
    ts = sorted({p[0] for p in storm_points})
    dts = pd.Series(ts).diff().dropna().dt.total_seconds()/3600.0
    if dts.empty:
        return None
    # round to nearest integer hour to be robust to tiny numeric quirks
    return int(np.round(dts.mode().iloc[0]))

# ---------------------------- Mask generation --------------------------------

def generate_mask_timewindow_fast(nc_path, storm_points, year, radius_km, timestep_hours, half_window_hours=12):
    """
    Build storm influence masks by union-of-discs around storm centers that fall
    within [t-half_window, t+half_window] of each analysis time.
    """
    ds = xr.open_dataset(nc_path)  # add chunks={'time': 24} if your backend supports dask
    lat = ds.latitude.values
    lon = ds.longitude.values
    times = pd.to_datetime(ds.time.values)

    # Early exit if nothing to do
    if len(storm_points) == 0:
        xr.Dataset(
            {'storm_mask': xr.DataArray(
                np.zeros((len(times), len(lat), len(lon)), dtype=np.int16),
                coords={'time': times, 'latitude': lat, 'longitude': lon},
                dims=['time','latitude','longitude']
            )}
        ).to_netcdf(f'./output/TRACK_ETCs_PRECIP_mask_{year}_{timestep_hours}h.nc')
        return

    # grid in radians
    latg, longg = np.meshgrid(lat, lon, indexing='ij')
    latg_r = np.deg2rad(latg)
    longg_r = np.deg2rad(longg)
    cos_latg = np.cos(latg_r)

    # Sort storms and precompute arrays
    sp = sorted(storm_points, key=lambda x: x[0])
    storm_times = np.array([p[0].to_datetime64() for p in sp])
    clon_r_all = np.deg2rad(np.array([p[1] for p in sp]))
    clat_r_all = np.deg2rad(np.array([p[2] for p in sp]))

    # Threshold for haversine "a" term (no arcsin):  a <= sin(radius/(2R))^2
    a_thresh = np.sin((radius_km / (2.0 * EARTH_RADIUS_KM)))**2

    mask = np.zeros((len(times), len(lat), len(lon)), dtype=bool)
    win = np.timedelta64(int(half_window_hours), 'h')

    st = storm_times
    for ti, t in enumerate(times.values.astype('datetime64[ns]')):
        left  = np.searchsorted(st, t - win, side='left')
        right = np.searchsorted(st, t + win, side='left')
        if right <= left:
            continue

        clat_active = clat_r_all[left:right]
        clon_active = clon_r_all[left:right]

        within_any = np.zeros_like(latg, dtype=bool)

        # Vectorize across grid; small loop over active centers
        for clatr, clonr in zip(clat_active, clon_active):
            dlat = clatr - latg_r
            dlon = ((clonr - longg_r + np.pi) % (2*np.pi)) - np.pi  # wrap to [-pi, pi]
            sin_dlat2 = np.sin(dlat * 0.5)
            sin_dlon2 = np.sin(dlon * 0.5)
            a = sin_dlat2*sin_dlat2 + cos_latg * np.cos(clatr) * sin_dlon2*sin_dlon2
            within_any |= (a <= a_thresh)
            # Optional early-exit:
            # if within_any.all(): break

        mask[ti] = within_any

    xr.Dataset(
        {'storm_mask': xr.DataArray(mask.astype(np.int16),
                                    coords={'time': times, 'latitude': lat, 'longitude': lon},
                                    dims=['time','latitude','longitude'])}
    ).to_netcdf(f'./output/TRACK_ETCs_PRECIP_mask_{year}_{timestep_hours}h.nc')

# --------------------------------- MAIN --------------------------------------

if __name__ == "__main__":
    # Read TRACK data (6-hour files)
    track_data = read_all_track_files(TRACK_DIR, START_YEAR, END_YEAR_INCLUSIVE)

    # Land-sea mask interpolator
    lsm_interpolator = load_land_sea_mask(LSM_PATH)

    # Spatial relevance filters
    track_data_land = filter_land_touching_tracks(
        track_data, lsm_interpolator, land_thresh=0.2,
        lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX
    )
    track_data_relevant = filter_conus_relevant(
        track_data, lsm_interpolator, land_thresh=0.2, buffer_km=400,
        lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX
    )

    # QC filters
    checks = QCChecks(lifetime=True, speed=False, path=True,
                      wind=False, retro_run=True, retro_frac=True, net_east=True)
    thr = QCThresh(min_hours=48, min_points=8, min_path_km=1000,
                   max_retro_run_hours=24, max_retro_total_frac=0.33,
                   require_net_east=True, min_net_east_km=100)

    track_data_qc, track_qc_stats = filter_quality_tracks(
        track_data_relevant, checks, thr, return_stats=True
    )

    # Build masks year by year from QC-kept tracks
    for year in range(START_YEAR, END_YEAR_INCLUSIVE + 1):
        print(year)
        nc_file = f'./ERA5/input/ERA5_tp_input_{year}_conus.nc'

        # Only tracks for this year
        tracks_kept_this_year = {year: track_data_qc.get(year, [])}
        storm_points = flatten_tracks_to_points(tracks_kept_this_year)

        # Cadence guard: expect 6-hour
        cad = infer_cadence_hours_from_points(storm_points)
        if cad is not None and cad != 6:
            raise ValueError(f"Unexpected track cadence for {year}: {cad}h (expect 6h). "
                             f"Check you are reading 6-hour TRACK files.")

        generate_mask_timewindow_fast(
            nc_path=nc_file,
            storm_points=storm_points,
            year=year,
            radius_km=RADIUS_KM,
            timestep_hours=TIMESTEP_H,
            half_window_hours=HALF_WINDOW_H
        )

