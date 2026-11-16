# Build 6-hourly storm-type labels on the ERA5 CONUS grid with precedence:
# 1=MCS, 4=AR, 2=ETC, 3=HUR, 5=Other convective (if cp>0 and none of the four), 0=Unidentified.
# Also writes per-year likelihood NetCDFs.

import os
import numpy as np
import xarray as xr
from datetime import datetime

VERSION = "TE"
YEARS = range(1980, 2025)

# ------------------------- helpers -------------------------

def _subset_conus(ds, lat_name="latitude", lon_name="longitude"):
    """Subset to ~CONUS box: lat 20–60N, lon (0..360: 225–295E) or (-180..180: -135..-65)."""
    lat = ds[lat_name]
    if float(lat.values[0]) > float(lat.values[-1]):
        ds = ds.sel({lat_name: slice(60, 20)})
    else:
        ds = ds.sel({lat_name: slice(20, 60)})

    lon = ds[lon_name]
    lon_max = float(lon.max())
    if lon_max > 180:  # 0..360°
        ds = ds.sel({lon_name: slice(225, 295)})
    else:              # -180..180°
        if float(lon.values[0]) > float(lon.values[-1]):
            ds = ds.sel({lon_name: slice(-65, -135)})
        else:
            ds = ds.sel({lon_name: slice(-135, -65)})
    return ds

def _infer_dt_hours(times):
    """Median step in hours (float)."""
    t = np.asarray(times)
    if t.size < 2:
        return np.nan
    dt = np.diff(t.astype('datetime64[h]')).astype('timedelta64[h]').astype(int)
    return float(np.median(dt)) if dt.size else np.nan

def _hourly_to_6h_logical_or(mask_da):
    """
    Convert an hourly (or near-hourly) 0/1 mask to 6-hourly presence by logical OR over each 6h block.
    If the mask is already ~6-hourly, just coerce to 0/1 and return.
    Otherwise, resample to 6h bins.
    """
    m = xr.where(mask_da.fillna(0) > 0, 1, 0).astype("int8")
    step = _infer_dt_hours(m.time.values)
    # Already ~6-hourly?
    if np.isfinite(step) and 5.0 <= step <= 7.0:
        return m.astype("int8")
    # Strict hourly and length multiple of 6 -> coarsen
    if np.isfinite(step) and 0.5 <= step <= 1.5 and (m.time.size % 6 == 0):
        m6 = m.coarsen(time=6, boundary="trim").max()
        return m6.astype("int8")
    # Fallback: resample to 6h anchored to epoch
    try:
        m6 = m.resample(time="6h", origin="epoch", offset="0h").max()
    except TypeError:
        m6 = m.resample(time="6h").max()
    return m6.astype("int8")

def _to_lon180(lon_1d):
    lon = np.asarray(lon_1d, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0

def _apply_ar_west_of_105(ar_da):
    """
    Keep AR flags only west of 105°W (lon < -105). Input is 6-hourly binary tag.
    Works regardless of 0..360 or -180..180 lon.
    """
    lon_1d = _to_lon180(ar_da["longitude"].values)
    lon2d = xr.DataArray(
        np.broadcast_to(lon_1d[None, :], (ar_da.sizes["latitude"], ar_da.sizes["longitude"])),
        coords={"latitude": ar_da["latitude"], "longitude": ar_da["longitude"]},
        dims=("latitude", "longitude"),
    )
    ar01 = xr.where(ar_da.fillna(0) > 0, 1, 0).astype("int8")
    return xr.where(lon2d < -105.0, ar01, 0).astype("int8")

def _force_tll(da):
    """Ensure (time, latitude, longitude) dimension order."""
    return da.transpose("time", "latitude", "longitude")

def _tinfo(da, label):
    """Print basic time coverage for debugging alignment issues."""
    try:
        t = da.time.values
        if t.size:
            print(f"{label:10s}: nt={t.size}, [{str(t.min())}  →  {str(t.max())}]")
        else:
            print(f"{label:10s}: nt=0")
    except Exception:
        print(f"{label:10s}: (no time coord)")

def _match_lon_domain(da, target_lon):
    """
    Re-map da.longitude to match target_lon domain (0..360 or -180..180), then sort ascending.
    """
    if "longitude" not in da.coords:
        return da
    targ = np.asarray(target_lon)
    targ_0360 = np.nanmin(targ) >= 0.0
    lon = da["longitude"].values
    # Determine if da is 0..360 currently
    da_0360 = np.nanmin(lon) >= 0.0
    if targ_0360 and not da_0360:
        # -180..180 -> 0..360
        new_lon = (lon % 360.0)
        da = da.assign_coords(longitude=("longitude", new_lon)).sortby("longitude")
    elif (not targ_0360) and da_0360:
        # 0..360 -> -180..180
        new_lon = ((lon + 180.0) % 360.0) - 180.0
        da = da.assign_coords(longitude=("longitude", new_lon)).sortby("longitude")
    else:
        # Just make sure it's sorted
        da = da.sortby("longitude")
    return da

def _ensure_ascending_lat(da):
    if "latitude" in da.coords:
        lat = da["latitude"].values
        if lat.size > 1 and lat[0] > lat[-1]:
            da = da.sel(latitude=slice(lat[0], lat[-1]))  # reverses to ascending
    return da

def _interp_mask_to_grid(mask_da, target_lat, target_lon, method="nearest"):
    """
    Interpolate a 0/1 mask (time, lat, lon) to target (lat, lon) grid with nearest-neighbor,
    keeping 0/1 values.
    """
    da = _ensure_ascending_lat(mask_da)
    da = _match_lon_domain(da, target_lon)
    # xarray.interp expects monotonic ascending coords
    if np.any(np.diff(da.latitude.values) < 0):
        da = da.sortby("latitude")
    if np.any(np.diff(da.longitude.values) < 0):
        da = da.sortby("longitude")
    # Cast to float for interp, then threshold back to 0/1
    da_f = da.astype("float32")
    out = da_f.interp(latitude=target_lat, longitude=target_lon, method=method)
    return xr.where(out > 0.5, 1, 0).astype("int8")

# ---------- likelihood helpers ----------

def compute_type_likelihoods(tp_mm, mcs6, ar6, etc6, hur6, cp_mm=None, precip_threshold=0.0):
    """
    Likelihood(type) = 1 / (# of active types) if that type's mask is active and precip>threshold, else 0.
    """
    tp_mm, mcs6, ar6, etc6, hur6 = xr.align(tp_mm, mcs6, ar6, etc6, hur6, join="exact")

    M = xr.where(mcs6 > 0, 1, 0).astype("int8")
    A = xr.where(ar6  > 0, 1, 0).astype("int8")
    E = xr.where(etc6 > 0, 1, 0).astype("int8")
    H = xr.where(hur6 > 0, 1, 0).astype("int8")

    precip_ok = (tp_mm > precip_threshold)
    # Order corresponds to stormtype codes:
# 1=MCS, 2=ETC, 3=HUR, 4=AR
    mask4 = xr.concat([M, E, H, A], dim="type")
    mask4 = mask4.assign_coords(
        type=("type", [1, 2, 3, 4])
        ).astype("float32")

    mask4["type"].attrs = {
    "long_name": "storm-type code",
    "flag_values": np.array([1, 2, 3, 4], dtype="int8"),
    "flag_meanings": "MCS ETC HUR AR",
    }
    mask4 = xr.where(precip_ok, mask4, 0.0)

    active_cnt = mask4.sum("type")
    active_cnt_safe = xr.where(active_cnt > 0, active_cnt, 1.0)

    likelihood = (mask4 / active_cnt_safe).astype("float32")
    identified_prob = likelihood.sum("type").astype("float32")

    other_conv_prob = None
    if cp_mm is not None:
        cp_mm = cp_mm.sel(time=likelihood.time, latitude=likelihood.latitude, longitude=likelihood.longitude)
        other_conv_prob = xr.where((identified_prob==0) & (cp_mm>0), 1.0, 0.0).astype("float32")

    return likelihood, identified_prob, other_conv_prob

def likelihood_to_dataset(lik, identified_prob, other_conv_prob=None, extra_attrs=None):
    ds = xr.Dataset(
        data_vars={
            "likelihood": lik.astype("float32"),
            "identified_prob": identified_prob.astype("float32"),
        },
        coords={
            "type": lik.coords["type"],
            "time": lik.coords["time"],
            "latitude": lik.coords["latitude"],
            "longitude": lik.coords["longitude"],
        },
    )
    if other_conv_prob is not None:
        ds["other_conv_prob"] = other_conv_prob.astype("float32")

    ds["likelihood"].attrs = {
        "long_name": "Storm-type likelihood (equal weight among active types)",
        "units": "1",
        "description": "If N types are active with precip>threshold, each gets 1/N; else 0.",
    }
    ds["identified_prob"].attrs = {
        "long_name": "Probability that precip is identified as one of the four storm types",
        "units": "1",
        "description": "Sum of type likelihoods; 0..1.",
    }
    if "other_conv_prob" in ds:
        ds["other_conv_prob"].attrs = {
            "long_name": "Other convective probability",
            "units": "1",
            "description": "1 where cp>0 and none of {MCS,AR,ETC,HUR} active; else 0.",
        }
    ds["type"].attrs = {
    "long_name": "storm-type code",
    "code_1": "MCS",
    "code_2": "ETC",
    "code_3": "HUR",
    "code_4": "AR",
    }
    ds["latitude"].attrs = {"units": "degrees_north"}
    ds["longitude"].attrs = {"units": "degrees_east"}
    ds.attrs.update({
        "title": "Storm-type likelihoods and identification probabilities",
        "source": "compute_type_likelihoods() from user workflow",
        "Conventions": "CF-1.8",
        "history": f"created {datetime.utcnow().isoformat()}Z",
    })
    if extra_attrs: ds.attrs.update(extra_attrs)
    return ds

def save_likelihood_netcdf(lik, identified_prob, other_conv_prob, out_path, extra_attrs=None, complevel=4):
    ds = likelihood_to_dataset(lik, identified_prob, other_conv_prob, extra_attrs=extra_attrs)
    enc = {
        "likelihood":       {"zlib": True, "complevel": complevel, "dtype": "float32"},
        "identified_prob":  {"zlib": True, "complevel": complevel, "dtype": "float32"},
    }
    if "other_conv_prob" in ds:
        enc["other_conv_prob"] = {"zlib": True, "complevel": complevel, "dtype": "float32"}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ds.to_netcdf(out_path, format="NETCDF4", engine="netcdf4", encoding=enc)
    return out_path

# ------------------------- main loop -------------------------

if __name__ == "__main__":
    for year in YEARS:
        print(f"\nProcessing {year} ...")

        # ---------- open & subset inputs ----------
        ds_tp = _subset_conus(xr.open_dataset(f"./ERA5/input/ERA5_tp_input_{year}_6hourly_conus.nc"))
        ds_cp = _subset_conus(xr.open_dataset(f"./ERA5/input/ERA5_cp_input_{year}_6hourly_conus.nc"))

        ds_hur = _subset_conus(xr.open_dataset(f"./Precip_Classification/output/{VERSION}_HUR_PRECIP_mask_{year}_6h.nc"))
        ds_etc = _subset_conus(xr.open_dataset(f"./Precip_Classification/output/{VERSION}_ETC_PRECIP_mask_{year}_6h.nc")) 

        ds_mcs = xr.open_dataset(f"./MCSs/{VERSION}/output/ERA5_sb_mcs_olr_tp_{year}_conus_min6h_pf.nc")
        # If needed: ds_mcs = _subset_conus(ds_mcs)

        ds_ar  = _subset_conus(xr.open_dataset(f"./ARs/{VERSION}/output/ERA5_AR_NFF_HUR_ETC_{year}_6hr.nc"))

        # ---------- extract variables ----------
        tp_mm = (ds_tp["tp"] * 1000.0)  # mm
        cp_mm = (ds_cp["cp"] * 1000.0)  # mm

        # Hourly masks to 6-hourly (or pass-through if already 6-hourly)
        hur6 = _hourly_to_6h_logical_or(ds_hur["storm_mask"])
        etc6 = _hourly_to_6h_logical_or(ds_etc["storm_mask"])
        mcs6 = _hourly_to_6h_logical_or(ds_mcs["MCS_pf"])

        # AR: keep only west of 105W, ensure 0/1 (already 6-hourly)
        ar6 = _apply_ar_west_of_105(ds_ar["binary_tag"])

        # ---------- ensure consistent (time, lat, lon) order ----------
        tp_mm  = _force_tll(tp_mm)
        cp_mm  = _force_tll(cp_mm)
        mcs6   = _force_tll(mcs6)
        etc6   = _force_tll(etc6)
        hur6   = _force_tll(hur6)
        ar6    = _force_tll(ar6)

        # ---------- diagnostics: raw time coverage ----------
        _tinfo(tp_mm, "TP_6h")
        _tinfo(mcs6,  "MCS6(raw)")
        _tinfo(etc6,  "ETC6(raw)")
        _tinfo(hur6,  "HUR6(raw)")
        _tinfo(ar6,   "AR6(raw)")

        # ---------- snap all masks to tp_mm.time (±3 h) ----------
        tol = np.timedelta64(3, "h")
        mcs6 = mcs6.reindex(time=tp_mm.time, method="nearest", tolerance=tol)
        etc6 = etc6.reindex(time=tp_mm.time, method="nearest", tolerance=tol)
        hur6 = hur6.reindex(time=tp_mm.time, method="nearest", tolerance=tol)
        ar6  = ar6 .reindex(time=tp_mm.time, method="nearest", tolerance=tol)

        # Fill NaNs produced by reindex misses with 0 and warn
        for name, da in {"MCS": mcs6, "ETC": etc6, "HUR": hur6, "AR": ar6}.items():
            if da.isnull().any():
                miss = int(da.isnull().any(dim=("latitude","longitude")).sum().item())
                if miss > 0:
                    print(f"⚠️  {name}: {miss} timesteps had no match within ±3h; filling with 0")
                da[:] = da.fillna(0)

        # ---------- REGRID masks to ERA5 tp grid (nearest) ----------
        tlat = tp_mm["latitude"]
        tlon = tp_mm["longitude"]
        mcs6 = _interp_mask_to_grid(_match_lon_domain(mcs6, tlon), tlat, tlon, method="nearest")
        etc6 = _interp_mask_to_grid(_match_lon_domain(etc6, tlon), tlat, tlon, method="nearest")
        hur6 = _interp_mask_to_grid(_match_lon_domain(hur6, tlon), tlat, tlon, method="nearest")
        ar6  = _interp_mask_to_grid(_match_lon_domain(ar6,  tlon), tlat, tlon, method="nearest")

        # After snapping & regridding, times and grids should match exactly
        tp_mm, cp_mm, mcs6, etc6, hur6, ar6 = xr.align(
            tp_mm, cp_mm, mcs6, etc6, hur6, ar6, join="exact", copy=False
        )


        # ---------- likelihoods ----------
        lik, ident_p, oc_p = compute_type_likelihoods(
            tp_mm=tp_mm, mcs6=mcs6, ar6=ar6, etc6=etc6, hur6=hur6,
            cp_mm=cp_mm, precip_threshold=0.0
        )
        out_lik = f"./likelihood/stormtype_likelihoods_{VERSION}_{year}_6hr.nc"
        save_likelihood_netcdf(
            lik, ident_p, oc_p, out_path=out_lik,
            extra_attrs={"era5_version": "HRES", "domain": "CONUS", "cadence": "6-hourly"}
        )
        print(f"Wrote likelihoods → {out_lik}")

        # ---------- close ----------
        ds_tp.close(); ds_cp.close()
        ds_hur.close(); ds_etc.close(); ds_mcs.close(); ds_ar.close()

    print("\nDone.")

