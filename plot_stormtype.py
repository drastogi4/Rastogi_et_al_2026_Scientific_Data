# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --------------------------- Config ---------------------------------
version = "TE"  # filename suffix for stormtype files
# --------------------------------------------------------------------

# -------------------------- Helpers ---------------------------------
def _subset_conus(ds, lat_name="latitude", lon_name="longitude"):
    """
    Subset to a CONUS-like box:
      LAT: 20–50°N
      LON: 235–295°E if 0..360; otherwise -125..-66
    Keeps input coordinate order (ascending/descending) in slicing.
    """
    # LAT
    if ds[lat_name].values[0] > ds[lat_name].values[-1]:  # descending (ERA-style)
        ds = ds.sel({lat_name: slice(50, 20)})
    else:
        ds = ds.sel({lat_name: slice(20, 50)})

    # LON
    lon = ds[lon_name]
    if float(lon.max()) > 180:  # 0..360 convention
        ds = ds.sel({lon_name: slice(235, 295)})
    else:  # -180..180
        if lon.values[0] > lon.values[-1]:  # descending
            ds = ds.sel({lon_name: slice(-66, -125)})
        else:
            ds = ds.sel({lon_name: slice(-125, -66)})
    return ds


def _ensure_ascending_for_interp(da, lat_name="latitude", lon_name="longitude"):
    """
    xarray.interp expects monotonic ascending coords.
    Return a view sorted ascending in each dimension if needed.
    """
    if np.any(np.diff(da[lat_name].values) < 0):
        da = da.sortby(lat_name)
    if np.any(np.diff(da[lon_name].values) < 0):
        da = da.sortby(lon_name)
    return da


def _interp_numeric_to_storm_grid(da, lat_st, lon_st, method="nearest"):
    """
    Interpolate a numeric DataArray to the (lat_st, lon_st) storm grid.
    Casts coords to ascending for xarray.interp first.
    """
    da = _ensure_ascending_for_interp(da, "latitude", "longitude")
    return da.interp(
        latitude=np.asarray(lat_st),
        longitude=np.asarray(lon_st),
        method=method
    )


def _interp_bool_to_storm_grid(rain_daily_bool, lat_st, lon_st, method="nearest"):
    """
    Regrid a boolean field: sort -> cast float -> interp -> threshold back to bool.
    """
    rain_daily_bool = _ensure_ascending_for_interp(rain_daily_bool, "latitude", "longitude")
    rain_daily_f = rain_daily_bool.astype("float32")
    out = rain_daily_f.interp(
        latitude=np.asarray(lat_st),
        longitude=np.asarray(lon_st),
        method=method
    )
    return out > 0.5


def _attach_season_coords(da):
    """
    Add:
      - season: 'DJF','MAM','JJA','SON' (xarray built-in)
      - syear:  assigns December to next year (for DJF spanning)
    """
    t = da['time']
    season = t.dt.season
    syear = xr.where(t.dt.month == 12, t.dt.year + 1, t.dt.year)
    return da.assign_coords(season=season, syear=syear)


def _accum_by_syear(acc, new):
    """Accumulate DataArray with a 'syear' coord by summing duplicates."""
    if acc is None:
        return new
    return xr.concat([acc, new], dim="syear").groupby("syear").sum()


# ------------------ Regridded “days” + rain-day panels ------------------
def load_stormtype_and_rain_days_regridded(
    stormtype_dir, precip_dir, years, threshold_mm=1.0, interp_method="nearest"
):
    """
    For each year:
      - Count per-grid-cell “days” for each storm type (1=MCS,2=ETC,3=HUR,4=AR,5=Other convective)
        by summing 6-hourly flags and dividing by 4.
      - Compute rain days (≥ threshold_mm/day) and “rain but no storm type”.

    Returns average across years:
      mcs_days, etc_days, hur_days, ar_days, oc_days, avg_rain_days, rain_only_no_type, lat_st, lon_st
    """
    acc = {1: None, 2: None, 3: None, 4: None, 5: None}
    sum_rain_days = None
    sum_rain_only_no_type = None
    lat_st, lon_st = None, None

    for y in years:
        # Stormtype
        st_fp = os.path.join(stormtype_dir, f"stormtype_{y}_6hr_{version}.nc")
        if not os.path.exists(st_fp):
            print(f"⚠️ Missing stormtype file: {st_fp}")
            continue

        ds_st = xr.open_dataset(st_fp)
        ds_st = _subset_conus(ds_st, "latitude", "longitude")

        if lat_st is None:
            lat_st = ds_st["latitude"].values
            lon_st = ds_st["longitude"].values

        st_da = ds_st["stormtype"]  # (time, lat, lon)

        # Per-type “days”: 6-hourly → daily-equivalent by /4
        for code in (1, 2, 3, 4, 5):
            days = (st_da == code).sum(dim="time") / 4.0
            acc[code] = days if acc[code] is None else (acc[code] + days)

        # Any storm (for rain-only-no-type)
        any_storm_daily = st_da.isin([1, 2, 3, 4, 5]).resample(time="1D").max(dim="time")

        # Precip -> daily rain boolean
        pr_fp = os.path.join(precip_dir, f"ERA5_tp_input_{y}_6hourly_conus.nc")
        if not os.path.exists(pr_fp):
            print(f"⚠️ Missing precip file: {pr_fp}")
            ds_st.close()
            continue

        ds_pr = xr.open_dataset(pr_fp)
        ds_pr = _subset_conus(ds_pr, "latitude", "longitude")
        pr_mm = ds_pr["tp"] * 1000.0
        rain_daily_bool = (pr_mm.resample(time="1D").sum() >= threshold_mm)
        rain_daily_bool_rg = _interp_bool_to_storm_grid(rain_daily_bool, lat_st, lon_st, interp_method)

        # Align timelines
        rain_daily_bool_rg, any_storm_daily = xr.align(rain_daily_bool_rg, any_storm_daily, join="inner")

        rain_days_y = rain_daily_bool_rg.sum(dim="time")
        rain_only_no_type_y = (rain_daily_bool_rg & (~any_storm_daily)).sum(dim="time")

        sum_rain_days = rain_days_y if sum_rain_days is None else (sum_rain_days + rain_days_y)
        sum_rain_only_no_type = (
            rain_only_no_type_y if sum_rain_only_no_type is None else (sum_rain_only_no_type + rain_only_no_type_y)
        )

        ds_st.close(); ds_pr.close()

    n_years = len(years)
    for code in acc:
        if acc[code] is not None:
            acc[code] = (acc[code] / n_years).values
    avg_rain_days_rg = (sum_rain_days / n_years).values if sum_rain_days is not None else None
    avg_rain_only_no_type = (sum_rain_only_no_type / n_years).values if sum_rain_only_no_type is not None else None

    return (acc[1], acc[2], acc[3], acc[4], acc[5],
            avg_rain_days_rg, avg_rain_only_no_type, lat_st, lon_st)


def plot_storms_and_rain_days_subplots(
    mcs, etc, hur, ar, oc, rain_days, rain_only_no_type,
    lat, lon, year, save_path=None
):
    """
    7-panel figure:
      [Rain days, MCS, ETC, HUR, AR, Other convective, Rain days (no storm type)]
    """
    data_list = [rain_days, mcs, etc, hur, ar, oc, rain_only_no_type]
    titles = [
        'Rain Days ≥ 1 mm', 'MCS Days', 'ETC Days', 'Hurricane Days',
        'AR Days', 'Other convective Days', 'Rain Days ≥ 1 mm (No Storm Type)'
    ]
    cmaps = ['Blues', 'Greens', 'OrRd', 'YlGnBu', 'PuRd', 'BuGn', 'Greys']
    vmaxs = [200, 50, 50, 10, 50, 50, 100]  # adjust if needed

    n = len(data_list)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.8*nrows),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = np.atleast_1d(axes).ravel()

    lon2d, lat2d = np.meshgrid(lon, lat)
    for i in range(n):
        ax = axes[i]
        C = np.asarray(data_list[i])
        im = ax.pcolormesh(lon2d, lat2d, C, cmap=cmaps[i], shading='nearest', vmin=0, vmax=vmaxs[i])
        ax.coastlines(); ax.add_feature(cfeature.STATES)
        ax.set_extent([-125, -66, 20, 50], crs=ccrs.PlateCarree())
        ax.set_title(titles[i])
        plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)

    for j in range(n, nrows*ncols):
        axes[j].axis('off')

    fig.suptitle(f"Storm Types and Rain Days – {year}", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ------------------ % of rainy days (annual mean) --------------------
def compute_percent_rainyday_types(
    stormtype_dir, precip_dir, years, threshold_mm=1.0, interp_method="nearest"
):
    """
    Average (over years) % of rainy days per grid cell labeled as:
      1 MCS, 2 ETC, 3 HUR, 4 AR, 5 Other convective, 0 Unidentified.
    """
    yearly_sums = {k: None for k in ["MCS", "ETC", "HUR", "AR", "OCONV", "UNID"]}
    lat_st, lon_st = None, None
    n_years_eff = 0

    for y in years:
        st_fp = os.path.join(stormtype_dir, f"stormtype_{y}_6hr_{version}.nc")
        pr_fp = os.path.join(precip_dir,   f"ERA5_tp_input_{y}_6hourly_conus.nc")
        if not (os.path.exists(st_fp) and os.path.exists(pr_fp)):
            print(f"⚠️ Skipping {y}: missing file(s).")
            continue

        ds_st = xr.open_dataset(st_fp)
        ds_st = _subset_conus(ds_st, "latitude", "longitude")
        if lat_st is None:
            lat_st = ds_st["latitude"].values
            lon_st = ds_st["longitude"].values
        st_da = ds_st["stormtype"]  # ints 0..5

        # Daily dominant type among 1..5; 0 if no type that day
        cat_bool = xr.concat([(st_da == 1), (st_da == 2), (st_da == 3), (st_da == 4), (st_da == 5)], dim="cat")
        daily_counts = cat_bool.resample(time="1D").sum("time")  # (time, cat, lat, lon)
        max_counts = daily_counts.max("cat")
        cat_idx = daily_counts.argmax("cat")  # 0..4
        daily_label = xr.where(max_counts > 0, (cat_idx + 1), 0).astype(np.int8)

        # Precip → daily rain boolean → regrid to storm grid
        ds_pr = xr.open_dataset(pr_fp)
        ds_pr = _subset_conus(ds_pr, "latitude", "longitude")
        pr_mm = ds_pr["tp"] * 1000.0
        rain_daily_bool = (pr_mm.resample(time="1D").sum() >= threshold_mm)
        rain_daily_bool_rg = _interp_bool_to_storm_grid(rain_daily_bool, lat_st, lon_st, interp_method)

        # Align time axes
        daily_label, rain_daily_bool_rg = xr.align(daily_label, rain_daily_bool_rg, join="inner")

        # Denominator + counts
        denom = rain_daily_bool_rg.sum("time")
        counts = {}
        counts["MCS"]   = ((daily_label == 1) & rain_daily_bool_rg).sum("time")
        counts["ETC"]   = ((daily_label == 2) & rain_daily_bool_rg).sum("time")
        counts["HUR"]   = ((daily_label == 3) & rain_daily_bool_rg).sum("time")
        counts["AR"]    = ((daily_label == 4) & rain_daily_bool_rg).sum("time")
        counts["OCONV"] = ((daily_label == 5) & rain_daily_bool_rg).sum("time")
        counts["UNID"]  = (rain_daily_bool_rg & (daily_label == 0)).sum("time")

        valid = denom > 0
        year_pct = {k: xr.where(valid, (counts[k] / denom) * 100.0, np.nan) for k in counts}

        # Equal-weight across years
        for k in yearly_sums:
            yearly_sums[k] = year_pct[k] if yearly_sums[k] is None else (yearly_sums[k] + year_pct[k])

        n_years_eff += 1
        ds_st.close(); ds_pr.close()

    if n_years_eff == 0:
        raise RuntimeError("No valid years processed.")

    return (yearly_sums["MCS"] / n_years_eff).values, \
           (yearly_sums["ETC"] / n_years_eff).values, \
           (yearly_sums["HUR"] / n_years_eff).values, \
           (yearly_sums["AR"] / n_years_eff).values, \
           (yearly_sums["OCONV"] / n_years_eff).values, \
           (yearly_sums["UNID"] / n_years_eff).values, \
           lat_st, lon_st


# -------------------- Precip amounts & percentages -------------------
def compute_precip_amount_by_stormtype(
    stormtype_dir, precip_dir, years, interp_method="nearest"
):
    """
    Attribute 6-hourly precipitation to the storm type at that timestep.
    Returns:
      - average-annual totals (mm/yr) per grid cell for MCS, ETC, HUR, AR, Other convective, UNID
      - average of annual percentages (%) per grid cell for the same categories
      - lat_st, lon_st
    """
    sum_yearly_amt = {k: None for k in ["MCS", "ETC", "HUR", "AR", "OCONV", "UNID"]}
    sum_yearly_pct = {k: None for k in ["MCS", "ETC", "HUR", "AR", "OCONV", "UNID"]}
    lat_st, lon_st = None, None
    n_years_eff = 0

    for y in years:
        st_fp = os.path.join(stormtype_dir, f"stormtype_{y}_6hr_{version}.nc")
        pr_fp = os.path.join(precip_dir,   f"ERA5_tp_input_{y}_6hourly_conus.nc")
        if not (os.path.exists(st_fp) and os.path.exists(pr_fp)):
            print(f"⚠️ Skipping {y}: missing file(s).")
            continue

        # Stormtype grid
        ds_st = xr.open_dataset(st_fp)
        ds_st = _subset_conus(ds_st, "latitude", "longitude")
        if lat_st is None:
            lat_st = ds_st["latitude"].values
            lon_st = ds_st["longitude"].values
        st_da = ds_st["stormtype"]

        # Precip → mm → interp to storm grid
        ds_pr = xr.open_dataset(pr_fp)
        ds_pr = _subset_conus(ds_pr, "latitude", "longitude")
        tp_mm = ds_pr["tp"] * 1000.0
        tp_rg = _interp_numeric_to_storm_grid(tp_mm, lat_st, lon_st, interp_method)

        # Align time
        st_da, tp_rg = xr.align(st_da, tp_rg, join="inner")

        # Annual totals by type
        parts_amt = {
            "MCS":   tp_rg.where(st_da == 1).sum("time"),
            "ETC":   tp_rg.where(st_da == 2).sum("time"),
            "HUR":   tp_rg.where(st_da == 3).sum("time"),
            "AR":    tp_rg.where(st_da == 4).sum("time"),
            "OCONV": tp_rg.where(st_da == 5).sum("time"),
            "UNID":  tp_rg.where(~st_da.isin([1, 2, 3, 4, 5])).sum("time"),
        }
        parts_amt = {k: v.astype("float32") for k, v in parts_amt.items()}

        # Percentages per year
        total_y = sum(parts_amt.values())
        year_pct = {
            k: xr.where(total_y > 0, (parts_amt[k] / total_y) * 100.0, np.nan).astype("float32")
            for k in parts_amt
        }

        # Accumulate
        for k in parts_amt:
            sum_yearly_amt[k] = parts_amt[k] if sum_yearly_amt[k] is None else (sum_yearly_amt[k] + parts_amt[k])
            sum_yearly_pct[k] = year_pct[k]  if sum_yearly_pct[k] is None else (sum_yearly_pct[k] + year_pct[k])

        n_years_eff += 1
        ds_st.close(); ds_pr.close()

    if n_years_eff == 0:
        raise RuntimeError("No valid years processed.")

    pr_mcs = (sum_yearly_amt["MCS"] / n_years_eff).values
    pr_etc = (sum_yearly_amt["ETC"] / n_years_eff).values
    pr_hur = (sum_yearly_amt["HUR"] / n_years_eff).values
    pr_ar  = (sum_yearly_amt["AR"]  / n_years_eff).values
    pr_oc  = (sum_yearly_amt["OCONV"] / n_years_eff).values
    pr_unid= (sum_yearly_amt["UNID"] / n_years_eff).values

    pct_mcs = (sum_yearly_pct["MCS"] / n_years_eff).values
    pct_etc = (sum_yearly_pct["ETC"] / n_years_eff).values
    pct_hur = (sum_yearly_pct["HUR"] / n_years_eff).values
    pct_ar  = (sum_yearly_pct["AR"]  / n_years_eff).values
    pct_oc  = (sum_yearly_pct["OCONV"] / n_years_eff).values
    pct_unid= (sum_yearly_pct["UNID"] / n_years_eff).values

    return (pr_mcs, pr_etc, pr_hur, pr_ar, pr_oc, pr_unid,
            pct_mcs, pct_etc, pct_hur, pct_ar, pct_oc, pct_unid,
            lat_st, lon_st)


# ----------------------------- Plot utils ----------------------------
def plot_eight_maps_2x4(data_list, titles, lat, lon, main_title,
                        cmaps=None, vmaxs=None, vmins=None, save_path=None):
    """
    8-panel (2x4) cartopy plot with per-panel vmin/vmax support.
    """
    assert len(data_list) == 8 and len(titles) == 8, "Need exactly 8 panels."

    if cmaps is None:
        cmaps = ['Greens','OrRd','YlGnBu','PuRd','BuGn','Greys','Blues','bwr']  # diverging for last
    if vmaxs is None:
        vmaxs = [np.nanpercentile(np.asarray(arr), 99) for arr in data_list]
    if vmins is None:
        vmins = [0.0]*8  # default 0 for first 7; override last for diffs

    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.8*nrows),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.ravel()
    lon2d, lat2d = np.meshgrid(lon, lat)

    for i, ax in enumerate(axes):
        C = np.asarray(data_list[i])
        im = ax.pcolormesh(lon2d, lat2d, C, cmap=cmaps[i],
                           shading='nearest', vmin=vmins[i], vmax=vmaxs[i])
        ax.coastlines(); ax.add_feature(cfeature.STATES)
        ax.set_extent([-125, -66, 20, 50], crs=ccrs.PlateCarree())
        ax.set_title(titles[i])
        plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)

    fig.suptitle(main_title, fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ----------------------------- Example run ---------------------------
if __name__ == "__main__":
    years = list(range(1980, 2025))

    # --- Regridded “days” & rain-day panels (7 plots)
    (mcs_days, etc_days, hur_days, ar_days, oc_days,
     avg_rain_days, rain_only_no_type, lat_st, lon_st) = \
        load_stormtype_and_rain_days_regridded(
            stormtype_dir="./release/",
            precip_dir="../../../Data/ERA5/input",
            years=years,
            threshold_mm=1.0,
            interp_method="nearest"
        )


    # --- Amounts & percentages (annual) → 8-panel; percent last panel = Identified − OC
    (pr_mcs, pr_etc, pr_hur, pr_ar, pr_oc, pr_unid,
     pct_mcs, pct_etc, pct_hur, pct_ar, pct_oc, pct_unid,
     lat_st, lon_st) = compute_precip_amount_by_stormtype(
        stormtype_dir="./netcdf",
        precip_dir="../../../Data/ERA5/input",
        years=years)

    # Amounts

    # Percentages 
    pct_ident = np.where(np.isfinite(pct_unid), 100.0 - pct_unid,
                         (pct_mcs + pct_etc + pct_hur + pct_ar + pct_oc))
    pct_ident_minus_oc = pct_ident - pct_oc

    data_pct8 = [pct_mcs, pct_etc, pct_hur, pct_ar, pct_oc, pct_unid, pct_ident, pct_ident_minus_oc]
    titles_pct8 = ['% MCS','% ETC','% HUR','% AR','% Other convective',
                   '% Unidentified','% Identified','% Identified − % OC']
    vmins = [0,0,0,0,0,0,0,0]
    vmaxs = [40,60,20,30,50,100,100,100]
    cmaps = ['Greens','OrRd','YlGnBu','PuRd','BuGn','Greys','Blues','Blues']

    plot_eight_maps_2x4(
        data_pct8, titles_pct8, lat_st, lon_st,
        main_title=f"Mean % of Precipitation by Storm Type (8 panels), {years[0]}–{years[-1]}",
        vmins=vmins, vmaxs=vmaxs, cmaps=cmaps,
        save_path=f"precip_percent_by_type_8panels_{years[0]}–{years[-1]}_{version}.pdf"
    )
