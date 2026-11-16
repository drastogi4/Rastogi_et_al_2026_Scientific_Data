# plot_mean_likelihoods_wetonly.py
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

version = "TE"
# ---------------- helpers ----------------
def _rename_latlon(obj):
    """
    Standardize spatial dims/coords to ('latitude','longitude').
    Supports common variants: lat/lon, y/x, nav_lat/nav_lon, Latitude/Longitude.
    Works for DataArray or Dataset.
    """
    dims = list(obj.dims)
    coords = list(obj.coords)

    # Find latitude
    lat_name = None
    for cand in ("latitude", "lat", "Latitude", "y", "nav_lat"):
        if cand in dims or cand in coords:
            lat_name = cand
            break
    # Find longitude
    lon_name = None
    for cand in ("longitude", "lon", "Longitude", "x", "nav_lon"):
        if cand in dims or cand in coords:
            lon_name = cand
            break

    ren = {}
    if lat_name and lat_name != "latitude":
        ren[lat_name] = "latitude"
    if lon_name and lon_name != "longitude":
        ren[lon_name] = "longitude"

    return obj.rename(ren) if ren else obj

def _require_spatial(da, src_name):
    """Raise a clear error if latitude/longitude missing or empty."""
    if "latitude" not in da.dims or "longitude" not in da.dims:
        raise ValueError(
            f"{src_name}: missing 'latitude'/'longitude' dims after normalization. "
            f"Found dims: {list(da.dims)}"
        )
    nlat = da.sizes["latitude"]
    nlon = da.sizes["longitude"]
    if nlat == 0 or nlon == 0:
        raise ValueError(
            f"{src_name}: zero-sized spatial dimension(s) "
            f"(latitude={nlat}, longitude={nlon})."
        )

def _ensure_ascending(da):
    """xarray.interp expects monotonic ascending coords."""
    if "latitude" in da.coords and np.any(np.diff(da["latitude"].values) < 0):
        da = da.sortby("latitude")
    if "longitude" in da.coords and np.any(np.diff(da["longitude"].values) < 0):
        da = da.sortby("longitude")
    return da

def _is_0360(lon):
    lonv = np.asarray(lon)
    return np.nanmin(lonv) >= 0.0

def _to_0360(lon):
    lon = np.asarray(lon)
    return np.mod(lon, 360.0)

def _to_m180_180(lon):
    lon = np.asarray(lon)
    return ((lon + 180.0) % 360.0) - 180.0

def _harmonize_lon_to_target(da, target_lon):
    """
    Map da.longitudes into target convention and interp onto target_lon points.
    Keeps other coords; ensures ascending.
    Works for DataArray or Dataset with lon coord 'longitude'.
    """
    da = _ensure_ascending(_rename_latlon(da))
    src = da["longitude"].values
    tgt = target_lon.values
    if _is_0360(tgt):
        new_lon = _to_0360(src)
    else:
        new_lon = _to_m180_180(src)
    da = da.assign_coords(longitude=new_lon)
    da = _ensure_ascending(da)
    # Interpolate onto exact target longitudes
    da = da.interp(longitude=target_lon, method="nearest")
    return da

def _regrid_to_reference(da, ref_lat, ref_lon):
    """
    Ensure da has same lat/lon coords as reference by interpolation.
    Assumes da already uses the same longitude convention as ref_lon.
    """
    da = _ensure_ascending(_rename_latlon(da))
    # First interp latitude if needed
    if not np.array_equal(da["latitude"].values, ref_lat.values):
        da = da.interp(latitude=ref_lat, method="linear")
    # Then interp longitude
    if not np.array_equal(da["longitude"].values, ref_lon.values):
        da = da.interp(longitude=ref_lon, method="nearest")
    return da

def _align_times_strict(lik, pr, year, tolerance="1H"):
    """
    Align likelihood and precip on a common set of times.
    1) Try exact intersection. If empty, 2) reindex likelihood to precip times
       with nearest within tolerance.
    Returns (lik_aligned, pr_aligned) with identical time coords, or (None, None) if fail.
    """
    # Exact intersection
    lik_i, pr_i = xr.align(lik, pr, join="inner")
    if lik_i.sizes.get("time", 0) > 0:
        return lik_i, pr_i

    # Try nearest reindex of lik to pr times, with tolerance
    try:
        lik2 = lik.reindex(time=pr.time, method="nearest", tolerance=np.timedelta64(1, "h"))
        # Drop any times that failed reindex
        good = lik2.time.where(lik2.notnull().any(dim=set(lik2.dims) - {"time"}), drop=True)
        lik2 = lik2.sel(time=good)
        pr2 = pr.sel(time=good)
        if lik2.sizes.get("time", 0) > 0:
            print(f"ℹ️  {year}: used nearest-time match within {tolerance} (len={lik2.sizes['time']}).")
            return lik2, pr2
    except Exception:
        pass

    print(f"⚠️  {year}: no overlapping times between likelihood and precip (even with tolerance). Skipping.")
    return None, None

# ---------------- core ----------------
def mean_likelihood_over_years(
    nc_dir,
    years,
    filename_fmt="./likelihood/stormtype_likelihoods_{version}_{year}_6hr.nc",
    version = "TE",
    varname="likelihood",
    expected_types=(1,2,3,4),
    # wet-hour conditioning (ERA5 tp)
    wet_only=True,
    precip_dir="../../../Data/ERA5/input",
    precip_filename_fmt="ERA5_tp_input_{year}_6hourly_conus.nc",
    precip_var="tp",
    precip_threshold=1e-4,  # meters per 6h (~0.1 mm) to avoid drizzle/numerical noise
):
    """
    Returns mean across years of per-year time means:
      mean_years( mean_time(likelihood | precip > threshold) ), equal weight per year.

    Robustness features:
    - normalizes lat/lon names
    - takes first valid file's grid as reference
    - reprojects remaining years to the reference grid (lon convention & lat/lon points)
    - aligns time between likelihood and precip per-year (exact intersection; nearest<=1h fallback)
    - skips malformed or empty files with readable warnings
    - NaN-aware accumulation across years so "no-wet-hours" cells don't bias low
    """
    ref_lat = ref_lon = None
    sum_year = None
    cnt_year = None
    used_years = []
    skipped = []

    for y in years:
        fn = os.path.join(nc_dir, filename_fmt.format(year=y,version=version))
        if not os.path.exists(fn):
            print(f"⚠️  Missing file: {fn}  (skipping)")
            skipped.append((y, "missing likelihood file"))
            continue

        try:
            ds = xr.open_dataset(fn)
        except Exception as e:
            print(f"⚠️  Could not open {fn}: {e}  (skipping)")
            skipped.append((y, "open error"))
            continue

        if varname not in ds:
            print(f"⚠️  '{varname}' not found in {fn}  (skipping)")
            skipped.append((y, f"{varname} not found"))
            ds.close(); continue

        lik = ds[varname]  # expect (type, time, lat, lon)
        lik = _rename_latlon(lik)

        # sanity checks
        if "type" not in lik.dims:
            print(f"⚠️  No 'type' dim in {fn} (dims={list(lik.dims)}). Skipping.")
            skipped.append((y, "no type dim"))
            ds.close(); continue
        if "time" not in lik.dims:
            print(f"⚠️  No 'time' dim in {fn} (dims={list(lik.dims)}). Skipping.")
            skipped.append((y, "no time dim"))
            ds.close(); continue

        try:
            _require_spatial(lik, fn)
        except ValueError as e:
            print(f"⚠️  {e}  (skipping year {y})")
            skipped.append((y, "empty spatial"))
            ds.close(); continue

        # subset to expected types (preserve order)
        have_types = [t for t in expected_types if t in lik.type.values]
        if len(have_types) == 0:
            print(f"⚠️  No expected types found in {fn} (have={list(lik.type.values)}). Skipping.")
            skipped.append((y, "no expected types"))
            ds.close(); continue
        lik = lik.sel(type=have_types)

        # ------- wet-hour conditioning (optional) -------
        if wet_only:
            pfn = os.path.join(precip_dir, precip_filename_fmt.format(year=y))
            if not os.path.exists(pfn):
                print(f"⚠️  {y}: precip file missing: {pfn}. Skipping year.")
                skipped.append((y, "precip missing"))
                ds.close(); continue

            try:
                dpr = xr.open_dataset(pfn)
            except Exception as e:
                print(f"⚠️  {y}: could not open precip {pfn}: {e}. Skipping year.")
                skipped.append((y, "precip open error"))
                ds.close(); continue

            if precip_var not in dpr:
                print(f"⚠️  {y}: '{precip_var}' not found in {pfn}. Skipping year.")
                skipped.append((y, "precip var missing"))
                dpr.close(); ds.close(); continue

            pr = _rename_latlon(dpr[precip_var])  # expect (time, lat, lon)
            try:
                _require_spatial(pr, pfn)
            except ValueError as e:
                print(f"⚠️  {e}  (skipping year {y})")
                skipped.append((y, "precip spatial error"))
                dpr.close(); ds.close(); continue

            # Regrid precip to the *likelihood grid for this year* before masking
            pr = _harmonize_lon_to_target(pr, lik["longitude"])
            pr = _regrid_to_reference(pr, lik["latitude"], lik["longitude"])

            # Align time dimensions strictly (intersection, or nearest<=1h)
            lik_al, pr_al = _align_times_strict(lik, pr, y, tolerance="1H")
            if lik_al is None:
                skipped.append((y, "no overlapping times"))
                dpr.close(); ds.close(); continue

            lik = lik_al
            pr = pr_al

            # ERA5 tp is meters/accumulation period; use a small threshold to avoid rounding noise
            wet = pr > float(precip_threshold)
            lik = lik.where(wet)
            dpr.close()

        # per-year time mean (after optional wet mask)
        if lik.sizes.get("time", 0) == 0:
            print(f"⚠️  {y}: no time steps after preprocessing. Skipping year.")
            skipped.append((y, "no time after preprocess"))
            ds.close(); continue

        yr_mean = lik.mean("time", skipna=True)  # (type, lat, lon)

        # establish reference grid from first valid file
        if ref_lat is None:
            # ensure longitude ascending (and reindex yr_mean consistently)
            ref_lat = yr_mean["latitude"]
            ref_lon = yr_mean["longitude"]
            ref_lat = _ensure_ascending(ref_lat)

            if np.any(np.diff(ref_lon.values) < 0):
                order = np.argsort(ref_lon.values)
                ref_lon = xr.DataArray(ref_lon.values[order], dims=("longitude",), name="longitude")
                yr_mean = yr_mean.isel(longitude=order)

            used_years.append(y)
            sum_year = yr_mean.fillna(0)
            cnt_year = xr.where(yr_mean.notnull(), 1, 0)
            ds.close()
            continue

        # Harmonize this year's per-year mean to reference grid
        yr_mean = _harmonize_lon_to_target(yr_mean, ref_lon)
        yr_mean = _regrid_to_reference(yr_mean, ref_lat, ref_lon)

        # Accumulate with NaN-aware counters
        sum_year = sum_year + yr_mean.fillna(0)
        cnt_year = cnt_year + xr.where(yr_mean.notnull(), 1, 0)
        used_years.append(y)
        ds.close()

    if not used_years:
        raise RuntimeError("No valid likelihood years found after screening.")

    # Compute final mean with valid-year counts per grid cell
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_da = (sum_year / xr.where(cnt_year > 0, cnt_year, np.nan)).astype("float32")

    print(f"✓ Used {len(used_years)} years: {used_years[0]} … {used_years[-1]} (total {len(used_years)})")
    if skipped:
        print("⚠️  Skipped years:")
        for y, why in skipped:
            print(f"   - {y}: {why}")

    return mean_da

def plot_likelihood_2x2(mean_lik, out_png=None, main_title=None):
    """
    mean_lik: DataArray (type, latitude, longitude)
              with type codes [1, 2, 3, 4]:
              1=MCS, 2=ETC, 3=HUR, 4=AR.
    """
    # Final sanity checks
    mean_lik = _rename_latlon(mean_lik)
    _require_spatial(mean_lik, "mean_lik")

    lat = mean_lik["latitude"].values
    lon = mean_lik["longitude"].values

    if lon.size == 0 or lat.size == 0:
        raise ValueError("Empty latitude/longitude arrays in mean_lik.")

    # Extent by longitude convention
    if np.nanmin(lon) < 0:  # -180..180
        extent = [-125, -66, 20, 50]
    else:                   # 0..360
        extent = [360-125, 360-66, 20, 50]

    LON2D, LAT2D = np.meshgrid(lon, lat)

    desired = [1,2,3,4]
    type_names = {
            1: "MCS",
            2: "ETC",
            3: "HUR",
            4: "AR",
            }

    present = [t for t in desired if t in mean_lik.type.values]

    vmax_dict = {
            1: 0.2,  # MCS
            2: 0.5,  # ETC
            3: 0.1,  # HUR
            4: 0.2,  # AR
            }


    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axes = axes.ravel()

    for i, t in enumerate(desired):
        ax = axes[i]

        if t in present:
            C = mean_lik.sel(type=t).values

            im = ax.pcolormesh(
                LON2D,
                LAT2D,
                C,
                cmap="Blues",
                vmin=0.0,
                vmax=vmax_dict[t],
                shading="nearest"
            )

            cb = plt.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                shrink=0.75
            )
            cb.set_label("Probability (0–1)")

        else:
            ax.text(
                0.5, 0.5,
                f"{type_names[t]} (missing)",
                transform=ax.transAxes,
                ha="center",
                va="center"
            )

        ax.coastlines()
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(f"Mean likelihood: {type_names[t]}")

    if main_title:
        fig.suptitle(main_title, fontsize=14)

    fig.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")

    plt.show()

# ---------------- main ----------------
if __name__ == "__main__":
    # Years to include
    years = list(range(1980, 2025))
    # Likelihood files directory/pattern
    nc_dir = "./"  # folder with stormtype_likelihoods_{year}_6hr.nc
    filename_fmt = "./likelihood/stormtype_likelihoods_{version}_{year}_6hr.nc"  # adjust to your filenames

    # ERA5 tp source (as you specified)
    precip_dir = "../../../Data/ERA5/input"
    precip_filename_fmt = "ERA5_tp_input_{year}_6hourly_conus.nc"

    # Compute wet-hour-conditioned mean
    mean_lik = mean_likelihood_over_years(
        nc_dir,
        years,
        filename_fmt=filename_fmt,
        version = version,
        varname="likelihood",
        expected_types=(1,2,3,4),
        wet_only=True,                 # key switch
        precip_dir=precip_dir,
        precip_filename_fmt=precip_filename_fmt,
        precip_var="tp",
        precip_threshold=1e-4          # meters per 6h (~0.1 mm)
    )

    title = f"Mean storm-type likelihood (wet hours only) {years[0]}–{years[-1]}"
    plot_likelihood_2x2(
        mean_lik,
        out_png=f"mean_likelihood_wetonly_{years[0]}-{years[-1]}_{version}.pdf",
        main_title=title
    )

