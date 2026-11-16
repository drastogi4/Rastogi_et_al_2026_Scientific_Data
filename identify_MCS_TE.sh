#!/bin/csh -f
# Identify mesoscale convective systems (MCSs) with TempestExtremes (TE)
# using OLR brightness temperature and hourly precipitation thresholds.
#
# Annual outputs:
#   output/ERA5_db_mcs_olr_tp_<year>_conus_min6h.txt
#   output/ERA5_sb_mcs_olr_tp_<year>_conus_min6h.nc
#   output/ERA5_sb_mcs_olr_tp_<year>_conus_min6h_pf.nc
#
# MCS_pf retains stitched MCS tags only where hourly precipitation is at
# least 1 mm/hr.

#SBATCH -A cli138
#SBATCH -J MCS
#SBATCH -N 1
#SBATCH -t 04:00:00

# --------------------------- User configuration ---------------------------
set data = "ERA5"
set start_year = 1980
set end_year = 2024
set timestep = "1hr"

set input_dir = "./ERA5/input"
set output_dir = "./output"
set work_dir = "./work/mcs_te"

set detectblobs = "DetectBlobs"
set stitchblobs = "StitchBlobs"

# OLR / brightness-temperature and rain thresholds.
set threshold = 191.28
set threshold_core = 145.33
set rain_threshold = 1.0

# Feng-style size and duration criteria.
set dx_km = 31
set min_area_km2 = 60000
@ min_size = $min_area_km2 / ( $dx_km * $dx_km )
set min_time = 6
set min_overlap = 50
set max_overlap = 90

# ----------------------------- Environment --------------------------------
module load netcdf-c
module load cdo

mkdir -p "$output_dir" "$work_dir/DN_files"
echo "Minimum area = ${min_area_km2} km2"
echo "Minimum size = ${min_size} grid cells"

@ year = $start_year
while ( $year <= $end_year )
    echo "============================================"
    echo "Running ${data} OLR + rain MCS tracking for ${year}"
    echo "============================================"

    set input_file = "${input_dir}/ERA5_olr_tp_input_${year}.nc"
    set input_list = "${work_dir}/tracking_files_${year}.txt"
    set blob_file = "${work_dir}/${data}_db_mcs_olr_tp_${year}_conus_min6h.txt"
    set stitched_file = "${work_dir}/${data}_sb_mcs_olr_tp_${year}_conus_min6h.nc"
    set precip_filtered_file = "${work_dir}/${data}_sb_mcs_olr_tp_${year}_conus_min6h_pf.nc"
    set merge_file = "${work_dir}/tmp_${data}_mcs_tp_${year}.nc"



    echo "$input_file" >! "$input_list"

    set strdetect = "--verbosity 0 --regional --thresholdcmd mtnlwrf,<=,${threshold},0 --filtercmd mtnlwrf,<=,${threshold_core},1 --filtercmd tp,>=,${rain_threshold},10"
    srun -n 1 $detectblobs --in_data_list "$input_list" --tagvar binary_tag \
        --out "$blob_file" --latname latitude --lonname longitude $strdetect


    srun -n 1 $stitchblobs --in "$blob_file" --out "$stitched_file" \
        --var binary_tag --outvar MCS --regional --mintime $min_time \
        --min_overlap_next $min_overlap --max_overlap_next $max_overlap \
        --minsize $min_size --latname latitude --lonname longitude


    # Combine stitched MCS tags with input precipitation and retain MCS tags
    # only for grid cells with at least the rain threshold.
    cdo -O merge "$stitched_file" "$input_file" "$merge_file"

    cdo -O -z zip_4 -expr,"MCS_pf=(MCS>0 && tp>=${rain_threshold})?MCS:0" \
        "$merge_file" "$precip_filtered_file"


    mv -f "$blob_file" "$output_dir/"
    mv -f "$stitched_file" "$output_dir/"
    mv -f "$precip_filtered_file" "$output_dir/"
    rm -f "$merge_file" "$input_list"

    echo "Finished ${year}"
    @ year++
end

echo "All ${data} OLR + rain MCS tracking complete."

