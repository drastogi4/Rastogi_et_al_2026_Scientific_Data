#!/bin/csh -f
# Identify extratropical cyclone (ETC) tracks with TempestExtremes (TE).
#
# Produces one GFDL-format track file per year:
#   output/ERA5_trajectories_ETC_conus_<year>_6hr.txt
#


#SBATCH -J ETCs
#SBATCH -N 1
#SBATCH -t 05:00:00

# --------------------------- User configuration ---------------------------
set start_year = 1980
set end_year = 2024
set data = "ERA5"
set timestep = "6hr"

# Paths are relative to the directory from which this script runs.
set track_input_dir = "./ERA5/input"
set output_dir = "./output"
set work_dir = "./work/etc_te"

# TempestExtremes executables must be available in PATH or loaded by your
# environment/module configuration before this script runs.
set detectnodes = "DetectNodes"
set stitchnodes = "StitchNodes"

# ----------------------------- Environment --------------------------------
module load netcdf-c

mkdir -p "$output_dir" "$work_dir/DN_files"

@ year = $start_year
while ( $year <= $end_year )
    echo "Processing ${data} ETC tracks for ${year}"

    set input_file = "${track_input_dir}/ERA5_allinput_${year}_forETC.nc"
    set input_list = "${work_dir}/tracking_files_${year}.txt"
    set dn_prefix = "${work_dir}/DN_files/out_${year}_"
    set dn_file = "${work_dir}/ERA5_DN_${year}.txt"
    set track_file = "${output_dir}/${data}_trajectories_ETC_conus_${year}_${timestep}.txt"

    echo "$input_file" >! "$input_list"

    # Detect candidate ETC centers.
    set dn_mslmag = 200.0
    set dn_msldist = 6.0
    set dn_zdiffmag = -6.0
    set dn_zdist = 6.5
    set dn_zmaxoffset = 1.0
    set dn_mergedist = 6.0
    set dn_timefilter = "$timestep"

    set strdetect = "--verbosity 0 --regional --timefilter ${dn_timefilter} --closedcontourcmd SLP,${dn_mslmag},${dn_msldist},0 --noclosedcontourcmd _DIFF(Z300,Z500),${dn_zdiffmag},${dn_zdist},${dn_zmaxoffset} --mergedist ${dn_mergedist} --searchbymin SLP --outputcmd SLP,min,0"

    rm -f ${dn_prefix}*
    srun -n 1 $detectnodes --in_data_list "$input_list" --out "$dn_prefix" \
        --latname "latitude" --lonname "longitude" $strdetect
    

    cat ${dn_prefix}* >! "$dn_file"

    # Stitch candidate centers into tracks. GFDL format is retained for
    # downstream TempestExtremes processing.
    set sn_range = 6.0
    set sn_trajmintime = "48h"
    set sn_trajmaxgap = "18h"

    srun -n 1 $stitchnodes --in_fmt "lon,lat,slp" --range $sn_range \
        --mintime $sn_trajmintime --maxgap $sn_trajmaxgap \
        --in "$dn_file" --out "$track_file" --out_file_format "gfdl" \
        --min_endpoint_dist 12.0
    
    echo "Finished ${year}: ${track_file}"
    @ year++
end

