#!/bin/csh -f
# Identify hurricane tracks with TempestExtremes (TE).
#
# Produces CSV and GFDL-format tracks per year:
#   output/ERA5_trajectories_hur_<year>_6hr_csv.txt
#   output/ERA5_trajectories_hur_<year>_6hr.txt
#


#SBATCH -J HUR
#SBATCH -N 1
#SBATCH -t 04:00:00

# --------------------------- User configuration ---------------------------
set start_year = 1980
set end_year = 2004
set data = "ERA5"
set timestep = "6hr"

# Input path is relative to the directory from which this script runs.
set track_input_dir = "./ERA5/input"
set output_dir = "./output"
set work_dir = "./work/hurricane_te"

# TempestExtremes executables must be available in PATH or loaded by your
# environment/module configuration before this script runs.
set detectnodes = "DetectNodes"
set stitchnodes = "StitchNodes"

# ----------------------------- Environment --------------------------------
module load netcdf-c

mkdir -p "$output_dir" "$work_dir/DN_files"

@ year = $start_year
while ( $year <= $end_year )
    echo "Processing ${data} hurricane tracks for ${year}"

    set input_file = "${track_input_dir}/ERA5_allinput_${year}.nc"
    set input_list = "${work_dir}/tracking_files_${year}.txt"
    set dn_prefix = "${work_dir}/DN_files/out_${year}_"
    set dn_file = "${work_dir}/ERA5_DN_${year}.txt"
    set track_csv = "${output_dir}/${data}_trajectories_hur_${year}_${timestep}_csv.txt"
    set track_gfdl = "${output_dir}/${data}_trajectories_hur_${year}_${timestep}.txt"

    echo "$input_file" >! "$input_list"

    # Detect candidate tropical cyclones.
    set dn_mslmag = 150.0
    set dn_msldist = 5.5
    set dn_zdiffmag = -6.0
    set dn_zdist = 6.5
    set dn_zmaxoffset = 1.0
    set dn_mergedist = 6.0
    set dn_timefilter = "$timestep"

    set strdetect = "--verbosity 0 --regional --timefilter ${dn_timefilter} --closedcontourcmd SLP,${dn_mslmag},${dn_msldist},0;_DIFF(Z300,Z500),${dn_zdiffmag},${dn_zdist},${dn_zmaxoffset} --mergedist ${dn_mergedist} --searchbymin SLP --outputcmd SLP,min,0;_VECMAG(U10,V10),max,2;PHIS,max,0"

    rm -f ${dn_prefix}*
    srun -n 1 $detectnodes --in_data_list "$input_list" --out "$dn_prefix" \
        --latname "latitude" --lonname "longitude" $strdetect


    cat ${dn_prefix}* >! "$dn_file"

    # Stitch detected centers into tracks. Apply the original lifetime,
    # location, wind, and surface-height filters.
    set sn_range = 8.0
    set sn_trajmintime = "48h"
    set sn_trajmaxgap = "24h"
    set sn_minwind = 10.0
    set sn_minlen = 5
    set track_threshold = "wind,>=,${sn_minwind},${sn_minlen};lat,<=,50.0,10;lat,>=,-50.0,10;lon,>=,-105.0,10;zs,<=,150.0,10"

    srun -n 1 $stitchnodes --in_fmt "lon,lat,slp,wind,zs" \
        --range $sn_range --mintime $sn_trajmintime --maxgap $sn_trajmaxgap \
        --in "$dn_file" --out "$track_csv" --out_file_format "csv" \
        --threshold "$track_threshold"


    srun -n 1 $stitchnodes --in_fmt "lon,lat,slp,wind,zs" \
        --range $sn_range --mintime $sn_trajmintime --maxgap $sn_trajmaxgap \
        --in "$dn_file" --out "$track_gfdl" --out_file_format "gfdl" \
        --threshold "$track_threshold"


    rm -f "$input_list" "$dn_file"
    echo "Finished ${year}: ${track_csv} and ${track_gfdl}"
    @ year++
end

