#!/bin/csh -f
# Identify atmospheric river (AR) features with TempestExtremes (TE),
# excluding areas associated with tracked hurricanes.
#
# Output per year:
#   ERA5_AR_NFF_HUR_<year>_6hr.nc
#


#SBATCH -A cli138
#SBATCH -J TE_AR
#SBATCH -N 1
#SBATCH -t 10:00:00

# --------------------------- User configuration ---------------------------
set start_year = 1980
set end_year = 2024
set data = "ERA5"
set timestep = "6hr"

# Input directories (relative to the directory from which this script runs).
set track_input_dir = "../../../../Data/ERA5/input"
set ivt_input_dir = "../../../../Data/ERA5/input"
set output_dir = "./output"

# TempestExtremes executables must be available in PATH or loaded by your
# environment/module configuration before running this script.
set detectnodes = "DetectNodes"
set stitchnodes = "StitchNodes"
set detectblobs = "DetectBlobs"
set nodefilefilter = "NodeFileFilter"

# ----------------------------- Environment --------------------------------
module load netcdf-c

mkdir -p "$output_dir" DN_files

@ year = $start_year
while ( $year <= $end_year )
    echo "Processing ${data} AR identification for ${year}"

    # --------------------- Hurricane track detection ----------------------
    @ yearplus1 = $year + 1
    set track_input = "${track_input_dir}/ERA5_allinput_${year}_conus.nc"
    set track_list = "tracking_files_${year}.txt"
    set dn_prefix = "DN_files/hur_${year}_"
    set dn_combined = "DN_files/hur_${year}.txt"
    set hur_tracks = "${output_dir}/${data}_trajectories_hur_conus_${year}.txt"

    echo "$track_input" >! "$track_list"

    # Detect candidate cyclones.
    set dn_mslmag = 150.0
    set dn_msldist = 5.5
    set dn_zdiffmag = -6.0
    set dn_zdist = 6.5
    set dn_zmaxoffset = 1.0
    set dn_mergedist = 6.0
    set dn_timefilter = "$timestep"

    set strdetect = "--verbosity 0 --regional --timefilter ${dn_timefilter} --closedcontourcmd SLP,${dn_mslmag},${dn_msldist},0;_DIFF(Z300,Z500),${dn_zdiffmag},${dn_zdist},${dn_zmaxoffset} --mergedist ${dn_mergedist} --searchbymin SLP --outputcmd SLP,min,0;_VECMAG(U10,V10),max,2;PHIS,max,0"

    rm -f ${dn_prefix}*
    srun -n 1 $detectnodes --in_data_list "$track_list" --out "$dn_prefix" \
        --latname "latitude" --lonname "longitude" $strdetect
  

    cat ${dn_prefix}* >! "$dn_combined"

    # Stitch candidate hurricanes into tracks.
    set sn_range = 8.0
    set sn_trajmintime = "48h"
    set sn_trajmaxgap = "24h"
    set sn_minwind = 10.0
    set sn_minlen = 5

    srun -n 1 $stitchnodes --in_fmt "lon,lat,slp,wind,zs" \
        --range $sn_range --mintime $sn_trajmintime --maxgap $sn_trajmaxgap \
        --in "$dn_combined" --out "$hur_tracks" --out_file_format "gfdl" \
        --threshold "wind,>=,${sn_minwind},${sn_minlen};lat,<=,50.0,10;lat,>=,-50.0,10;lon,>=,-105.0,10;zs,<=,150.0,10"


    # ------------------------- AR blob detection ---------------------------
    set ivt_input = "${ivt_input_dir}/ERA5_ivt_input_${year}_conus.nc"
    set ivt_list = "ERA5_IVT_files_${year}.txt"
    set ar_blobs = "ERA5_AR_files_${year}_${timestep}.nc"
    set ar_data_list = "ERA5_AR_input_${year}_${timestep}.txt"
    set ar_mask_list = "ERA5_AR_output_${year}_${timestep}.txt"
    set ar_mask = "${output_dir}/ERA5_AR_NFF_HUR_${year}_${timestep}.nc"

    echo "$ivt_input" >! "$ivt_list"

    srun -n 1 $detectblobs --in_data_list "$ivt_list" --out "$ar_blobs" \
        --timefilter "$timestep" --regional \
        --thresholdcmd "_LAPLACIAN{8,10}(_VECMAG(viwve,viwvn)),<=,-20000,0" \
        --minabslat 15 --geofiltercmd "area,>=,4e5km2" \
        --latname "latitude" --lonname "longitude"
    if ( $status != 0 ) then
        echo "ERROR: DetectBlobs failed for ${year}" >&2
        exit 1
    endif

    # Remove AR detections within 8 degrees of hurricane tracks.
    echo "$ar_blobs" >! "$ar_data_list"
    echo "$ar_mask" >! "$ar_mask_list"
    srun -n 1 $nodefilefilter --in_nodefile "$hur_tracks" \
        --in_fmt "lon,lat,slp,wind,zs" --in_data_list "$ar_data_list" \
        --out_data_list "$ar_mask_list" --var "binary_tag" \
        --bydist 8.0 --invert --regional



    # remove temporary list files created by this script.
    rm -f "$track_list" "$dn_combined" "$ivt_list" \
        "$ar_data_list" "$ar_mask_list"
    mv -f "$ar_blobs" "$output_dir/"

    echo "Finished ${year}: ${ar_mask}"
    @ year++
end

