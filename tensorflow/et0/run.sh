# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


so_file="matrix_solve/matrix_solve_ce_op/libmatrix_solve_ce_op.so"
gp_file="matrix_solve/matrix_solve_ce_op/matrix_solve_codelets.gp"
save_folder="test_result"
if [ ! -f "$so_file" ] || [ ! -f "$gp_file" ]; then
 cd matrix_solve/matrix_solve_ce_op
 make
 cd ../../
fi
if [ ! -x "$save_folder" ]; then
 mkdir "$save_folder"
fi

START_TIME=`date +%s`
file_root="test_chinese_map/"
prs_file="raw_data/SURF_CLI_CHN_MUL_DAY-PRS-10004-20210403.TXT"
rhu_file="raw_data/SURF_CLI_CHN_MUL_DAY-RHU-13003-20210403.TXT"
tem_file="raw_data/SURF_CLI_CHN_MUL_DAY-TEM-12001-20210403.TXT"
win_file="raw_data/SURF_CLI_CHN_MUL_DAY-WIN-11002-20210403.TXT"
file_dem="resource/DEM_china_1km.tif"
file_latit="latitude/china_latitude.tif"
tif_save_1=$save_folder"/ET0_generate_0103_cvt_total_1.tif"
tif_save_2=$save_folder"/ET0_generate_0103_cvt_total_2.tif"
tif_save_3=$save_folder"/ET0_generate_0103_cvt_total_3.tif"
tif_save_4=$save_folder"/ET0_generate_0103_cvt_total_4.tif"
tif_save_5=$save_folder"/ET0_generate_0103_cvt_total_5.tif"
tif_save_6=$save_folder"/ET0_generate_0103_cvt_total_6.tif"
tif_save_7=$save_folder"/ET0_generate_0103_cvt_total_7.tif"
tif_save_8=$save_folder"/ET0_generate_0103_cvt_total_8.tif"
tif_save_9=$save_folder"/ET0_generate_0103_cvt_total_9.tif"
tif_save_10=$save_folder"/ET0_generate_0103_cvt_total_10.tif"
tif_save_11=$save_folder"/ET0_generate_0103_cvt_total_11.tif"
tif_save_12=$save_folder"/ET0_generate_0103_cvt_total_12.tif"
tif_save_13=$save_folder"/ET0_generate_0103_cvt_total_13.tif"
tif_save_14=$save_folder"/ET0_generate_0103_cvt_total_14.tif"

python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_1 & \
python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_2 & \
python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_3 & \
python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_4                                                                                                                                                                   

python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_5 & \
python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_6 & \
python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_7 & \
python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_8 

python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_9 & \
python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_10 & \
python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_11 & \
python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_12 

python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_13 & \
python $file_root"kriging_test.py" --prs-file $prs_file --rhu-file $rhu_file --tem-file $tem_file --win-file $win_file \
                                   --file-dem $file_dem --file-latit $file_latit --tif-save $tif_save_14


END_TIME=`date +%s`
 
EXECUTING_TIME=`expr $END_TIME - $START_TIME`
UNIT=" seconds"
echo "*********total cost**********: "$EXECUTING_TIME$UNIT