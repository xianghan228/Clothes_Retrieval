sh feature_extract_general.sh /data/yyxue/exp/0zky_36_rawproto_vggs_8attri/36_att8_only_2000_iter_13500.caffemodel deploy_skirt_query.txt pool5_93 skirt_query 60 0
sh feature_extract_general.sh /data/yyxue/exp/0zky_36_rawproto_vggs_id/36_ali_iter_500.caffemodel deploy_skirt_query.txt pool5_93 skirt_query2 60 0
sh feature_extract_general.sh /data/yyxue/exp/0zky_36_rawproto_vggs_8attri/36_att8_only_2000_iter_13500.caffemodel deploy_skirt_database.txt pool5_93 skirt_database 13688 0
sh feature_extract_general.sh /data/yyxue/exp/0zky_36_rawproto_vggs_id/36_ali_iter_500.caffemodel deploy_skirt_database.txt pool5_93 skirt_database2 13688 0
nohup python JD_retrieval_ggnet_tianchi.py --rlmdb_path ./skirt_database --qlmdb_path ./skirt_query --rlmdb_path2 ./skirt_database2 --qlmdb_path2 ./skirt_query2 --fw_result ./_skirt_result.txt > skirt.out &


sh feature_extract_general.sh /data/yyxue/exp/0zky_62_rawproto_vggs_8attri/62_att8_only_2000_iter_14000.caffemodel deploy_down_database.txt pool5_93 down_database 8433 0
sh feature_extract_general.sh /data/yyxue/exp/0zky_62_rawproto_vggs_id/62_ali_iter_400.caffemodel deploy_down_database.txt pool5_93 down_database2 8433 0
sh feature_extract_general.sh /data/yyxue/exp/0zky_62_rawproto_vggs_8attri/62_att8_only_2000_iter_14000.caffemodel deploy_down_query.txt pool5_93 down_query 57 0
sh feature_extract_general.sh /data/yyxue/exp/0zky_62_rawproto_vggs_id/62_ali_iter_400.caffemodel deploy_down_query.txt pool5_93 down_query2 57 0
nohup python JD_retrieval_ggnet_tianchi.py --rlmdb_path ./down_database --qlmdb_path ./down_query --rlmdb_path2 ./down_database2 --qlmdb_path2 ./down_query2 --fw_result ./_down_result.txt > down.out &

sh feature_extract_general.sh /data/yyxue/exp/0zky_rawproto_vggs_8attri/googlenet_clothes_woman_0617_3_5d_iter_14000.caffemodel deploy_upper_database.txt pool5_93 upper_database 23954 0
sh feature_extract_general.sh /data/yyxue/exp/0zky_rawproto_vggs_id/googlenet_clothes_woman_0617_3_5d_iter_500.caffemodel deploy_upper_database.txt pool5_93 upper_database2 23954 0
sh feature_extract_general.sh /data/yyxue/exp/0zky_rawproto_vggs_8attri/googlenet_clothes_woman_0617_3_5d_iter_14000.caffemodel deploy_upper_query.txt pool5_93 upper_query 100 0
sh feature_extract_general.sh /data/yyxue/exp/0zky_rawproto_vggs_id/googlenet_clothes_woman_0617_3_5d_iter_500.caffemodel deploy_upper_query.txt pool5_93 upper_query2 100 0
nohup python JD_retrieval_ggnet_tianchi.py --rlmdb_path ./upper_database --qlmdb_path ./upper_query --rlmdb_path2 ./upper_database2 --qlmdb_path2 ./upper_query2 --fw_result ./_upper_result.txt > upper.out &
