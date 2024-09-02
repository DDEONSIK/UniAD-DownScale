#!/bin/bash

# python ./tools/analysis_tools/visualize/run.py \
#     --predroot PATH_TO_YOUR_PREDISION_RESULT_PKL \
#     --out_folder PATH_TO_YOUR_OUTPUT_FOLDER \
#     --demo_video FILENAME_OF_OUTPUT_VIDEO \
#     --project_to_cam True


# 실행 명령어: bash /home/hyun/local_storage/code/UniAD/tools/uniad_vis_result.sh

PYTHONPATH=$PYTHONPATH:/home/hyun/local_storage/code/UniAD python ./tools/analysis_tools/visualize/run.py \
   --predroot /home/hyun/local_storage/code/UniAD/output/results.pkl \
   --out_folder /home/hyun/local_storage/code/UniAD/visualize \
   --demo_video /home/hyun/local_storage/code/UniAD/visualize/output/2cData_e2e_video.mp4 \
   --project_to_cam True

# 에러: AssertionError: Database version not found: data/nuscenes/v1.0-mini


#    python ./tools/analysis_tools/visualize/run.py --predroot ./output/results.pkl --out_folder /home/hyun/local_storage/code/UniAD/visualize --demo_video /home/hyun/local_storage/code/UniAD/visualize/output/2cData_e2e_video.mp4 --project_to_cam True
#    python ./tools/analysis_tools/visualize/run.py --predroot ./output/results.pkl --out_folder ./output_visualize --demo_video 2cData_e2e_video.avi --project_to_cam True