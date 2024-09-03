import numpy
import torch
import mmcv
import os

print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"PyTorch version: {torch.__version__}")
print(f"MMCV version: {mmcv.__version__}")
print("CUDA_HOME:", os.environ.get('CUDA_HOME'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))

from mmdet3d.core.bbox.coders import build_bbox_coder

print("\n")

"""
ps -ef | grep hyun
kill -9 
watch -d -n 0.5 nvidia-smi

#Train e2e 3090 GPU 3

nohup ./tools/uniad_dist_train.sh ./projects/configs/stage2_e2e/base_e2e.py 3 > output.log 2>&1 &
disown -h
watch -d -n 0.5 nvidia-smi

2024-08-20
3090
2차 DownScale Train data 불러와서 e2e 학습 진행 중
-> 24.08.29 완료
24.09.02 재학습 (mapping 값 이상)

#test
nohup ./tools/uniad_dist_eval.sh ./projects/configs/stage2_e2e/base_e2e.py ./projects/work_dirs/stage2_e2e/base_e2e/latest.pth 3 > output.log 2>&1 &
disown -h
메모리: 평균 5GB
(24.08.29 10시 시작) (30분정도 소요)
/home/hyun/local_storage/code/UniAD/test/base_e2e/2차 data test: Thu_Aug_29_10_27_49_2024/2차 data TEST output.log


############ DV-1 재검증
nohup ./tools/uniad_dist_train.sh ./projects/configs/stage1_track_map/base_track_map.py 3 > output.log 2>&1 &
disown -h


#Visualization
# please refer to  ./tools/uniad_vis_result.sh
python ./tools/analysis_tools/visualize/run.py \
    --predroot /PATH/TO/YOUR/RESULTS.pkl \
    --out_folder /PATH/TO/YOUR/OUTPUT \
    --demo_video test_demo.avi \
    --project_to_cam True

"""


"""
# 파일 압축 해제 및 폴더 병합, log 작성, 오류catch 
import os
import tarfile
import shutil
import logging

# 로깅 설정
logging.basicConfig(filename='extraction_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the source and destination directories
source_dir = "/home/data/hyun/datasets/download"
destination_dir = "/home/data/hyun/datasets/nuscenes"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

def merge_folder(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                os.makedirs(d)
            merge_folder(s, d)
        else:
            if not os.path.exists(d):
                shutil.copy2(s, d)

# 처리된 파일 목록을 저장할 파일
processed_files_log = 'processed_files.txt'

# 이전에 처리된 파일 목록 불러오기
processed_files = set()
if os.path.exists(processed_files_log):
    with open(processed_files_log, 'r') as f:
        processed_files = set(f.read().splitlines())

# Iterate over all .tar files in the source directory
for file_name in os.listdir(source_dir):
    if file_name.endswith(".tar") and file_name not in processed_files:
        # Construct the full file path
        file_path = os.path.join(source_dir, file_name)
        
        # Create a temporary directory for extraction
        temp_dir = os.path.join(source_dir, "temp_extraction")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Open and extract the tar file
            with tarfile.open(file_path, "r") as tar:
                tar.extractall(path=temp_dir)
            logging.info(f"Extracted: {file_name} to {temp_dir}")
            
            # Merge the extracted content with the destination directory
            merge_folder(temp_dir, destination_dir)
            logging.info(f"Merged: {file_name} content to {destination_dir}")
            
            # 처리 완료된 파일 기록
            with open(processed_files_log, 'a') as f:
                f.write(f"{file_name}\n")
            
        except Exception as e:
            logging.error(f"Error processing {file_name}: {str(e)}")
        finally:
            # Remove the temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

logging.info("All .tar files have been processed.")
print("Process completed. Check 'extraction_log.txt' for details.")

"""
"""
nuScenes-map-expansion-v1.3.zip
can_bus.zip
압축 해제

python -m zipfile -e nuScenes-map-expansion-v1.3.zip /home/data/hyun/datasets/download
python -m zipfile -e can_bus.zip /home/data/hyun/datasets/download


파일 삭제 명령어
rm -rf /home/data/hyun/datasets/download/expansion

"""





