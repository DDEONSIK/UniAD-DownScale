# 파일 압축 해제 및 폴더 병합, log 작성, 오류catch 
# 1시간30분정도 소요

import os
import tarfile
import zipfile
import shutil
import logging

# 로깅 설정
logging.basicConfig(filename='extraction_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 소스 디렉토리와 대상 디렉토리 설정
source_dir = "/home/data/hyun/datasets/download"
destination_dir = "/home/data/hyun/datasets/download/nuscenes"

# 대상 디렉토리가 존재하지 않으면 생성
os.makedirs(destination_dir, exist_ok=True)

def merge_folder(src, dst):
    """폴더 병합 함수: src에서 dst로 파일과 폴더를 병합"""
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

# 처리된 파일 목록을 저장할 파일 경로
processed_files_log = '/home/data/hyun/datasets/download/processed_files.txt'

# 이전에 처리된 파일 목록 불러오기
processed_files = set()
if os.path.exists(processed_files_log):
    with open(processed_files_log, 'r') as f:
        processed_files = set(f.read().splitlines())

# 소스 디렉토리 내의 모든 .tar, .tgz, 및 .zip 파일을 순회
for file_name in os.listdir(source_dir):
    if (file_name.endswith(".tar") or file_name.endswith(".tgz") or file_name.endswith(".zip")) and file_name not in processed_files:
        # 파일의 전체 경로 구성
        file_path = os.path.join(source_dir, file_name)
        
        # 임시 디렉토리 생성
        temp_dir = os.path.join(source_dir, "temp_extraction")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # .tar 및 .tgz 파일 처리
            if file_name.endswith(".tar") or file_name.endswith(".tgz"):
                with tarfile.open(file_path, "r:*") as tar:
                    tar.extractall(path=temp_dir)
                logging.info(f"Extracted: {file_name} to {temp_dir}")
            
            # .zip 파일 처리
            elif file_name.endswith(".zip"):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                logging.info(f"Extracted: {file_name} to {temp_dir}")
            
            # 압축 해제된 내용 병합
            merge_folder(temp_dir, destination_dir)
            logging.info(f"Merged: {file_name} content to {destination_dir}")
            
            # 처리 완료된 파일 기록
            with open(processed_files_log, 'a') as f:
                f.write(f"{file_name}\n")
            
        except Exception as e:
            logging.error(f"Error processing {file_name}: {str(e)}")
        finally:
            # 임시 디렉토리 삭제
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

logging.info("All .tar, .tgz, and .zip files have been processed.")
print("Process completed. Check 'extraction_log.txt' for details.")
