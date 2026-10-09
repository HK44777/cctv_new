import scipy.io
import os
import shutil
from tqdm import tqdm

def main():
    info_dir = 'MARS-evaluation/info'
    base_dir = 'C:/Users/PC1/Downloads/mars_dataset/OneDrive_1_2022-4-2'
    bbox_test_dir = os.path.join(base_dir, 'bbox_test', 'bbox_test')
    query_dir = os.path.join(base_dir, 'query', 'query')

    if not os.path.exists(query_dir):
        os.makedirs(query_dir)

    print('Loading MARS info files...')
    # test_name.txt has image names
    with open(f'{info_dir}/test_name.txt', 'r') as f:
        test_names = f.read().splitlines()
        
    # tracks_test_info.mat contains [start_index, end_index, person_id, camera_id] (1-indexed)
    track_info = scipy.io.loadmat(f'{info_dir}/tracks_test_info.mat')['track_test_info']
    
    # query_IDX.mat contains indices of query tracklets (1-indexed)
    query_idx = scipy.io.loadmat(f'{info_dir}/query_IDX.mat')['query_IDX'].squeeze()

    print(f'Found {len(query_idx)} query tracklets.')

    # Collect all image names that belong to query tracklets
    query_images = set()
    for idx in query_idx:
        # idx is 1-indexed, so we subtract 1 for python arrays
        track = track_info[idx - 1] 
        start_idx = track[0] - 1
        end_idx = track[1] - 1
        
        for i in range(start_idx, end_idx + 1):
            query_images.add(test_names[i])

    print(f'Identified {len(query_images)} query images. Moving them to the query folder...')

    moved_count = 0
    missing_count = 0
    for img_name in tqdm(query_images):
        pid = img_name[:4]
        src_path = os.path.join(bbox_test_dir, pid, img_name)
        dst_folder = os.path.join(query_dir, pid)
        dst_path = os.path.join(dst_folder, img_name)
        
        if os.path.exists(src_path):
            os.makedirs(dst_folder, exist_ok=True)
            shutil.move(src_path, dst_path)
            moved_count += 1
        else:
            missing_count += 1

    print(f'Successfully moved {moved_count} images to the query folder!')
    if missing_count > 0:
        print(f'Note: {missing_count} images were already moved or missing from bbox_test.')

if __name__ == "__main__":
    main()
