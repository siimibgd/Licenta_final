import os
import shutil
import random
import math


def move_10_percent_jpgs_from_each_subdir(main_dir, test_dir_name="test", sample_ratio=0.10):
    test_dir = os.path.join(main_dir, test_dir_name)
    os.makedirs(test_dir, exist_ok=True)

    for subdir_name in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir_name)

        if not os.path.isdir(subdir_path) or subdir_path == test_dir:
            continue

        jpg_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(".jpg")]
        if not jpg_files:
            continue

        num_to_move = max(1, math.ceil(sample_ratio * len(jpg_files)))
        sampled_files = random.sample(jpg_files, num_to_move)

        for file_name in sampled_files:
            src_path = os.path.join(subdir_path, file_name)
            dst_path = os.path.join(test_dir, file_name)

            if os.path.exists(dst_path):
                base, ext = os.path.splitext(file_name)
                counter = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(test_dir, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.move(src_path, dst_path)
            print(f"Moved: {src_path} -> {dst_path}")


main_directory = "../EuroSAT_RGB"
move_10_percent_jpgs_from_each_subdir(main_directory)
