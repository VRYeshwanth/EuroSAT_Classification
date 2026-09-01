import os
import shutil
import random

random.seed(42)

dir_path = os.path.join(os.getcwd(), "EuroSAT_RGB")

classes = os.listdir(dir_path)
print(f"Present Classes:\n{classes}")

dest_dir = os.path.join(os.getcwd(), "data")
os.makedirs(dest_dir)

print("Performing splitting of dataset :-\n------------------------------")

for class_name in classes:
    directory = os.path.join(dir_path, class_name)
    files = os.listdir(directory)
    random.shuffle(files)

    train_dir = os.path.join(dest_dir, "train", class_name)
    val_dir = os.path.join(dest_dir, "val", class_name)
    test_dir = os.path.join(dest_dir, "test", class_name)

    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)

    n = len(files)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    for train_file in train_files:
        shutil.copy(os.path.join(directory, train_file), train_dir)

    for val_file in val_files:
        shutil.copy(os.path.join(directory, val_file), val_dir)

    for test_file in test_files:
        shutil.copy(os.path.join(directory, test_file), test_dir)

    print(f"Class: {class_name}")
    print(f"No of images in {train_dir}: {len(os.listdir(train_dir))}")
    print(f"No of images in {val_dir}: {len(os.listdir(val_dir))}")
    print(f"No of images in {test_dir}: {len(os.listdir(test_dir))}\n")