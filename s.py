import os

train_path = "archive/fruits-360_dataset_original-size/fruits-360-original-size/Training"
class_names = sorted(os.listdir(train_path))  # Sorting ensures consistent ordering

dataset_yaml = f"train: {train_path}\nval: /home/arshlaan/Downloads/Frutly/images/val\nnc: {len(class_names)}\nnames: {class_names}"

with open("/home/arshlaan/Downloads/Frutly/dataset.yaml", "w") as f:
    f.write(dataset_yaml)

print("Dataset.yaml created!")
