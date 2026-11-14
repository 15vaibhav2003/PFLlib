# generating_PathMNIST.py and building pathmnist.py for PFlib: downloading and spliting among clients 

import numpy as np
import os
import sys
import random
from medmnist import PathMNIST
from utils.dataset_utils import check, separate_data, split_data, save_file

# Here  we have fixed the seed for a each client so that they get same data in each round this is neceassry as due to this each client seed same data throughout.
random.seed(1)
np.random.seed(1)
num_clients = 5            # 5 simulated hospitals
dir_path = "PathMNIST/"    # output folder inside dataset/


#downloading one official spit 
def load_split(split, root):
   
    ds = PathMNIST(split=split, download=True, root=root, size=28)
    x = ds.imgs                
    y = ds.labels.reshape(-1)   
    return x, y


def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    #skipping the datset if one already exist
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    root = dir_path + "rawdata"
    os.makedirs(root, exist_ok=True)

  # laoding all three offical splits and merging them like te mnist template
  
    xs = []
    ys = []
    for split in ["train", "val", "test"]:
        x, y = load_split(split, root)
        print(f"{split}: {len(y)} images")
        xs.append(x)
        ys.append(y)
    x = np.concatenate(xs)
    y = np.concatenate(ys).astype(np.int64)
    del xs, ys

   
    x = x.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5

  
    x = x.transpose(0, 3, 1, 2)

    num_classes = len(set(y))
    print(f"Number of classes: {num_classes}")
    print(f"Total images: {len(y)}, image shape: {x.shape[1:]}")

    # pflib's functions for drichlet split
    X, Y, statistic = separate_data((x, y), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, Y)
    save_file(config_path, train_path, test_path, train_data, test_data,
              num_clients, num_classes, statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
