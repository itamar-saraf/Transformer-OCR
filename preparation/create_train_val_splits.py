import argparse
import os

import random

from preparation.utils import get_path


def main(args):
    list_path = get_path('list', args)
    with open(list_path) as f:
        gt_data = f.readlines()

    number_of_images = len(gt_data)
    num_validation_images = int(number_of_images * args.val_ratio)

    random.shuffle(gt_data)

    validation_images = gt_data[:num_validation_images]
    train_images = gt_data[num_validation_images:]

    gt_dir = os.path.dirname(args.gt_file)

    with open(os.path.join(get_path('val', args) + gt_dir, "validation.txt"), "a") as f:

        for item in validation_images:
            f.write(item)

    with open(os.path.join(get_path('train', args) + gt_dir, "train.txt"), "a") as f:

        for item in train_images:
            f.write(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool that takes a gt json file and creates a training, validation and reference gt")
    parser.add_argument("gt_file")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="ratio for validation images")

    args = parser.parse_args()
    main(args)
