import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from preparation.utils import get_path
from preparation.bbox import BBox, Vec2
import re
import scipy.io
from PIL import Image
from tqdm import tqdm


def split_text(text):
    all_words = [word for instance in text for word in re.split(r'\s+', instance) if len(word) > 0]
    return all_words


def get_bboxes(bboxes):
    if len(bboxes.shape) == 2:
        bboxes = bboxes[..., None]

    bboxes = bboxes.transpose(2, 1, 0)
    boxes = []
    aabbs = []
    for box in bboxes:
        box = BBox._make([Vec2._make(b).to_int() for b in box])
        boxes.append(box)

        aabbs.append(box.to_aabb())

    return boxes, aabbs


def get_relative_box_position(anchor, box):
    anchor_vec = Vec2(anchor.left, anchor.top)
    return BBox(
        box.top_left - anchor_vec,
        box.top_right - anchor_vec,
        box.bottom_right - anchor_vec,
        box.bottom_left - anchor_vec,
    )


def main(args):
    path = get_path('gt.mat', args)
    gt_data = scipy.io.loadmat(path)

    image_names = gt_data['imnames'][0]
    word_bboxes = gt_data['wordBB'][0]
    char_bboxes = gt_data['charBB'][0]
    text = gt_data['txt'][0]

    iterator = tqdm(zip(image_names, word_bboxes, char_bboxes, text), total=len(image_names))

    for image_name, word_boxes, char_boxes, words in iterator:
        if '85' != image_name[0].split('/')[0]:
            continue
        words = split_text(words)
        image_name = image_name[0]
        bboxes, aabbs = get_bboxes(word_boxes)

        try:
            with Image.open(os.path.join(get_path('images', args), image_name)) as the_image:
                for i in range(len(aabbs)):
                    crop = aabbs[i].crop_from_image(the_image)
                    destination_dir = os.path.join(
                        get_path('saved_images', args) + args.destination + '/' + os.path.dirname(image_name))
                    os.makedirs(destination_dir, exist_ok=True)
                    destination_file_name = f"{os.path.splitext(image_name)[0]}_{i}.jpg"

                    crop.save(
                        os.path.join(get_path('saved_images', args) + args.destination + '/' + destination_file_name))

                    with open(os.path.join(get_path('saved_images', args) + args.destination + '/' + "list.txt"),
                              "a") as f:
                        path = './dataset/' + destination_file_name

                        f.write(path + '\t' + words[i] + '\n')

        except Exception as e:
            print(e)
            print(image_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool that takes oxford detection gt and crops all words using their aabb")
    parser.add_argument("gt_file", help="path to oxford gt file")
    parser.add_argument("destination", help="path to destination dir, where you want to save the cropped images")

    args = parser.parse_args()
    main(args)
