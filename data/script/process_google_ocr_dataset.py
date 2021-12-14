# python script/process_google_ocr_dataset.py
import os
import json
import argparse
import numpy as np
import random
import pickle
from tqdm import tqdm
from IPython import embed
from PIL import Image

def load_json(fn):
    f = open(fn)
    data = json.load(f)
    return data["anns"]

def crop_and_save_image(js, output_folder):
    input_image_path = "TextOCR_raw/train_images/{}.jpg".format(js["image_id"])
    output_image_path = "{}/{}.jpg".format(output_folder, js["id"])
    im = Image.open(input_image_path)
    bbox = js["bbox"]
    cropped_im = im.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])) 
    try:
        cropped_im.save(output_image_path)
        return output_image_path, cropped_im.size, True
    except:
        # seems like the boundign box with 0 height :/
        print("{} failed".format(output_image_path))
        return None, None, False


def split_train_val(input_json):
    infile = open("TextOCR/basic/validation_gt.txt", "r")
    val_keys = set()
    for l in infile.readlines():
        id = l.split("\t")[0]
        id = os.path.basename(id).replace(".jpg", "")
        val_keys.add(id)

    train_json =  {k: input_json[k] for k in input_json.keys() if k not in val_keys}
    val_json =  {k: input_json[k] for k in input_json.keys() if k in val_keys}

    return train_json, val_json

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def remove_overlapping_examples(train_examples, val_examples, th=0.3, verbose=False):
    filtered_examples = []
    removed_examples = []
    for train_ex in train_examples:
        passed = True
        bboxinfo_train = train_ex["bbox"]
        box_train = [bboxinfo_train[0], bboxinfo_train[1], bboxinfo_train[0] + bboxinfo_train[2], bboxinfo_train[1] + bboxinfo_train[3]]
        for val_ex in val_examples:
            bboxinfo_val = val_ex["bbox"]
            box_val = [bboxinfo_val[0], bboxinfo_val[1], bboxinfo_val[0] + bboxinfo_val[2], bboxinfo_val[1] + bboxinfo_val[3]]
            iou = bb_intersection_over_union(box_train, box_val)
            # print(iou)
            if iou > th:
                # print(train_ex, val_ex)
                passed = False
                removed_examples.append(val_ex)
                break
        if passed:
            filtered_examples.append(train_ex)

    if verbose:
        print("Removed {} examples: {}, Val {}".format(len(train_examples) - len(filtered_examples), removed_examples, val_examples))
    return filtered_examples, len(train_examples) - len(filtered_examples)

def create_dataset(train_json, val_json, folder_path, dataset_name):
    output_folder = "{}/{}".format(folder_path, dataset_name)
    outfile = open("{}/{}_gt.txt".format(folder_path, dataset_name), "w")
    os.makedirs(output_folder, exist_ok=True)

    im_heights = []
    im_widths = []
    text_lens = []
    total_removed = 0

    # get all image ids
    all_image_ids = set()
    for k in train_json:
        all_image_ids.add(train_json[k]["image_id"])

    # validation data
    val_examples = {}
    for k in val_json:
        image_id = val_json[k]["image_id"]
        val_examples.setdefault(image_id, [])
        val_examples[image_id].append(val_json[k])
    
    # iterate all images
    parsed_pkl_path = "TextOCR_GoogleOCR/parsed_pkls"

    """
    all_image_ids = []
    for fn in os.listdir(parsed_pkl_path):
        all_image_ids.append(fn.replace("result_","").replace(".pkl",""))
    """

    for id in tqdm(all_image_ids):
        # get training example from Google OCR
        output_parsed_path = "{}/result_{}.pkl".format(parsed_pkl_path, id)
        if not os.path.exists(output_parsed_path):
            continue
        else:
            data = pickle.load(open(output_parsed_path, "rb"))

        # remove validation images
        train_examples = data["annts"]
        if id in val_examples:
            train_examples, num_removed = remove_overlapping_examples(train_examples, val_examples[id])
            total_removed += num_removed

        # crop image and save
        for idx, d in enumerate(train_examples):
            example_id = "{}_{}".format(id, idx)
            data = {
                "image_id": id,
                "id": example_id,
                "bbox": d["bbox"],
            }
            txt = d["text"]
            output_image_path, im_size, success = crop_and_save_image(data, output_folder)
            if not success:
                continue
            im_heights.append(im_size[0])
            im_widths.append(im_size[1])
            text_lens.append(len(txt))

            # write groundtruth
            outfile.write("{}\t{}\n".format(output_image_path.replace("{}/".format(folder_path), ""), txt))

    outfile.close()    
    print("Max txt_len: {}, im_height: {}, im_width: {}, total_removed: {}".format(np.max(text_lens), np.max(im_heights), np.max(im_widths), total_removed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="basic", help='Where to store logs and models')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    opt = parser.parse_args()

    random.seed(opt.manualSeed)

    # create dataset folder
    output_folder = "TextOCR_GoogleOCR/{}".format(opt.dataset_name)
    os.makedirs(output_folder, exist_ok=True)

    # loading json
    train_val_json = load_json("TextOCR_raw/TextOCR_0.1_train.json")
    train_json, val_json = split_train_val(train_val_json)    

    # create dataset
    create_dataset(train_json, val_json,  output_folder, "training")

