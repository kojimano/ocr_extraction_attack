# python script/process_text_ocr_dataset.py
import os
import json
import argparse
import numpy as np
import random
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
    cropped_im.save(output_image_path)
    #print(js["utf8_string"])
    #print(output_image_path)
    return output_image_path, cropped_im.size


def split_train_val(input_json,  val_ratio = 0.01, debug: bool = False):
    keys = list(input_json.keys())
    random.shuffle(keys)
    num_examples = len(keys)
    train_index = int(num_examples * (1-val_ratio))
    train_keys, val_keys = keys[:train_index], keys[train_index:]

    if debug:
        train_keys = train_keys[:40]
        val_keys = val_keys[:40]

    train_json =  {k: input_json[k] for k in train_keys}
    val_json =  {k: input_json[k] for k in val_keys}
    return train_json, val_json

def create_dataset(input_json, folder_path, dataset_name):
    output_folder = "{}/{}".format(folder_path, dataset_name)
    outfile = open("{}/{}_gt.txt".format(folder_path, dataset_name), "w")
    os.makedirs(output_folder, exist_ok=True)

    im_heights = []
    im_widths = []
    text_lens =[ ]
    # iterate thorugh examples
    for js in tqdm(input_json):
        txt = input_json[js]["utf8_string"].strip()

        # crop image and save
        output_image_path, im_size = crop_and_save_image(input_json[js], output_folder)
        im_heights.append(im_size[0])
        im_widths.append(im_size[1])
        text_lens.append(len(txt))

        # write groundtruth
        outfile.write("{}\t{}\n".format(output_image_path.replace("{}/".format(folder_path), ""), txt))
    
    print("Max txt_len: {}, im_height: {}, im_width: {}".format(np.max(text_lens), np.max(im_heights), np.max(im_widths)))
    outfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="basic", help='Where to store logs and models')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    opt = parser.parse_args()

    random.seed(opt.manualSeed)

    # create dataset folder
    output_folder = "TextOCR/{}".format(opt.dataset_name)
    os.makedirs(output_folder, exist_ok=True)

    # loading json
    train_val_json = load_json("TextOCR_raw/TextOCR_0.1_train.json")
    train_json, val_json = split_train_val(train_val_json)
    test_json = load_json("TextOCR_raw/TextOCR_0.1_val.json")

    # create dataset
    #create_dataset(train_json, output_folder, "training")
    #create_dataset(val_json, output_folder, "validation")
    create_dataset(test_json, output_folder, "evaluation")

