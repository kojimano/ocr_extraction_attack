# python script/annotate_via_google_ocr.py
# export GOOGLE_APPLICATION_CREDENTIALS=~/key.json

from __future__ import print_function

import os
from google.cloud import vision
from IPython import embed
import io
import pickle
import proto
import json
import cv2
from tqdm import tqdm
from google.protobuf.json_format import MessageToDict

client = vision.ImageAnnotatorClient()
image_folder = "TextOCR_raw/train_images"
output_folder = "TextOCR_GoogleOCR"
annotation_dict = {}


def main(image_folder, save_image = True):
    for i, im_name in enumerate(tqdm(os.listdir(image_folder))):
        if i == 20:
            break
        im_path = os.path.join(image_folder, im_name)

        # save raw response
        output_pkl_folder = "{}/raw_pkls".format(output_folder)
        os.makedirs(output_pkl_folder, exist_ok=True)
        output_path = "{}/response_{}".format(output_pkl_folder, im_name.replace(".jpg", ".json"))

        if not os.path.exists(output_path):
            continue #! remove this to make this script working.
            with io.open(im_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.text_detection(image=image)   # query
            json_string = proto.Message.to_json(response)
            response_json = json.loads(json_string)
            json.dump(response_json, open(output_path, "w"))

        else:
            response_json = json.load(open(output_path, "r"))


        # save parsed response
        output_parsed_pkl_folder = "{}/parsed_pkls".format(output_folder)
        os.makedirs(output_parsed_pkl_folder, exist_ok=True)
        output_parsed_path = "{}/result_{}".format(output_parsed_pkl_folder, im_name.replace(".jpg", ".pkl"))

        """
        results = {
            "img_name": im_name,
            "annts": []
        }
        for idx, text in enumerate(response_json['textAnnotations'][1:]):
            output_string = text["description"]
            xs, ys = [], []
            for v in text["boundingPoly"]["vertices"]:
                xs.append(v["x"])
                ys.append(v["y"])

            x, y = min(xs), min(ys)
            x2, y2 =  max(xs), max(ys)
            bbox = [x, y, x2-x, y2-y]

            image_id = im_name.replace(".jpg","")
            id = im_name.replace(".jpg","") + "_" + str(idx)
            results["annts"].append(
                {   
                    "text": output_string,
                    "image_id": image_id,
                    "id": id,
                    "bbox": bbox
                }
            )
        pickle.dump(results, open(output_parsed_path, "wb"))
        """

        # save annotated images (debugging)
        if save_image:
            output_images_folder = "{}/annotated_images".format(output_folder)
            os.makedirs(output_images_folder, exist_ok=True)
            output_image_path = "{}/image_{}".format(output_images_folder, im_name)

            image = cv2.imread(im_path)

            for text in response_json['textAnnotations'][1:]:
                output_string = text["description"]
                xs, ys = [], []
                for v in text["boundingPoly"]["vertices"]:
                    xs.append(v["x"])
                    ys.append(v["y"])

                x, y = min(xs), min(ys)
                x2, y2 =  max(xs), max(ys)
                image = cv2.rectangle(image, (x, y), (x2, y2), (36,255,12), 2)
                #image = cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
                cv2.putText(image, output_string, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 4)
            cv2.imwrite(output_image_path, image)
        

if __name__ == "__main__":
    main(image_folder)