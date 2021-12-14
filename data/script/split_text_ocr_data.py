# python script/split_text_ocr_data.py --training_perc 0.05
# python script/split_text_ocr_data.py --training_perc 0.1
# python script/split_text_ocr_data.py --training_perc 0.2
# python script/split_text_ocr_data.py --training_perc 0.4
# python script/split_text_ocr_data.py --training_perc 0.7

import os
import argparse
from IPython import embed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_perc', type=float, default=1.0, help='Where to store logs and models')
    opt = parser.parse_args()
    source_folder = "TextOCR_GoogleOCR"

    # create dataset folder
    output_folder = "{}/basic/training_{}".format(source_folder, opt.training_perc)

    # sample from traning folder 
    input_folder = "{}/basic/training".format(source_folder)
    
    # read examples
    infile = open("{}/basic/training_gt.txt".format(source_folder), "r")
    outfile = open("{}/basic/training_{}_gt.txt".format(source_folder, opt.training_perc), "w")

    lines = infile.readlines()
    num_lines = len(lines)
    split_index = int(num_lines * opt.training_perc)
    print(split_index)
    for l in lines[:split_index]:
        outfile.write(l)
    outfile.close()

