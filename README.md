# ocr_extraction_attack

# Data
Data processing and annotation scripts are located under `/data/script`. We are undecided whther we will release our Google OCR annotation due to the ethical concerns and potential the term of use violation.
```
cd `/data
mkdir TextOCR
mkdir TextOCR/basic
cd TextOCR/basic
wget https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
```

# Models
'''
mkdir models
cd models
git clone https://github.com/clovaai/deep-text-recognition-benchmark.git
'''
