from torch.utils.data import Dataset
from torchvision import transforms
import torch
from typing import Optional, Callable
import xml.etree.ElementTree as ET
from PIL import Image
import os

class Pet_Model_Dataset(Dataset):
    def __init__(self, bbox_dir: str, images_dir: str, transform: Optional[Callable] = None):
        self.bbox_dir = bbox_dir  
        self.images_dir = images_dir  
        self.transform = transform
        self.bbox_files: list[str] = sorted([f for f in os.listdir(bbox_dir) if f.endswith('.xml')])

    def __len__(self) -> int:
        return len(self.bbox_files)

    def __getitem__(self, index: int) -> dict:
        bbox_path = os.path.join(self.bbox_dir, self.bbox_files[index])
        try:
            tree = ET.parse(bbox_path)
        except ET.ParseError as e:
            raise ValueError(f"XML parse error in file: {bbox_path} — {str(e)}")
        root = tree.getroot()

        filename = root.find('filename').text.strip() #filename has breed name in it
        image_path = os.path.join(self.images_dir, filename)

        image = Image.open(image_path).convert("RGB")

        breed = "_".join(filename.split('_')[:-1]) #extracts from file name the breed and assigns teh corresponding label
        #i know this isn't a very good practice however i just didnt know how to get the label number from the xml files in another way

        label = None
        for known_breed in breed_to_label:
            if known_breed.lower() == breed.lower():
                label = breed_to_label[known_breed]
                break

        size_tree = root.find('size')
        height = float(size_tree.find('height').text) ##the bboxes are normalized in a way that upon reshaping and transforming images, this doesnt affect them
        width = float(size_tree.find('width').text)

        obj = root.find('object')
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        xmax = float(bbox.find('xmax').text)
        ymin = float(bbox.find('ymin').text)
        ymax = float(bbox.find('ymax').text)

        cx = (xmin + xmax) / (2 * width)
        cy = (ymin + ymax) / (2 * height)
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        box_tensor = torch.tensor([cx, cy, w, h], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "category": label,
            "bbox": box_tensor}
    
breed_to_label = {  #breeds with corresponding labels, to use when breed is extracted from .xml file, 
    "Abyssinian": 0,
    "Bengal": 5,
    "Birman": 6,
    "Bombay": 7,
    "British_Shorthair": 9,
    "Egyptian_Mau": 11,
    "Maine_Coon": 20,
    "Persian": 23,
    "Ragdoll": 26,
    "Russian_Blue": 27,
    "Siamese": 32,
    "Sphynx": 33,
    "american_bulldog": 1,
    "american_pit_bull_terrier": 2,
    "basset_hound": 3,
    "beagle": 4,
    "boston_terrier": 15,
    "boxer": 8,
    "chihuahua": 10,
    "english_cocker_spaniel": 12,
    "english_setter": 13,
    "german_shorthaired": 14,
    "great_pyrenees": 15,
    "havanese": 16,
    "japanese_chin": 17,
    "keeshond": 18,
    "leonberger": 19,
    "miniature_pinscher": 21,
    "newfoundland": 22,
    "pomeranian": 24,
    "pug": 25,
    "saint_bernard": 28,
    "samoyed": 29,
    "scottish_terrier": 30,
    "shiba_inu": 31,
    "staffordshire_bull_terrier": 34,
    "wheaten_terrier": 35,
    "yorkshire_terrier": 36}

#REFERNCE FOR THE XML + NORMALIZATION OF BBOX: "WRITE FUNCTION TO EXTRACT BOUNDING BOXES FROM EACH IMAGE"
#  https://medium.com/@saptarshimt/yolo-v1-pascal-voc-simplistic-pytorch-implementation-from-scratch-961fa36f4d4d