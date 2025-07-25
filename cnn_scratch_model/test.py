import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import Pet_Model_Dataset  
from model import CNN_model
from utils import *
import cv2
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
class_number = 37
best_model_path = "CNN_SAVED/best_model.pth"
bbox_dir = "C:/Users/saidy/Desktop/Christina_Saidy_Assigment_5/Christina_Saidy_Assignment_5+6/oxford-iiit-pet/annotations/xmls"
imag_dir = "C:/Users/saidy/Desktop/Christina_Saidy_Assigment_5/Christina_Saidy_Assignment_5+6/oxford-iiit-pet/images"

model = CNN_model(class_number=class_number).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

dataset = Pet_Model_Dataset(bbox_dir=bbox_dir, images_dir=imag_dir, transform=transform)
x, test_data = splits(dataset)
dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

def plot_predictions(dataloader, model, n=5):
    model.eval()

    for i, sample in enumerate(dataloader):
        if i >= n:
            break

        image_tensor = sample['image'].to(device)
        actual_class = sample['category'].item()
        actual_bbox = sample['bbox'][0].tolist()

        with torch.no_grad():
            pred_class, pred_bbox = model(image_tensor)
            pred_class = torch.argmax(pred_class, dim=1).item()  ######## dim =1 same as axis =1 in numpy
            pred_bbox = pred_bbox[0].cpu().tolist()

        image = image_tensor[0].cpu().permute(1, 2, 0).numpy()##  gives err without permute changes shape to cv2 friendly 
        print("image", image) 
        image = (image * 255).astype(np.uint8)  ##### change normalized image and convert it to int cause floats arent cv2 friendly
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]

        actual_box = bbox_to_pix(actual_bbox, w, h) 
        pred_box = bbox_to_pix(pred_bbox, w, h)

        #IOU CALCULATION
        iou = intersection_over_union(actual_box, pred_box)
        print(f"IoU: {iou}")

        cv2.rectangle(image,
                      (int(actual_box[0]), int(actual_box[1])),
                      (int(actual_box[2]), int(actual_box[3])),
                      color=(0, 255, 0), thickness=2)
        cv2.putText(image, f"Actual: {actual_class}",
            (int(actual_box[0]), int(actual_box[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(image,
                      (int(pred_box[0]), int(pred_box[1])),
                      (int(pred_box[2]), int(pred_box[3])),
                      color=(0, 0, 255), thickness=2)
        cv2.putText(image, f"Predicted: {pred_class}",
            (int(pred_box[0]), int(pred_box[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        os.makedirs("saved_cnn_output", exist_ok=True)
        output_path = os.path.join("saved_cnn_output", f"prediction_{i+1}.jpg")
        cv2.imwrite(output_path, image)
        cv2.imshow(f'Prediction {i+1}', image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

plot_predictions(dataloader, model, n=5)