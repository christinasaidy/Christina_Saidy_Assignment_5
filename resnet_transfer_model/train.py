import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Pet_Model_Dataset  
from model import ResNet_Model 
import os
from utils import *
from torch.utils.tensorboard import SummaryWriter
import tensorboard 

######CONFIGS
batch_size = 32
class_number = 37
epochs = 15
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bbox_dir = "C:/Users/saidy/Desktop/Christina_Saidy_Assigment_5/Christina_Saidy_Assignment_5+6/oxford-iiit-pet/annotations/xmls"
imag_dir = "C:/Users/saidy/Desktop/Christina_Saidy_Assigment_5/Christina_Saidy_Assignment_5+6/oxford-iiit-pet/images"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

##DATASET
dataset = Pet_Model_Dataset(bbox_dir=bbox_dir,
    images_dir=imag_dir,
    transform=transform)

# Get train + val splits
train_dataset, test_dataset = splits(dataset, train_ratio=0.7)
train, val = trainval_splits(train_dataset, val_ratio=0.2)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

model = ResNet_Model(num_classes=class_number).to(device)

criterion_class = nn.CrossEntropyLoss() # classification loss
criterion_bbox = nn.MSELoss()            # bounding box loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

writer = SummaryWriter(log_dir="runs/pet_model_losses")

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device) 
        labels = batch['category'].to(device)
        bboxes = batch['bbox'].to(device)

        optimizer.zero_grad()
        outputs_label, outputs_bbox = model(images)  

        loss_class = criterion_class(outputs_label, labels)
        loss_bbox = criterion_bbox(outputs_bbox, bboxes)

        loss = loss_class + loss_bbox
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        _, predicted = outputs_label.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.state_dict(), f"saved_models/resnet_epoch{epoch + 1}.pth")

    avg_train_loss = running_train_loss / len(train_loader)
    train_accuracy = 100. * correct_train / total_train

    print(f"Epoch [{epoch+1}] Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
    writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch)
    writer.add_scalar('Epoch/Train_Accuracy', train_accuracy, epoch)

    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels, bboxes in val_loader:
            images = batch['image'].to(device) 
            labels = batch['category'].to(device)
            bboxes = batch['bbox'].to(device)

            outputs_label, outputs_bbox = model(images)

            loss_class = criterion_class(outputs_label, labels)
            loss_bbox = criterion_bbox(outputs_bbox, bboxes)

            loss = loss_class + loss_bbox
            running_val_loss += loss.item()

#ACCURACY
            _, predicted = outputs_label.max(1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100. * correct_val / total_val

    print(f"Epoch [{epoch+1}] Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    writer.add_scalar('Epoch/Val_Loss', avg_val_loss, epoch)
    writer.add_scalar('Epoch/Val_Accuracy', val_accuracy, epoch)

writer.close()