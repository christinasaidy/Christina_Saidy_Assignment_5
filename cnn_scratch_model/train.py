import torch
from torch.utils.data import DataLoader
from model import CNN_model
from dataset import Pet_Model_Dataset
from torchvision import transforms
import os
from utils import *
from torch.utils.tensorboard import SummaryWriter
import tensorboard 

# CONFIGS
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
learning_rate = 1e-4
bbox_dir = "C:/Users/saidy/Desktop/Christina_Saidy_Assignment_5+6/oxford-iiit-pet/annotations/xmls"
imag_dir = "C:/Users/saidy/Desktop/Christina_Saidy_Assignment_5+6/oxford-iiit-pet/images"
epochs = 35
class_number = 37
save_dir = "CNN_SAVED"
best_model_path = os.path.join(save_dir, "best_model.pth")  # path to save best model

os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dataset = Pet_Model_Dataset(bbox_dir=bbox_dir,
    images_dir=imag_dir,
    transform=transform)

# get train + val
train_dataset, test_dataset = splits(dataset, train_ratio=0.7)  # function in utils that splits dataset into train + test
train, val = trainval_splits(train_dataset, val_ratio=0.2)     # splits train into train + val

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

model = CNN_model(class_number=class_number).to(device)
criterion_category = torch.nn.CrossEntropyLoss()
criterion_bbox = torch.nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer = SummaryWriter(log_dir="runs/cnn_model_losses")  # using tensorboard to track losses

best_val_loss = 3733773  #big random number

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch in train_loader:
        images = batch['image'].to(device)
        labels = batch['category'].to(device)
        bboxes = batch['bbox'].to(device)  

        optimizer.zero_grad()

        outputs_category, outputs_bbox = model(images)

        loss_class = criterion_category(outputs_category, labels)
        loss_bbox = criterion_bbox(outputs_bbox, bboxes)

        total_loss = 0.6 * loss_class + loss_bbox

        total_loss.backward()
        optimizer.step()

        running_train_loss += total_loss.item()

#ACCURACY
        _, predicted = outputs_category.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_accuracy = 100. * correct_train / total_train

    print(f"Epoch {epoch+1}/{epochs}, Total Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    writer.add_scalar('Loss/train', avg_train_loss, epoch + 1)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch + 1)

    save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), save_path)

    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['category'].to(device)
            bboxes = batch['bbox'].to(device)  

            outputs_category, outputs_bbox = model(images)

            loss_class = criterion_category(outputs_category, labels)
            loss_bbox = criterion_bbox(outputs_bbox, bboxes)

            total_loss = loss_class + loss_bbox
            running_val_loss += total_loss.item()

            _, predicted = outputs_category.max(1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100. * correct_val / total_val

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    writer.add_scalar('Loss/Val', avg_val_loss, epoch + 1)
    writer.add_scalar('Accuracy/Val', val_accuracy, epoch + 1)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print("Best model updated!")

writer.close()