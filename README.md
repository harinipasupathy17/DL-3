# DL-Convolutional-Deep-Neural-Network-for-Image-Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### 1.Load and preprocess the MNIST dataset (normalize pixel values, reshape images).
### 2.Design CNN architecture with convolutional, pooling, flatten, and fully connected layers.
### 3.Compile the model with an optimizer (Adam), loss function (categorical crossentropy), and metrics (accuracy).
### 4.Train the model using training data with defined epochs and batch size.
### 5.Evaluate model performance using test data, accuracy, and confusion matrix.
### 6.Predict digits for new images and analyze misclassifications.



## PROGRAM
### Name: Harini P
### Register Number:212224230082
```
import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset=torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset=torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)

image,label=train_dataset[0]
print("Image shape:",image.shape)
print("Number of training samples:",len(train_dataset))

image,label=test_dataset[0]
print("Image shape:",image.shape)
print("Number of testing samples:",len(test_dataset))
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

class CNNClassifier(nn.Module):
  def __init__(self):
    super(CNNClassifier,self).__init__()
    self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
    self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
    self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
    self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
    self.fc1=nn.Linear(128*3*3,128)
    self.fc2=nn.Linear(128,64)
    self.fc3=nn.Linear(64,10)

  def forward(self,x):
    x=self.pool(t.relu(self.conv1(x)))
    x=self.pool(t.relu(self.conv2(x)))
    x=self.pool(t.relu(self.conv3(x)))
    x=x.view(x.size(0),-1)
    x=nn.functional.relu(self.fc1(x))
    x=nn.functional.relu(self.fc2(x))
    x=self.fc3(x)
    return x

from torchsummary import summary
model=CNNClassifier()
if t.cuda.is_available():
  device=t.device("cuda")
  model.to(device)
print("Name: Harini P")
print("Reg.no: 212224230082")
summary(model,input_size=(1,28,28))
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
def train_model(model,train_loader,num_epochs):
  for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    for images,labels in train_loader:
      if t.cuda.is_available():
        images,labels=images.to(device),labels.to(device)
      optimizer.zero_grad()
      outputs=model(images)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
print("Name: Harini P")
print("Reg.no: 212224230082")
train_model(model,train_loader,num_epochs=10)

def test_model(model, test_loader):
  model.eval()
  correct = 0
  total = 0
  all_preds = []
  all_labels = []
  with t.no_grad():
    for images, labels in test_loader:
      if t.cuda.is_available():
        images, labels = images.to(device), labels.to(device)

      outputs = model(images)
      _, predicted = t.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      all_preds.extend(predicted.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  accuracy = correct/total
  print("Name: Harini P")
  print("Reg.no: 212224230082")
  print(f"Test Accuracy: {accuracy:.4f}")
  return all_labels, all_preds

# Compute confusion matrix
all_labels, all_preds = test_model(model, test_loader)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))

print('Name: Harini P')
print('Register Number: 212224230082')

# Use numbers 0–9 for labels
classes = [str(i) for i in range(10)]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print('Name: Harini P')
print('Register Number: 212224230082')
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

def predict_image(model,image_index,dataset):
  model.eval()
  image,label=dataset[image_index]
  if t.cuda.is_available():
    image=image.to(device)

  with t.no_grad():
    output=model(image.unsqueeze(0))
    _,predicted=t.max(output,1)
  class_names=[str(i) for i in range(10)]
  print("Name: Harini P")
  print("Reg.no: 212224230082")
  plt.imshow(image.cpu().squeeze(0),cmap='gray')
  plt.title(f"Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}")
  plt.axis("off")
  plt.show()
  print(f"Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}")
predict_image(model,image_index=80,dataset=test_dataset)

```


# OUTPUT
## Training Loss per Epoch

<img width="695" height="765" alt="Screenshot 2025-09-02 163659" src="https://github.com/user-attachments/assets/0ad4e7a9-eeb4-422f-9117-40dcb6c76294" />


## Confusion Matrix

<img width="904" height="673" alt="Screenshot 2025-09-02 154412" src="https://github.com/user-attachments/assets/a7d8f651-d1f4-4774-9ba3-49c6cace1597" />


## Classification Report


<img width="681" height="434" alt="Screenshot 2025-09-02 163844" src="https://github.com/user-attachments/assets/2999872f-64dd-48ee-a888-ed88ea8b02b7" />


## New Sample Data Prediction


<img width="629" height="641" alt="Screenshot 2025-09-02 163902" src="https://github.com/user-attachments/assets/97fac4ed-7723-43c3-b284-630f80930630" />


# RESULT

Thus the CNN model was trained and tested successfully on the MNIST dataset.
