import numpy as np
import matplotlib.pyplot as plt


# Import PyTorch packages
import torch,os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid


import torch.nn.functional as F



#####################################################################
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        

        self.conv2 = nn.Conv2d(6, 16,5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        

        self.fcnl1 = nn.Linear(400, 32)
        self.relu3 = nn.ReLU()
        
        self.fcnl2 = nn.Linear(32, 15)
        self.relu4 = nn.ReLU()

        self.fcnl3 = nn.Linear(15, 15)
        self.relu5 = nn.ReLU()
    
    def forward(self, img):
        img = self.conv1(img)
        img = self.relu1(img)
        img = self.pool1(img)
        
        #print(img)
        #print(self.conv1(img).shape)

        img = self.conv2(img)
        img = self.relu2(img)
        img = self.pool2(img)

        img = img.view(-1, 400)
        #img = img.view(img.size(0),-1)

        img = self.fcnl1(img)
        img = self.relu3(img)

        img = self.fcnl2(img)
        img = self.relu4(img)

        img = self.fcnl3(img)
        img = self.relu5(img)

        return img


        
        #img = F.softmax(img, dim=1)
        #return img

    def loadModel(self, model_path):
        self.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    #def saveModel(self, model_path):
    #    torch.save(self.state_dict(), model_path)


'''

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
    
    def forward(self, img):
        output = self.model(img)
        return output
'''

def calcAccuracy(scores, label):
    _, prediction = torch.max(scores, dim=1)
    return torch.tensor(torch.sum(prediction == label).item()/len(scores))

# Cross validate
def validate(validate_ds, model,softmax):
    validate_length = 0
    accuracy = 0
    for img, lbl in validate_ds:
        scores = model(img)
        loss = softmax(scores, lbl)
        accuracy += calcAccuracy(scores, lbl)
        validate_length += 1
    accuracy /= validate_length
    return loss, accuracy

# Run the training and cross validation
def fit(train_ds, validate_ds, no_epochs, optimizer, model):
    history = []
    softmax = nn.CrossEntropyLoss()
    for index in range(no_epochs):
        # Train
        for img, lbl in train_ds:
            scores = model(img)
            loss = softmax(scores, lbl)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Validate
        valid_loss, valid_acr = validate(validate_ds, model, softmax)
            
        # Print epoch record
        print(f"Epoch [{index + 1}/{no_epochs}] => loss: {loss}, val_loss: {valid_loss}, val_acc: {valid_acr}")
        history.append({"loss": loss,
                       "valid_loss": valid_loss,
                       "valid_acr": valid_acr
                       })
    return history


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class GPUDataLoader():
    def __init__(self, ds, device):
        self.ds = ds
        self.device = device
    
    def __iter__(self):
        for batch in self.ds:
            yield to_device(batch, self.device)


def run():
    torch.multiprocessing.freeze_support()



    TRAIN_PATH = os.path.join(os.getcwd(),'train')
    VALIDATE_PATH = os.path.join(os.getcwd(),'test')


    transform = Compose([ToTensor(),Resize((32,32))])
    #transform = Compose([ToTensor(),Resize((128,128))])

    # Load train data and test data
    train_data = ImageFolder(root=TRAIN_PATH, transform=transform)
    validate_data = ImageFolder(root=VALIDATE_PATH, transform=transform)

    print("Train dataset has {0} images".format(len(train_data)))

    # View image size and class
    fst_img, fst_lbl = train_data[0]
    print("First image has size: {0} and class: {1}.".format(fst_img.shape, fst_lbl))

    sc_img, sc_lbl = train_data[5000]
    print("Another random image has size: {0} and class: {1}.".format(sc_img.shape, sc_lbl))

    # View all classes
    classes = train_data.classes
    print("There are {0} classes in total: ".format(len(classes)))
    print(classes)

    train_ds = DataLoader(train_data, shuffle=True, pin_memory=True, num_workers=8)
    validate_ds = DataLoader(validate_data, shuffle=True, pin_memory=True, num_workers=8)

    count = 0
    figure, axis = plt.subplots(1, 4, figsize=(16,8))

    for batch in train_ds:
        img, lbl = batch
        axis[count].imshow(img[0].permute(1, 2, 0))
        axis[count].set_title(classes[lbl.item()])
        axis[count].axis("off")
        if count == 3:
            break
        else:
            count += 1
    #plt.show()



    
    model = SimpleNet()

    device = torch.device('cuda:0')
    model = model.to(device)
    train_ds = GPUDataLoader(train_ds, device)
    validate_ds = GPUDataLoader(validate_ds, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000025)
    no_epochs = 10
    history = fit(train_ds, validate_ds, no_epochs, optimizer, model)

    print(history)

    train_loss = []
    valid_loss = []
    valid_acr = []
    for x in history:
        train_loss.append(x["loss"])
        valid_loss.append(x["valid_loss"])
        valid_acr.append(x["valid_acr"])
        
    train_loss = [x.item() for x in train_loss]
    valid_loss = [x.item() for x in valid_loss]
    valid_acr = [x.item() for x in valid_acr]
    epochs = np.arange(no_epochs)




    torch.save(model.state_dict(),'newCNNmodel.pkl',_use_new_zipfile_serialization=False)

    plt.plot(epochs, train_loss)
    plt.plot(epochs, valid_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss of training and validation over iterations")
    plt.legend(["training", "validation"])
    plt.show()

    #plt.plot(epochs, valid_acr)
    #plt.xlabel("Epochs")
    #plt.ylabel("Accuracy")
    #plt.title("Cross validation accuracy")
    #plt.show()


    

if __name__ == '__main__':
    run()
