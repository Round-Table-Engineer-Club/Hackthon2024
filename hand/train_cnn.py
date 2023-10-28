# %% [code]
import torch
import torch.optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layers3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layers4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*128,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,26)
        )
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x

def train_model():
    train_data = pd.read_csv('sign_mnist_train.csv', dtype=np.float32)

    train_labels = train_data.label.values
    train_features = train_data.loc[:, train_data.columns != 'label'].values / 255  # scaling feature values

    # 这里没有用到train_test_split，因为我们有一个独立的测试集文件
    # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    train_x = torch.from_numpy(train_features)
    train_y = torch.from_numpy(train_labels).type(torch.LongTensor)  # converting the target into torch format

    # 加载测试数据
    test_data = pd.read_csv('sign_mnist_test.csv', dtype=np.float32)  # 确保路径正确

    test_labels = test_data.label.values
    test_features = test_data.loc[:, test_data.columns != 'label'].values / 255  # scaling feature values

    test_x = torch.from_numpy(test_features)
    test_y = torch.from_numpy(test_labels).type(torch.LongTensor)  # converting the target into torch format


    train = TensorDataset(train_x,train_y)
    test = TensorDataset(test_x,test_y)

    train_loader = DataLoader(train,batch_size=100,shuffle=False,drop_last=True)
    test_loader = DataLoader(test,batch_size=100,shuffle=False,drop_last=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using {} device.".format(device))
    cnn = CNN().to(device)

    error = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(cnn.parameters(),lr=0.1)

    for epoch in range(100):  # using 'epoch' here as it refers to multiple cycles
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)  # moving data to GPU or CPU
            img = img.view(img.shape[0], 1, 28, 28)  # using img.shape[0] instead of hardcoded batch size
            optimizer.zero_grad()
            output = cnn(img)
            loss = error(output, label)
            loss.backward()
            optimizer.step()

            # Optional: You might want to print the training loss every few batches

        # Calculating validation/test accuracy after each epoch
        cnn.eval()  # setting the model to evaluation mode
        with torch.no_grad():  # turning off gradients to save memory
            num_correct = 0
            num_samples = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                x = x.view(x.shape[0], 1, 28, 28)
                out = cnn(x)
                _, predictions = torch.max(out, 1)  # getting the predicted class
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            accuracy = float(num_correct) / float(num_samples)  # calculating accuracy
            print(f'Epoch [{epoch+1}/100], Accuracy: {accuracy:.4f}')

            # Saving the best model
            if accuracy > 0.975:  # based on your accuracy requirement
                torch.save(cnn.state_dict(), 'weight/best.pth')
                if accuracy > 0.983:
                    break  # stop training if accuracy exceeds this threshold

        cnn.train()  # setting the model back to training mode if you're continuing training afterwards


if __name__ == "__main__":
    print("Training script is running...")
    train_model()