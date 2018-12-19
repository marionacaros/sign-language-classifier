
import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import CustomDataset
import model
import os

train = pd.read_csv('sign-language-mnist/sign_mnist_train.csv')
test = pd.read_csv('sign-language-mnist/sign_mnist_test.csv')

# The data set is given in the form of labels and pixel value ranging from pixel 1 to pixel 784 which is 28 * 28 image.

labels_val = train['label'].values
labels_unique = np.unique(np.array(labels_val))
print('Labels:', labels_unique)

print('There are {} images in the train set'.format(len(train)))
print('There are {} images in the test set'.format(len(test)))

# ### Hyperparameters

learning_rate = 0.01
batch_size = 32
num_classes = 25
epochs = 7

# #### Show image

train.drop('label', axis=1, inplace=True)

images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])
plt.imshow(images[0].reshape(28, 28))


# Define transforms
transformations = transforms.Compose([transforms.ToTensor()])
# Define custom dataset
imag_from_csv = CustomDataset.CustomDatasetFromCSV('sign-language-mnist/sign_mnist_train.csv', 28, 28,
                                     transformations)
# Define data loader
mn_dataset_loader = torch.utils.data.DataLoader(dataset=imag_from_csv,
                                                batch_size=batch_size,
                                                shuffle=True)

# Call model
net = model.CNN(25)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# ### Train the network
# -----------------------------------------------------------------------------------------------------------------


for epoch in range(epochs):  # loop over the dataset multiple times
    losses = []
    running_loss = 0.0
    for i, (images, labels) in enumerate(mn_dataset_loader):
        # Feed the data to the model

        # get the inputs
        inputs = images
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # print(CNN())
        outputs = net(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # print statistics
        if (i + 1) % 100 == 0:
            print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % ( epoch + 1, epochs, i + 1, len(train) // batch_size, loss.data[0]))
    plt.plot(losses)

print('Finished Training')
cwd = os.getcwd()+"/model.txt"
print(cwd)
torch.save(net.state_dict(), cwd)

