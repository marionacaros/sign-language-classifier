import torch
import CustomDataset
import torchvision.transforms as transforms
import Classifier_sign_language_MNIST
import model
import os

num_classes = 25

cwd = os.getcwd()+"/model.txt"
net = model.CNN(num_classes)
net.load_state_dict(torch.load(cwd))
net.eval()

# ### Hyperparameters

learning_rate = 0.01
batch_size = 32
num_classes = 25
epochs = 4


# Define transforms
transformations = transforms.Compose([transforms.ToTensor()])

# Define custom dataset
test_from_csv = CustomDataset.CustomDatasetFromCSV('sign-language-mnist/sign_mnist_test.csv', 28, 28, transformations)
# Define data loader
test_loader = torch.utils.data.DataLoader(dataset=test_from_csv,
                                          batch_size=batch_size,
                                          shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        # images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))