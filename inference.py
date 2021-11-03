# using

# bagging: 10

import torch
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from torchvision import utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import time
import copy
import sys

data_path = sys.argv[1]
model_path = sys.argv[2]


def default_loader(P):
	return Image.open(P).convert('RGB')


def make_output(result):
    # store result into answer.txt
    labels = []
    f = open(data_path + '/classes.txt')
    for line in f:
        line = line.strip('\n')
        labels.append(line)
        # grab all the labels
    file_content = open(data_path + '/testing_img_order.txt', 'r')
    # grab the test order
    output_file = open('./answer.txt', 'w')
    lines = []
    for line in file_content:
        line = line.strip('\n')
        lines.append(line)
        # get test file name
    cnt = 0
    for predicts in result:
        for predict in predicts:
            output_file.write(lines[cnt] + ' ' + labels[predict] + '\n')
            # connect the test file name and prediction
            # write into answer.txt
            cnt += 1
    return


def vote(preds):
    # Performing the vote based on predictions made by all 10 models
    bag_size = len(preds)
    batch_num = len(preds[0])
    vote_result = []

    for i in range(batch_num):
        unit = []
        batch_size = len(preds[0][i])
        for j in range(batch_size):
            vote_dict = {}
            winner = 0
            now_best = 0
            for k in range(bag_size):
                # iterate over each image
                if preds[k][i][j] in vote_dict:
                    vote_dict[preds[k][i][j]] += 1
                else:
                    vote_dict[preds[k][i][j]] = 1
                # each model vote for their own prediction
            for key in vote_dict:
                if vote_dict[key] > now_best:
                    winner = key
                    now_best = vote_dict[key]
                    # get the label that most of the model vote
            unit.append(winner)
        vote_result.append(unit)
        # store all results

    return vote_result


# same as in train.py,slightly changing to read in test data but not train data
class birdDataset(Dataset):
    def __init__(self, path, transform=None, loader=default_loader):
        data = []
        file_content = open(path, 'r')
        for line in file_content:
            line = line.strip('\n')
            words = line.split(' ', 1)
            if len(words) == 2:
                data.append((data_path + "/train/" + words[0], words[1]))
            else:
                data.append((data_path + "/test/" + words[0]))
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        if len(self.data[index]) == 2:
            path, label = self.data[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            return img, int(label[0:3], 10) - 1
        else:
            path = self.data[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            return img

    def __len__(self):
        return len(self.data)


def predict(model, device, dataloader):
    # Store one model's predict result
    model.eval()
    running_correct = 0
    whole_preds = []
    for i, data in enumerate(dataloader, 0):
        input_imgs = data
        input_imgs = input_imgs.to(device)
        outputs = model(input_imgs)
        _, preds = torch.max(outputs, 1)
        # Predict
        whole_preds.append(preds.cpu().numpy())
        # Store the Predict result
    return whole_preds


valid_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((270, 270)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

vali_set = birdDataset(path=data_path + '/testing_img_order.txt',
                       transform=valid_transform)
dataloader = DataLoader(vali_set, batch_size=20, shuffle=False)
# Load data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(pretrained=True)
number_features = model.fc.in_features
model.fc = torch.nn.Linear(number_features, 200)
# Initialize the model

results = []
for i in range(10):
    # For each model we store their predictions
    p = model_path + '/best_model' + str(i)
    print('testing {}'.format(p))
    model.load_state_dict(torch.load(p))
    model = model.to(device)
    result = predict(model, device, dataloader)
    results.append(result)
print(len(results))
result = vote(results)
# result shows the final prediction dicided by bagging method(vote)
make_output(result)
