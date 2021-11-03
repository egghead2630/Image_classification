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

# 0.Load in the datas
def default_loader(P):
	return Image.open(P).convert('RGB')
# define our own dataset to load in data
# The idea come from some code in internet, I have list the references in report
class birdDataset(Dataset):
	def __init__(self,path,transform=None,loader=default_loader):
		data = []
		file_content = open(path,'r')
            # read from the indicated path
        for line in file_content:
			line = line.strip('\n')
			words = line.split(' ',1)
			data.append(("../data/train/" + words[0],words[1]))
                # make proper path to load in data
        
		self.data = data
		self.transform = transform
		self.loader = loader
	def __getitem__(self,index):
        # define the method for dataloader to load in data
		path, label = self.data[index]
		img = self.loader(path)
            # use default_loader to load image
		if self.transform != None:
			img = self.transform(img)
                # transform if needed
		return img, int(label[0:3],10) - 1
                # return image along with integer label derived from the first three characters
	def __len__(self):
		return len(self.data)
	
def train_model(model, device, criterion, optimizer,scheduler,dataloader,train_set,num_epoches = 20):
        # The function used to train our model
        # This function refers some code on internet, I have list the references in the report
    since = time.time()
        # record the time
	best_model = copy.deepcopy(model.state_dict())
        # copy now model state
	best_acc = 0.0
        # record accuracy
	for epoch in range(num_epoches):
            # train epoch
		print('Epoch {} / {}'.format(epoch,num_epoches-1))
		print('-' * 10)
            # output to split each Epoch
		for phase in ['train' , 'vali']:
			print(phase)
			if phase == 'train':
				model.train()
                # if it's train phase, then train()
			else:
				model.eval()
                # evaluation phase then eval()
			running_loss = 0.0
			running_correct = 0
                # record the loss and the predict correct numbers
			dataloader['train'] = DataLoader(train_set,batch_size = 10, shuffle = True)
                # reload the data every epoch, to accomplish the data augmentation
                # by performing transform repeatedly
			for i, data in enumerate(dataloader[phase],0):			
				input_imgs, labels = data
                    # split img and label stores them seperatedly
				input_imgs = input_imgs.to(device)
				labels = labels.to(device)
				
				outputs = model(input_imgs)
				loss = criterion(outputs,labels)
                    # Get output and loss
				if phase == 'train':
                    # if in train mode, then update the model
					optimizer.zero_grad()	
					loss.backward()
					optimizer.step()
                        # Classical three step to update the model
                        # 1. Initialize the gradient
                        # 2. Perform back propagation
                        # 3. Update the model by the calculation result
				_, preds = torch.max(outputs, 1)
				running_loss += loss.item() * input_imgs.size(0)
				running_correct += torch.sum(preds == labels.data)
                    # Calculate number of correct predictions
        
				if i % 50 == 1:
                    # every 50 batches we check running_loss
					print('running loss {}'.format(running_loss))
			
			if phase == 'train':
				scheduler.step()
				# Count to reduce learning rate after few steps
			epoch_loss = running_loss / (len(dataloader[phase]) * 10)
			epoch_acc = running_correct.double() / (len(dataloader[phase]) * 10)
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
                # Calulate and print the over all loss and accuracy in the epoch
			if phase == 'vali' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model = copy.deepcopy(model.state_dict())
                # record the best model so far
		print()
	time_elapsed = time.time() - since
        # Calculate  passed time
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
        # Print some info
	model.load_state_dict(best_model)
        # Let model be the best model and return
	return model
	

# 1.do the data normalization and augmentation simultaneously and split to test and train data

transform = torchvision.transforms.Compose([
	torchvision.transforms.Resize((300,300)),
	torchvision.transforms.RandomCrop((256,256)),	
	torchvision.transforms.ToTensor(),
	torchvision.transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225]),
	torchvision.transforms.RandomHorizontalFlip(),
	torchvision.transforms.RandomRotation(30)
])
    # Transform for the training part
valid_transform = torchvision.transforms.Compose([
	torchvision.transforms.Resize((256,256)), 
	torchvision.transforms.ToTensor(),
	torchvision.transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225]),
])
    # Transform for the validation part
whole_dataset = birdDataset(path='../data/training_labels.txt',transform=transform)
    # Define for dataset 
train_size = int(len(whole_dataset) * 0.99)
vali_size = len(whole_dataset) - train_size
    # Define the size for two parts
for i in range(10):
    # Train 10 models to predict test data
    train_set,vali_set = torch.utils.data.random_split(whole_dataset,[train_size, vali_size])
        # Split them as the proportion 0.99: 0.01 randomly
    vali_set.transform = valid_transform    
    train_data_load = DataLoader(train_set,batch_size = 10, shuffle = True)
    vali_data_load = DataLoader(vali_set,batch_size = 10, shuffle = False)
        # Load in the data

    dataload = {}
    dataload['train'] = train_data_load
    dataload['vali'] = vali_data_load
        # Manually store them into a list
    labels = []
    f = open('../data/classes.txt')
    for line in f:
        line = line.strip('\n')
        labels.append(line)
            # record all labels
	
# 2. setting hyperparamter (learning rate & optimizer )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(pretrained = True)
        # Set to use gpu if available otherwise cpu
        # Use pretrained resnet50 here
    number_features = model.fc.in_features
    model.fc = torch.nn.Linear(number_features,len(labels))
        # Resize the output to same number as our label numbers
        # to predict for our data
    model = model.to(device)
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(), lr = 0.02)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 8, gamma = 0.3)
        # Define 
        # 1. Criterion to be CrossEntropyLoss() for it's a classification problem
        # 2. Stochastic Gradient Descent to be optimizer of my model
        # 3. Scheduler to lower the learning rate properly
# 3. train model 
    best_model = train_model(model,device,criterion,optimizer,exp_lr_scheduler,dataload,train_set=train_set)
    path = './best_model'
    index = str(i)
    torch.save(best_model.state_dict(),path + index)
        # Store the model one by one
    