# VRDL_HW1
-------------------------------------------------------------------------
This repository is the official implementation of My VRDL_HW1: Image classification


Requirements
-------------------------------------------------------------------------
Python version:
　　Python 3.9.7

pytorch:
    pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

sklearn:
    pip3 install scikit-learn

PIL:
    python3 -m pip install --upgrade Pillow

matplotlib:
    pip3 install matplotlib








Training
-------------------------------------------------------------------------
To train the models, run this command:

python3 train.py <path_to_data> 

    Then the the program would perform training and save the ten best models locally, from best_model0 to best_model9
    
    In the <path_to_data> you shall indicate the path to the directory that contains following files and directories:
            1. train(folder): contain all of the images for training
            2. test(folder): contain all of the images for testing
            3. classes.txt: contain all the labels
            4. testing_img_order.txt: contain the order of testing image coming in
            5. training_labels.txt: contain all the training image number aloing with their labels
    
    About train process, I specify the transform as:

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize((300,300)),
		torchvision.transforms.RandomCrop((270,270)),	
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225]),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.RandomRotation(30)
	])
  with batch_size = 10, for more detailed explaination about the whole process, please refer to the code comment and report.
 
 Warning: You the data path shall not contain / or \ in the tail.

 	Correct: /data
 	Error: /data/
 
 If you mistype, then error would occur, this format is the same for all the path in this implementation.
  
 
train.py's link: 






Evaluation
-------------------------------------------------------------------------
To evaluate my model on, run:

python3 inference.py <path_to_data> <path_to_model>
    
    <path_to_data> is the same as train part
    
    In the <path_to_model> you shall indicate the path to the directory that contains following files:
            1. All of the 10 models from best_model0 to best_model9
    
To get the 10 models, go to this google cloud to download: https://drive.google.com/drive/folders/1pyId96KiD4XHN25f7EQhh9PYi792z6Pi?usp=sharing

After running, an answer.txt would be generated and stored locally.
    
Compress the answer.txt to .zip file and upload to codalab is enough to evaluate my model's prediction
    
For convinience, I write a checktxt.py to check if two answer.txt is the same
    
Run checktxt.py: python3 checktxt.py <path_to_first_answer> <path_to_second_answer>
    
    <path_to_first_answer> should be filled with the path to first answer.txt   ex: /answer1/answer.txt
    
    <path_to_second_answer> should be filled with the path to second answer.txt   ex: /answer2/answer.txt
    
    If Output is True, means the two answer.txt are exactly the same, in False case, they're not.

inference.py's link





Pre-trained Models
-------------------------------------------------------------------------
You can download and use pretrained models by simply running this command in python3:
    
    model = torchvision.models.resnet50(pretrained = True)
    
    This will load a pretrained resnet50 to your model, and the resnet50 will be automatically downloaded if have not been downloaded yet.
    
    



Results
-------------------------------------------------------------------------
Our model achieves the following performance on :

HW1 challenge on codalab	

![img1](https://github.com/egghead2630/VRDL_HW1/blob/main/results/result.png)



leaderboard:
https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07#results


Reproducing without retraining
-------------------------------------------------------------------------
Please refer to evaluation part, there is a detailed explaination.





Thanks for reading this README.
