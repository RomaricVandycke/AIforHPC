# AIforHPC

BaseLine.py

The BaseLine implementation I used is a Deep Neural Network I customized from the website https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
The Neural network detect according to certain criteria foun in the first colum if one patient has diabetes. 0 mean no, 1 means yes

The Neural network has a lot of different layers with some dropout in order to enhance the time requiers to train the model. The Dataset only contain a bit less than 1k Lines which is small. 
In order to artificially increase the training time

Finally for the BaseLine I get after multiple training, a mean of 65sec. (100s in local on mu computer..)

To launch the BaseLine I simply activate the conda/pytorch environement..
source $STORE/mytorchdist/bin/activate
python BaseLine.py

DDPv3.py

The neural network consist of 1 input layer 2 hidden layer and 1 output layer. I use Relu function for the Hiden layers and sigmoide function for output layer.

Function Train_with_args initialize a model of the class PimaClassifier inside the DDP.
I Used Binary Cross entropy as Loss function and Adam as Optimizer (Commonly used in my degree..)

to launch the code just write:
sbatch DDPv3.sh



Of Course don't forget to change the path to the csv file in the first Lines of DDPv3.py
