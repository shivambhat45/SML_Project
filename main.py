import argparse
from datetime import datetime
from train import main
import os
import shutil
from subprocess import call
import pickle
import json
"""
C : Convolutional Layer syntax : ('C', out_channels, kernel_size, padding, )
M : MaxPooling Layer
fc : Fully Connected Layer

"""
archs = {
    'LeNet5_padding': [('C', 6, 5, 'same', 3),
                ('M',2,2),
                ('C', 16, 5, 'same'),
                ('M',2,2),
                ('fc' , 1024 , 120 , ),
                 ('fc' , 120 , 84),
                ('fc' , 84 , 10)] ,

    'VGG8': [('C', 128, 3, 'same', 3),
                ('M',2,2),
                ('C', 256, 3, 'same', 1.0),
                ('M',2,2),
                ('C', 512, 3, 'same', 1.0),
                ('M',2,2),
                ('fc' , 8192 , 1024),
                ('fc' , 1024 , 10)],
    'LeNet5_without_padding': [('C', 6, 5, 'not_same', 3),
                ('M',2,2),
                ('C', 16, 5, 'not_same'),
                ('M',2,2),
                ('fc' , 400 , 120 , ),
                 ('fc' , 120 , 84),
                ('fc' , 84 , 10)] ,
 
    'CNN8' : [('C', 16, 3, 'same', 3),
                ('M',2,2),
                ('C', 32, 3, 'same', 16),
                ('M',2,2),
                ('C', 64, 3, 'same', 32),
                ('M',2,2),
                ('fc' , 1024 , 256),
                ('fc' , 256 , 10)],
   
    'CNN10' : [('C', 32, 3, 'same', 3),
                ('M',2,2),
                ('C', 64, 3, 'same', 32),
                ('M',2,2),
                ('C', 128, 3, 'same', 64),
                ('M',2,2),
                ('C', 256, 3, 'same', 128),
                ('M',2,2),
                ('fc' , 1024 , 512),
                ('fc' , 512 , 10)],
   
    'CNN4' : [('C', 16, 3, 'same', 3),
                ('M',2,2),
                ('fc' , 4096 , 128),
                ('fc' , 128 , 10)],
 
    'CNN6' : [('C', 16, 3, 'same', 3),
                ('M',2,2),
                ('C', 32, 3, 'same', 16),
                ('M',2,2),
                ('fc' , 2048 , 128),
                ('fc' , 128 , 10)],
   
    'CNN8_4' : [('C', 16, 3, 'same', 3),
                ('M',2,2),
                ('C', 32, 3, 'same', 16),
                ('M',2,2),
                ('C', 64, 3, 'same', 32),
                ('M',2,2),
                ('fc' , 1024 , 512),
                ('fc' , 512 , 256),
                ('fc' , 256 , 128),
                ('fc' , 128 , 10)],
   
    'CNN6_6' : [('C', 16, 3, 'same', 3),
                ('M',2,2),
                ('C', 32, 3, 'same', 16),
                ('M',2,2),
                ('fc' , 2048 , 1024),
                ('fc' , 1024 , 512),
                ('fc' , 512 , 256),
                ('fc' , 256 , 128),
                ('fc' , 128 , 64),
                ('fc' , 64 , 10)],
   
     'CNN4_8' : [('C', 16, 3, 'same', 3),
                ('M',2,2),
                ('fc' , 4096 , 2048),
                ('fc' , 2048 , 1024),
                ('fc' , 1024 , 512),
                ('fc' , 512 , 256),
                ('fc' , 256 , 128),
                ('fc' , 128 , 64),
                ('fc' , 64 , 32),
                ('fc' , 32 , 10)],
    
}

import subprocess

def invoke_make(value):
    os.chdir('NeuroSIM')
    try:
        
        subprocess.run(['make'], check=True)
        print("Make command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing make command: {e}")
    os.chdir("../")



def parse_layers_from_arch(layers ):
    with open('config.json', 'r') as f:
        config = json.load(f)


    embedding_size = config["embedding_size"]
    layer2vec = [0] * embedding_size
    index = 0
    print(layers)
    for layer in layers : 
        if layer[0] == 'C' : 
             layer2vec[index] = layer[1]
             layer2vec[index+1] = layer[2]
             layer2vec[index+2] = config[layer[3]] * (layer[2]) // 2
             layer2vec[index+3] = config["end"]
             index += 4
        elif layer[0] == 'M' : 
             layer2vec[index] = layer[1]
             layer2vec[index+1] = layer[2]
             layer2vec[index+2] = config["end"]
             index += 3
        elif layer[0] == 'fc' : 
            layer2vec[index] = layer[1]
            layer2vec[index+1] = layer[2]
            layer2vec[index+2] = config["end"]
            index += 3
        else : 
            raise ValueError("Invalid layer type")
    return layer2vec
    

def make_network_file(arch):
    input_width = 32
    input_height = 32
    input_channel = 3
    network_file = []
    flag = 0
    for i , layer in enumerate(arch):
        if layer[0] == 'C':
            
            filters = layer[1]
            kernel_size = layer[2]
            network_file.append([input_height, input_width, input_channel, kernel_size, kernel_size, filters, flag , 1])
            padding =  layer[2]//2 if layer[3] == 'same' else 0
            input_width = (input_width - kernel_size + 2*padding) + 1
            input_height = (input_height - kernel_size + 2*padding)  + 1
            input_channel = filters
        elif layer[0] == 'M':
            kernel_size = layer[1]
            stride = layer[2]
            input_width = (input_width - kernel_size )//stride + 1
            input_height = (input_height - kernel_size ) // stride  + 1
        elif layer[0] == 'fc':
            network_file.append([1, 1, layer[1], 1, 1, layer[2], 0, 1])
        flag = 1
    file = []
    for net in network_file:
        file.append(','.join(map(str, net)))
    return file  , network_file 

if __name__ == '__main__':
    """
    This script is used for training a PyTorch model on the CIFAR-X dataset.
    It takes various command-line arguments to configure the training process.
    """

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
    parser.add_argument('--type', default='cifar10', help='dataset for training')
    parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 200)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 32)')
    parser.add_argument('--grad_scale', type=float, default=1, help='learning rate for wage delta calculation')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 117)')
    parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status default = 100')
    parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test (default = 1)')
    parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
    parser.add_argument('--decreasing_lr', default='200,250', help='decreasing strategy')
    parser.add_argument('--wl_weight', type = int, default=2)
    parser.add_argument('--wl_grad', type = int, default=8)
    parser.add_argument('--wl_activate', type = int, default=8)
    parser.add_argument('--wl_error', type = int, default=8)
    parser.add_argument('--inference', default=0)
    parser.add_argument('--onoffratio', default=10)
    parser.add_argument('--cellBit', default=1)
    parser.add_argument('--subArray', default=128)
    parser.add_argument('--ADCprecision', default=5)
    parser.add_argument('--vari', default=0)
    parser.add_argument('--t', default=0)
    parser.add_argument('--v', default=0)
    parser.add_argument('--detect', default=0)
    parser.add_argument('--target', default=0)
    parser.add_argument('--nonlinearityLTP', default=0.01)
    parser.add_argument('--nonlinearityLTD', default=-0.01)
    parser.add_argument('--max_level', default=100)
    parser.add_argument('--d2dVari', default=0)
    parser.add_argument('--c2cVari', default=0)
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    parser,current_time

    for (key, value) in archs.items():
        folder_name = f'Results/{key}/NeuroSIM'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else :
            shutil.rmtree(folder_name)
            os.makedirs(folder_name)

        src = 'NeuroSIM'
        trg = folder_name
        
        files=os.listdir(src)
        
        # 	int numMemInRow = (netStructure[maxIFMLayer][0]-netStructure[maxIFMLayer][3]+1)*(netStructure[maxIFMLayer][1]-netStructure[maxIFMLayer][4]+1);
		# int numMemInCol = netStructure[maxIFMLayer][2]*param->numBitInput;
  
#   numRowSubArrayWG = 128; // # of rows of single SRAM subArray in "gradient calculation of weight"
# 	numColSubArrayWG = 128;
        numMemInRow = 0
        numMemInCol = 0
        index  = 0
        maxi = 0
        netfile = make_network_file(value)[1]
        for i , item in enumerate(netfile):
            if(item[0] * item[1] * item[2] > maxi):
                maxi = item[0] * item[1] * item[2]
                index = i
        numMemInRow = (netfile[index][0] - netfile[index][3] + 1) * (netfile[index][1] - netfile[index][4] + 1)
        numMemInCol = netfile[index][2] * 8
        
        if(key == 'LeNet5_padding'):
            numRowSubArray = 32 
            numColSubArray = 32
        elif (key == 'LeNet5_without_padding'):
            numRowSubArray = 64 
            numColSubArray = 64 
        else :
             numRowSubArray = 16
             numColSubArray = 32 
            
        for fname in files:
            shutil.copy2(os.path.join(src,fname), trg)
            if(fname == 'Param.cpp'):
                 with open(os.path.join(src, fname), 'r') as file:
                    filedata = file.read()
                    filedata = filedata.replace('numRowSubArray = 32;', f'numRowSubArray = {numRowSubArray};')
                    filedata = filedata.replace('numColSubArray = 32;', f'numColSubArray = {numColSubArray};')
                    filedata = filedata.replace('numRowSubArrayWG = 128;', f'numRowSubArrayWG = {min(numMemInRow , 128)};')
                    filedata = filedata.replace('numColSubArrayWG = 128;', f'numColSubArrayWG = {min(128 , numMemInCol)};')
                    with open(os.path.join(trg, fname), 'w') as file:
                        file.write(filedata)      
                
        with open(f'Results/{key}/NeuroSIM/NetWork.csv', 'w') as f:
            for item in make_network_file(value)[0]:
                f.write("%s\n" % item) 
        folder2 = f'Results/{key}/'
        
        with open(f'Results/{key}/arch.pkl', 'wb') as f:
           pickle.dump(parse_layers_from_arch(value), f)
        os.chdir(folder2)
        # Call the function to invoke make
        invoke_make(key)

        print("=="*5 )
        print(key) 
        main(parser,current_time, value)
        print(key)
        print("\n\n"*5)
        os.chdir('../../')

 