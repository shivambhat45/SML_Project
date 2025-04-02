

import pandas as pd

# fead all the files in the folder results
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pickle
import json

# open config

# open the file
with open("config.json", "r") as file:
    config = json.load(file)


folders = os.listdir("Results")

for folder in folders:
    path = "Results/"+folder+"/NeuroSim_Results_Each_Epoch"
    files = os.listdir(path)
    if(len(files) == 0):
        continue
    with open("Results/" + folder + "/arch.pkl", 'rb') as file:
        input = pickle.load(file)
    
    
    # create input to numpy array
    input = np.array(input)
    
    with open(path +"/NeuroSim_Breakdown_Epock_0.csv", 'r') as file:
        csv_content = file.read()
        delimiter = "\n\n"  # Replace this with the appropriate delimiter/pattern
        csv_file = StringIO(csv_content.split(delimiter)[1])
        df = pd.read_csv(csv_file)
        latency = (df[' latency_FW(s)'])[:-1]
        energy = (df[' energy_FW(J)'])[:-1]
        
        
        latency = np.array(latency)
        energy = np.array(energy)
        if len(latency) < config["output_size"]:
            # pad with zeroes
            latency = np.pad(latency, (0, config["output_size"]-len(latency)), 'constant')
            energy = np.pad(energy, (0, config["output_size"]-len(energy)), 'constant')
        if(latency.dtype == np.object or energy.dtype == np.object):
            continue
        if not os.path.exists("./dataset.pkl"):
            with open ("./dataset.pkl" , "wb") as file:
                pickle.dump([(input , (latency , energy))] , file)
        else : 
            with open ("./dataset.pkl" , "rb") as file:
                
                existing_data = pickle.load(file)
            with open ("./dataset.pkl" , "wb") as file:
                existing_data.append((input , (latency , energy)))
                pickle.dump(existing_data , file)
            