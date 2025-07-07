import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import constants
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter
import scipy.io
import argparse

from utils_lab_6basicECMclassification import *

def parse_args():
    parser = argparse.ArgumentParser(description="Process circuit data and save to .mat file.")
    
    parser.add_argument('--data_ver', type=str, default="v2", help="Data version identifier.")
    # parser.add_argument('--circuit_model', type=int, default=6, help="Which circuit model to use (1-7).")
    parser.add_argument('--is_Test', type=int, default=0, help="Flag to indicate if the data is for testing.")
    
    return parser.parse_args()

if __name__ == "__main__":
    np.random.seed(0)
    args = parse_args()
    if args.is_Test == 1:
        np.random.seed(1)

    # For Classification C1-C6
    all_param=[]
    all_param.append(Circuit0_param.tolist())
    all_param.append(Circuit1_param.tolist())
    all_param.append(Circuit2_param.tolist())
    all_param.append(Circuit3_param.tolist())
    all_param.append(Circuit4_param.tolist())
    all_param.append(Circuit5_param.tolist())

    df = pd.DataFrame(all_param)
    df.transpose()
    k=df.stack()
    k.to_csv('paramc1-c6_gRange.csv', index=False)

    data_ver="v2"
    if size_number >= 1000: data_num_n = str("%.0f%s" % (size_number/1000.0, 'k'))
    else : data_num_n = str(size_number)

    File_name="xy_data_"+data_num_n+"_"+str(number_of_circuit)+"circuit_"+data_ver+".mat"
    if args.is_Test == 1:
        File_name="xy_data_"+data_num_n+"_"+str(number_of_circuit)+"circuit_"+data_ver+"_test_set.mat"


    x_data,y_data = export_data(Circuit_spec,size_number,number_of_point,numc=number_of_circuit)
    mdic={"x_data":x_data,"y_data":y_data}
    scipy.io.savemat(File_name, mdic)
    print("finished")
