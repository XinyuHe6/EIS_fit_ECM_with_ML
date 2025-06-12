import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import constants
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter
import scipy.io
import argparse

from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Process circuit data and save to .mat file.")
    
    parser.add_argument('--data_ver', type=str, default="v2", help="Data version identifier.")
    parser.add_argument('--circuit_model', type=int, default=6, help="Which circuit model to use (1-7).")
    parser.add_argument('--is_Test', type=int, default=0, help="Flag to indicate if the data is for testing.")
    
    return parser.parse_args()

if __name__ == "__main__":
    np.random.seed(0)
    # np.random.seed(1)

    # For Classification C1-C7
    # all_param=[]
    # all_param.append(Circuit0_param.tolist())
    # all_param.append(Circuit1_param.tolist())
    # all_param.append(Circuit2_param.tolist())
    # all_param.append(Circuit3_param.tolist())
    # all_param.append(Circuit4_param.tolist())
    # all_param.append(Circuit5_param.tolist())
    # all_param.append(Circuit6_param.tolist())

    # df = pd.DataFrame(all_param)
    # df.transpose()
    # k=df.stack()
    # k.to_csv('paramc1-c7_gRange.csv', index=False)

    # data_ver="v2"
    # if size_number >= 1000: data_num_n = str("%.0f%s" % (size_number/1000.0, 'k'))
    # else : data_num_n = str(size_number)

    # File_name="xy_data_"+data_num_n+"_"+str(number_of_circuit)+"circuit_"+data_ver+".mat"
    # # File_name="xy_data_"+data_num_n+"_"+str(number_of_circuit)+"circuit_"+data_ver+"_test_set.mat"


    # x_data,y_data = export_data(Circuit_spec,size_number,number_of_point,numc=number_of_circuit)
    # mdic={"x_data":x_data,"y_data":y_data}
    # scipy.io.savemat(File_name, mdic)
    # print("finished")



    

    # # For Regression Ci
    # data_ver = "v2"
    # circuit_param = [Circuit0_param, Circuit1_param, Circuit2_param, Circuit3_param, Circuit4_param]
    # for i in range(5):
    #     all_param = []
    #     all_param.append(circuit_param[i].tolist())
    #     df = pd.DataFrame(all_param)
    #     df.transpose()
    #     k = df.stack()
    #     k.to_csv('paramc' + str(i+1) + '.csv', index=False)

    #     if size_number >= 1000: data_num_n = str("%.0f%s" % (size_number/1000.0, 'k'))
    #     else : data_num_n = str(size_number)

    #     File_name="xy_data_"+data_num_n+"_regC"+str(i+1)+"_"+data_ver+".mat"

    #     x_data,_ = arrange_data(Circuit_spec[i],(i),size_number, number_of_point)
    #     y_data = circuit_param[i]
    #     print(y_data.shape)
    #     mdic={"x_data":x_data,"y_data":y_data}
    #     scipy.io.savemat(File_name, mdic)
        
    #     # Split data into train and test sets
    #     test_size_number = int(0.2 * size_number)
    #     if test_size_number >= 1000: test_data_num_n = str("%.0f%s" % (test_size_number/1000.0, 'k'))
    #     else : test_data_num_n = str(test_size_number)

    #     test_file_name = "xy_data_"+test_data_num_n+"_regC"+str(i+1)+"_"+data_ver+"_test.mat"
    #     x_test_data = x_data[-test_size_number:]
    #     y_test_data = y_data[-test_size_number:]
    #     mdic={"x_data":x_test_data,"y_data":y_test_data}
    #     scipy.io.savemat(test_file_name, mdic)


    # # For Regression C6
    # data_ver = "v2"
    # # circuit_param = [Circuit0_param, Circuit1_param, Circuit2_param, Circuit3_param, Circuit4_param]
    
    # all_param = []
    # all_param.append(Circuit5_param.tolist())
    # df = pd.DataFrame(all_param)
    # df.transpose()
    # k = df.stack()
    # k.to_csv('paramc6.csv', index=False)

    # if size_number >= 1000: data_num_n = str("%.0f%s" % (size_number/1000.0, 'k'))
    # else : data_num_n = str(size_number)

    # File_name="xy_data_"+data_num_n+"_regC6_"+data_ver+".mat"

    # x_data,_ = arrange_data(Circuit_spec[5],5,size_number, number_of_point)
    # y_data = Circuit5_param
    # print(y_data.shape)
    # mdic={"x_data":x_data,"y_data":y_data}
    # scipy.io.savemat(File_name, mdic)
    
    # # Split data into train and test sets
    # test_size_number = int(0.2 * size_number)
    # if test_size_number >= 1000: test_data_num_n = str("%.0f%s" % (test_size_number/1000.0, 'k'))
    # else : test_data_num_n = str(test_size_number)

    # test_file_name = "xy_data_"+test_data_num_n+"_regC6_"+data_ver+"_test.mat"
    # x_test_data = x_data[-test_size_number:]
    # y_test_data = y_data[-test_size_number:]
    # mdic={"x_data":x_test_data,"y_data":y_test_data}
    # scipy.io.savemat(test_file_name, mdic)

    # # For Regression C7
    # data_ver = "v2"
    
    # all_param = []
    # all_param.append(Circuit5_param.tolist()) #X
    # df = pd.DataFrame(all_param)
    # df.transpose()
    # k = df.stack()
    # k.to_csv('paramc7.csv', index=False)

    # if size_number >= 1000: data_num_n = str("%.0f%s" % (size_number/1000.0, 'k'))
    # else : data_num_n = str(size_number)

    # File_name="xy_data_"+data_num_n+"_regC7_"+data_ver+".mat"

    # x_data,_ = arrange_data(Circuit_spec[6],6,size_number, number_of_point)
    # y_data = Circuit6_param
    # print(y_data.shape)
    # mdic={"x_data":x_data,"y_data":y_data}
    # scipy.io.savemat(File_name, mdic)
    
    # # Split data into train and test sets
    # test_size_number = int(0.2 * size_number)
    # if test_size_number >= 1000: test_data_num_n = str("%.0f%s" % (test_size_number/1000.0, 'k'))
    # else : test_data_num_n = str(test_size_number)

    # test_file_name = "xy_data_"+test_data_num_n+"_regC7_"+data_ver+"_test.mat"
    # x_test_data = x_data[-test_size_number:]
    # y_test_data = y_data[-test_size_number:]
    # mdic={"x_data":x_test_data,"y_data":y_test_data}
    # scipy.io.savemat(test_file_name, mdic)


    # For Regression Ci New (using another seed to generate test data)
    np.random.seed(0)
    args = parse_args()
    if args.is_Test:
        np.random.seed(1)
    circuit_idx = args.circuit_model - 1  # Convert to zero-based index
    curcuit_label = args.circuit_model

    params = [Circuit0_param, Circuit1_param, Circuit2_param, Circuit3_param, Circuit4_param, Circuit5_param, Circuit6_param, Circuit7_param, Circuit8_param]
    param_csvs = ['G1_paramc1.csv', 'G1_paramc2.csv', 'G1_paramc3.csv', 'G1_paramc4.csv', 'G1_paramc5.csv', 'G1_paramc6.csv', 'G1_paramc7.csv', 'G1_paramc8.csv', 'G1_paramc9.csv']
    if args.is_Test:
        param_csvs = [csv.replace('.csv', '_test.csv') for csv in param_csvs]
    file_names = ["_regC1_", "_regC2_", "_regC3_", "_regC4_", "_regC5_", "_regC6_", "_regC7_", "_regC8_", "_regC9_"]


    data_ver = args.data_ver
    
    all_param = []
    all_param.append(params[circuit_idx].tolist())
    df = pd.DataFrame(all_param)
    df.transpose()
    k = df.stack()
    k.to_csv(param_csvs[circuit_idx], index=False)

    if size_number >= 1000: data_num_n = str("%.0f%s" % (size_number/1000.0, 'k'))
    else : data_num_n = str(size_number)

    File_name="G1_xy_data_"+data_num_n+file_names[circuit_idx]+data_ver+".mat"
    if args.is_Test:
        File_name="G1_xy_data_"+data_num_n+file_names[circuit_idx]+data_ver+"_test.mat"


    x_data,_ = arrange_data(Circuit_spec[circuit_idx],circuit_idx,size_number, number_of_point)
    y_data = params[circuit_idx]
    print(y_data.shape)
    mdic={"x_data":x_data,"y_data":y_data}
    scipy.io.savemat(File_name, mdic)

    # save freq
    np.savetxt("angular_freq.csv", angular_frequency, delimiter=",", fmt="%s") 
    


        
    
