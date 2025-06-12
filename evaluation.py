import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.io
from tensorflow import keras
import tensorflow as tf
import datetime
from numpy import unique
from numpy import argmax
from pandas import read_csv

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

##### Evaluation #####
# Load EIS data-set
filename="EIS_131k_regC9_v2_model_G1_RegC9_1024/G1_xy_data_33k_regC9_v2_test.mat"

x=scipy.io.loadmat(filename)["x_data"]
y=scipy.io.loadmat(filename)["y_data"]
y=np.squeeze(y)
x=np.swapaxes(x, 1, 2)


new_shape=x.shape
new_shape=np. asarray(new_shape)
new_shape[-1]=new_shape[-1]+3
new_shape=tuple(new_shape)
new_x = np.zeros(new_shape)
new_x[:, :, :3] = x


y[:,0]=y[:,0]*10**3 # R0
y[:,1]=y[:,1]*10**3 # R1
y[:,2]=y[:,2]*10**3 # R2
y[:,3]=y[:,3]*10**3 # R3
y[:,4]=y[:,4]*10**3 # C1
y[:,5]=y[:,5]*10**3 # C2
y[:,6]=y[:,6]*10**3 # C3
y[:,7]=y[:,7]*10**3 # sigma


new_x[:,:,3]=x[:,:,0]*-1
new_x[:,:,4]=x[:,:,1]*-1
new_x[:,:,5]=x[:,:,2]*-1


#split data
x_train, x_test, y_train, y_test = train_test_split(new_x, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

#Load Model
model_to_load="EIS_131k_regC9_v2_model_G1_RegC9_1024/G1_RegC9_alpha_BN.h5"
predict_model = tf.keras.models.load_model(model_to_load)


y_pred=predict_model.predict(x_test)
y_pred=np.asarray(y_pred)



df_true = pd.DataFrame(y_test/ 1e3, columns=[f"true_{i}" for i in range(8)])
df_pred = pd.DataFrame(y_pred/ 1e3, columns=[f"pred_{i}" for i in range(8)])


df_combined = pd.concat([df_true, df_pred], axis=1)

# Step 4: Export to CSV
df_combined.to_csv("EIS_131k_regC9_v2_model_G1_RegC9_1024/simulation_params_true_vs_pred.csv", index=False)