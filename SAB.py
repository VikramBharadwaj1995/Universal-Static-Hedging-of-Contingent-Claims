import numpy as np
import math
from scipy.stats import norm
import keras
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers as opt
import sobol_seq
import time
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
import sys
import gc
import os
from keras.models import load_model
from keras.models import save_model


# Setting Initial Parameters
no_of_paths = int(sys.argv[1])
no_of_hidden_nodes = int(sys.argv[2])
value = float(sys.argv[3])
no_of_epochs = 100
lr = 0.0005
no_of_assets = 1
cor_mat = [[1]]
vol_list = np.array([0.2])
curr_stock_price = np.ones(no_of_assets) * value
t = 1
k = 1
no_of_exercise_days = 10
r = 0.06
w = np.array([1])
w = w.reshape(-1, 1)
no_of_output_nodes = 1
batch_size = int(no_of_paths / 10)
time_taken = timelb_taken = 0


# Defining Dicts and Lists
w_dict = {}
num_cores = 32
weights_dict = {}
price_list = []
pricelb_list = []
time_list = []
timelb_list = []
ub_list = []
dt = float(t / no_of_exercise_days)


# CPU and GPU configuration to run on Tensorflow-GPU
GPU = 1
CPU = not GPU
if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True, device_count={'CPU': num_CPU, 'GPU': num_GPU})
session = tf.Session(config=config)
K.set_session(session)
es = EarlyStopping(monitor='val_loss', mode='min', patience=20)


# Generates a Sobol sequence of quasi-random low-discrepancy sequences
def i4_sobol_generate_std_normal(dim_num, n, skip=1):
    """
    Generates multivariate standard normal quasi-random variables.
    Parameters:
      Input, integer dim_num, the spatial dimension.
      Input, integer n, the number of points to generate.
      Input, integer SKIP, the number of initial points to skip.
      Output, real np array of shape (n, dim_num).
    """
    sobols = sobol_seq.i4_sobol_generate(dim_num, n, skip)
    normals = norm.ppf(sobols)
    return normals


# Generate Covariance Matrix
def generate_covariance_from_correlation(cor_mat, vol_list, dt):
    vol_diag_mat = np.diag(vol_list)
    cov_mat = np.dot(np.dot(vol_diag_mat, cor_mat), vol_diag_mat) * dt
    return cov_mat


# Simulate the Stock Matrix using GBM
def multi_variate_gbm_simulation(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets,
                                 curr_stock_price, r, vol_list, cov_mat, t):
    zero_mean = np.zeros(no_of_assets)
    dw_matx = np.random.multivariate_normal(zero_mean, cov_mat, (int(no_of_paths / 2), no_of_exercise_days))
    dw_mat = np.concatenate((dw_matx, -dw_matx), axis=0)
    dw_mat = dw_mat.reshape(no_of_paths, no_of_exercise_days)
    sim_ln_stock_mat = np.zeros((no_of_paths, no_of_exercise_days + 1))
    sim_ln_stock_mat[:, 0] = np.log(curr_stock_price)
    base_drift = (r - 0.5 * np.square(vol_list[0])) * dt

    for day in range(1, no_of_exercise_days + 1):
        curr_drift = sim_ln_stock_mat[:, day - 1] + base_drift
        sim_ln_stock_mat[:, day] = curr_drift + dw_mat[:, day - 1]

    sim_stock_mat = np.exp(sim_ln_stock_mat)
    return sim_stock_mat


# Computes the Price
def pricer_bermudan_options_by_nn(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets, w, sim_stock_mat,
                                  batch_size, no_of_epochs, no_of_hidden_nodes, no_of_output_nodes, t, nnet_model):
    continuation_value = np.zeros((no_of_paths, 1))
    # Finding intrinsic value of the option for all paths and exercise days
    dt = t / no_of_exercise_days
    es = EarlyStopping(monitor='val_loss', mode='min', patience=6)
    for day in range(no_of_exercise_days - 1, -1, -1):
        # Neural Network Function using RELU activaion function
        # Build Neural Network model(using Stock prices at next exercise period and their option value)
        stock_vec = sim_stock_mat[:, day + 1]

        intrinsic_value = np.maximum(k - stock_vec, 0)
        intrinsic_value = np.asarray(intrinsic_value)
        intrinsic_value = intrinsic_value.reshape(-1, 1)

        option_value = np.maximum(intrinsic_value, continuation_value)

        X_train = np.log(stock_vec)
        X_train = X_train.reshape(-1, no_of_assets)

        Y_train = option_value
        Y_train = np.asarray(Y_train)
        Y_train.reshape(-1, 1, 1)

        nnet_model.fit(X_train, Y_train, epochs=no_of_epochs, batch_size=batch_size, verbose=0,
                                     validation_split=0.3, callbacks=[es])

        w1 = nnet_model.layers[0].get_weights()
        w2 = nnet_model.layers[1].get_weights()
        nnet_model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mean_squared_error')
        nnet_model.layers[0].set_weights(w1)
        nnet_model.layers[1].set_weights(w2)

        w_dict[day] = ([w1, w2])
        w_vect = np.array(nnet_model.layers[0].get_weights()[0]).reshape(-1)
        w_vect_2 = np.array(nnet_model.layers[1].get_weights()[0]).reshape(-1)
        strikes = np.array(nnet_model.layers[0].get_weights()[1]).reshape(-1)
        bias_2 = np.array(nnet_model.layers[1].get_weights()[1]).reshape(-1)
        strikes = np.asarray(strikes)
        weights_dict[day] = ([nnet_model.layers[0].get_weights()[0], nnet_model.layers[1].get_weights()[0],
                              nnet_model.layers[0].get_weights()[1], nnet_model.layers[1].get_weights()[1]])

        stock_vec = sim_stock_mat[:, day]
        x = np.log(stock_vec) + ((r - 0.5 * np.square(vol_list[0])) * dt)
        opt_val = np.zeros((no_of_paths, 1))

        for node in range(0, no_of_hidden_nodes):
            w_o = w_vect[node]
            mu = x * w_o + strikes[node]
            var = (w_o * cov_mat[0] * w_o)
            sd = var ** 0.5
            ft = mu * (1 - norm(0, sd).cdf(-mu))
            st = (sd / (2 * math.pi) ** 0.5) * np.exp(-0.5 * (mu / sd) ** 2)
            opt_val[:, 0] = opt_val[:, 0] + w_vect_2[node] * (ft + st)

        continuation_value = (opt_val + bias_2) * np.exp(-r * dt)

    return np.mean(continuation_value)


# Pre-training the Neural Network model
def pricer_bermudan_options_by_nn_pre(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets, w, sim_stock_mat,
                                      batch_size, no_of_epochs, no_of_hidden_nodes, no_of_output_nodes, t, nnet_model):
    continuation_value = np.zeros((no_of_paths, 1))

    for day in range(no_of_exercise_days - 1, no_of_exercise_days - 2, -1):
        stock_vec = sim_stock_mat[:, day + 1]

        intrinsic_value = np.maximum(k - stock_vec, 0)
        intrinsic_value = np.asarray(intrinsic_value)
        intrinsic_value = intrinsic_value.reshape(-1, 1)

        option_value = np.maximum(intrinsic_value, continuation_value)

        X_train = np.log(stock_vec)
        X_train = X_train.reshape(-1, no_of_assets)

        Y_train = option_value
        Y_train = np.asarray(Y_train)
        Y_train.reshape(-1, 1, 1)

        nnet_model.fit(X_train, Y_train, epochs=no_of_epochs, batch_size=batch_size, verbose=0,
                                     validation_split=0.3, callbacks=[es])

    return nnet_model


# Computes both the Upper Bound and the Lower Bound
def pricer_bermudan_options_by_nn_lb(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets, w, sim_stock_mat,
                                     batch_size, no_of_epochs, no_of_hidden_nodes, no_of_output_nodes, t, nnet_model):
    continuation_value = np.zeros((no_of_paths, 1))
    # es = EarlyStopping(monitor='val_loss', mode='min', patience=9)
    ExerciseTime = np.ones((no_of_paths, 1)) * t
    Cashflow = np.zeros((no_of_paths, 1))
    Delta_M = np.zeros((no_of_paths, no_of_exercise_days + 1))
    IV = np.zeros((no_of_paths, no_of_exercise_days + 1))
    
        
    for day in range(no_of_exercise_days - 1, -1, -1):
        stock_vec = sim_stock_mat[:, day + 1]
        stock_vec = stock_vec.reshape(-1, 1)
        intrinsic_value = np.maximum(k - stock_vec, 0)
        X_train = np.log(stock_vec)
        nnet_model.layers[0].set_weights(w_dict[day][0])
        nnet_model.layers[1].set_weights(w_dict[day][1])
        V_t = nnet_model.predict(X_train)
        ExerciseTime[intrinsic_value > continuation_value] = (day + 1) * dt
        Cashflow[intrinsic_value > continuation_value] = intrinsic_value[intrinsic_value > continuation_value]

        IV[:, day + 1] = (np.exp(-r * (day + 1) * dt)) * intrinsic_value.reshape(-1, )
        w_vect = weights_dict[day][0]
        w_vect_2 = weights_dict[day][1]
        strikes = weights_dict[day][2]
        bias_2 = weights_dict[day][3]

        stock_vec = sim_stock_mat[:, day]
        x = np.log(stock_vec) + ((r - 0.5 * np.square(vol_list[0])) * dt)
        opt_val = np.zeros((no_of_paths, 1))

        for node in range(0, no_of_hidden_nodes):
            w_o = w_vect[0, node]
            mu = x * w_o + strikes[node]
            var = (w_o * cov_mat[0] * w_o)
            sd = var ** 0.5
            ft = mu * (1 - norm(0, sd).cdf(-mu))
            st = (sd / (2 * math.pi) ** 0.5) * np.exp(-0.5 * (mu / sd) ** 2)
            opt_val[:, 0] = opt_val[:, 0] + w_vect_2[node] * (ft + st)

        continuation_value = (opt_val + bias_2) * np.exp(-r * dt)

        V_t = V_t.reshape(-1, 1)
        continuation_value = continuation_value.reshape(-1, 1)
        Delta_M[:, day + 1] = (
                (np.exp(-r * (day + 1) * dt)) * V_t - (np.exp(-r * (day) * dt)) * continuation_value).reshape(-1, )

    IV[:, 0] = np.maximum(k - sim_stock_mat[:, 0], 0).reshape(-1, )
    Martingale = np.cumsum(Delta_M, axis=1)
    FM = IV - Martingale
    ub = np.mean(np.maximum(np.max(FM, axis=1), 0))
    lb = np.mean(Cashflow * (np.exp(-r * ExerciseTime)))
    return lb, ub


exercise_days = np.array([float(i / no_of_exercise_days) for i in range(1, no_of_exercise_days + 1)])
dt = t / no_of_exercise_days

# Generate covariance matrix without considering dt
cov_mat = generate_covariance_from_correlation(cor_mat, vol_list, dt)

# Updating current stock price as first column
sim_stock_mat = multi_variate_gbm_simulation(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets,
                                             curr_stock_price, r, vol_list, cov_mat, t)

# Define the Neural Network model and set the weights from the pre-trained network
nnet_model = Sequential()
nnet_model.add(Dense(no_of_hidden_nodes, activation='relu',
                     kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)))
nnet_model.add(Dense(1, activation='linear',
                     kernel_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None)))
nnet_model.compile(optimizer=opt.Adam(lr=0.001), loss='mean_squared_error')


# Initial learning rate list. The for loop will use the learning rates in the list and train the model.
# The weights will be set again once the training is complete.
lr_list = [0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004,0.0003, 0.0002, 0.0001, 0.0001]*2
for i in range(10):
    nnet_model = pricer_bermudan_options_by_nn_pre(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets, w, sim_stock_mat,
                                      batch_size, no_of_epochs * 5, no_of_hidden_nodes, no_of_output_nodes, t, nnet_model)
    w1 = nnet_model.layers[0].get_weights()
    w2 = nnet_model.layers[1].get_weights()
    nnet_model.compile(optimizer=keras.optimizers.Adam(lr=lr_list[i]), loss='mean_squared_error')
    nnet_model.layers[0].set_weights(w1)
    nnet_model.layers[1].set_weights(w2)


w1_pre = nnet_model.layers[0].get_weights()
w2_pre = nnet_model.layers[1].get_weights()
nnet_model.compile(optimizer=keras.optimizers.Adam(lr), loss='mean_squared_error')
nnet_model.layers[0].set_weights(w1_pre)
nnet_model.layers[1].set_weights(w2_pre)

# Compute Price, Price_Lower_Bound, Price_Upper_Bound for 30 runs. Draw the mean and standard deviation for the same.
# The model will be deleted by clear_session(). The model that was saved in the previous iteration will be used again.
for i in range(30):
    
    start = time.time()
    sim_stock_mat = multi_variate_gbm_simulation(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets,
                                                 curr_stock_price, r, vol_list, cov_mat, t)

    price = pricer_bermudan_options_by_nn(no_of_paths, no_of_exercise_days, exercise_days,
                                          no_of_assets, w, sim_stock_mat,
                                          batch_size, no_of_epochs,
                                          no_of_hidden_nodes, no_of_output_nodes, t, nnet_model)
    end = time.time()
    start_lb = time.time()
    sim_stock_mat = multi_variate_gbm_simulation(200000, no_of_exercise_days, exercise_days, no_of_assets,
                                                 curr_stock_price, r, vol_list, cov_mat, t)
    price_lb, price_ub = pricer_bermudan_options_by_nn_lb(200000, no_of_exercise_days, exercise_days, no_of_assets, w,
                                                          sim_stock_mat,
                                                          batch_size, no_of_epochs, no_of_hidden_nodes,
                                                          no_of_output_nodes, t, nnet_model)
    end_lb = time.time()
    nnet_model.save('model.h5')
    del nnet_model
    K.clear_session()
    nnet_model = load_model('model.h5')
    nnet_model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mean_squared_error')
    nnet_model.layers[0].set_weights(w1_pre)
    nnet_model.layers[1].set_weights(w2_pre)
    price_list.append(price * 40)
    pricelb_list.append(price_lb * 40)
    ub_list.append(price_ub * 40)
    time_list.append(end-start)
    timelb_list.append(end_lb-start_lb)
    print("Price - ", price*40, "Price_LB - ", price_lb*40, "Price_UB - ", price_ub*40)
    print("Non-LB - ",end-start)
    print("LB - ", end_lb-start_lb)

print("30 iterations: ", no_of_hidden_nodes, ";", value)
print("Price List:")
print(price_list)
print("Price LB List:")
print(pricelb_list)
print("Price UB List:")
print(ub_list)
print("*********************************************************")
print("*********************************************************")

for tx in time_list:
    time_taken += tx

for txl in timelb_list:
    timelb_taken += txl

text_file = open("New_Output_SAB.txt", "a")
strx = "\n" + str(no_of_hidden_nodes) + ", " + str(np.mean(price_list)) + ", " + str(np.std(price_list)) + ", "\
       + str(np.mean(pricelb_list)) + ", " + str(np.std(pricelb_list)) + ", " + str(np.mean(ub_list)) + ", " +\
       str(np.std(ub_list)) + ", " + str(time_taken) + ", " + str(timelb_taken)
text_file.write(strx)
gc.collect()
