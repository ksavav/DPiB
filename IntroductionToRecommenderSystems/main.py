import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statistics import mean 
from math import sqrt, pow

# const values 
path = 'E:/Desktop/PGa/Semestr_9/PDW/rekomendacejsracje/ml-100k/ml-100k/'
num_users = 943
num_items = 1682
neighbors = 3

#get dataset 
data = pd.read_csv(path + 'u.data', sep="\t", header=None)
data.columns = ['user id',  'item id', 'rating', 'timestamp']
data = data.drop(columns=['timestamp'])
data = data.pivot(index='user id', columns='item id', values='rating').fillna(0)

# Pearson correlation coefficient algorithm
def predict(data, user, movie, k):
    neighbors = []
    neighbors_full = []
    if user[movie] != 0:
        print('this user already seen this movie')
    else:
        for i in range(len(data)):
            if data[i][movie] != 0:
                sim = pearson_correlation_coeff(user, data[i])
                if sim != 0:
                    neighbors_full.append(data[i])
                    neighbors.append([i, sim, [value for value in data[i] if value != 0], data[i][movie]])

        neighbors.sort(key=lambda x: x[1], reverse=True)

        predicted_value = pred(user, [[item[1], item[2], item[3]] for item in neighbors[:k]], k, movie)

        #print(f'this user would rate movie {movie + 1}: {round(predicted_value, 3)}')

        return predicted_value

# sim from Pearson correlation coefficient algorithm
def pearson_correlation_coeff(x, y):
    x_list = []
    y_list = []

    nominator = 0
    denominatorx = 0
    denominatory = 0
    for i in range(len(x)):
        if y[i] !=0 and x[i] != 0:
            x_list.append(x[i])
            y_list.append(y[i])
    
    if len(y_list) < 2:
        return 0

    for i in range(len(x_list)):
        nominator += ((x_list[i] - mean(x_list)) * (y_list[i] - mean(y_list)))
        denominatorx += pow((x_list[i] - mean(x_list)), 2)
        denominatory += pow((y_list[i] - mean(y_list)), 2)

    try:
        pcc = nominator/(sqrt(denominatorx) * sqrt(denominatory))
    except ZeroDivisionError:
        return 0
    
    return pcc

# predicition from Pearson correlation coefficient algorithm
def pred(x, neighbors, k, movie):
    meter = 0
    denominator = 0
    for i in range(k):
        #print(neighbors[i][0])
        meter += neighbors[i][0] * (neighbors[i][2] - mean(neighbors[i][1])) #meter += neighbors[i][0] * (neighbors[i][1][movie] - mean(neighbors[i][1]))
        denominator += neighbors[i][0]

    x_non_zero = [value for value in x if value != 0]
    pred = mean(x_non_zero) + meter/denominator
    return pred

# implementation of naive algorithm
def naive_algorithm(train_list, movie):
    temp_list = []

    for i in range(len(train_list)):
        if train_list[i][movie] != 0:
            temp_list.append(train_list[i][movie])

    return np.mean(np.array(temp_list))

# MAE
def mae(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs(actual - predicted))

# RMSE
def rmse(actual, predicted):
    mse = np.square(np.subtract(actual, predicted)).mean()
    return sqrt(mse)  

# user = [3, 1, 4, 4, 0]
# test = [[4, 2, 5, 4, 5], 
#         [1, 5, 5, 4, 3], 
#         [5, 3, 4, 3, 4], 
#         [3, 4, 2, 1, 2]]

train, test = sklearn.model_selection.train_test_split(data, test_size=0.05)

train_list = train.values.tolist()
test_list = test.values.tolist()

actual_values = []
predicted_values = []
naive_values= []
temp_act = []
validation = []
chosen_movie = []
ks = [2, 3, 5, 10]

# to make sure, that for every k, test list is the same
for i in range(len(test_list)):
    for j in range(len(test_list[i])):
        if test_list[i][j] != 0:
            temp_act.append(test_list[i][j])
            test_list[i][j] = 0
            chosen_movie.append(j)
            break
    validation.append(test_list[i])    

# validate model with different k and naive algorithm for comparison
for k in ks:
    temp_pred = []
    temp_naive = []
    print(f'--------- k = {k} ---------')
    for i in range(len(test_list)):
        print(f'iteration: {i+1}')
        predict_val = predict(train_list, validation[i], chosen_movie[i], k)
        temp_naive.append(naive_algorithm(train_list, chosen_movie[i]))
        temp_pred.append(predict_val)
    actual_values.append([k, temp_act])
    predicted_values.append([k, temp_pred])
    naive_values.append(temp_naive)
    
# print first few predictions
for i in range(5):
    print(f'actual = {actual_values[3][1][i]}, predicted = {predicted_values[3][1][i]}')

print('\n---------------------- results ----------------------\n')

# print RMSE and MAE for 4 different ks and native algoithm
for i in range(len(ks)):
    print(f'for k = {actual_values[i][0]} RMSE = {rmse(actual_values[i][1], predicted_values[i][1])} MAE = {mae(actual_values[i][1], predicted_values[i][1])}')
print(f'\nnaive algorithm RMSE = {rmse(actual_values[i][1], naive_values[i])} MAE = {mae(actual_values[i][1], naive_values[i])}\n')    

# actual = 1.0, predicted = 1.4963812930244715
# actual = 3.0, predicted = 3.8384894318210216
# actual = 3.0, predicted = 3.3445732769379695
# actual = 4.0, predicted = 3.871396084416542
# actual = 5.0, predicted = 4.458746298543071
# 
# ---------------------- results ----------------------
# 
# for k = 2 RMSE = 1.383602465309964 MAE = 1.124602752383544
# for k = 3 RMSE = 1.1545948173989025 MAE = 0.9129103731574363
# for k = 5 RMSE = 1.0855062671547442 MAE = 0.8843795338060437
# for k = 10 RMSE = 1.0316068552576314 MAE = 0.8460636337548998
# 
# naive algorithm RMSE = 1.0002526514724541 MAE = 0.7732615574274057
