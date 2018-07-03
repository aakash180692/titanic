# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:16:12 2018

@author: agupt489
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import datetime
from sklearn import *
from multiprocessing import *

test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')
gender_subm = pd.read_csv('gender_submission.csv')

train_data['Cabin_letter'] = train_data.Cabin.str[0:1]
test_data['Cabin_letter'] = test_data.Cabin.str[0:1]

train_data.shape, test_data.shape

msk = np.random.rand(len(train_data)) < 0.6
train = train_data[msk]
temp = train_data[~msk]

msk2 = np.random.rand(len(temp)) < 0.5
test = temp[msk2]
val = temp[~msk2]

X_train = train.drop(['PassengerId','Name', 'Ticket', 'Cabin','Survived'], axis=1)
y_train = train['Survived']
X_val = val.drop(['PassengerId','Name', 'Ticket', 'Cabin','Survived'], axis=1)
y_val = val['Survived']
X_test = test.drop(['PassengerId','Name', 'Ticket', 'Cabin','Survived'], axis=1)
y_test = test['Survived']

# Get indices of categorical variables
d_t = X_train.dtypes

list_cat = [x for x in d_t[d_t == object].axes[0]]

print(list_cat)

cat_indices = [X_train.columns.tolist().index(col) for col in list_cat]
print(cat_indices)

# Filling null values with 0

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
X_val = X_val.fillna(0)

def catboost_run(X_train,y_train,X_val,y_val,X_test,y_test,cat_indices,n_tr,rsm,lrn_rt,dep,l2_reg,num_split,cat_split,bag_temp):
    
    print("model train start")
    
    t1 = datetime.datetime.now()
    print(t1)
    
    model = CatBoostClassifier(iterations=n_tr,
                               rsm=rsm,
                               learning_rate=lrn_rt, 
                               depth=dep,
                               l2_leaf_reg=l2_reg,
                               bagging_temperature = bag_temp,
                               random_seed=2)
    
    model.fit(X_train, y_train,cat_indices, use_best_model=True, eval_set=(X_val, y_val), verbose=True)
    # Predicitng and calculating performance on test data
    predict_prob = model.predict_proba(X_test)[:,1]

    pred_list = [1 if i > 0.5 else 0 for i in predict_prob.tolist()]

    y_list = y_test.tolist()

    counter = 0
    counter_1 = 0
    for i in range(len(pred_list)):
        if pred_list[i] == y_list[i]:
            counter = counter+1
            if y_list[i] == 1:
                counter_1 = counter_1+1
                
    accuracy = counter/len(pred_list)
    
    return accuracy

# Lists of paramenter values

rsm_pv = [0.7,0.8,0.9]
print(len(rsm_pv))
lrn_rt_pv = [0.075,0.1,0.15]
print(len(lrn_rt_pv))
dep_pv = [6,7,8]
print(len(dep_pv))
l2_reg_pv = [10,15,50]
print(len(l2_reg_pv))

result_col_list = ['rsm',
                   'learning_rate',
                   'depth',
                   'l2_regularization',
                   'accuracy']

results_df = pd.DataFrame(data=None,columns=result_col_list)

t_a = datetime.datetime.now()

cntr_prog = 0
for rsm in rsm_pv:
    for lrn_rt in lrn_rt_pv:
        for dep in dep_pv:
            for l2_reg in l2_reg_pv:
                
                t1 = datetime.datetime.now()
                model = CatBoostClassifier(iterations=100,
                                           rsm=rsm,
                                           learning_rate=lrn_rt, 
                                           depth=dep,
                                           l2_leaf_reg=l2_reg,
                                           random_seed=2)
    
                model.fit(X_train, y_train,cat_indices, use_best_model=True, eval_set=(X_val, y_val),logging_level='Silent')
                # Predicitng and calculating performance on test data
                predict_prob = model.predict_proba(X_test)[:,1]

                pred_list = [1 if i > 0.5 else 0 for i in predict_prob.tolist()]

                y_list = y_test.tolist()

                counter = 0
                for i in range(len(pred_list)):
                    if pred_list[i] == y_list[i]:
                        counter = counter+1
                
                accuracy = counter/len(pred_list)
                
                result_df_temp = pd.DataFrame(data=None,columns=result_col_list)
                
                result_df_temp.loc[0,'rsm'] = rsm
                result_df_temp.loc[0,'learning_rate'] = lrn_rt
                result_df_temp.loc[0,'depth'] = dep
                result_df_temp.loc[0,'l2_regularization'] = l2_reg
                
                result_df_temp.loc[0,'accuracy'] = accuracy

                results_df = results_df.append(result_df_temp)
                
                t2 = datetime.datetime.now()
                
                itr_tm = t2-t1
                
                cntr_prog = cntr_prog+1
                print(str(itr_tm) + " -> " + str(cntr_prog) + "/" + str(3*3*3*3))
                
t_b = datetime.datetime.now()

serial_time = t_b - t_a
print("total time taken = " + str(serial_time))

results_df2 = results_df

results_df2 = results_df2.reset_index().drop(columns = ["index"])

final_result = results_df2[results_df2.accuracy == max(results_df2['accuracy'])].reset_index().drop(columns = ["index"]).head(1)

final_result

rsm_f = final_result['rsm'][0]
lrn_rt_f = final_result['learning_rate'][0]
dep_f = final_result['depth'][0]
l2_reg_f = final_result['l2_regularization'][0]

n_tree = 1000
print(str(n_tree) + " TREES")

model = CatBoostClassifier(iterations=n_tree,
                           rsm=rsm_f,
                           learning_rate=lrn_rt_f, 
                           depth=dep_f,
                           l2_leaf_reg=l2_reg_f,
                           random_seed=2)

t1 = datetime.datetime.now()
print(t1)
model.fit(X_train, y_train,cat_indices, use_best_model=True, eval_set=(X_val, y_val), verbose=True)

t2 = datetime.datetime.now()
print(t2)
print(t2-t1)

# Predicitng and calculating performance on test data
predict_prob = model.predict_proba(X_test)[:,1]

pred_list = [1 if i > 0.5 else 0 for i in predict_prob.tolist()]

y_list = y_test.tolist()

counter = 0
counter_1 = 0
for i in range(len(pred_list)):
    if pred_list[i] == y_list[i]:
        counter = counter+1
        if y_list[i] == 1:
            counter_1 = counter_1+1

print("Result total = "+str(counter/len(pred_list)))

print("Result survived = "+str(counter_1/sum(pred_list)))

print("Result dead = "+str((counter - counter_1)/(len(pred_list) - sum(pred_list))))