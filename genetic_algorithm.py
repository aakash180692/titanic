# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:31:50 2018

@author: agupt489
"""

import math as math
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import datetime
import random as random
from copy import deepcopy

# data preparation
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


# function which gives the number of bits required for representing a parameter in the individual genome
def num_trait_bin_len(min_val,max_val,step):
    num_val = math.ceil((max_val-min_val)/step + 1)
#    print(num_val)
    num_val_bin = str(bin(num_val))[2:]
#    print(num_val_bin)
    num_val_bin_len = len(num_val_bin)
    return num_val_bin_len

# testing trait_bin_len function
#print(num_trait_bin_len(0.01,0.55,0.02))

# function for converting a binary string value from a given individual into real value
def bin2num(min_val,max_val,decimal_precision,bin_str):
    Nbit = len(bin_str)
    Dval = int(bin_str,2)
    
    real_val = round(min_val + ((max_val-min_val)/((2**Nbit)-1))*Dval,decimal_precision)
    return real_val

# population initialization function
def dna_initialization(param_df):
    dna_str = ""
    for i in range(param_df.shape[0]):
        dna_str = dna_str+bin(np.random.randint(2**param_df.loc[i]['nBits']))[2:].zfill(int(param_df.loc[i]['nBits']))
    return dna_str

# function for roulette wheel selection
def roullete_selection(pop_df, n_select):
    pop_df2 = deepcopy(pop_df)
    pop_df2 = pop_df2.sort_values(by='fitness', axis=0, ascending=False)    
    pop_df2['fit_cum'] = pop_df2['fitness'].cumsum()    
    fit_sum = pop_df2['fitness'].sum()    
    return_selection = pd.DataFrame(columns=pop_df.columns.tolist())
    
    for i in range(n_select):
        rand_num = random.uniform(0, fit_sum)
        selection = pop_df2[pop_df2['fit_cum'] >= rand_num].head(1)
        return_selection = return_selection.append(selection[return_selection.columns.tolist()])
    
    return return_selection

# function for ranking selection
def ranking_selection(pop_df, n_select):
    pop_df2 = deepcopy(pop_df)
    pop_df2['fitness_2'] = 1/pop_df2['fitness'].rank(ascending=False)
    pop_df2['fitness'] = pop_df2['fitness_2']
    
    return_selection = roullete_selection(pop_df2[pop_df.columns.tolist()],n_select)
    
    return return_selection

# function for tournament selection
def tournament_selection(pop_df, n_select, tournament_size):
    pop_df2 = deepcopy(pop_df)
    return_selection = pd.DataFrame(columns=pop_df.columns.tolist())
    for i in range(n_select):
        tournament_pool = pop_df2.sample(n=tournament_size)
        winner = tournament_pool.sort_values(by='fitness', axis=0, ascending=False).head(1) 
        return_selection = return_selection.append(winner)
    
    return return_selection

# random bit mutation function
def mutation(dna,p_mut):
    dna2 = ''
    for i in range(len(dna)):
        if random.random() < p_mut:
            if dna[i] == '1':
                dna2 = dna2 + '0'
            else:
                dna2 = dna2 + '1'
        else:
            dna2 = dna2 + dna[i]
    return dna

# function for uniform crossover
def uniform_crossover(par1,par2):
    child = ''
    for i in range(len(par1)):
        if random.random() < 0.5:
            child = child + par1[i]
        else:
            child = child + par2[i]
    return mutation(child,p_mut)
        
def fit_prop_crossover(par1,fit1,par2,fit2):
    prob_par1 = fit1/(fit1+fit2)
    child = ''
    for i in range(len(par1)):
        if random.random() < prob_par1:
            child = child + par1[i]
        else:
            child = child + par2[i]
    return mutation(child,p_mut)

# GA parameters
n_pop = 100             # population size
n_gen = 100              # number of generations
p_mut = 0.05           # mutation probability
p_cross = 0.5          # crossover probability
child_ratio = 0.5      # share of children to be included in each generation
tournament_size = 4    # size pool for tournament selection, needed only in case of tournament selection

# catboost parameter
n_tree = 100

# list of parameters in format [parameter name, minimum value, maximum value, step size]
num_param_list = [['rsm',0.5,1,0.05,2],
                  ['depth',4,10,1,0],
                  ['learning_rate',0.01,0.5,0.01,2],
                  ['l2_leaf_reg',1,100,1,0]]

param_df = pd.DataFrame(num_param_list, 
                        columns=['parameter_name', 
                                 'min_val',
                                 'max_val',
                                 'step_size',
                                 'decimal_precision'])

for i in range(param_df.shape[0]):
    nBits = num_trait_bin_len(param_df.loc[i]['min_val'],
                              param_df.loc[i]['max_val'],
                              param_df.loc[i]['step_size'])
    param_df.loc[i,'nBits'] = nBits

dna_len = int(param_df['nBits'].sum())

# result collection df
hyper_param_list = list(param_df['parameter_name'])
result_col_list = ['dna_str'] + hyper_param_list + ['accuracy']

#results_df = pd.DataFrame(data=None,columns=result_col_list)

t_a = datetime.datetime.now()

def ga_catboost(pop_list,individual,gen):  
    
    t1 = datetime.datetime.now()
    dna = pop_list[individual]
    
    hyper_param_str = ''
    bit_cntr = 0
    
    res_dict = {"dna":dna}
    
    for i in range(param_df.shape[0]):
        parameter = param_df.loc[i]['parameter_name']
        param_min = param_df.loc[i]['min_val']
        param_max = param_df.loc[i]['max_val']
        decimal_precision = param_df.loc[i]['decimal_precision']
        bit_len = int(param_df.loc[i]['nBits'])
        
        bit_start = bit_cntr
        bit_end = bit_start+bit_len
        
        param_str = dna[bit_start:bit_end]        
        param_val = bin2num(param_min,param_max,decimal_precision,param_str)      
        hyper_param_str = hyper_param_str + parameter + "=" + str(param_val) + ","
        res_dict[parameter] = param_val
        
    if dna in list(result_df['dna']):
        accuracy = result_df[result_df['dna'] == dna]['fitness'].tolist()[0]
    else:
        model = eval("CatBoostClassifier(iterations=" + str(n_tree) + "," + hyper_param_str + "random_seed=2)")
        model.fit(X_train, y_train,cat_indices,logging_level='Silent')
    
        # Predicitng and calculating performance on test data
        predict_prob = model.predict_proba(X_test)[:,1]
    
        pred_list = [1 if i > 0.5 else 0 for i in predict_prob.tolist()]
    
        y_list = y_test.tolist()
    
        counter = 0
        for i in range(len(pred_list)):
            if pred_list[i] == y_list[i]:
                counter = counter+1
    
        accuracy = counter/len(pred_list)
    
    res_dict['fitness'] = accuracy
    
    t2 = datetime.datetime.now()    
    print('Calculated fitness for Generation ' + str(gen) + ' Individual ' + 
          str(individual) + ' = ' + str(accuracy) + ' -> ' + str(t2-t1))
    
    return res_dict


t1 = datetime.datetime.now()
   
# initialize population
pop_list = [dna_initialization(param_df) for i in range(n_pop)]

result_col_list = ['dna'] + list(param_df['parameter_name']) + ['fitness']
result_df = pd.DataFrame(columns=result_col_list)

print(len(pop_list))
n_child_needed = math.floor(child_ratio*n_pop)

# initial run
gen = 0
results = [ga_catboost(pop_list,i,gen) for i in range(len(pop_list))]

result_temp = pd.DataFrame(results,columns=result_col_list)

result_df = result_df.append(result_temp)
result_df = result_df.drop_duplicates()
result_df = result_df.reset_index(drop=True)

pop_df = result_temp[['dna','fitness']]

gen_best_performance = []
while gen < n_gen:
    
    gen = gen+1
    
#    par_pool1 = tournament_selection(pop_df,n_child_needed,tournament_size)
#    par_pool2 = tournament_selection(pop_df,n_child_needed,tournament_size)
    
    par_pool1 = ranking_selection(pop_df,n_child_needed)
    par_pool2 = ranking_selection(pop_df,n_child_needed)
    
    child_pool = [uniform_crossover(par1,par2) for par1,par2 in zip(par_pool1['dna'].tolist(),
                                    par_pool2['dna'].tolist())]
    
    # removing n lowest performing individuals where n is the number of children
    pop_df = pop_df.sort_values(by='fitness',ascending=False)
    pop_list = pop_df['dna'].tolist()[0:(n_pop-n_child_needed)]
    
    pop_list = pop_list + child_pool
      
    # Calculate fitness values
    results = [ga_catboost(pop_list,i,gen) for i in range(len(pop_list))]
    
    result_temp = pd.DataFrame(results,columns=result_col_list)
    
    result_df = result_df.append(result_temp)
    result_df = result_df.drop_duplicates()
    result_df = result_df.reset_index(drop=True)
    
    pop_df = result_temp[['dna','fitness']]
    
    print('Best fitness for generation ' + str(gen) + ' is ' + str(pop_df['fitness'].max()))
    gen_best_performance = gen_best_performance + [[gen,pop_df['fitness'].max()]]
    
    
t2 = datetime.datetime.now()

print("total time -> " + str(t2-t1))

#    results = [pool.apply_async(ga_catboost, args=(x,)) for x in list(pop_df['population'])]
#    final_result = [result.get() for result in results]










    
    
    
    
    
    