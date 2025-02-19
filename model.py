#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:03:50 2024

@author: sowmya
"""

import pandas as pd
import numpy as np
import scipy


#input data
mfa_data = pd.read_excel('mfa_data.xlsx', sheet_name = 'mfa_data')
eq = pd.read_excel('mfa_data.xlsx', sheet_name = 'equations')
input_flows = pd.read_excel('mfa_data.xlsx', sheet_name = 'input_flows')
tcs = pd.read_excel('mfa_data_excel.xlsx', sheet_name = 'tcs')

# Data Cleaning
input_flows.rename(columns = {"Unnamed: 0": "Flow Name"}, inplace = True)
tcs.rename(columns = {"Unnamed: 0": "TC Name"}, inplace = True)

input_flows= input_flows.round(0)

eq.fillna(0, inplace=True)

#%% Model Functions
#%%% Model estimation function that returns solution of MFA + value of balancing termn for sanity check
def model_estimation(eq, inputs):
    
    check = inputs.loc[inputs["Name"]=='F6']
    inputs.loc[inputs["Name"]=='F6', 'Value']=0
    
    eq= eq.replace('tc_5', inputs.loc[inputs["Name"]=='TC_5', 'Value'].values[0]*-1)
    eq= eq.replace('tc_11', inputs.loc[inputs["Name"]=='TC_11', 'Value'].values[0]*-1)
    eq= eq.replace('tc_17', inputs.loc[inputs["Name"]=='TC_17', 'Value'].values[0]*-1)
    
    eq= eq.replace('tc_19', inputs.loc[inputs["Name"]=='TC_19', 'Value'].values[0]*-1)
    eq= eq.replace('tc_23', inputs.loc[inputs["Name"]=='TC_23', 'Value'].values[0]*-1)
    index = inputs["Name"].iloc[:25]
        
    eq = eq.iloc[:,1:].to_numpy()
    inputs = inputs["Value"].iloc[:25].to_numpy()
    
    solution = np.linalg.solve(eq, inputs)
    solution = pd.DataFrame(solution, index)
    solution.rename(columns = {0:'Value'}, inplace = True)
    return solution, check

#%%# Rate estimation function to calculate all the model indicators

def rate_estimation(solution, beta_pet):
    mis_rate = (solution.loc["F22"].values+solution.loc["F23"].values + 0.6* (solution.loc["F21"].values-solution.loc["F7"].values))/solution.loc["F1"].values
    man_rate = (solution.loc["F14"].values+solution.loc["F15"].values + 0.4* (solution.loc["F21"].values-solution.loc["F7"].values))/solution.loc["F1"].values
    recov_rate = (solution.loc["F14"].values+solution.loc["F15"].values)/solution.loc["F1"].values
    recyc_rate = (solution.loc["F14"].values+solution.loc["F17"].values)/solution.loc["F1"].values
    rec_rp = (solution.loc["F14"].values+solution.loc["F17"].values)/(solution.loc["F1"].values * 0.32)
    rec_nrp = (solution.loc["F15"].values - solution.loc["F17"].values)/(solution.loc["F1"].values * 0.68)
    rec_iws = solution.loc["F14"].values/solution.loc["F1"].values
    rec_pet = 0.25*(solution.loc["F14"].values+solution.loc["F17"].values)/(beta_pet*solution.loc["F1"].values)
    per_cap_gen_min = solution.loc["F1"].values/(30*7.1)
    per_cap_gen_max = solution.loc["F1"].values/(30*6.407)
    rates = pd.DataFrame()
    rates['Mismanagement Rate'] = mis_rate
    rates['Management Rate'] = man_rate
    rates['Recovery Rate'] = recov_rate
    rates['Recycling Rate'] = recyc_rate
    rates['RP Recovery Rate'] = rec_rp
    rates['NRP Recovery Rate'] = rec_nrp
    rates['IWS Recovery Rate'] = rec_iws
    rates['PET Recovery Rate'] = rec_pet
    rates['Per cap Plastic Waste Gen (min)'] = per_cap_gen_min
    rates['Per cap Plastic Waste Gen (max)'] = per_cap_gen_max
    return rates


#%%% Function with conditions to check validity of estimation of MCS

def conditions(solution, rates):
    valid = True

    #check for negative flow values
    if any(solution["Value"]<=0):
        valid = False
        
    #check if balancing terms lies within range
    if not(input_flows.loc[input_flows['Flow Name']=='F6']['Minimum'].values[0] <= solution.loc['F6']['Value'] <= input_flows.loc[input_flows['Flow Name']=='F6']['Maximum'].values[0]):
        valid = False
        
    #check if all rates are between 0 and 1
    if any(rates.iloc[0,:8]>=1.0) or any(rates.iloc[0,:8]<=0):
        valid= False
    
    return valid

#%% Execution: Monte Carlo Simulation

# Variables that have a data range and need to be statistically tested
var_mc = input_flows.loc[input_flows["Minimum"]!=0][["Flow Name", "Minimum", "Maximum"]].values
var_mc = np.append(var_mc,tcs.loc[tcs["Minimum"].notna()][["TC Name", "Minimum", "Maximum"]].values, axis = 0)

# Number of runs of Monte Carlo Simulation
runs = 10000


#Random data inputs for variables that need to be tested
random_var = pd.DataFrame()

#Create input Data from random vars and generate monte carlo simualtion 

montecarlo_result = pd.DataFrame(columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11',
       'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21',
       'F22', 'F23', 'F24', 'F25', 'Mismanagement Rate', 'Management Rate',
       'Recovery Rate', 'Recycling Rate', 'RP Recovery Rate',
       'NRP Recovery Rate', 'IWS Recovery Rate', 'PET Recovery Rate',
       'Per cap Plastic Waste Gen (min)', 'Per cap Plastic Waste Gen (max)'])
invalid_result = pd.DataFrame(columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11',
       'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21',
       'F22', 'F23', 'F24', 'F25', 'Mismanagement Rate', 'Management Rate',
       'Recovery Rate', 'Recycling Rate', 'RP Recovery Rate',
       'NRP Recovery Rate', 'IWS Recovery Rate', 'PET Recovery Rate',
       'Per cap Plastic Waste Gen (min)', 'Per cap Plastic Waste Gen (max)'])

np.random.seed(59) #Setting a seed for reproducibility 

i=0
valid_count= 0 
while valid_count < runs:
    for var in range(len(var_mc)):
        random_var[var_mc[var,0]] = scipy.stats.uniform.rvs(loc = var_mc[var,1], scale = var_mc[var,2]-var_mc[var,1], size = 1) 
    random_i = random_var.iloc[0]
    input_i = pd.DataFrame(columns =['Name','Value'])
    input_i['Name'] = pd.concat([input_flows['Flow Name'], tcs["TC Name"]], ignore_index = True)
    input_i['Value'] = 0
    input_i['Value'] = input_i['Name'].map(random_i).fillna(input_i['Value']).astype(float)

    solution_i, check_i =  model_estimation(eq, input_i)
    rates_i = rate_estimation(solution_i, 0.055)
    valid = conditions(solution_i, rates_i)
    
    if valid:
        solution_i = solution_i.transpose()
        result_i = pd.concat([solution_i.reset_index().drop(columns='index'), rates_i], axis = 1)
        montecarlo_result = pd.concat([montecarlo_result,result_i])
        valid_count=valid_count+1
        
    else:
        solution_i = solution_i.transpose()
        result_i = pd.concat([solution_i.reset_index().drop(columns='index'), rates_i], axis = 1)
        invalid_result = pd.concat([invalid_result,result_i])
    
    i = i+1

#%% Save the output

montecarlo_result.to_csv('MCS_results.csv')










