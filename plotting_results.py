#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:36:35 2025

@author: marriyapillais
"""
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns


input_flows = pd.read_excel('data/mfa_data.xlsx', sheet_name = 'input_flows')
input_flows.rename(columns = {"Unnamed: 0": "Flow Name"}, inplace = True)
montecarlo_result = pd.read_csv('results/MCS_results.csv', index_col=0)


#%% Estimating optimal solution


montecarlo_result.reset_index(inplace = True)
montecarlo_result.drop(columns = 'index', inplace = True)

from scipy.stats import gaussian_kde

opt_df = montecarlo_result[['F2', 'F4', 'F6', 'F7','F9', 'F12','F14']]

#opt_df = montecarlo_result.iloc[:,:25]

in_densities = [gaussian_kde(opt_df[column]) for column in opt_df.columns]  # Calculate KDE for each column

joint_prob_density = [
    np.prod([in_densities[i](row[i]) for i in range(len(row))]) for idx, row in opt_df.iterrows()
]  # Calculate joint probability density for each row

max_index = np.argmax(joint_prob_density)  # Find index of the max probability density

optimal_combination = opt_df.iloc[max_index]  # Select the corresponding row
final_result = montecarlo_result.iloc[max_index]


#%% Plot 1: input variables

in_df = montecarlo_result[['F2', 'F4', 'F6', 'F7','F9', 'F12','F14']]  # Select input variable from the results

#in_df_column_names = {'F2':'Collection by Itinerant Waste Buyers', 'F6':'Collection by wastepickers from bins',
#                 'F9':'Collection of Recyclable Plastic by BOVs', 'F14':'Aggregation in scrap shops',
#                 'F7':'Collection by wastepickers from landfills',
#                 'F12':'Collection of Non-Recyclable Plastic by BOVs', 'F4':'Collection by FS in corporation bins '
#                       }
in_df_columns = {'F2':'F2: Collection by Itinerant Waste Buyers', 'F6':'F6: Collection by wastepickers from bins',
                 'F9':'F9: Collection of Recyclable Plastic by BOVs', 'F14':'F14: Aggregation in scrap shops',
                 'F7':'F7: Collection by wastepickers from landfills',
                 'F12':'F12: Collection of Non-Recyclable Plastic by BOVs', 'F4':'F4: Collection by FS in corporation bins '
                       }


in_densities = [gaussian_kde(in_df[column]) for column in list(in_df_columns.keys())]


#input plots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))  
fig.tight_layout(pad=4.0)  
i=0  

for row in range(2):
    for col in range(4):
        if i < len(in_df.columns):  # Ensure we only plot 7 subplots
            column = list(in_df_columns.keys())[i]  # Get the column name
            
            
            # Plot the histogram
            axes[row, col].hist(in_df[column], bins=30, density=True,alpha=1, color= '#caf0f8', label='Histogram')
            x_values = np.linspace(0, in_df[column].max(), 1000)

            # Plot the KDE
            axes[row, col].plot(x_values, in_densities[i](x_values), color='#003049', label='PDF')
            
            #Plot the input and output range
            axes[row, col].axvline(input_flows.loc[input_flows["Flow Name"]==column]['Minimum'].values, color='#f77f00', linestyle='--', linewidth=2, label='Minimum of input range')
            axes[row, col].axvline(input_flows.loc[input_flows["Flow Name"]==column]['Maximum'].values, color='#d62828', linestyle='--', linewidth=2, label='Maximum of input range')

            #Plot the optimum solution
            axes[row, col].axvline(optimal_combination[column], color='black', linestyle='--', linewidth=2, label='Most probable value')

            # Add labels and title
            axes[row, col].set_title(in_df_columns[column])
            axes[row, col].set_ylabel('Density')
            axes[row, col].set_xlabel('Quantity in tons per month')
            
        elif i == len(in_df.columns):  # Add legend in the last subplot
            axes[row, col].axis('off')  # Turn off the axis for empty plots
            handles, labels = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from any subplot
            axes[row, col].legend(handles, labels, loc='center', fontsize='large', title_fontsize='13')

        else:
            axes[row, col].axis('off')  # Turn off the axis for empty plots

        i += 1
        
plt.savefig('plots/input_variables.png', dpi = 300)
plt.show()

#%%% Plot 2: output rates

out_rates_df = montecarlo_result[['Mismanagement Rate', 'Management Rate', 'Recovery Rate', 'Recycling Rate','RP Recovery Rate', 'NRP Recovery Rate','IWS Recovery Rate','PET Recovery Rate']]
out_rates_densities = [gaussian_kde(out_rates_df[column]) for column in out_rates_df.columns]  # Calculate KDE for each column

#input plots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))  
fig.tight_layout(pad=4.0)  # Adjust layout 
i=0  

for row in range(2):
    for col in range(4):
        if i < len(out_rates_df.columns):  # Ensure we only plot 7 subplots
            column = out_rates_df.columns[i]  # Get the column name
            
            # Plot the histogram
            axes[row, col].hist(out_rates_df[column], bins=30, density=True,alpha=1, color= '#caf0f8', label='Histogram')
            x_values = np.linspace(0, out_rates_df[column].max(), 1000)

            # Plot the KDE
            axes[row, col].plot(x_values, out_rates_densities[i](x_values), color='#003049', label='PDF')
            
        
            # Add labels and title
            axes[row, col].set_title(column)
            axes[row, col].set_ylabel('Density')
            axes[row, col].set_xlabel('Quantity in tons per month')
        
        else:
            axes[row, col].axis('off')  # Turn off the axis for empty plots

        i += 1

# Create a common legend at the bottom
handles, labels = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from one subplot
fig.legend(handles, labels, loc='lower center', fontsize='large', title_fontsize='13', ncol=2)  # Adjust legend position and style


plt.show()
#plt.savefig('plots/output_rates_pdf.png', dpi = 300)


#%% Plot 3: output vs input 

in_df_columns = {'F2':'F2: Collection by IWBs', 'F4':'F4: Collection by FS in corporation bins ', 
                       'F6':'F6: Collection by wastepickers from bins', 'F7':'F6: Collection by wastepickers from landfills',
                       'F9':'F9: Collection of RP by BOVs', 'F12':'F9: Collection of NRP by BOVs',
                       'F14':'F14: Aggregation in scrap shops'}
for output in out_rates_df.columns:
    print(output)
    
    i=0  
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  
    fig.tight_layout(pad=4.0)  # Adjust layout 
    for row in range(2):
        for col in range(4):
            if i < len(in_df.columns):  # Ensure we only plot 7 subplots
                column = in_df.columns[i]  # Get the column name
                
                # Plot the histogram
                axes[row, col].plot(montecarlo_result[column],montecarlo_result[output],'o',alpha = 0.75, color = '#0E4A6C' )
                
            
                # Add labels and title
                axes[row, col].set_ylabel(output+' (in %)')
                axes[row, col].set_xlabel(in_df_columns[column])
                
                axes[row, col].set_ylim(0)
                axes[row, col].set_xlim(0)
            
            else:
                axes[row, col].axis('off')  # Turn off the axis for empty plots
    
            i += 1

    # Create a common legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from one subplot
    fig.legend(handles, labels, loc='lower center', fontsize='large', title_fontsize='13', ncol=2)  # Adjust legend position and style
    
    
    plt.show()
    #plt.savefig('plots/pet_recovery_rate_vs_inputs.png', dpi = 300)

#%% Getting the output with max likelihood values


#%%% Getting the 90% confidence intervals 

densities = [gaussian_kde(montecarlo_result[column]) for column in montecarlo_result.columns]  # Calculate KDE for each column
ranges = pd.DataFrame()
for idx, col in enumerate(montecarlo_result.columns):
    
    x_values = np.linspace(min(montecarlo_result[col]), max(montecarlo_result[col]), 1000)
    kde_values = densities[idx](x_values)
    percentile_5 = np.percentile(montecarlo_result[col], 5)
    percentile_95 = np.percentile(montecarlo_result[col], 95)
    ranges[col]=[percentile_5, percentile_95]    

#%%% Combining results and generating output
ranges.rename(index={0: "5th Percentile", 1: "95th Percentile"},inplace=True)
ranges=ranges.transpose()
ranges.reset_index(names=['Variable'],inplace = True)

final_result=pd.DataFrame(final_result).reset_index(names=['Variable'])

ranges = pd.merge(ranges,final_result,how='left', on='Variable')

#Update columns NAme
col_name=ranges.columns[3]
ranges.rename(columns={col_name:'Max Likelihood solution'},inplace= True)

ranges['Unit']='tons per month'
for var in ranges['Variable'].unique():
    if var.endswith('Rate'):
        ranges.loc[ranges['Variable']==var,'Unit']='%'
    elif var.endswith(')'):
        ranges.loc[ranges['Variable']==var,'Unit']='kg per person per day'

columns = ranges.columns
ranges = ranges[[columns[0],columns[1],columns[3],columns[2],columns[4]]]

ranges.to_csv('Results.csv',index=False)
