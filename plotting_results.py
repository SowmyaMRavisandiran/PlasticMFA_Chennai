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




#%% Set the font dictionaries (for plot title and axis titles)
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker

title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'14'}

# Set the font properties (for use in legend)   
font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
font_prop = font_manager.FontProperties(fname=font_path, size=14)


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
        ranges.loc[ranges['Variable']==var,'95th Percentile']*=100
        ranges.loc[ranges['Variable']==var,'5th Percentile']*=100
        ranges.loc[ranges['Variable']==var,'Max Likelihood solution']*=100
        ranges.loc[ranges['Variable']==var,'Unit']='%'
    elif var.endswith(')'):
        ranges.loc[ranges['Variable']==var,'Unit']='kg per person per day'

columns = ranges.columns
ranges = ranges[[columns[0],columns[1],columns[3],columns[2],columns[4]]]

ranges.to_csv('results/Max_Likelihood_results.csv',index=False)

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
fig, axes = plt.subplots(2, 4, figsize=(24, 12))  
plt.subplots_adjust(hspace=0.3) 
#fig.tight_layout(pad=4.0)  
i=0  

for row in range(2):
    for col in range(4):
        if i < len(in_df.columns):  # Ensure we only plot 7 subplots
            column = list(in_df_columns.keys())[i]  # Get the column name
            
            
            # Plot the histogram
            axes[row, col].hist(in_df[column], bins=30, density=True,alpha=1, color= '#caf0f8', label='Histogram')
            x_values = np.linspace(0, in_df[column].max(), 1000)

            # Plot the KDE
            axes[row, col].plot(x_values, in_densities[i](x_values), color='#003049', label='Posterior PDF of input variables')
            
            #Plot the input and output range
            axes[row, col].axvline(input_flows.loc[input_flows["Flow Name"]==column]['Minimum'].values, color='#f77f00', linestyle='--', linewidth=2, label='Minimum of input range')
            axes[row, col].axvline(input_flows.loc[input_flows["Flow Name"]==column]['Maximum'].values, color='#d62828', linestyle='--', linewidth=2, label='Maximum of input range')

            #Plot the optimum solution
            axes[row, col].axvline(optimal_combination[column], color='black', linestyle='--', linewidth=2, label='Max Likelihood Solution')

            # Add labels and title
            axes[row, col].set_title(in_df_columns[column], **title_font)
            axes[row, col].set_ylabel('Density', **axis_font)
            axes[row, col].set_xlabel('Quantity in tons per month', **axis_font)
            
            
            
            
            # Set the tick labels font
            for label in (axes[row,col].get_xticklabels() + axes[row,col].get_yticklabels()):
                label.set_fontname('Arial')
                label.set_fontsize(13)
            
            #Setting uniform tick labels
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-1, 1))  # Forces scientific notation for small numbers
            axes[row, col].xaxis.set_major_formatter(formatter)
            axes[row,col].yaxis.set_major_formatter(formatter)

        elif i == len(in_df.columns):  # Add legend in the last subplot
            axes[row, col].axis('off')  # Turn off the axis for empty plots
            handles, labels = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from any subplot
            axes[row, col].legend(handles, labels, loc='center', prop=font_prop)

        else:
            axes[row, col].axis('off')  # Turn off the axis for empty plots

        i += 1

#plt.savefig('plots/input_variables.pdf', dpi = 300, format = 'pdf')        
#plt.savefig('plots/input_variables.png', dpi = 300)
plt.show()

#%%% Plot 2: output rates

out_rates_df = montecarlo_result[['Mismanagement Rate', 'Management Rate', 'Recovery Rate', 'Recycling Rate','RP Recovery Rate', 'NRP Recovery Rate','IWS Recovery Rate','PET Recovery Rate']]
out_rates_df = out_rates_df*100
out_rates_densities = [gaussian_kde(out_rates_df[column]) for column in out_rates_df.columns]  # Calculate KDE for each column

#input plots
fig, axes = plt.subplots(3, 3, figsize=(15, 13))  
fig.tight_layout(pad=4.0)  # Adjust layout 
i=0  

for row in range(3):
    for col in range(3):
        if i < len(out_rates_df.columns):  # Ensure we only plot 7 subplots
            column = out_rates_df.columns[i]  # Get the column name
            
            # Plot the histogram
            axes[row, col].hist(out_rates_df[column], bins=30, density=True,alpha=1, color= '#caf0f8', label='Histogram')
            x_values = np.linspace(0, out_rates_df[column].max(), 1000)

            # Plot the KDE
            axes[row, col].plot(x_values, out_rates_densities[i](x_values), color='#003049', label='Predicted PDF for model indicators')
            axes[row, col].axvline(ranges.loc[ranges['Variable']==column]['Max Likelihood solution'].values[0], color='black', linestyle='--', linewidth=2, label='Max Likelihood Solution')

        
            # Add labels and title
            axes[row, col].set_title(column, **title_font)
            axes[row, col].set_ylabel('Density', **axis_font)
            axes[row, col].set_xlabel('Percentage', **axis_font)
        
        elif i == len(out_rates_df.columns):  # Add legend in the last subplot
            axes[row, col].axis('off')  # Turn off the axis for empty plots
            handles, labels = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from any subplot
            axes[row, col].legend(handles, labels, loc='center', fontsize='large', title_fontsize='13')
        else:
            axes[row, col].axis('off')  # Turn off the axis for empty plots

        i += 1

#plt.savefig('plots/output_rates.png', dpi = 300, format = 'png')
#plt.savefig('plots/output_rates.pdf', dpi = 300, format = 'pdf')
plt.show()


#%% Plot 3: output vs input 
in_df_columns = {'F2':'F2: Collection by Itinerant Waste Buyers', 'F6':'F6: Collection by wastepickers from bins',
                 'F9':'F9: Collection of Recyclable Plastic by BOVs', 'F14':'F14: Aggregation in scrap shops',
                 'F7':'F7: Collection by wastepickers from landfills',
                 'F12':'F12: Collection of Non-Recyclable Plastic by BOVs', 'F4':'F4: Collection by FS in corporation bins '
                       }
out_rates_df = out_rates_df/100
for output in out_rates_df.columns:
    i=0  
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  
    fig.tight_layout(pad=4.0)  # Adjust layout 
    for row in range(2):
        for col in range(4):
            if i < len(in_df.columns):  # Ensure we only plot 7 subplots
                column = in_df.columns[i]  # Get the column name
                
                # Plot the histogram
                axes[row, col].plot(montecarlo_result[column],montecarlo_result[output],'o',alpha = 0.5, color = '#0E4A6C', label='Model indicator value for 1 run')
                
                axes[row, col].axvline(input_flows.loc[input_flows["Flow Name"]==column]['Minimum'].values, color='#f77f00', linestyle='--', linewidth=2, label='Minimum of input range')
                axes[row, col].axvline(input_flows.loc[input_flows["Flow Name"]==column]['Maximum'].values, color='#d62828', linestyle='--', linewidth=2, label='Maximum of input range')

            
                # Add labels and title
                axes[row, col].set_title(in_df_columns[column])
                axes[row, col].set_ylabel(output+' (in %)')
                axes[row, col].set_xlabel('Quantity in tons per months')
                
                axes[row, col].set_ylim(0)
                axes[row, col].set_xlim(0)
            elif i == len(in_df.columns):  # Add legend in the last subplot
                axes[row, col].axis('off')  # Turn off the axis for empty plots
                handles, labels = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from any subplot
                axes[row, col].legend(handles, labels, loc='center', prop=font_prop)
            else:
                axes[row, col].axis('off')  # Turn off the axis for empty plots
    
            i += 1
 
    #plt.savefig('plots/'+output+'_vs_inputs.pdf', dpi = 300, format='pdf')
    #plt.savefig('plots/'+output+'_vs_inputs.png', dpi = 300)
    plt.show()
    


