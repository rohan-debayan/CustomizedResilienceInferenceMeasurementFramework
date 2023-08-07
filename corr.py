import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple
import ipywidgets as widgets
from IPython.display import display

def matrix(folder_path: str, start_year: int, end_year: int, final_df: pd.DataFrame):
    data = pd.DataFrame()
    for year in range(start_year, end_year):
        filename = os.path.join(folder_path, str(year) + '.csv')
        if not os.path.exists(filename):
            print(f"No CSV file found for the year {year}.")
        else:
            year_data = pd.read_csv(filename)
            year_data = year_data.replace(np.nan, 0)
            year_data['Year'] = year
            year_data['GEOID'] = year_data['STCOU'].astype(str).str.zfill(5)
            year_data['uniqueID'] = year_data['GEOID'] + "_" + year_data['Year'].astype(str)
            data = data.append(year_data, ignore_index=True)
    X=data
    extra_row = final_df[~final_df['FIPScounty'].isin(data['GEOID'])]

    drop_columns = ['GEOID', 'STCOU', 'Areaname', 'Year']
    for column in drop_columns:
        if column in data.columns:
            data = data.drop(column, axis=1)

    corr_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm")
    plt.show()

    return X, data, corr_matrix

def makedf(X: pd.DataFrame, prgr: pd.DataFrame) -> pd.DataFrame:

    # drop unnecessary columns
    drop_columns = ['GEOID', 'STCOU', 'Areaname', 'Year']
    for column in drop_columns:
        if column in X.columns:
            X = X.drop(column, axis=1)

    # create copy of input dataframe
    save_df = X.copy()            
            
    # map additional columns from prgr dataframe
    save_df['Y_Class'] = save_df['uniqueID'].map(prgr.set_index('uniqueID')['Resilience_num'])
    save_df['YA_Class'] = save_df['uniqueID'].map(prgr.set_index('uniqueID')['Adaptability_num'])
    save_df['YV_Class'] = save_df['uniqueID'].map(prgr.set_index('uniqueID')['Vulnerability_num'])
    save_df['YA_Value'] = save_df['uniqueID'].map(prgr.set_index('uniqueID')['Adaptability'])
    save_df['YV_Value'] = save_df['uniqueID'].map(prgr.set_index('uniqueID')['Vulnerability'])
    save_df['FIPSstate'] = save_df['uniqueID'].map(prgr.set_index('uniqueID')['FIPSstate'])
    save_df['YR_Value'] = save_df['uniqueID'].map(prgr.set_index('uniqueID')['Resilience'])

    # count and print the number of entries with no value
    num_rows_with_nan = save_df.isna().any(axis=1).sum()
    print(f"Number of entries with no value: {num_rows_with_nan}")

    # drop rows with missing values
    save_df = save_df.dropna()
    X = X[X['uniqueID'].isin(save_df['uniqueID'])]
    X = X.drop('uniqueID', axis=1)
    return save_df,X

def remopt():
    print("Enter method for dropping columns: ")
    
    options = widgets.RadioButtons(
        options=['Remove by selecting names', 'Remove by correlation index'],
        description='Options:',
        disabled=False
    )

    display(options)
    return options

def optionA(data):
    checkboxes = []
    for col in data.columns:
        checkbox = widgets.Checkbox(description=col)
        checkboxes.append(checkbox)

    checkboxes_widget = widgets.VBox(checkboxes)
    display(checkboxes_widget)

    button = widgets.Button(description="Drop Columns")
    display(button)

    def drop_columns(button):
        columns_to_drop = []
        for i in range(len(checkboxes)):
            if checkboxes[i].value:
                columns_to_drop.append(data.columns[i])

        return columns_to_drop

    button.on_click(drop_columns)
    
def optionB(data):
    to_remove = set()
    threshold = input("Enter the correlation index (0.7 is usual):")
    print("Columns being removed because of high correlation:\n")
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) >= float(threshold):
                colname = corr_matrix.columns[i]
                to_remove.add(colname)

    for colname in to_remove:
        print(colname)
    
    return list(to_remove)

def dropcol(df, columns_to_drop):
    df = df.drop(columns_to_drop, axis=1)
    return df

