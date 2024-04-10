from ipywidgets import widgets, VBox, HBox, Layout
from IPython.display import HTML, display
import pandas as pd
from typing import Dict, Any
import numpy as np
import geopandas as gpd
import folium
from folium.features import GeoJson, GeoJsonTooltip
import json

def create_gui1():
    # List of widget names and descriptions
    widget_details = [
        ('ini', 'Initial year:', 'IntText'),
        ('fin', 'Final year:', 'IntText'),
        ('avalanche', 'Avalanches:', 'RadioButtons'),
        ('coastal', 'Coastal disasters:', 'RadioButtons'),
        ('drought', 'Droughts:', 'RadioButtons'),
        ('earthquake', 'Earthquake:', 'RadioButtons'),
        ('flooding', 'Flooding:', 'RadioButtons'),
        ('fog', 'Fog:', 'RadioButtons'),
        ('hail', 'Hail:', 'RadioButtons'),
        ('heat', 'Heat:', 'RadioButtons'),
        ('hurricane', 'Hurricane:', 'RadioButtons'),
        ('landslide', 'Landslide:', 'RadioButtons'),
        ('lightning', 'Lightning:', 'RadioButtons'),
        ('severestorm', 'Severe storms:', 'RadioButtons'),
        ('tornado', 'Tornados:', 'RadioButtons'),
        ('tsunami', 'Tsunamis:', 'RadioButtons'),
        ('wildfire', 'Wildfire:', 'RadioButtons'),
        ('wind', 'Wind:', 'RadioButtons'),
        ('winterweather', 'Winter weather:', 'RadioButtons'),
        ('countygrp', 'County Group:', 'Dropdown'),
        ('submit', 'Submit:', 'Button')
    ]

    # Creating widgets
    widgets_dict = {}
    for name, description, widget_type in widget_details:
        if widget_type == 'IntText':
            widget = widgets.IntText(description=description)
        elif widget_type == 'RadioButtons':
            options = ['Yes', 'No'] if description != 'County Group:' else ['All', 'Mostly Urban','Mostly Rural', 'Completely Rural','Custom']
            widget = widgets.RadioButtons(options=options, description=description)
        elif widget_type == 'Button':
            widget = widgets.Button(description=description, button_style='success')
        elif widget_type == 'Dropdown':
            options = ['All', 'Mostly Urban','Mostly Rural', 'Completely Rural']# Might add custom on a later update
            widget = widgets.Dropdown(options=options, description=description)

        widget.style = {'description_width': '50%'}
        widgets_dict[name] = widget

    # Grouping widgets into columns
    input1 = widgets.HBox([widgets_dict['ini'], widgets_dict['fin']])
    input_col2 = widgets.VBox(list(widgets_dict.values())[2:8])
    input_col3 = widgets.VBox(list(widgets_dict.values())[8:14])
    input_col4 = widgets.VBox(list(widgets_dict.values())[14:-1])
    input_col5 = widgets.HBox([widgets_dict['submit']], layout=Layout(justify_content='center'))
    
    # Display widgets
    input_row2 = widgets.HBox([input_col2, input_col3, input_col4]) 
    display(input1)
    display(input_row2)
    display(input_col5)
    
    # Return the dictionary of widgets
    return widgets_dict

def process_data(widget_dict):
    # Extract values from the widget dictionary
    ini_value = widget_dict['ini'].value
    fin_value = widget_dict['fin'].value
    countygrp_value = widget_dict['countygrp'].value
    hazards_list = [
        "Avalanche","Tornado","Coastal","Flooding","SevereStorm","Wind","Drought",
        "Heat","Earthquake","Fog","WinterWeather","Hail","Landslide","Lightning",
        "Tsunami","Wildfire","Hurricane"
    ]

    # Load and preprocess the population data
    pl = pd.read_csv('data/population/population.csv')
    pl['popini'] = pl[f'{ini_value}']
    pl['popfin'] = pl[f'{fin_value}']

    # Load and preprocess the disaster data
    df = pd.read_csv('data/disaster/Disasters.csv')
    index_names = df[(df['Year'] < int(ini_value)) | (df['Year'] > int(fin_value))].index
    df.drop(index_names, inplace=True)
    df['Disaster'] = df['Disaster'].str.replace(" ", "").str.lower().str.split('/').str[0]
    
    for hazard in hazards_list:
        if widget_dict[hazard.lower()].value == 'No':
            index_names = df[df['Disaster'].str.contains(hazard.lower())].index
            df.drop(index_names, inplace=True)
            print("Records being removed:" + hazard)

    pl['FIPScounty'] = pl.apply(lambda row : str(row['STCOU']).zfill(5), axis = 1)
    df['UniqueCode'] = df.apply(lambda row : row['UniqueCode'].strip('\''), axis = 1)

    # Filter based on county group
    if countygrp_value != 'All':
        df_orig = df.copy()
        ru = pd.read_csv('data/ruralUrban/RUdiv.csv')
        ru['FIPScounty'] = ru.apply(lambda row: str(row['FIPScounty']).zfill(5), axis=1)
        ru = ru[ru['Class'] == countygrp_value[:2]]
        df = df[df['UniqueCode'].isin(ru['FIPScounty'])]
        print(f"{len(df_orig) - len(df)} rows removed from original dataframe from {len(df_orig)}")

    # Merge the dataframes
    fn = pd.merge(df, pl, left_on='UniqueCode', right_on='FIPScounty', how='inner')

    return fn

def print1(widget_dict):
    print('The inputted parameters are:')
    print('Duration to be computed:' + str(widget_dict['ini'].value) + ' to ' + str(widget_dict['fin'].value) +' \n')
    hazards_list = [
        "Avalanche","Tornado","Coastal","Flooding","SevereStorm","Wind","Drought",
        "Heat","Earthquake","Fog","WinterWeather","Hail","Landslide","Lightning",
        "Tsunami","Wildfire","Hurricane"
    ]
    print('The hazards considered in this computation are:')
    for i, hazard in enumerate(hazards_list):
        if (widget_dict[hazard.lower()].value=='Yes'):
            print(f"{hazard}", end='')
            if i != len(hazards_list) - 1:
                print(', ', end='')

    print(f"\n\n{widget_dict['countygrp'].value} counties will be evaluated")
    


def empfac(fn: pd.DataFrame, widget_dict: Dict[str, Any]) -> pd.DataFrame:
    ini_value = widget_dict['ini'].value
    fin_value = widget_dict['fin'].value

    # Grouping and calculating required metrics
    fn['Perdaycapdmg'] = fn['DamageRIM'] / fn['Duration(Day)']
    groupby_disaster = fn.groupby('Disaster')['Perdaycapdmg']
    weight = pd.DataFrame({
        'Count': groupby_disaster.count(),
        'Sum': groupby_disaster.sum(),
        'Mean': groupby_disaster.mean()
    })
    weight['Probability'] = weight['Count'] / ((fin_value - ini_value) * 365)
    weight.to_csv('calculation/Weight.csv')

    # Calculating exposure and damage for each year and creating edr_per_year
    ex_per_year, dg_per_year, edr_per_year = {}, {}, {}
    pl_per_year = {}

    for year in range(ini_value, fin_value + 1):
        fn_year = fn[fn['Year'] == year]
        fn_year['Mean'] = fn_year['Disaster'].map(weight['Mean'])
        fn_year['Probability'] = fn_year['Disaster'].map(weight['Probability'])
        fn_year['Exposure'] = fn_year['Duration(Day)'] * fn_year['Mean'] * fn_year['Probability']
        fn.loc[fn['Year'] == year, 'Exposure'] = fn_year['Exposure']
        ex_per_year[year] = fn_year.groupby('FIPScounty')['Exposure'].sum().to_frame()
        dg_per_year[year] = fn_year.groupby('FIPScounty')['DamageRIM'].sum().to_frame()

        # Creating pl_per_year
        columns_to_select = ['FIPScounty', 'Areaname', str(year), str(year + 1)]
        pl_per_year[year] = fn[columns_to_select]

    for year in range(ini_value, fin_value):
        pl_year = pl_per_year[year]
        edr = pd.merge(pl_year, dg_per_year[year], on=["FIPScounty"], how="inner")
        edr = pd.merge(edr, ex_per_year[year], on=["FIPScounty"], how="inner")
        edr_per_year[year] = edr

    # Calculating Recovery
    for year in range(ini_value, fin_value):
        edr_per_year[year]['Recovery'] = edr_per_year[year][str(year)].where(edr_per_year[year][str(year)] == 0, 
                                       (edr_per_year[year][str(year+1)]-edr_per_year[year][str(year)])/edr_per_year[year][str(year)])
        edr_per_year[year].dropna(subset=['Recovery', 'DamageRIM', 'Exposure'], inplace=True)

    df_list = []
    for year, df in enumerate(edr_per_year.values(), start=ini_value):
        df = df[['FIPScounty', 'Areaname', 'DamageRIM', 'Exposure', 'Recovery']]
        df['Year'] = year
        df_list.append(df)

    edr_tot = pd.concat(df_list)

    # Min Max Scaling
    edr_tot['Normalized_Damage'] = (edr_tot['DamageRIM'] - edr_tot['DamageRIM'].min()) / (edr_tot['DamageRIM'].max() - edr_tot['DamageRIM'].min())
    edr_tot['Normalized_Recovery'] = (edr_tot['Recovery'] - edr_tot['Recovery'].min()) / (edr_tot['Recovery'].max() - edr_tot['Recovery'].min())
    #Ver1
    edr_tot['Normalized_Exposure'] = (edr_tot['Exposure'] - edr_tot['Exposure'].min()) / (edr_tot['Exposure'].max() - edr_tot['Exposure'].min())

    # Vulnerability
    #edr_tot['Vul'] = edr_tot['DamageRIM'] - edr_tot['Exposure']
    #edr_tot['NVul'] = (edr_tot['Vul'] - edr_tot['Vul'].min()) / (edr_tot['Vul'].max() - edr_tot['Vul'].min())
    edr_tot_uniq = edr_tot.drop_duplicates()
    return edr_tot_uniq


def disres(edr_tot: pd.DataFrame) -> pd.DataFrame:
    edr_tot['Adaptability'] = edr_tot['Normalized_Recovery'] -edr_tot['Normalized_Damage']
    edr_tot['NAdap'] = (edr_tot['Adaptability'] - edr_tot['Adaptability'].min()) / (edr_tot['Adaptability'].max() - edr_tot['Adaptability'].min())
    #Ver1
    edr_tot['Vulnerability'] = edr_tot['Normalized_Damage'] -edr_tot['Normalized_Exposure']
    #edr_tot['Vulnerability'] = edr_tot['NVul']
    edr_tot['Resilience'] = edr_tot['NAdap'] - edr_tot['NVul']

    columns_to_quantize = ['Adaptability', 'Vulnerability', 'Resilience']
    for col in columns_to_quantize:
        q1, q2, q3 = edr_tot[col].quantile([0.25, 0.5, 0.75])
        
        conditions = [
            (edr_tot[col] <= q1),
            (edr_tot[col] > q1) & (edr_tot[col] <= q2),
            (edr_tot[col] > q2) & (edr_tot[col] <= q3),
            (edr_tot[col] > q3)
        ]

        values = ['Low', 'Medium', 'High', 'Extremely High']
        valuesnum = [1, 2, 3, 4]

        edr_tot[col+'_tier'] = np.select(conditions, values)
        edr_tot[col+'_num'] = np.select(conditions, valuesnum)

    prgr = edr_tot
    prgr['uniqueID'] = edr_tot['FIPScounty'] + "_" + edr_tot['Year'].astype(str)
    prgr['FIPSstate'] = edr_tot['FIPScounty'].str[:2]
    prgr.to_csv('calculation/Priorigrp.csv')
    return edr_tot, prgr

def gencpleth(prgr: pd.DataFrame, county_geo: str, ini_value: int, fin_value: int) -> folium.Map:
    map_df = prgr.groupby('FIPScounty')[['Vulnerability', 'Adaptability', 'Resilience']].mean().reset_index()
    geoJSON_df = gpd.read_file(county_geo)
    geoJSON_df = geoJSON_df.rename(columns = {"id":"FIPScounty"})
    map_df = geoJSON_df.merge(map_df, on="FIPScounty")
    map_df = map_df.dropna(subset=['geometry'])
    map_df.to_csv('calculation/Map.csv')

    quantile_labels = ["Low", "Medium", "High", "Extremely High"]
    quantiles = [0, .25, .5, .75, 1]
    for col in ['Resilience', 'Vulnerability', 'Adaptability']:
        map_df[col + '_quantile_label'] = pd.qcut(map_df[col], q=quantiles, labels=quantile_labels)

    quantile_mapping = {"Low": 1, "Medium": 2, "High": 3, "Extremely High": 4}
    for col in ['Resilience_quantile_label', 'Vulnerability_quantile_label', 'Adaptability_quantile_label']:
        map_df[col + '_numeric'] = map_df[col].map(quantile_mapping)

    map = folium.Map(location=[37.0902, -95.7129], zoom_start=4.4)
    geo_data = json.load(open(county_geo))

    for col, color in zip(['Resilience', 'Vulnerability', 'Adaptability'], ['YlGnBu', 'Reds', 'Blues']):
        folium.Choropleth(
            geo_data=geo_data,
            data=map_df,
            columns=['FIPScounty', col + '_quantile_label_numeric'],
            key_on='feature.id',
            fill_color=color,
            fill_opacity=0.7, 
            line_opacity=0.2,
            legend_name=col + ' Score',
            smooth_factor=0,
            line_color="#0000",
            name=col + ' Score',
            show=False,
            overlay=True,
            nan_fill_color="Grey",
        ).add_to(map)

    style_function = lambda x: {'fillColor': '#ffffff', 'color':'#000000', 'fillOpacity': 0.1, 'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.50, 'weight': 0.1}

    hov = GeoJson(
        data=map_df,
        style_function=style_function, 
        control=False,
        highlight_function=highlight_function, 
        tooltip=GeoJsonTooltip(
            fields=['NAME', 'Resilience_quantile_label', 'Adaptability_quantile_label', 'Vulnerability_quantile_label'],
            aliases=['County:', 'Resilience Score:', 'Adaptability:', 'Vulnerability:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    )
    map.add_child(hov)
    map.keep_in_front(hov)
    folium.LayerControl(collapsed=False).add_to(map)

    #return map
    title = f"This displays the aggregate resilience indexes from {ini_value} to {fin_value}"
    display(HTML(f"<b><p style='text-align: center; font-size: 30px'>{title}</p></b>"))
    display(map)
    return map_df