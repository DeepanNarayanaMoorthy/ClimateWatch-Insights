import requests
import shutil

emm_paramss=["CW_HistoricalEmissions_CAIT.csv", "AFG", 
 "Total excluding LUCF", "All GHG"]

agri_paramss=["CW_Agriculture_emissions.csv", "AFG", 
"total_emissions_agr_9", ""]

areas=["USA", "CAN", "AUS", "MEX"]
typee=['Building',
       'Waste ',
        'Transportation ',
        'Other Fuel Combustion ']

typee_agri=['total_emissions_agr_1',
'total_emissions_agr_10',
'total_emissions_agr_11',
'total_emissions_agr_12',
'total_emissions_agr_13']

year="2011"
single_cat_violin_emm='Other Fuel Combustion '
single_cat_violin_agri='total_emissions_agr_10'

#============ VIZ
# plott="return_radar_agri"
plott='violinplot'
# plott='seasonal_decom_agri'
# plott='get_treemap_agri'
# plott="parallel_cate_adap"
# plott="adaptation_3d"
# plott="worldmap_timeseries"

############### TIME SERIES

# plott="generalplot_agri"
# plott="seasonal_decom_agri"
# plott="detrend_agri"
# plott="autocor_agri"
# plott="cyclicvar_agri"
# plott="double_expsmooth_agri"
# plott="partialautocor_agri"
# plott="arima_agri"
# plott="anomaly_agri"
# plott="rollmean_agri"
# plott="autocor_compare_agri"
# plott="RNN_forecast"
# plott="kalman_filter_multiple_data"

partition_dict={"poverty_14":4,"climate_risks":4,"climate_risks_rank":4,"vulnerability":4,"vulnerability_rank":4,"readiness":4,"readiness_rank":4}

attributes=['poverty_14','climate_risks_rank', 'vulnerability_rank', 'readiness_rank' ]
rollmeanwindow=5

emm_dictt={
    "plot": plott,
    "indicator": "emm",
    "rollmeanwindow":rollmeanwindow,
    "paramss":emm_paramss,
    "areas":areas,
    "typee":typee,
    'year':year,
    "single_cat_violin": single_cat_violin_emm,
    "partition_dict": partition_dict,
    "attributes":attributes
}

agri_dictt={
    "plot": plott,
    "indicator": "agri",
    "rollmeanwindow":rollmeanwindow,
    "paramss":agri_paramss,
    "areas":areas,
    "typee":typee_agri,
    'year':year,
    "single_cat_violin": single_cat_violin_agri,
    "partition_dict": partition_dict,
    "attributes":attributes
}

url = 'http://127.0.0.1:5000/getjson'

x = requests.post(url, json = agri_dictt)

with open("sample.html", 'wb') as f:
    f.write(x.content)


# print(x.text)

