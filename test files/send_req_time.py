import requests
import shutil

emm_paramss=["CW_HistoricalEmissions_CAIT.csv", "AFG", 
 "Total excluding LUCF", "All GHG"]

agri_paramss=["CW_Agriculture_emissions.csv", "AFG", 
"total_emissions_agr_9", ""]

# plott="generalplot_agri"
# plott="seasonal_decom_agri"
# plott="detrend_agri"
# plott="autocor_agri"
# plott="cyclicvar_agri"
plott="double_expsmooth_agri"
# plott="partialautocor_agri"
# plott="arima_agri"
# plott="anomaly_agri"
# plott="rollmean_agri"
# plott="autocor_compare_agri"
# plott="RNN_forecast"
# plott="kalman_filter_multiple_data"

emm_dictt={
    "plot": plott,
    "indicator": "emm",
    "rollmeanwindow":5,
    "paramss":emm_paramss
}

agridict_dictt={
    "plot": plott,
    "indicator": "agri",
    "rollmeanwindow":5,
    "paramss":agri_paramss
}

url = 'http://127.0.0.1:5000/getjson'

x = requests.post(url, json = agridict_dictt)

with open("sample.png", 'wb') as f:
    f.write(x.content)


# print(x.text)

