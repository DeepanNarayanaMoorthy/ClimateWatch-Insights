import sys
sys.path.insert(0, 'E:\BOOKSDUMP2\ClimateWatch_FinalNoteBooks\Time Series Code\\')
sys.path.insert(0, 'E:\BOOKSDUMP2\ClimateWatch_FinalNoteBooks\Viz Code\\')

from Agri_Emmision_Time_Series_Code import *
from Agri_Emmision_Visualization_Code import *

import io
import base64
from flask import send_file
from flask import Flask, request, json, render_template

app = Flask(__name__)

def sendbyteimage(plott):
    bytes_image = io.BytesIO()
    plott.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return send_file(bytes_image,
        attachment_filename='plot.png',
        mimetype='image/png')

def agri_emmision_series(json_data, paramss1, paramss2, paramss3, paramss4):
    if(json_data["indicator"]=="emm"):
        return return_series_emmision(paramss1, paramss2,
                                        paramss3, paramss4)
    elif(json_data["indicator"]=="agri"):
        return return_series_agri(paramss1, paramss2,
                                paramss3)

def agri_emmision_multiple(json_data, paramss1, paramss2, paramss3):
    if(json_data["indicator"]=="emm"):
        return return_multiplegas_emmision(paramss1, paramss2,
                                                paramss3)
    elif(json_data["indicator"]=="agri"):
        return return_series_wrt_country_agri(paramss1, paramss2)

def agri_dict_viz_data(json_data, paramss1, paramss2, paramss3, paramss4):
    if(json_data["indicator"]=="emm"):
        return return_vizdata_emmision(paramss1, paramss2, paramss3, paramss4)
    elif(json_data["indicator"]=="agri"):  
          df=pd.read_csv("data\ClimateWatch_AgricultureProfile\\"+paramss1)
          return(df)


                

@app.route('/getjson', methods=['GET', 'POST'])
def my_route():
    json_data = request.get_json()

    if(json_data["plot"]=="generalplot_agri"):
        paramss=json_data["paramss"]
        try:
            # finalseries=return_series_emmision(paramss[0], paramss[1],
            #                                     paramss[2], paramss[3])
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            return(str(generalplot_agri(finalseries, str(paramss)).to_html(full_html=True)))
        except Exception as e:
            return({"error": str(e)})

    if(json_data["plot"]=="seasonal_decom_agri"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            plott=seasonal_decom_agri(finalseries, str(paramss))
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})

    if(json_data["plot"]=="detrend_agri"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            plott=detrend_agri(finalseries)
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})

    if(json_data["plot"]=="autocor_agri"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            plott=autocor_agri(finalseries)
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})

    if(json_data["plot"]=="cyclicvar_agri"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            plott=cyclicvar_agri(finalseries)
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})

    if(json_data["plot"]=="double_expsmooth_agri"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            plott=double_expsmooth_agri(finalseries)
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})

    if(json_data["plot"]=="partialautocor_agri"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            plott=partialautocor_agri(finalseries)
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})

    if(json_data["plot"]=="partialautocor_agri"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            plott=partialautocor_agri(finalseries)
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})

    if(json_data["plot"]=="arima_agri"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            plott=arima_agri(finalseries)
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})

    if(json_data["plot"]=="anomaly_agri"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            plott=anomaly_agri(finalseries)
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})

    if(json_data["plot"]=="rollmean_agri"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            plott=rollmean_agri(finalseries, json_data["rollmeanwindow"], str(paramss))
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})

    if(json_data["plot"]=="autocor_compare_agri"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            plott=autocor_compare_agri(finalseries)
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})


    if(json_data["plot"]=="RNN_forecast"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_series(json_data, paramss[0], paramss[1],
                                                paramss[2], paramss[3])
            plott=RNN_forecast(finalseries)
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})

    if(json_data["plot"]=="kalman_filter_multiple_data"):
        paramss=json_data["paramss"]
        try:
            finalseries=agri_emmision_multiple(json_data, paramss[0], paramss[1],
                                                paramss[2])
            plott=kalman_filter_multiple_data(finalseries)
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(sys.exc_info())})

#============VIZ PART STARTS==============

    if(json_data["plot"]=="return_radar_agri"):
        paramss=json_data["paramss"]
        try:
            df=agri_dict_viz_data(json_data, paramss[0], paramss[1], paramss[2], paramss[3])
            finalseries=radar_bar_dict_agri(df, json_data['areas'], json_data['typee'], json_data['year'])
            return(str(return_radar_bar_agri(finalseries, True).to_html(full_html=True)))
        except Exception as e:
            return({"error": str(e)})

    if(json_data["plot"]=="return_bar_agri"):
        paramss=json_data["paramss"]
        try:
            df=agri_dict_viz_data(json_data, paramss[0], paramss[1], paramss[2], paramss[3])
            finalseries=radar_bar_dict_agri(df, json_data['areas'], json_data['typee'], json_data['year'])
            return(str(return_radar_bar_agri(finalseries, False).to_html(full_html=True)))
        except Exception as e:
            return({"error": str(e)})

    if(json_data["plot"]=="violinplot"):
        paramss=json_data["paramss"]
        try:
            df=agri_dict_viz_data(json_data, paramss[0], paramss[1], paramss[2], paramss[3])
            # print(len(df))
            df=df.fillna(1)
            df.to_html('Student.html')
            plott=violin_agri(df, json_data['areas'], json_data['single_cat_violin'])
            print(plott)
            return sendbyteimage(plott)
        except Exception as e:
            return({"error": str(e)})

    if(json_data["plot"]=="get_treemap_agri"):
        paramss=json_data["paramss"]
        try:
            df=agri_dict_viz_data(json_data, paramss[0], paramss[1], paramss[2], paramss[3])
            df=treemap_agri_data(df, json_data['typee'][:2], json_data['year'])

            return(str(get_treemap_agri(df, json_data['typee'][:2], json_data['year']).to_html()))
        except Exception as e:
            return({"error": str(e)})

    if(json_data["plot"]=="parallel_cate_adap"):
        try:
            filename="data\ClimateWatch_Adaptation\CW_adaptation.csv"
            plott=parallel_cate_adap(filename, json_data['partition_dict'])
            return(str(plott.to_html()))
        except Exception as e:
            return({"error": str(e)})

    if(json_data["plot"]=="adaptation_3d"):
        try:
            filename="data\ClimateWatch_Adaptation\CW_adaptation.csv"
            df=pd.read_csv(filename).fillna(0)
            fig = px.scatter_3d(df, x=json_data['attributes'][0], 
                            y=json_data['attributes'][1], z=json_data['attributes'][2],
                                color=json_data['attributes'][3], hover_name ="country") 

            return(str(fig.to_html()))
        except Exception as e:
            return({"error": str(e)})


    if(json_data["plot"]=="worldmap_timeseries"):
        paramss=json_data["paramss"]
        try:
            df=agri_dict_viz_data(json_data, paramss[0], paramss[1], paramss[2], paramss[3])
            fig=worldmap_timeseries(df, json_data["typee"][0])
            return(str(fig.to_html()))
        except Exception as e:
            return({"error": str(e)})



if __name__ == '__main__':   
    app.run(debug=True)