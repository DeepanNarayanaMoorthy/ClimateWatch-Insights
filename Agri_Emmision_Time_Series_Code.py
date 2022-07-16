import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import matplotlib
warnings.filterwarnings("ignore")
from scipy import signal
import warnings
# %matplotlib inline
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.filters.hp_filter import hpfilter
import numpy as np
from sklearn import metrics
from timeit import default_timer as timer
from statsmodels.tsa.api import Holt
from alibi_detect import od
import statsmodels.api as sm
from alibi_detect.od import SpectralResidual
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import simdkalman


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.framework.ops import disable_eager_execution


from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D


def return_series_agri(agri_file, country, factor):
    df=pd.read_csv(r"data\ClimateWatch_AgricultureProfile\\"+agri_file)
    
    for j in range(len(df.columns)):
        if (df.columns[j].strip(" ")=="short names"):
            shortnamecol=j
            break
            
    for j in range(len(df.columns)):
        if (df.columns[j].strip(" ")=="area"):
            areacol=j
            break
    # print(shortnamecol, areacol) 
    df=df.loc[(df[df.columns[areacol]] == country) & (df[df.columns[shortnamecol]] == factor)]
    df=df.dropna(axis='columns')
    df=df.iloc[:,2:].iloc[0]
    dates = pd.Series(["01/01/"+i for i in df.keys()])
    dates=list(pd.to_datetime(dates, format='%d/%m/%Y'))
    dictt={i:df.get(str(i.year)) for i in dates}
    finalseries=pd.Series(dictt)
    return(finalseries)


def generalplot_agri(seriess, titlee):
    try:
        dff=seriess
        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(
            go.Scatter(x=dff.keys(), y=dff.tolist(), name="Scatter Plot"), 
            row=1, col=1,

        )

        fig.add_trace(
            go.Bar(x=dff.keys(), y=dff.tolist(), name="Bar Chart"),
            row=2, col=1,

        )

        fig.update_layout(height=600, width=1700, title_text=titlee)
        # fig.show()
        return(fig)
    except Exception as e:
        return "Data Insufficient", str(e)




def seasonal_decom_agri(finalseries, titlee):
    try:
        result = seasonal_decompose(finalseries, model='add')

        results_df = pd.DataFrame({'trend': result.trend, 'seasonal': result.seasonal, 'resid': result.resid, 'observed': result.observed})
        f, ax = plt.subplots(4, 1, sharex=True, figsize=(20, 10))

        ax[0].plot(finalseries.keys(), results_df['trend'], 'b')
        ax[0].set_title("Seasonal Decomposition plot for "+titlee)
        ax[0].set_xlabel('Trend')
        ax[0].set_ylabel('Value')

        ax[1].plot(finalseries.keys(), results_df['seasonal'], 'b')
        ax[1].set_xlabel('Seasonal')
        ax[1].set_ylabel('Value')

        ax[2].plot(finalseries.keys(), results_df['resid'], 'b')
        ax[2].set_xlabel('Residual')
        ax[2].set_ylabel('Value')

        ax[3].plot(finalseries.keys(), results_df['observed'], 'b')
        ax[3].set_xlabel('Observed')
        ax[3].set_ylabel('Value')
        # plt.savefig(f'{title}.png')
        # f.show()
        return(f)
    except Exception as e:
        return "Data Insufficient", str(e)





def detrend_agri(seriess):
    try:
        detrended = signal.detrend(seriess.values)
        plt.figure(figsize=(15,6))
        plt.plot(detrended)
        plt.xlabel('EXINUS')
        plt.ylabel('Frequency')
        plt.title('Detrending using Scipy Signal', fontsize=16)
        # plt.show()
        return(plt)
    except Exception as e:
        return "Data Insufficient", str(e)




def autocor_agri(seriess):
    try:
        plt.rcParams.update({'figure.figsize':(15,6), 'figure.dpi':220})
        autocor = autocorrelation_plot(seriess.tolist())
        autocor.plot()
        plt.title("Autocorrelation Chart")
        return(plt)
    except Exception as e:
        return "Data Insufficient", str(e)





def cyclicvar_agri(seriess):
    try:
        EXINUS_cycle,EXINUS_trend = hpfilter(seriess, lamb=1600)
        dfff=pd.DataFrame()
        dfff['cycle'] =EXINUS_cycle
        dfff['trend'] =EXINUS_trend
        dfff[['cycle']].plot(figsize=(15,6)).autoscale(axis='x',tight=True)
        plt.title('Extracting Cyclic Variations', fontsize=16)
        plt.xlabel('Year')
        plt.ylabel('Variation')
        # plt.show()
        return(plt)
    except Exception as e:
        return "Data Insufficient", str(e)





def double_expsmooth_agri(seriess):
    try:
        finalseries=seriess
        test = finalseries.iloc[-(int(finalseries.size*(3/10))):]
        train =finalseries.iloc[:-(int(finalseries.size*(3/10)))]
        def timeseries_evaluation_metrics_func(y_true, y_pred):

            def mean_absolute_percentage_error(y_true, y_pred): 
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            # print('Evaluation metric results:-')
            # print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
            # print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
            # print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
            # print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
            # print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')

        from sklearn.model_selection import ParameterGrid
        param_grid = {'smoothing_level': [0.10, 0.20,.30,.40,.50,.60,.70,.80,.90], 'smoothing_slope':[0.10, 0.20,.30,.40,.50,.60,.70,.80,.90],
                      'damping_slope': [0.10, 0.20,.30,.40,.50,.60,.70,.80,.90],'damped' : [True, False]}
        pg = list(ParameterGrid(param_grid))

        df_results_moni = pd.DataFrame(columns=['smoothing_level', 'smoothing_slope', 'damping_slope','damped','RMSE','r2'])
        start = timer()
        for a,b in enumerate(pg):
            smoothing_level = b.get('smoothing_level')
            smoothing_slope = b.get('smoothing_slope')
            damping_slope = b.get('damping_slope')
            damped = b.get('damped')
            # print(smoothing_level, smoothing_slope, damping_slope,damped)
            fit1 = Holt(train,damped =damped ).fit(smoothing_level=smoothing_level, smoothing_slope=smoothing_slope, damping_slope = damping_slope ,optimized=False)
            #fit1.summary
            z = fit1.forecast(len(test))
            # print(z)
            df_pred = pd.DataFrame(z, columns=['Forecasted_result'])
            RMSE = np.sqrt(metrics.mean_squared_error(test, df_pred.Forecasted_result))
            r2 = metrics.r2_score(test, df_pred.Forecasted_result)
            # print( f' RMSE is {np.sqrt(metrics.mean_squared_error(test, df_pred.Forecasted_result))}')
            df_results_moni = df_results_moni.append({'smoothing_level' :smoothing_level, 
                                                      'smoothing_slope':smoothing_slope, 
                                                      'damping_slope' :damping_slope,
                                                      'damped':damped,'RMSE': RMSE,'r2':r2}, ignore_index=True)
        end = timer()
        print(f' Total time taken to complete grid search in seconds: {(end - start)}')

        # print(f' Below mentioned parameter gives least RMSE and r2')
        df_results_moni.sort_values(by=['RMSE','r2']).head(1)
        fit1 = Holt(train,damped =False ).fit(smoothing_level=0.9, smoothing_slope=0.6, damping_slope = 0.1 ,optimized=False)
        Forecast_custom_pred = fit1.forecast(len(test))
        # fit1.summary()
        timeseries_evaluation_metrics_func(test,Forecast_custom_pred)
        # Automated Parameter
        fitESAUTO = Holt(train).fit(optimized= True, use_brute = True)
        # fitESAUTO.summary()
        fitESAUTOpred = fitESAUTO.forecast(len(test))
        timeseries_evaluation_metrics_func(test,fitESAUTOpred)
        plt.rcParams["figure.figsize"] = [16,9]
        plt.plot( train, label='Train')
        plt.plot(test, label='Test')
        plt.plot(fitESAUTOpred, label='Automated grid search')
        plt.plot(Forecast_custom_pred, label='Double Exponential Smoothing with custom grid search')
        plt.legend(loc='best')
        # plt.show()
        return(plt)
    except Exception as e:
        return "Data Insufficient", str(e)




def partialautocor_agri(seriess):
    try:
        finalseries=seriess
        fig, axs = plt.subplots(2)
        sm.graphics.tsa.plot_pacf(finalseries, lags=14, ax=axs[0])
        axs[0].set_ylabel('R')
        axs[0].set_xlabel('Lag')
        sm.graphics.tsa.plot_acf(finalseries, lags=14, ax=axs[1]);
        axs[1].set_ylabel('R')
        axs[1].set_xlabel('Lag')
        fig.tight_layout()
        return(fig)
    except Exception as e:
        return "Data Insufficient", str(e)




def arima_agri(seriess):
    try:
        finalseries=seriess
        mod = sm.tsa.arima.ARIMA(endog=finalseries, order=(1, 0, 0))
        res = mod.fit()
        # print(res.summary())
        STEPS = 20
        forecasts_df = res.get_forecast(steps=STEPS).summary_frame() 
        ax = finalseries.plot(figsize=(12, 6))
        plt.ylabel('ForeCast')
        forecasts_df['mean'].plot(style='k--')
        ax.fill_between(
            forecasts_df.index,
            forecasts_df['mean_ci_lower'],
            forecasts_df['mean_ci_upper'],
            color='k',
            alpha=0.1
        )
        # plt.show()
        return(plt)
    except Exception as e:
        return "Data Insufficient", str(e)



def anomaly_agri(seriess):
    finalseries=seriess
    anomalyser=np.array(np.array(finalseries.tolist()))

    od = SpectralResidual(
     threshold=1.,
     window_amp=20,
     window_local=20,
     n_est_points=10,
     n_grad_points=5
    )
    scores = od.score(anomalyser)
    intrusion_outliers = od.predict(anomalyser)

    ax = pd.Series(anomalyser, name="data").plot(
        legend=False, figsize=(12, 6)
    )
    ax2 = ax.twinx()
    ax = pd.Series(scores, name="scores").plot(
        ax=ax2, legend=False, color="r", marker=matplotlib.markers.CARETDOWNBASE
    )
    ax.figure.legend(bbox_to_anchor=(1, 1), loc="upper left");
    # plt.show()
    return(plt)
    return scores, intrusion_outliers




def rollmean_agri(seriess, window, titlee):
    years = [r for r in range(seriess.shape[0] )]
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.ylabel('Data')
    plt.xlabel('Year Line')
    plt.title("Rolling Mean of "+ titlee +" with Window Size "+ str(window))
    ax.plot(years,seriess.values,color='grey')
    ax.plot(np.convolve(seriess, np.ones((window,))/window, mode='valid'),color='black')
    ax.set_yscale('log')
    return(plt)




def autocor_compare_agri(seriess):
    def label(ax, string):
        ax.annotate(string, (1, 1), xytext=(-8, -8), ha='right', va='top',
                    size=14, xycoords='axes fraction', textcoords='offset points')

    np.random.seed(1977)
    data = np.array(np.array(seriess.tolist()))

    fig, axes = plt.subplots(nrows=4, figsize=(8, 12))
    fig.tight_layout()

    axes[0].plot(data)
    label(axes[0], 'Raw Data')

    axes[1].acorr(data, maxlags=data.size-1)
    label(axes[1], 'Matplotlib Autocorrelation')

    tsaplots.plot_acf(data, axes[2])
    label(axes[2], 'Statsmodels Autocorrelation')

    autocorrelation_plot(data, ax=axes[3])
    label(axes[3], 'Pandas Autocorrelation')

    for ax in axes.flat:
        ax.set(title='', xlabel='')
    # plt.show()
    return(plt)




def return_series_wrt_country_agri(agri_file,  country):
    df=pd.read_csv(r"data\ClimateWatch_AgricultureProfile\\"+agri_file)
    
    for j in range(len(df.columns)):
        if (df.columns[j].strip(" ")=="short names"):
            shortnamecol=j
            break
            
    for j in range(len(df.columns)):
        if (df.columns[j].strip(" ")=="area"):
            areacol=j
            break
            
    df=df.loc[df[df.columns[areacol]] == country].drop([df.columns[areacol], df.columns[shortnamecol]], axis="columns")
    return(df)



def kalman_filter_multiple_data(data):    
    splitter=int(data.shape[1]*(7/10))-1
    X = data.iloc[:,:splitter]
    y = data.iloc[:,data.shape[1]-splitter+1:]
    X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, 
                                                      test_size=0.1, 
                                                      random_state=42)
    smoothing_factor = 5.0
    n_seasons = 7
    state_transition = np.zeros((n_seasons+1, n_seasons+1))
    state_transition[0,0] = 1
    state_transition[1,1:-1] = [-1.0] * (n_seasons-1)
    state_transition[2:,1:-1] = np.eye(n_seasons-1)
    observation_model = [[1,1] + [0]*(n_seasons-1)]
    level_noise = 0.2 / smoothing_factor
    observation_noise = 0.2
    season_noise = 1e-3
    process_noise_cov = np.diag([level_noise, season_noise] + [0]*(n_seasons-1))**2
    observation_noise_cov = observation_noise**2
    kf = simdkalman.KalmanFilter(state_transition = state_transition,
                                 process_noise = process_noise_cov,
                                 observation_model = observation_model,
                                 observation_noise = observation_noise_cov)
    result = kf.compute(X_train[0], 50)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(np.arange(1,len (X_train[0,])),X_train[0,1:], label='X')
    ax.plot(
        np.arange(len (X_train[0,])+1,len(X_train[0,])+len(y_train[0])+1),
        y_train[0],
        label='True')

    ax.plot(np.arange(len (X_train[0,])+1,len(X_train[0,])+len(result.predicted.observations.mean)+1),
            result.predicted.observations.mean,
            label='Predicted observations')
    ax.plot(np.arange(len (X_train[0,])+1,len(X_train[0,])+len(result.predicted.states.mean[:,0])+1),
            result.predicted.states.mean[:,0],
            label='Predicted states')
    ax.plot(np.arange(1,len (X_train[0,])),
            result.smoothed.observations.mean[1:],
            label='Expected Observations')
    ax.plot(np.arange(1,len (X_train[0,])),
            result.smoothed.states.mean[1:,0],
            label='States')
    ax.legend()
    ax.set_yscale('log')
    # plt.show()
    return(plt)




def RNN_forecast(finalseries):
    factor_data= pd.Series(finalseries.tolist(), index=finalseries.keys(), name="factor_data").to_frame()
    X_train, X_test, y_train, y_test = train_test_split(factor_data, factor_data.factor_data.shift(-1), shuffle=False)

    DROPOUT_RATIO = 0.1
    HIDDEN_NEURONS = 5


    def create_model(factor_data):
      scale = tf.constant(factor_data.factor_data.std())

      continuous_input_layer = keras.layers.Input(shape=1)

      categorical_input_layer = keras.layers.Input(shape=1)
      embedded = keras.layers.Embedding(factor_data.size, 5)(categorical_input_layer)
      embedded_flattened = keras.layers.Flatten()(embedded)

      year_input = keras.layers.Input(shape=1)
      year_layer = keras.layers.Dense(1)(year_input)

      hidden_output = keras.layers.Concatenate(-1)([embedded_flattened, year_layer, continuous_input_layer])
      output_layer = keras.layers.Dense(1)(hidden_output)
      output = output_layer * scale + continuous_input_layer

      model = keras.models.Model(inputs=[
        continuous_input_layer, categorical_input_layer, year_input
      ], outputs=output)

      model.compile(loss='mse', optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
      return model

    def show_result(y_test, predicted):
      plt.figure(figsize=(16, 6))
      plt.plot(y_test.index, predicted, 'o-', label="predicted")
      plt.plot(y_test.index, y_test, '.-', label="actual")

      plt.ylabel("factor_data")
      plt.legend()
      return(plt)
      # plt.show()


    disable_eager_execution()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


    factor_data = pd.Series(finalseries.tolist(), index=finalseries.keys(), name="factor_data").to_frame()
    factor_data["year"] = factor_data.index.year.values - factor_data.index.year.values.min()
    factor_data["month"] = factor_data.index.month.values - 1

    X_train, X_test, y_train, y_test = train_test_split(factor_data.astype(np.float32), 
                                                        factor_data.factor_data.shift(-1).astype(np.float32), 
                                                        shuffle=False)

    model = create_model(X_train)
    model.fit(
      (X_train["factor_data"], X_train["year"], X_train["month"]),
      y_train, epochs=1000,
      callbacks=[callback],
        verbose=0
    )
    predicted = model.predict((X_test["factor_data"], X_test["year"], X_test["month"]))

    return show_result(y_test, predicted)


def univariate_1d_cnn_model(finalseries, no_pred):
    def split_sequence(sequence, n_steps_in, n_steps_out):
      X, y = list(), list()
      for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
          break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
      return array(X), array(y)
    raw_seq = finalseries.tolist()
    n_steps_in, n_steps_out = no_pred, 2
    X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in,
    n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(105, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=7000, verbose=0)
    return(model)


def return_multiplegas_emmision(emm_file,  country, factor): 
    try:
        df=pd.read_csv(str(r"data\ClimateWatch_HistoricalEmissions\\") + emm_file)

        for j in range(len(df.columns)):
            # print(df.columns[j].strip(" ").lower())
            if (df.columns[j].strip(" ").lower()=="country"):
                countrycol=j
                break

        for j in range(len(df.columns)):
            if (df.columns[j].strip(" ").lower()=="sector"):
                sectorcol=j
                break

        for j in range(len(df.columns)):
            if (df.columns[j].strip(" ").lower()=="gas"):
                gascol=j
                break

        colls=list(df.columns)
        colls[countrycol]="area"
        colls[sectorcol]="short names"
        colls[gascol]="gas"
        df.columns=[i.lower() for i in colls]
        # pprint(df)
        df=df.loc[(df["area"] == country) & (df["short names"] == factor) ].drop(['area', 'short names', 'gas'], axis=1)
        try:
            df=df.drop(["source"], axis=1)
        except:
            pass
        return(df)
    except Exception as e:
        return "Data Incorrect", str(e)
    
def return_series_emmision(emm_file, country, factor,  gas):
    try:
        df=pd.read_csv(str(r"data\ClimateWatch_HistoricalEmissions\\") + emm_file)

        for j in range(len(df.columns)):
            # print(df.columns[j].strip(" ").lower())
            if (df.columns[j].strip(" ").lower()=="country"):
                countrycol=j
                break

        for j in range(len(df.columns)):
            if (df.columns[j].strip(" ").lower()=="sector"):
                sectorcol=j
                break

        for j in range(len(df.columns)):
            if (df.columns[j].strip(" ").lower()=="gas"):
                gascol=j
                break

        colls=list(df.columns)
        colls[countrycol]="area"
        colls[sectorcol]="short names"
        colls[gascol]="gas"
        df.columns=[i.lower() for i in colls]
        # pprint(df)
        df=df.loc[(df["area"] == country) & (df["short names"] == factor) & (df["gas"] == gas)].drop(['area', 'short names', 'gas'], axis=1)
        try:
            df=df.drop(["source"], axis=1)
        except:
            pass
        df=df.dropna(axis='columns')
        df=df.iloc[0]
        dates = pd.Series(["01/01/"+i for i in df.keys()])
        dates=list(pd.to_datetime(dates, format='%d/%m/%Y'))
        dictt={i:df.get(str(i.year)) for i in dates}
        finalseries=pd.Series(dictt)

        return(finalseries)
    except Exception as e:
        return "Data Incorrect", str(e)