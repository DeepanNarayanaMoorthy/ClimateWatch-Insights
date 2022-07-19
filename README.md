
# Climatewatch Insights

A Time-series analytics dashboard built to forecast and visualise data associated with socio-economic
indicators and historic emission of 140 nations. RNNs have been used to perform forecasting.
Seasonal decomposition and comparison of autocorrelation methods along with an API version
has been implemented to detect anomalies. Kalman filters have been used to estimate the best
developing regions by combining various emission indicators.

## About Climatewatch
ClimateWatch is an open online platform designed to empower users with the climate data, visualizations and resources they need to gather insights on national and global progress on climate change, sustainable development, and help advance the goals of the paris agreement.

## Run Locally

Clone the project

```bash
  git clone https://github.com/DeepanNarayanaMoorthy/ClimateWatch-Insights.git
```

Go to the project directory

```bash
  cd ClimateWatch-Insights
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  streamlit rum dashboard.py
```


## Acknowledgements

 - [ClimateWatch Data Source](https://www.climatewatchdata.org/)


## Tech Stack

**Client:** StreamLit

**Server:** Python, scikit-learn, Tensorflow

