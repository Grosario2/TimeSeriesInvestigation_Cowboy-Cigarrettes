import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

cigData = pd.read_csv('CowboyCigsData.csv')

cigData.drop(columns=['Unnamed: 0'], inplace=True)

cigData.rename(columns={'Time':'Month'},inplace=True)

cigData['Month'] = pd.to_datetime(cigData['Month'],format= '%Y-%m', errors='coerce')

cigData.set_index('Month',drop=True,inplace=True)

y = cigData['#CigSales']

cigSales_log = np.log(y)

cigSales_log_diff = cigSales_log.diff().dropna()

# Make a function called evaluate_arima_model to find the MSE of a single ARIMA model 
def evaluate_arima_model(dataset, order):
    # Needs to be an integer because it is later used as an index.
    # Use int()
    split=int(len(dataset) * 0.8) 
    # Make train and test variables, with 'train, test'
    train, test = dataset[0:split], dataset[split:len(dataset)]
    past=[x for x in train]
    test.reset_index(drop=True, inplace=True)
    # make predictions
    predictions = list()
    for i in range(len(test)):#timestep-wise comparison between test data and one-step prediction ARIMA model. 
        model = ARIMA(past, order=arima_order)
        model_fit = model.fit(disp=0)
        future = model_fit.forecast()[0]
        predictions.append(future)
        past.append(test[i])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    # Return the error
    return error
    
# Make a function called evaluate_models to evaluate different ARIMA models with several different p, d, and q values.
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    # Iterate through p_values
    for p in p_values:
        # Iterate through d_values
        for d in d_values:
            # Iterate through q_values
            for q in q_values:
                # p, d, q iterator variables in that order
                order = (p,d,q)
                try:
                    # Make a variable called mse for the Mean squared error
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    return print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    
# Now, we choose a couple of values to try for each parameter.
p_values = [x for x in range(0, 3)]
d_values = [x for x in range(0, 3)]
q_values = [x for x in range(0, 3)]

# Finally, we can find the optimum ARIMA model for our data.
# Nb. this can take a while...!
import warnings
warnings.filterwarnings("ignore")

evaluate_models(cigSales_log.values, p_values, d_values, q_values)    
    
