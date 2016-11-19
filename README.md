# Reinforcement Learning for Finance
Generate Financial Indicator like S&P500 based on selected stock data


## Fetch Data Example
```
import utils 
# fetch symbols from yahoo finance
symbols = utils.get_sap_symbols('sap500')
# fetch Open value from 01/04/2015 to 01/04/2016
input_data = utils.get_data_list_key('2015-04-01', '2016-04-01', symbols, 'Open')
# fetch Open S&P500
target_data = utils.get_data('^GSPC', start_date, end_date)['Open']
```

For Regularization, used [Batch Normalization](https://arxiv.org/pdf/1502.03167v3.pdf) and [Drop Out](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf).

To determine which regularization and hyper parameters, you can use [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).
```
import config
best_conf = config.random_search(train_input, train_target, test_input, test_target, '/path/to/your/save/directory', '/gpu:0')
```

![Result](https://github.com/jjakimoto/Indicator_Analysis/blob/master/assets/compare.jpg)

The figure is the result of prediction.
After learned with data from 01/10/2013 to 03/31/2015, predict S&P500 from 01/04/2015 to 01/10/2016.
The number of label corresponds to the number of stock data as input.
