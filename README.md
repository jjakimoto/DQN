# Reinforcement Learning for Finance
We apply reinforcement learning for stock trading. 


## Fetch Data Example
```
import utils 
# fetch symbols from yahoo finance
symbols = utils.get_sap_symbols('sap500')
# fetch Open value from 01/04/2015 to 01/04/2016
input_data = utils.get_data_list_key('2015-04-01', '2016-04-01', symbols, 'Open')
```

We have two models:
(1) Exit Rule
Determing optimal times to sell out stocks is elementary and at the same time challenging task. I implemented an alogrithm learning optimal times to sell out stocks based on [DQN](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html). To give stability, I introduced [Double Q-Learning](https://www.aaai.org/Conferences/AAAI/2016/Papers/12vanHasselt12389.pdf).

![Result](https://github.com/jjakimoto/DQN/tree/master/assets/trade_result.jpg)

(2)Optimal Portfolio
Constructing optimal portfolio is one of the most chanllenging and interesting tasks. I implemented an algorithm to prdocue optimal portfolios based on [DDPG](
The figure is the result of prediction.
After learned with data from 01/10/2013 to 03/31/2015, predict S&P500 from 01/04/2015 to 01/10/2016.
The number of label corresponds to the number of stock data as input.

![Result](https://github.com/jjakimoto/DQN/tree/master/assets/exit_result.jpg)
