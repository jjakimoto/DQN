import wanings
import numpy as np
import pandas as pd
import time
# local library
import utils
from model import DDPG
from config import DDPGConfig


def main():
    st = time.time()
    symbols = utils.get_sap_symbols('sap500')
    np.random.shuffle(symbols)
    chosen_symbols = symbols[:10]
    start_date="2010-04-01"
    end_date="2016-03-31"
    # use Open data
    input_data = utils.get_data_list_key(chosen_symbols, start_date, end_date)
    elapsed = time.time() - st
    print ("time for getting data:", elapsed)

    train_st = pd.Timestamp("2010-04-01")
    train_end = pd.Timestamp("2013-03-31")
    test_st = pd.Timestamp("2013-04-01")
    test_end = pd.Timestamp("2016-03-31")

    train_input = input_data.loc[(input_data.index >= train_st) & (input_data.index <= train_end)]
    test_input = input_data.loc[(input_data.index >= test_st) & (input_data.index <= test_end)]
    
    # training
    config = DDPGConfig()
    ddpg = DDPG(config)
    values = ddpg.train(train_input)
    
    # prediction
    profit = []
    date = []
    index = test_input.index
    values = test_input.values
    old_value = values[0]
    prof = 0
    count = 0
    for i in range(1, len(index)):
        value = values[i]
        action = ddpg.predict_action(old_value)
        ddpg.update_memory(old_value, value)
        gain = np.sum((value - old_value) * action)
        prof += gain
        profit.append(prof)
        date.append(index[i])
        if count%10 == 0:
            result = pd.DataFrame(profit, index=pd.DatetimeIndex(date))
            result.to_csv("test_result.csv")
        count += 1
        if count%10 == 0:
            print('time:', index[i])
            print('portfolio:', action)
            print('profit:', prof)
        print('***************************')
        for i in range(100):
            ddpg.update_weight()
        old_value = value
    
    
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
