from yahoo_finance import Share
import numpy as np
import pandas as pd
import datetime

def get_data_by_key(key, data):
    data_it = iter(data)
    return_data = []
    flag = True
    for d in data_it:
        if key !="Date":
            return_data.append(float(d[key]))
        else:
            return_data.append(d[key])
        
    return np.array(return_data)
    
def get_data_by_list(name_list, start_date, end_date, data_type="Open"):
    share_list = []
    new_name_list = []
    for name in name_list:
        try:
            share_list.append(Share(name))
            new_name_list.append(name)
        except:
            pass
    
    stock_data_list = []
    date = []
    flag = True
    N_data = 0
    fail_idx_list = []
    fail_name_list = []
    for idx, share in enumerate(share_list):
        name = new_name_list[idx]
        try:
            hist_data = share.get_historical(start_date=start_date, end_date=end_date)[::-1]
            stock_data = map(float, get_data_by_key(key=data_type, data=hist_data))
            n_data = len(stock_data)
            if n_data == 0:
                fail_name_list.append(name)
                fail_idx_list.append(idx)
            date.append(get_data_by_key(key='Date', data=hist_data))
            stock_data_list.append(stock_data)
        except:
            pass
    print ("fail_name_list: ", fail_name_list)
    return np.array(stock_data_list), date, fail_idx_list

def get_fixed_data(name_list, start_date, end_date, data_type="Open"):
    stock_data, date, fail_idx = get_data_by_list(name_list, start_date, end_date, data_type="Open")
    count = 0
    stock_data = list(stock_data)
    date = list(date)
    for i in fail_idx:
        del stock_data[i - count]
        del date[i - count]
        del name_list[i - count]
        count += 1
        
    new_fail_idx = []
    for i in range(len(stock_data) - 1):
        if len(stock_data[i]) < len(stock_data[i + 1]):
            new_fail_idx.append(i)
        if i == len(stock_data) - 2:
            if len(stock_data[i]) > len(stock_data[i + 1]):
                new_fail_idx.append(i)
                
    count = 0         
    for i in new_fail_idx:
        del stock_data[i - count]
        del date[i - count]
        del name_list[i - count]
        count += 1
    date_label = get_datetime_list(date[0])
    stock_data = pd.DataFrame(np.array(stock_data).T, index=date_label)
    
    return stock_data, name_list   

def convert_time_format(date):
    date_tilde = date.split("-")
    date_tilde = map(int, date_tilde)
    return datetime.datetime(*date_tilde)

def get_datetime_list(date):
    date_label=[]
    for i in range(len(date)):
        date_label.append(convert_time_format(date[i]))
    return date_label