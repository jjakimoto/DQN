class MultiDDPGConfig(object):
    activation = 'relu'
    gamma = 0.95
    history_length = 10
    n_stock = n_stock
    n_smooth = 3
    n_down = 3
    k_w = 3
    n_hidden = 100
    n_batch = 32
    n_epochs = 100
    n_feature = 5
    update_rate = 0.5
    learning_rate = 1e-3