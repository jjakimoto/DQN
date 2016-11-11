class DDPGConfig(object):
    def __init__(self, n_stock):
        device = '/gpu:0'
        save_path = '/home/tomoaki/work/github/DQN/DDPG_model.ckpt'
        is_load = False
        activation = 'relu'
        gamma = 1.0
        history_length = 10
        n_stock = n_stock
        n_smooth = 5
        n_down = 5
        k_w = 3
        n_hidden = 100
        n_batch = 32
        n_epochs = 100
        n_feature = 32
        alpha = 0.7
        beta = 0.5
        update_rate = 1e-2
        learning_rate = 1e-3
    
        # memory_config
        memory_length = 200
        n_memory = 10
        action_scale = 10