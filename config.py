class DDPGConfig(object):
    def __init__(self, n_stock):
        self.device = '/gpu:0'
        self.save_path = '/path/to/your/save/path/model.ckpt'
        self.is_load = False
        self.gamma = 1.0
        self.history_length = 10
        self.n_stock = n_stock
        self.n_smooth = 5
        self.n_down = 5
        self.k_w = 3
        self.n_hidden = 100
        self.n_batch = 32
        self.n_epochs = 100
        self.n_feature = 32
        self.alpha = 0.7
        self.beta = 0.5
        self.update_rate = 1e-1
        self.learning_rate = 1e-3
    
        # memory_config
        self.memory_length = 200
        self.n_memory = 10
        self.noise_scale = 0.2
        
class DQNConfig(object):
    def __init__(self, n_stock):
        self.device = '/gpu:0'
        self.save_path = '/path/to/your/save/path/model.ckpt'
        self.is_load = False
        self.gamma = 0.999
        self.history_length = 10
        self.n_stock = n_stock
        self.n_smooth = 5
        self.n_down = 5
        self.k_w = 3
        self.n_hidden = 100
        self.n_batch = 32
        self.n_epochs = 100
        self.n_feature = 32
        self.alpha = 0.7
        self.beta = 0.5
        self.update_rate = 1e-1
        self.learning_rate = 1e-3
    
        # memory_config
        self.memory_length = 200
        self.n_memory = 1
        