from yaml import load, dump, Loader, Dumper
import sys

class Config:
    def __init__(self, path):
        stream = open(path, 'r')
        data = load(stream, Loader=Loader)
        try:
            self.episodes = data["episodes"]
        except:
            print("please, specify a number of episodes. Aborting execution")
            return
        try:
            self.learning_rate = data["learning_rate"]
        except:
            self.learning_rate = 1e-4
        try:
            self.save_plot = data["save_plot"]
        except:
            self.save_plot = True
        try:
            self.show_plot = data["show_plot"]
        except:
            self.show_plot = False
        try:
            self.seed = data["seed"]
        except:
            self.seed = 42
        try:
            self.model_path = data["model_path"]
        except:
            self.model_path = "/models/Pong_REINFORCE_{}.torch".format(self.episodes)
        try:
            self.max_step = data["max_step"]
        except:
            self.max_step = sys.maxsize
        try:
            self.log_flush_freq = data["log_flush_freq"]
        except:
            self.log_flush_freq = 1
        try:
            self.log_window_size = data["log_window_size"]
        except:
            self.log_window_size = 100
        try:
            self.use_loaded_model = data["use_loaded_model"]
        except:
            self.use_loaded_model = False
        try:
            self.override_model = data["override_model"]
        except:
            self.override_model = False
        try:
            self.torch_rand = data["torch_rand"]
        except:
            self.torch_rand = True
        try:
            self.baseline = data["baseline"]
        except:
            self.baseline = False