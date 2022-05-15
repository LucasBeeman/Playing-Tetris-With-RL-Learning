from stable_baselines3.common.callbacks import BaseCallback
import os

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        with open('currentModel.txt', 'r') as f:
            self.n_calls = int(f.read()[11: len(f.read()) - 1] + '0')

#makes sure that the directories don't get recreated
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

#loads the 10,000th model and saves it to a zip file
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            #saves the model to the currentModel.txt file
            with open('currentModel.txt', 'w') as f:
                f.write('best_model_{}'.format(self.n_calls))
        return True