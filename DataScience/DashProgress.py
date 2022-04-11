from transformers import ProgressCallback


class DashProgress(ProgressCallback):

    def __init__(self):
        self.current_step = 0
        self.done = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.current_step += 1

    def on_train_end(self, args, state, control, **kwargs):
        self.done = True
