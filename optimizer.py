import tensorflow as tf
import keras



class WarmUpAndDecaySchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 init_lr: float,
                 num_train_steps: int,
                 num_warmup_steps: int):
        super().__init__()
        self.init_lr = init_lr
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps

    # TODO: resolve discontinuity at the end of warmup
    def __call__(self, step):
        # Warm up.
        if step < self.num_warmup_steps: 
            return self.init_lr * (step / self.num_warmup_steps)
        # Linear decay after warm up
        step = min(step, self.num_train_steps)
        learning_rate = self.init_lr * (1 - step / self.num_train_steps)
        return learning_rate
        


def get_optimizer(
        init_lr: float,
        num_train_steps: int,
        num_warmup_steps: int
):
    lr_schedule = WarmUpAndDecaySchedule(init_lr, num_train_steps, num_warmup_steps)
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, # type: ignore
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6
    )
    optimizer.exclude_from_weight_decay(var_names=["LayerNorm", "layer_norm", "bias"])
    return optimizer
