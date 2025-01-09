import math


# Define decay functions
def constant(initial_value):
    return initial_value

def linear_decay(initial_value, decay_rate, step):
    return initial_value - decay_rate * step

def cosine_decay(initial_value, step, max_steps):
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    return initial_value * cosine_decay

def exponential_decay(initial_value, decay_rate, step):
    return initial_value * (decay_rate ** step)

def step_decay(initial_value, decay_rate, step, decay_steps):
    factor = step // decay_steps
    return initial_value * (decay_rate ** factor)

def polynomial_decay(initial_value, step, max_steps, power=2.0):
    if step > max_steps:
        return 0.0
    return initial_value * ((1 - step / max_steps) ** power)


# Function to get the decayed value based on decay type
def get_decayed_value(step, config):
    decay_type = config.sample.decay.type
    initial_value = config.sample.decay.initial_value

    if decay_type == "constant":
        return constant(initial_value)
    elif decay_type == "linear":
        decay_rate = config.sample.decay.linear.decay_rate
        return linear_decay(initial_value, decay_rate, step)
    elif decay_type == "cosine":
        max_steps = config.sample.decay.cosine.max_steps
        return cosine_decay(initial_value, step, max_steps)
    elif decay_type == "exponential":
        decay_rate = config.sample.decay.exponential.decay_rate
        return exponential_decay(initial_value, decay_rate, step)
    elif decay_type == "step":
        decay_rate = config.sample.decay.step.decay_rate
        decay_steps = config.sample.decay.step.decay_steps
        return step_decay(initial_value, decay_rate, step, decay_steps)
    elif decay_type == "polynomial":
        max_steps = config.sample.decay.polynomial.max_steps
        power = config.sample.decay.polynomial.power
        return polynomial_decay(initial_value, step, max_steps, power)
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")