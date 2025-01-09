import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def compressibility():
    
    config = base.get_config()
    
    config.score_fixed = True
    
    config.run_name = "ddpo_compressibility"
    
    config.use_regularization = True
    config.penalty_constant = 0.1

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.
    config.sample.batch_size = 4 #8
    config.sample.num_batches_per_epoch = 4

    # this corresponds to (4 * 4) / (4 * 2) = 2 gradient updates per epoch.
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }
    return config


def incompressibility():
    config = compressibility()
    
    #config.run_name = "ddpo_incompressibility_regularization_0.1"
    config.run_name = "ddpo_incompressibility_regularization_0.1"
    
    if config.score_fixed == True:
        config.run_name += "_score"
    else:
        config.run_name += "_noise" 
    
    config.reward_fn = "jpeg_incompressibility"
    return config


def aesthetic():
    config = compressibility()
    
    config.penalty_constant = 0.0001
    
    config.run_name = "ddpo_aesthetic_regularization_0.0001"
    
    if config.score_fixed == True:
        config.run_name += "_score"
    else:
        config.run_name += "_noise" 
    
    config.num_epochs = 100
    config.reward_fn = "aesthetic_score"
    
    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.
    config.sample.batch_size = 4 #8
    config.sample.num_batches_per_epoch = 4

    # this corresponds to (4 * 4) / (4 * 2) = 2 gradient updates per epoch.
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 4

    config.prompt_fn = "simple_animals"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    return config


def prompt_image_alignment():
    config = compressibility()
    
    config.run_name = "ddpo_aesthetic_regularization_0.01"

    config.num_epochs = 200
    # for this experiment, I reserved 2 GPUs for LLaVA inference so only 6 could be used for DDPO. the total number of
    # samples per epoch is 8 * 6 * 6 = 288.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 6

    # again, this one is harder to optimize, so I used (8 * 6) / (4 * 6) = 2 gradient updates per epoch.
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 6

    # prompting
    config.prompt_fn = "nouns_activities"
    config.prompt_fn_kwargs = {
        "nouns_file": "simple_animals.txt",
        "activities_file": "activities.txt",
    }

    # rewards
    config.reward_fn = "llava_bertscore"

    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }

    return config

def compressibility_continuous():
    config = base.get_config()

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    config.sample.batch_size = 4
    config.sample.num_batches_per_epoch = 2

    # this corresponds to (4 * 2) / (2 * 2) = 2 gradient updates per epoch.
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    config.use_continuous = True
    config.penalty_alpha = 0.00001

    return config

def get_config(name):
    return globals()[name]()
