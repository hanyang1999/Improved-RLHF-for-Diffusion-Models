from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from reward_model import ValueMulti
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob, pipeline_with_logprob_regularizedReward
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import torch.nn.functional as F
import tempfile
from PIL import Image
import copy
from transformers import CLIPModel, CLIPProcessor  # pylint: disable=g-multiple-import
from transformers import CLIPTextModel, CLIPTokenizer  # pylint: disable=g-multiple-import
from collections import OrderedDict


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def bp():
    import pdb; pdb.set_trace()


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config
    debug = 0
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch", config=config.to_dict(), init_kwargs={"wandb": {"name": config.run_name, "entity": "contiRL4diffusion"}}
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
    
    pipeline_original = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
    pipeline_original.vae.requires_grad_(False)
    pipeline_original.text_encoder.requires_grad_(False)
    pipeline_original.unet.requires_grad_(False)
    
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # Load reward models
    reward_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    reward_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    reward_tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    
    reward_clip_model.requires_grad_(False)

    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained.model,
        subfolder="tokenizer",
        revision=config.pretrained.revision,
    )
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)
        pipeline_original.unet.to(accelerator.device, dtype=inference_dtype)

    reward_clip_model.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        pipeline.unet.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet
    
    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) <= 2
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) <= 2
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    def _train_value_func(value_function, samples_batched, accelerator, config):
        """Trains the value function."""
        """
        "prompt_ids": prompt_ids,
        "prompt_embeds": prompt_embeds,
        "timesteps": timesteps,
        "latents": latents[:, :-1],  # each entry is the latent before timestep t
        "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
        "log_probs": log_probs,
        "rewards": rewards,
        "rewards_regularized": None,
        "regularization": regularization_sum
        "prompt_embeds_value"
        """
        batch_state_original = samples_batched["latents"].squeeze(1)
        new_first_dim = batch_state_original.shape[0] * batch_state_original.shape[1] 
        batch_state = batch_state_original.reshape(new_first_dim, *batch_state_original.shape[2:])
        batch_timestep = samples_batched["timesteps"].squeeze(1).reshape(new_first_dim, *samples_batched["timesteps"].shape[3:])
        batch_final_reward = samples_batched["rewards"].squeeze(1)
        batch_txt_emb_value = samples_batched["prompt_embeds_value"].squeeze(1)

        batch_timestep = (batch_timestep - 1) * config.sample.num_steps / 1000
        batch_timestep = batch_timestep.int() 

        repeats = int(new_first_dim / batch_txt_emb_value.shape[0])
        repeated_txt_emb_value = []
        for i in range(batch_txt_emb_value.shape[0]):
            row = batch_txt_emb_value[i].unsqueeze(0)
            repeated_row = row.repeat(repeats, 1)
            repeated_txt_emb_value.append(repeated_row)

        batch_txt_emb_value = torch.cat(repeated_txt_emb_value, dim=0)

        pred_value = value_function(
            batch_state.cuda().detach(),
            batch_txt_emb_value.cuda().detach(),
            batch_timestep.cuda().detach()
        )
        # calculate summation of regularization reward after timestep till end
        batch_final_reward = batch_final_reward.cuda().float()

        repeated_final_reward = []
        for i in range(batch_final_reward.shape[0]):
            row = batch_final_reward[i].unsqueeze(0)
            repeated_row = row.repeat(repeats, 1)
            repeated_final_reward.append(repeated_row)

        batch_final_reward = torch.cat(repeated_final_reward, dim=0)

        # batch_final_reward += regularization_sum
        value_loss = F.mse_loss(
            pred_value.float(),
            batch_final_reward.cuda().detach())
        accelerator.backward(value_loss/config.v_step)
        del pred_value
        del batch_state
        del batch_timestep
        del batch_final_reward
        del batch_txt_emb_value
        return (value_loss.item() / config.v_step)
    

    def _train_value_func_with_regularization(value_function, samples_batched, accelerator, config):
        """Trains the value function."""
        """
        "prompt_ids": prompt_ids,
        "prompt_embeds": prompt_embeds,
        "timesteps": timesteps,
        "latents": latents[:, :-1],  # each entry is the latent before timestep t
        "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
        "log_probs": log_probs,
        "rewards": rewards,
        "rewards_regularized": None,
        "regularization": regularization_sum
        "prompt_embeds_value"
        """
        batch_state_original = samples_batched["latents"]
        new_first_dim = batch_state_original.shape[0] * batch_state_original.shape[1] 
        batch_state = batch_state_original.reshape(new_first_dim, *batch_state_original.shape[2:])
        batch_timestep = samples_batched["timesteps"].reshape(new_first_dim, *samples_batched["timesteps"].shape[2:])
        batch_final_reward = samples_batched["rewards"]
        batch_txt_emb_value = samples_batched["prompt_embeds_value"]
        batch_regularization_reward = samples_batched["traj_regularization_cumsum_inv"]

        # scale timestep to 0-50
        batch_timestep = (batch_timestep - 1) * config.sample.num_steps / 1000
        batch_timestep = batch_timestep.int() 

        # repeat txt_emb_value for each timestep
        repeats = int(new_first_dim / batch_txt_emb_value.shape[0])
        repeated_txt_emb_value = []
        for i in range(batch_txt_emb_value.shape[0]):
            row = batch_txt_emb_value[i].unsqueeze(0)
            repeated_row = row.repeat(repeats, 1)
            repeated_txt_emb_value.append(repeated_row)

        batch_txt_emb_value = torch.cat(repeated_txt_emb_value, dim=0)

        pred_value = value_function(
            batch_state.cuda().detach(),
            batch_txt_emb_value.cuda().detach(),
            batch_timestep.cuda().detach()
        )

        batch_final_reward = batch_final_reward.cuda().float()

        # repeat batch_final_reward for each timestep
        repeated_final_reward = []
        for i in range(batch_final_reward.shape[0]):
            row = batch_final_reward[i].unsqueeze(0)
            repeated_row_final_reward = row.repeat(repeats, 1)
            # repeated_row += batch_regularization_reward[i].reshape(repeated_row.shape)
            repeated_row_final_reward += - config.penalty_constant / 2.0 * batch_regularization_reward[i].reshape(repeated_row_final_reward.shape)
            repeated_final_reward.append(repeated_row_final_reward)

        batch_final_reward = torch.cat(repeated_final_reward, dim=0)

        # batch_final_reward += regularization_sum
        value_loss = F.mse_loss(
            pred_value.float(),
            batch_final_reward.cuda().detach())
        accelerator.backward(value_loss/config.v_step)
        del pred_value
        del batch_state
        del batch_timestep
        del batch_final_reward
        del batch_txt_emb_value
        return (value_loss.item() / config.v_step)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()
    
    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)


    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.per_prompt_stat_tracking.buffer_size,
            config.per_prompt_stat_tracking.min_count,
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)

    value_function = ValueMulti(50, (4, 64, 64))
    value_optimizer = torch.optim.AdamW(value_function.parameters(), lr=config.v_lr)
    value_function, value_optimizer = accelerator.prepare(
        value_function, value_optimizer
    )

    state_dict = torch.load('/home/hanyang/ddpo/logs/ddpo_incompressibility_pretrain_2024.05.18_09.54.15/checkpoints/checkpoint_18/value_func_pretrain.pth')
    if accelerator.num_processes == 1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        value_function.load_state_dict(new_state_dict)
    else:
        value_function.load_state_dict(state_dict)

    
    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Num Processes = {accelerator.num_processes}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch), # num_batches_per_epoch = 4
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
            )
            # encode prompts
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            
            
            inputs = reward_tokenizer(
                prompts,
                max_length=tokenizer.model_max_length,
                padding="do_not_pad",
                truncation=True,
            )
            input_ids = inputs.input_ids
            padded_tokens = reward_tokenizer.pad(
                {"input_ids": input_ids}, padding=True, return_tensors="pt"
            )

            txt_emb_value = reward_clip_model.get_text_features(
                input_ids=padded_tokens.input_ids.to("cuda")
            )     
                   
            # sample
            with autocast():
                if config.use_regularization:
                    images, _, latents, log_probs, all_regularization_terms = pipeline_with_logprob_regularizedReward(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=config.sample.eta,
                        output_type="pt",
                        original_unet= pipeline_original.unet,
                        debug = debug,
                    )
                else:
                    images, _, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=config.sample.eta,
                        output_type="pt",
                    )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            
            # len(all_regularization_terms) = num_steps
            if config.use_regularization:
                regularizations = torch.stack(all_regularization_terms, dim=-1) # (batch_size, num_steps)
                regularization_sum = regularizations.sum(dim=-1) # (batch_size, 1)
                # reverse and compute cumulated sum
                reversed_regularizations = torch.flip(regularizations, [1])
                reversed_regularizations = torch.cumsum(reversed_regularizations, dim=1)
                regularizations_cumsum_inv = torch.flip(reversed_regularizations, [1])
                del reversed_regularizations

            
            #print(reward_regularization) 
            
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
            # import pdb; pdb.set_trace()
            # yield to to make sure reward computation starts
            time.sleep(0)
            
            if config.use_regularization:
                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "prompt_embeds_value": txt_emb_value,
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                        "log_probs": log_probs,
                        "rewards": rewards,
                        "rewards_regularized": None,
                        "regularization": regularization_sum,
                        "traj_regularization": regularizations,
                        "traj_regularization_cumsum_inv": regularizations_cumsum_inv
                    }
                )
            else:
                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "prompt_embeds_value": txt_emb_value,
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                        "log_probs": log_probs,
                        "rewards": rewards,
                    }
                )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)
            if config.use_regularization:
                sample["rewards_regularized"] = sample["rewards"] - config.penalty_constant / 2.0 * sample["regularization"]
        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            accelerator.log(
                {
                    "images": [
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                        for i, (prompt, reward) in enumerate(zip(prompts, rewards))  # only log rewards from process 0
                    ],
                },
                step=global_step,
            )

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        # log rewards and images
        accelerator.log(
            {"reward": rewards, "epoch": epoch, "reward_mean": rewards.mean(), "reward_std": rewards.std()},
            step=global_step,
        )
        
        # Modify the rewards with regularization
        if config.use_regularization:
            rewards = accelerator.gather(samples["rewards_regularized"]).cpu().numpy()
        
        # import pdb;pdb.set_trace()
        
        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # import pdb;pdb.set_trace()
        
        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        # del samples["rewards"]
        # del samples["rewards_regularized"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
        assert num_timesteps == config.sample.num_steps


        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            
            # Training Value Functions given samples
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples_pm = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples_pm[key] = samples_pm[key][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]
            
            if config.use_regularization:
                samples_pm["traj_regularization"] = samples_pm["traj_regularization"][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]
                samples_pm["traj_regularization_cumsum_inv"] = samples_pm["traj_regularization_cumsum_inv"][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]


            samples_pm_batched = {k: v for k, v in samples_pm.items()}

            if config.v_flag == 1:
                tot_val_loss = 0
                value_optimizer.zero_grad()
                if config.use_regularization:
                    for v_step in range(config.v_step):
                        if v_step < config.v_step-1:
                            with accelerator.no_sync(value_function):
                                tot_val_loss += _train_value_func_with_regularization(
                                    value_function, samples_pm_batched, accelerator, config
                                )
                        else:
                            tot_val_loss += _train_value_func_with_regularization(
                                value_function, samples_pm_batched, accelerator, config
                            )
                else:
                    for v_step in range(config.v_step):
                        if v_step < config.v_step-1:
                            with accelerator.no_sync(value_function):
                                tot_val_loss += _train_value_func(
                                    value_function, samples_pm_batched, accelerator, config
                                )
                        else:
                            tot_val_loss += _train_value_func(
                                value_function, samples_pm_batched, accelerator, config
                            )
                value_optimizer.step()
                value_optimizer.zero_grad()
                if accelerator.is_main_process:
                    print("value_loss", tot_val_loss)
                    accelerator.log({"value_loss": tot_val_loss}, step=inner_epoch)
                del tot_val_loss
                torch.cuda.empty_cache()

            
            # rebatch for training
            samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]
              
            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):  
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        with autocast():
                            if config.train.cfg:
                                noise_pred = unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
                                    noise_pred_text - noise_pred_uncond
                                )
                            else:
                                noise_pred = unet(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample
                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )
                        
                        # Add value function
                        scaled_timestep = ((sample["timesteps"][:,j] - 1) * 50 / 1000).int().cuda()
                        value_func = value_function(sample["latents"][:,j], sample["prompt_embeds"][:,j], scaled_timestep)
                        
                        # calculate next latent's value function
                        next_timestep = torch.clamp(scaled_timestep - 1, min=0).cuda()
                        value_func_next = value_function(sample["next_latents"][:,j], sample["prompt_embeds"][:,j], next_timestep)
                        if j==num_train_timesteps-1:
                            value_func_next = sample["rewards"][:]
                        
                        # ppo logic
                        # advantages = torch.clamp(
                        #     value_func_next - value_func, -config.train.adv_clip_max, config.train.adv_clip_max
                        # )
                        
                        sample_regularization = sample["traj_regularization"][:,j]
                        # TD_error = sample_regularization + value_func_next - value_func
                        TD_error = sample["rewards"][:] + sample["traj_regularization_cumsum_inv"][:,j] - value_func
                        advantages = TD_error # shall we scale with mean and std like for DDPO original implementation?
                        
                        # advantages = torch.clamp(
                        #         advantages, -config.train.adv_clip_max, config.train.adv_clip_max)
                        
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()


if __name__ == "__main__":
    app.run(main)
