"""
Critic-free training loop for Condition C.

Identical to WebRL's offpolicy_train_loop but:
  - No critic model (saves ~16GB VRAM)
  - No update_critic() phase (saves training time)
  - Advantage = MC return directly (binarized), not critic-based

The actor loss computation reuses WebRL's actor_loss() structure
but replaces the critic-based advantage with MC returns.
"""

import os
import sys
import math
import time
import copy
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from webrl.data import ReplayBuffer, DummyDataset
from webrl.environment.env_utils import add_mc_return
from webrl.misc import colorful_print

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import shutil
import deepspeed


def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            if "min" in key:
                mean_dict[key] = min(d[key] for d in dict_list if key in d)
            elif "max" in key:
                mean_dict[key] = max(d[key] for d in dict_list if key in d)
            else:
                tmp_list = [d[key] for d in dict_list if key in d]
                if len(tmp_list) != 0:
                    mean_dict[key] = sum(tmp_list) / len(tmp_list)
                else:
                    mean_dict[key] = 0
    return mean_dict


class CriticFreeTrainer:
    """
    Actor-only trainer using MC returns for advantage estimation.

    Advantage = mc_return (binarized: ≥0 → 1.0, <0 → -0.6)
    Loss = SmoothL1(β * log_ratio, advantage_binary) — same as WebRL Eq. 5

    This is equivalent to KL-constrained REINFORCE with MC returns (GRPO-style).
    """

    def __init__(self, agent, accelerator, tokenizer,
                 lm_lr=1e-5, max_grad_norm=0.01,
                 use_wandb=False, checkpointing_steps=400,
                 save_path='logs', actor_epochs=1):
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.AdamW(agent.model.parameters(), lr=lm_lr)
        self.ref_optimizer = torch.optim.AdamW(agent.ref_model.parameters(), lr=lm_lr)
        self.mse_loss = torch.nn.SmoothL1Loss(reduction='none')
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.softmax = torch.nn.Softmax(dim=-1)
        self.use_wandb = use_wandb
        self.checkpointing_steps = checkpointing_steps
        self.save_path = save_path
        self.actor_epochs = actor_epochs
        self.step = 0

    def actor_loss(self, observation, action, next_observation, mc_return, reward,
                   validation=False, **kwargs):
        """
        Compute actor loss using MC returns as advantage (no critic).

        Mirrors WebRL's actor_loss() but replaces:
            advantage = nv - v - 0.05 + reward + mc_return  (critic-based)
        with:
            advantage = mc_return  (precomputed from milestone rewards)
        """
        device = self.accelerator[0].unwrap_model(self.agent.model).device
        dtype = self.accelerator[0].unwrap_model(self.agent.model).dtype
        mc_return = torch.Tensor(mc_return).to(device, dtype=dtype).flatten()

        # Advantage from MC returns directly (no critic)
        advantage = mc_return
        advantage = torch.clamp(advantage, -1, 1)

        # Actor log-probs
        log_prob, action_attention_mask = self.agent.get_log_prob(observation, action)
        ref_log_prob = self.agent.get_log_prob_ref(observation, action).to(log_prob.device)
        non_zero_counts = action_attention_mask.sum(dim=1)

        # Binarize advantage (same as WebRL)
        advantages = advantage.flatten()
        advantages = torch.where(
            advantages >= 0, torch.tensor(1.0), torch.tensor(-0.6)
        ).to(log_prob.device)

        log_prob = log_prob.sum(dim=1).flatten()
        ref_log_prob = ref_log_prob.sum(dim=1).flatten()
        ref_prob = torch.exp(ref_log_prob / non_zero_counts)

        # Adaptive beta (same as WebRL)
        beta = torch.ones_like(advantages).to(advantages.device)
        cond1 = (advantages >= 0) & (ref_prob >= 0.8)
        cond2 = (advantages < 0) & (ref_prob < 0.9)
        beta[cond1] = 5.0
        beta[cond2] = 5.0

        # Mask: only update on positive advantages
        mask = (advantages > 0).to(dtype=log_prob.dtype, device=log_prob.device)

        non_zero_counts_mask = non_zero_counts[mask.bool()].sum()
        safe_count = torch.where(
            non_zero_counts_mask > 0, non_zero_counts_mask,
            torch.tensor(1.0, dtype=non_zero_counts_mask.dtype)
        )

        log_prob_ratio = (log_prob - ref_log_prob) / safe_count

        ratio = torch.abs(advantages / log_prob_ratio)
        beta = torch.where(beta <= ratio, beta, ratio).detach()

        loss = self.mse_loss(beta * log_prob_ratio, advantages)
        loss = torch.reciprocal(beta) * loss * mask
        loss = loss.mean()

        if not validation and loss is not None:
            self.accelerator[0].backward(loss)

        advantages = advantages.detach().cpu()
        info = {
            "pg.loss": loss.detach().cpu().item() if loss is not None else 0,
            "advantages.mean": advantages.mean(),
            "log_prob": (log_prob / safe_count).cpu().mean(),
            "ref_prob": ref_prob.cpu().mean(),
            "ref_log_prob": (ref_log_prob / safe_count).cpu().mean(),
            "log_prob_ratio": log_prob_ratio.cpu().mean(),
            "mask": mask.cpu().sum(),
            "beta": beta.cpu().mean(),
            "non_zero_counts": torch.sum(non_zero_counts).cpu(),
        }
        if validation:
            return {f"validation.{k}": v for k, v in info.items()}
        return info

    def update_policy(self, replay_buffer, validation_buffer=None):
        """Train actor using critic-free advantage (MC returns)."""
        self.step = 0
        info = {}
        info_list = []

        data = [replay_buffer.get(i) for i in range(len(replay_buffer))]
        if self.accelerator[0].is_main_process:
            print(f'Training data size: {len(data)}')
        for d in data:
            for k, v in d.items():
                d[k] = v[0]

        dataloader = DataLoader(DummyDataset(data), batch_size=replay_buffer.batch_size)
        self.agent.model.gradient_checkpointing_enable()
        self.agent.model.train()

        # Prepare with DeepSpeed (actor = accelerator[0], ref = accelerator[1])
        self.agent.model, self.lm_optimizer, dataloader = self.accelerator[0].prepare(
            self.agent.model, self.lm_optimizer, dataloader
        )
        self.agent.ref_model, self.ref_optimizer, _ = self.accelerator[1].prepare(
            self.agent.ref_model, self.ref_optimizer, dataloader
        )

        num_update_steps_per_epoch = math.ceil(
            len(dataloader) / self.accelerator[0].gradient_accumulation_steps
        )
        max_train_steps = self.actor_epochs * num_update_steps_per_epoch
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        progress_bar = tqdm(range(max_train_steps),
                           disable=not self.accelerator[0].is_local_main_process)

        valid_data = [validation_buffer.get(i) for i in range(len(validation_buffer))]
        for d in valid_data:
            for k, v in d.items():
                d[k] = v[0]
        valid_dataloader = DataLoader(DummyDataset(valid_data),
                                      batch_size=replay_buffer.batch_size)

        for epoch in range(num_train_epochs):
            for batch in dataloader:
                with self.accelerator[0].accumulate(self.agent.model):
                    info = self.actor_loss(**batch)
                    info_list.append(info)

                    self.accelerator[0].clip_grad_norm_(
                        self.agent.model.parameters(), self.max_grad_norm
                    )
                    self.lm_optimizer.step()
                    self.lm_optimizer.zero_grad()

                if self.accelerator[0].sync_gradients:
                    self.step += 1
                    progress_bar.update(1)

                    if self.step == max_train_steps - 1:
                        self.save_actor(os.path.join(self.save_path, 'actor'))

                    if self.accelerator[0].is_main_process:
                        if self.use_wandb:
                            for i in info_list:
                                wandb.log(i)
                    info.update(dict_mean(info_list))
                    info_list = []

                    if self.step % self.checkpointing_steps == 0:
                        if self.accelerator[0].is_main_process:
                            os.makedirs(
                                os.path.join(self.save_path, 'actor', f'steps_{self.step}'),
                                exist_ok=True
                            )
                        self.save_actor(
                            os.path.join(self.save_path, 'actor', f'steps_{self.step}')
                        )

                    if validation_buffer is not None and self.step % self.checkpointing_steps == 0:
                        valid_info_list = []
                        with torch.no_grad():
                            for batch in valid_dataloader:
                                valid_info_list.append(
                                    self.actor_loss(validation=True, **batch)
                                )
                        valid_info = dict_mean(valid_info_list)
                        if self.use_wandb and self.accelerator[0].is_main_process:
                            wandb.log(valid_info)

        return info

    def save_actor(self, path):
        torch.cuda.empty_cache()
        self.accelerator[0].wait_for_everyone()
        self.agent.model.save_checkpoint(path)
        if self.accelerator[0].is_main_process:
            state_dict = get_fp32_state_dict_from_zero_checkpoint(path)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            torch.save(new_state_dict, os.path.join(path, 'pytorch_actor.bin'))

            latest_path = os.path.join(path, 'latest')
            if os.path.isfile(latest_path):
                with open(latest_path, 'r') as fd:
                    tag = fd.read().strip()
                ds_checkpoint_dir = os.path.join(path, tag)
                shutil.rmtree(ds_checkpoint_dir)
        self.tokenizer.save_pretrained(path)


def critic_free_train_loop(
    agent, tokenizer, accelerator,
    batch_size=2, capacity=500000,
    lm_lr=1e-5, gamma=0.9,
    use_wandb=False, max_grad_norm=0.01,
    save_path=None, offline_data_path=None,
    checkpointing_steps=400, actor_epochs=1,
    **kwargs
):
    """
    Training loop without critic. Only 2 accelerators needed (actor, ref).

    Mirrors offpolicy_train_loop but skips critic entirely.
    """
    accelerator_actor = accelerator[0]

    trainer = CriticFreeTrainer(
        agent=agent,
        accelerator=accelerator,
        tokenizer=tokenizer,
        lm_lr=lm_lr,
        max_grad_norm=max_grad_norm,
        use_wandb=use_wandb,
        checkpointing_steps=checkpointing_steps,
        save_path=save_path,
        actor_epochs=actor_epochs,
    )

    replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)

    all_trajectories = torch.load(offline_data_path, weights_only=False)
    if accelerator_actor.is_main_process:
        print(f"Loaded {len(all_trajectories)} offline trajectories")

    if '_filter' not in offline_data_path:
        all_trajectories = [add_mc_return(t, gamma=gamma) for t in all_trajectories]

    train_trajectories = all_trajectories[:int(len(all_trajectories) * 0.95)]
    val_trajectories = all_trajectories[int(len(all_trajectories) * 0.95):]

    validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)

    if '_filter' not in offline_data_path:
        data = sum(train_trajectories, [])
        val_data = sum(val_trajectories, [])
    else:
        data = train_trajectories
        val_data = val_trajectories

    for d in data:
        if d['action'].endswith('<|eot_id|>') is False:
            d['action'] = d['action'] + '<|eot_id|>'
        replay_buffer.insert(**d)
    for d in val_data:
        validation_buffer.insert(**d)

    # No critic training — go straight to actor
    if accelerator_actor.is_main_process:
        print(">>>Training Policy (critic-free)")
    info = trainer.update_policy(replay_buffer, validation_buffer)

    return info
