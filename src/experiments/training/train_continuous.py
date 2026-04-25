# Leonardo Barazza, acse-lb1223
# the PPO implemntation in the train_continuous function was adapted from https://github.com/vwxyzjn/cleanrl

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from celare import AgentContinuous, oscill_coef
from experiments.config.config import ArgsContinuous
from experiments.utils import make_env_continuous, save_checkpoint
from datetime import datetime
import wandb

# Function implementing the training loop using PPO for continuous action spaces
def train_continuous(args, verbose=False):

    if args.agent_mode == 0: # baseline mode
        args.detach = True
    elif args.agent_mode == 1 or args.agent_mode == 2: # constant or oscillation mode
        args.detach = False

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # print the arguments and hyperparameters used for reference
    if verbose:
        print("\n" + "#" * 90)
        print("Arguments: ")
        print(args)
        print("#" * 90 + "\n")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{current_time}__{args.exp_name}__{args.env_id}__{args.seed}"
    print("Run name: ", run_name)

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env_continuous(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = AgentContinuous(envs, n_codebook_vectors=args.num_codebook_vectors, latent_dim=args.latent_dim, detach=args.detach, alpha=args.alpha).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # count the number of checkpoints saved so far
    checkpoint_counter = 1

    # update step for the oscillation mode
    update_step = 0

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        agent.eval()

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # action logic
            with torch.no_grad():
                action, logprob, _, value, _, _, _, _, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # execute the game and log data
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        wandb.log({"charts/episodic_return": info["episode"]["r"], "global_step": global_step})
                        wandb.log({"charts/episodic_length": info["episode"]["l"], "global_step": global_step})

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        agent.train()

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, quant_loss, recons_loss, clust_entropy_loss, _, _\
                    = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                freq = args.oscill_freq
                mu = args.oscill_mu
                amp = args.oscill_amp
                oscill_n = args.oscill_n

                # in the case of baseline mode, the main network is detached so the clustering
                # coeff only affects the VQ-VAE
                if args.agent_mode == 0:   # baseline
                    clust_coef = mu
                elif args.agent_mode == 1: # constant mode
                    clust_coef = mu 
                elif args.agent_mode == 2: # oscillation mode
                    clust_coef = oscill_coef(update_step, freq=freq, mean=mu, amp=amp, n=oscill_n)

                update_step += 1

                # PPO loss
                loss_ppo = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # CELaRe loss
                loss_clust = clust_coef * (1.0 * quant_loss + 1.0 * recons_loss - args.clust_entropy_coef * clust_entropy_loss)

                loss = loss_ppo + loss_clust

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        wandb.log({
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/quantization_loss": quant_loss.item(),
            "losses/reconstruction_loss": recons_loss.item(),
            "losses/vqvae_entropy_loss": clust_entropy_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "charts/update_step": update_step,
            "global_step": global_step
        })

        if verbose and iteration % 10 == 0:
            # only print stuff to check for determinisim if wandted
            print("iteration: ", iteration, "avg. return: ", y_true.mean())
            print(f"global_step: {global_step}")

        # save checkpoint
        if args.save_model and (global_step // (args.checkpoint_interval * checkpoint_counter) >= 1):
            checkpoint_counter += 1
            save_checkpoint(agent, run_name, global_step, verbose=verbose)

    # save final checkpoint
    if args.save_model:
        save_checkpoint(agent, run_name, global_step, verbose=verbose)

    envs.close()

# run experiments for continuous action space sweep
def run_continuous_sweep(config=None):
    with wandb.init(config=config):

        # Update the args with sweep parameters
        config = wandb.config

        args = tyro.cli(ArgsContinuous)

        # Update the args with sweep parameters        
        if config is not None:
            for key, value in config.items():
                setattr(args, key, value)

        # set environment specific parameters for this sweep
        if config.env_id == "Pendulum-v1":
            args.total_timesteps = 400_000
        elif config.env_id == "MountainCarContinuous-v0":
            args.total_timesteps = 700_000
        elif config.env_id == "BipedalWalker-v3":
            args.total_timesteps = 2_500_000
        elif config.env_id == "LunarLander-v2":
            args.total_timesteps = 2_500_000

        # run the training
        train_continuous(args, verbose=True)


if __name__ == "__main__":
    # use args provided by the user in the command line
    args = tyro.cli(ArgsContinuous)
    train_continuous(args=args, verbose=True)