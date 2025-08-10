import torch
from torch import nn
import gym
import torch.optim as optim
import time
import numpy as np
import os
from copy import deepcopy
from tensordict import TensorDict
from einops import rearrange

from finworld.registry import TRAINER
from finworld.registry import ENVIRONMENT
from finworld.environment import make_env
from finworld.utils import build_storage, PortfolioRecords
from finworld.log import logger

@TRAINER.register_module(force=True)
class PPOPortfolioTrainer():

    def __init__(self,
                 config = None,
                 dataset = None,
                 agent = None,
                 metrics = None,
                 device = None,
                 dtype = None,
                 **kwargs):
        self.config = config

        train_environment_config = deepcopy(config.train_environment)
        train_environment_config.update({"dataset": dataset})
        self.train_environment = ENVIRONMENT.build(train_environment_config)

        valid_environment_config = deepcopy(config.valid_environment)
        valid_environment_config.update({"dataset": dataset})
        self.valid_environment = ENVIRONMENT.build(valid_environment_config)

        test_environment_config = deepcopy(config.test_environment)
        test_environment_config.update({"dataset": dataset})
        self.test_environment = ENVIRONMENT.build(test_environment_config)

        env_name = self.train_environment.__class__.__name__

        self.train_environments = gym.vector.AsyncVectorEnv([
            make_env(env_name, env_params=dict(env=deepcopy(self.train_environment),
                                                   transition_shape=config.transition_shape, seed=config.seed + i)) for
            i in range(config.num_envs)
        ])

        self.valid_environments = gym.vector.AsyncVectorEnv([
            make_env(env_name, env_params=dict(env=deepcopy(self.valid_environment),
                                                   transition_shape=config.transition_shape, seed=config.seed + i)) for
            i in range(1)
        ])

        self.test_environments = gym.vector.AsyncVectorEnv([
            make_env(env_name, env_params=dict(env=deepcopy(self.test_environment),
                                                   transition_shape=config.transition_shape, seed=config.seed + i)) for
            i in range(1)
        ])

        self.metrics = metrics
        self.agent = agent

        self.policy_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, list(agent.actor.parameters())),
                                       lr=config.policy_learning_rate, eps=1e-5, weight_decay=0)
        self.value_optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(agent.critic.parameters())),
                                     lr=config.value_learning_rate, eps=1e-5)

        self.device = device
        self.dtype = dtype

        self.global_step = 0
        self.check_index = 0
        self.save_index = 0

        self.storage = self.set_storage()

        self.use_bc = self.config.use_bc

    def __str__(self):
        return f"PPOPortfolioTrainer(task_type={self.config.task_type})"

    def __repr__(self):
        return self.__str__()

    def set_storage(self):

        transition = self.config.transition
        transition_shape = self.config.transition_shape

        storage = TensorDict({}, batch_size=self.config.num_steps).to(self.device)
        for name in transition:
            assert name in transition_shape
            shape = (self.config.num_steps, * transition_shape[name]["shape"])
            type = transition_shape[name]["type"]
            storage[name] = build_storage(shape, type, self.device)

        return storage

    def flatten_storage(self, storage):
        flat_storage = {}
        for key, value in storage.items():
            flat_storage[key] = rearrange(value, 'b n ... -> (b n) ...')
        flat_storage = TensorDict(flat_storage, batch_size=self.config.num_steps * self.config.num_envs).to(self.device)
        return flat_storage

    def explore_environment(self,
                            init_state = None,
                            init_info = None,
                            reset = False):
        mode = "train"
        if reset:
            state, info = self.train_environments.reset()
        else:
            state, info = init_state, init_info

        # To TensorDict
        next_obs = TensorDict({key: torch.Tensor(value) for key, value in state.items()},
                              batch_size=self.config.num_envs).to(self.device)

        next_done = torch.zeros(self.config.num_envs).to(self.device)

        # Exploring the environment
        for step in range(0, self.config.num_steps):
            self.global_step += 1 * self.config.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():

                # Construct the input tensor
                input_tensor = TensorDict(
                    {
                        "dense": next_obs["features"],
                        "sparse": next_obs["times"].to(torch.int32),
                        "cashes": next_obs["policy_portfolio_cashes"],
                        "positions": next_obs["policy_portfolio_positions"].to(torch.float32),
                        "actions": next_obs["policy_portfolio_actions"].to(torch.float32),
                        "rets": next_obs["policy_portfolio_rets"],
                    }, batch_size=next_obs.batch_size
                ).to(next_obs.device)

                action, logprob, _, value = self.agent.get_action_and_value(input_tensor)

                self.storage["training_values"][step] = value.flatten()

            for key, value in next_obs.items():
                self.storage[key][step] = value
            self.storage["training_dones"][step] = next_done
            self.storage["training_actions"][step] = action
            self.storage["training_logprobs"][step] = logprob

            if self.use_bc:
                expert_action = torch.tensor(info["expert_action"]).to(self.device)
                self.storage["training_expert_actions"][step] = expert_action

            next_obs, reward, done, truncted, info = self.train_environments.step(action.cpu().numpy())
            next_obs = TensorDict({key: torch.Tensor(value) for key, value in next_obs.items()},
                                  batch_size=self.config.num_envs).to(self.device)

            reward = torch.tensor(reward).to(self.device).view(-1)
            self.storage["training_rewards"][step] = reward

            next_done = torch.Tensor(done).to(self.device)

            if "final_info" in info:
                for info_item in info["final_info"]:
                    if info_item is not None:
                        logger.info(f"| global_step={self.global_step}, total_return={info_item['total_return']:.4f}, total_profit = {info_item['total_profit']:.4f}")

                        metrics = {
                            f"{mode}/total_return": info_item["total_return"],
                            f"{mode}/total_profit": info_item["total_profit"],
                        }

                        logger.log_metric(metrics)

        # Bootstrap value if not done
        with torch.no_grad():
            input_tensor = TensorDict(
                {
                    "dense": next_obs["features"],
                    "sparse": next_obs["times"].to(torch.int32),
                    "cashes": next_obs["policy_portfolio_cashes"],
                    "positions": next_obs["policy_portfolio_positions"].to(torch.float32),
                    "actions": next_obs["policy_portfolio_actions"].to(torch.float32),
                    "rets": next_obs["policy_portfolio_rets"],
                }, batch_size=next_obs.batch_size
            ).to(next_obs.device)
            next_value = self.agent.get_value(input_tensor).reshape(1, -1)
            lastgaelam = 0
            for t in reversed(range(self.config.num_steps)):
                if t == self.config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.storage["training_dones"][t + 1]
                    nextvalues = self.storage["training_values"][t + 1]

                delta = self.storage["training_rewards"][t] + self.config.gamma * nextvalues * nextnonterminal - self.storage["training_values"][t]

                # Expert-guided advantage computation
                if self.use_bc:  # Use behavioral cloning
                    expert_action = self.storage["training_expert_actions"][t].unsqueeze(-1).long()
                    current_action = self.storage["training_actions"][t].unsqueeze(-1).long()

                    # Calculate the log-probability of the expert action
                    _, expert_logprob, _, _ = self.agent.get_action_and_value(input_tensor, expert_action)
                    _, current_logprob, _, _ = self.agent.get_action_and_value(input_tensor, current_action)

                    # Expert advantage as log-prob difference
                    expert_advantage = (expert_logprob - current_logprob).detach()

                    # KL divergence penalty
                    kl_divergence = torch.mean(expert_logprob - current_logprob).detach()
                else:
                    expert_advantage = 0.0
                    kl_divergence = 0.0

                # Combine standard and expert advantages with KL penalty
                combined_advantage = delta + self.config.expert_weight * expert_advantage - self.config.kl_penalty_weight * kl_divergence

                # Generalized Advantage Estimation (GAE)
                self.storage["training_advantages"][t] = lastgaelam = combined_advantage + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
            self.storage["training_returns"] = self.storage["training_advantages"] + self.storage["training_values"]

    def update_value(self, flat_storage, b_inds, info):
        v_loss = info["v_loss"]
        value = info["value"]

        for start in range(0, self.config.batch_size, self.config.value_minibatch_size):
            end = start + self.config.value_minibatch_size
            mb_inds = b_inds[start:end]

            input_tensor = TensorDict({
                "dense": flat_storage["features"][mb_inds],
                "sparse": flat_storage["times"][mb_inds].to(torch.int32),
                "cashes": flat_storage["policy_portfolio_cashes"][mb_inds],
                "positions": flat_storage["policy_portfolio_positions"][mb_inds].to(torch.float32),
                "actions": flat_storage["policy_portfolio_actions"][mb_inds].to(torch.float32),
                "rets": flat_storage["policy_portfolio_rets"][mb_inds],
            }, batch_size=self.config.value_minibatch_size).to(self.device)
            newvalue = self.agent.get_value(input_tensor)

            value = newvalue.view(-1).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if self.config.clip_vloss:
                v_loss_unclipped = (newvalue - flat_storage["training_returns"][mb_inds]) ** 2
                v_clipped = flat_storage["training_values"][mb_inds] + torch.clamp(
                    newvalue - flat_storage["training_values"][mb_inds],
                    -self.config.clip_coef,
                    self.config.clip_coef,
                )
                v_loss_clipped = (v_clipped - flat_storage["training_returns"][mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - flat_storage["training_returns"][mb_inds]) ** 2).mean()

            loss = v_loss * self.config.vf_coef

            self.value_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
            self.value_optimizer.step()

        res_info = {
            "v_loss": v_loss,
            "value": value.mean(),
        }

        return res_info

    def update_policy(self,
                      flat_storage,
                      b_inds,
                      info):
        policy_update_steps = info["policy_update_steps"]
        clipfracs = info["clipfracs"]
        kl_explode = info["kl_explode"]
        pg_loss = info["pg_loss"]
        entropy_loss = info["entropy_loss"]
        if self.use_bc:
            bc_loss = info["bc_loss"]
        total_approx_kl = info["total_approx_kl"]
        logprob = info["logprob"]

        self.policy_optimizer.zero_grad()

        # update policy
        for start in range(0, self.config.batch_size, self.config.policy_minibatch_size):
            if policy_update_steps % self.config.gradient_checkpointing_steps == 0:
                total_approx_kl = 0
            policy_update_steps += 1
            end = start + self.config.policy_minibatch_size

            mb_inds = b_inds[start:end]

            input_tensor = TensorDict({
                "dense": flat_storage["features"][mb_inds],
                "sparse": flat_storage["times"][mb_inds].to(torch.int32),
                "cashes": flat_storage["policy_portfolio_cashes"][mb_inds],
                "positions": flat_storage["policy_portfolio_positions"][mb_inds].to(torch.float32),
                "actions": flat_storage["policy_portfolio_actions"][mb_inds].to(torch.float32),
                "rets": flat_storage["policy_portfolio_rets"][mb_inds],
            }, batch_size=self.config.policy_minibatch_size).to(self.device)

            input_actions = flat_storage["training_actions"][mb_inds].to(torch.float32)

            _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(input_tensor, input_actions)

            logprob = newlogprob.view(-1).mean()

            logratio = newlogprob - flat_storage["training_logprobs"][mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                total_approx_kl += approx_kl / self.config.gradient_checkpointing_steps
                clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]

            mb_advantages = flat_storage["training_advantages"][mb_inds]
            if self.config.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            if self.use_bc:
                input_tensor = TensorDict(
                    {
                        "dense": flat_storage["features"][mb_inds],
                        "sparse": flat_storage["times"][mb_inds].to(torch.int32),
                        "cashes": flat_storage["expert_trading_cashes"][mb_inds],
                        "positions": flat_storage["expert_trading_positions"][mb_inds].to(torch.int32),
                        "actions": flat_storage["expert_trading_actions"][mb_inds].to(torch.int32),
                        "rets": flat_storage["expert_trading_rets"][mb_inds],
                    }, batch_size=self.config.policy_minibatch_size).to(self.device
                )
                input_actions = flat_storage["training_expert_actions"][mb_inds].unsqueeze(-1).long()
                _, expert_logprobs, _, _ = self.agent.get_action_and_value(input_tensor, input_actions)
                bc_loss = - expert_logprobs.mean()

            entropy_loss =  - entropy.mean()
            loss = pg_loss + self.config.ent_coef * entropy_loss

            if self.use_bc:
                loss = loss + self.config.bc_coef * bc_loss

            loss /= self.config.gradient_checkpointing_steps

            loss.backward()

            if policy_update_steps % self.config.gradient_checkpointing_steps == 0:
                if self.config.target_kl is not None:
                    if total_approx_kl > self.config.target_kl:
                        self.policy_optimizer.zero_grad()
                        kl_explode = True
                        policy_update_steps -= self.config.gradient_checkpointing_steps
                        break

                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                self.policy_optimizer.zero_grad()

        info = {
            "policy_update_steps": policy_update_steps,
            "clipfracs": clipfracs,
            "kl_explode": kl_explode,
            "old_approx_kl": old_approx_kl,
            "approx_kl": approx_kl,
            "total_approx_kl": total_approx_kl,
            "pg_loss": pg_loss,
            "entropy_loss": entropy_loss,
            "logprob": logprob,
        }

        if self.use_bc:
            info["bc_loss"] = bc_loss

        return info

    def train(self):
        start_time = time.time()

        state, info = self.train_environments.reset()

        num_updates = self.config.total_steps // self.config.batch_size
        warm_up_updates = self.config.warm_up_steps // self.config.batch_size

        is_warmup = True
        mode = "train"
        for update in range(1, num_updates + 1 + warm_up_updates):
            if is_warmup and update > warm_up_updates:
                is_warmup = False

            # Annealing the rate if instructed to do so.
            if self.config.anneal_lr and not is_warmup:
                frac = 1.0 - (update - 1.0 - warm_up_updates) / num_updates
                self.policy_optimizer.param_groups[0]["lr"] = frac * self.config.policy_learning_rate
                self.value_optimizer.param_groups[0]["lr"] = frac * self.config.value_learning_rate

            # Explore the environment to collect data
            self.explore_environment(init_state=state, init_info=info, reset=False)

            # Flatten the storage
            flat_storage = self.flatten_storage(self.storage)
            b_inds = np.arange(self.config.batch_size)

            portfolio_records = {
                "clipfracs": [],
                "kl_explode": False,
                "policy_update_steps": 0,
                "entropy_loss": torch.tensor(0),
                "old_approx_kl": torch.tensor(0),
                "approx_kl": torch.tensor(0),
                "total_approx_kl": torch.tensor(0),
                "v_loss": torch.tensor(0),
                "pg_loss": torch.tensor(0),
                "value": torch.tensor(0),
                "logprob": torch.tensor(0),
            }
            if self.use_bc:
                portfolio_records["bc_loss"] = torch.tensor(0)

            for epoch in range(self.config.update_epochs):
                if portfolio_records["kl_explode"]:
                    break

                np.random.shuffle(b_inds)

                # Update value
                update_value_res_info = self.update_value(flat_storage, b_inds, portfolio_records)
                portfolio_records.update(update_value_res_info)

                if is_warmup:
                    continue

                # Update policy
                update_policy_info = self.update_policy(flat_storage,
                                          b_inds,
                                          portfolio_records)
                portfolio_records.update(update_policy_info)

            y_pred, y_true = flat_storage["training_values"].cpu().numpy(), flat_storage["training_returns"].cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if len(portfolio_records["clipfracs"]) == 0:
                num_clipfracs = 0
            else:
                num_clipfracs = np.mean(portfolio_records["clipfracs"])

            metrics = {
                f"{mode}/policy_learning_rate": self.policy_optimizer.param_groups[0]["lr"],
                f"{mode}/value_learning_rate": self.value_optimizer.param_groups[0]["lr"],
                f"{mode}/value_loss": portfolio_records["v_loss"].item(),
                f"{mode}/policy_loss": portfolio_records["pg_loss"].item(),
                f"{mode}/value": portfolio_records["value"].item(),
                f"{mode}/logprob": portfolio_records["logprob"].item(),
                f"{mode}/entropy": portfolio_records["entropy_loss"].item(),
                f"{mode}/old_approx_kl": portfolio_records["old_approx_kl"].item(),
                f"{mode}/approx_kl": portfolio_records["approx_kl"].item(),
                f"{mode}/total_approx_kl": portfolio_records["total_approx_kl"].item(),
                f"{mode}/policy_update_times": portfolio_records["policy_update_steps"]// self.config.gradient_checkpointing_steps,
                f"{mode}/clipfrac": num_clipfracs,
                f"{mode}/explained_variance": explained_var,
                f"{mode}/SPS": self.global_step / (time.time() - start_time),
            }
            if self.use_bc:
                metrics[f"{mode}/bc_loss"] = portfolio_records["bc_loss"].item()
            logger.log_metric(metrics)

            logger.info(f"SPS: {self.global_step}, {(time.time() - start_time):.4f}")

            if self.global_step % self.config.check_steps >= self.check_index:
                self.valid()
                self.check_index += 1

            if self.global_step % self.config.save_steps >= self.save_index:
                torch.save(self.agent.state_dict(), os.path.join(self.config.checkpoint_path, "{:08d}.pth".format(self.global_step // self.config.save_steps)))
                self.save_index += 1

        self.valid()

        torch.save(self.agent.state_dict(), os.path.join(self.config.checkpoint_path, "{:08d}.pth".format(self.global_step // self.config.save_steps + 1)))

        self.train_environments.close()
        self.valid_environments.close()
        self.test_environments.close()

    def valid(self):

        mode = "valid"

        portfolio_records = PortfolioRecords()

        # TRY NOT TO MODIFY: start the game
        state, info = self.valid_environments.reset()
        # Update the trading records with initial state information
        portfolio_records.add(
            dict(
                timestamp=info["timestamp"][0],
                price=info["price"][0],
                cash=info["cash"][0],
                position=info["position"][0],
                value=info["value"][0],
            ),
        )

        next_obs = TensorDict({key: torch.Tensor(value) for key, value in state.items()}, batch_size=1).to(self.device)

        while True:

            # ALGO LOGIC: action logic
            with torch.no_grad():

                input_tensor = TensorDict(
                    {
                        "dense": next_obs["features"],
                        "sparse": next_obs["times"].to(torch.int32),
                        "cashes": next_obs["policy_portfolio_cashes"],
                        "positions": next_obs["policy_portfolio_positions"].to(torch.float32),
                        "actions": next_obs["policy_portfolio_actions"].to(torch.float32),
                        "rets": next_obs["policy_portfolio_rets"],
                    }, batch_size=next_obs.batch_size
                ).to(next_obs.device)

                action = self.agent.get_action(input_tensor)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncted, info = self.valid_environments.step(action.cpu().numpy())

            portfolio_records.add(
                dict(
                    action=info["action"][0],
                    ret=info["ret"][0],
                    total_profit=info["total_profit"][0],
                    timestamp=info["timestamp"][0],  # next timestamp
                    price=info["price"][0],  # next price
                    cash=info["cash"][0],  # next cash
                    position=info["position"][0],  # next position
                    value=info["value"][0],  # next value
                ),
            )

            next_obs = TensorDict({key: torch.Tensor(value) for key, value in next_obs.items()}, batch_size=1).to(self.device)

            if "final_info" in info:
                for info_item in info["final_info"]:
                    if info_item is not None:
                        logger.info(f"| total_return={info_item['total_return']:.4f}, total_profit = {info_item['total_profit']:.4f}")

                        metrics = {
                            f"{mode}/total_return": info_item["total_return"],
                            f"{mode}/total_profit": info_item["total_profit"],
                        }

                        logger.log_metric(metrics)
                break

        # End of the environment, add the final record
        portfolio_records.add(
            dict(
                action=info["action"][0],
                ret=info["ret"][0],
                total_profit=info["total_profit"][0],
            )
        )

        rets = portfolio_records.data["ret"]
        rets = np.array(rets, dtype=np.float32)

        metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            arguments = dict(
                ret=rets,
            )
            metric_item = metric_fn(**arguments)
            metrics[metric_name] = metric_item

        metrics = {f"{mode}/{k}": v for k, v in metrics.items()}

        logger.log_metric(metrics)

        log_string = f"{mode.capitalize()} Metrics:\n"
        metrics_string = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"| {log_string} - {metrics_string}")

        # Save the trading records
        records_df = portfolio_records.to_dataframe()
        records_df.to_json(os.path.join(self.config.exp_path, f"{mode}_records.jsonl"), orient="records", lines=True)