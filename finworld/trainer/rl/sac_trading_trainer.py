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
from finworld.utils import build_storage
from finworld.utils import ReplayBuffer
from finworld.utils import TradingRecords
from finworld.log import logger


@TRAINER.register_module(force=True)
class SACTradingTrainer():

    def __init__(self,
                 *args,
                 config=None,
                 dataset=None,
                 agent=None,
                 metrics=None,
                 device=None,
                 dtype=None,
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

        self.policy_optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(agent.actor.parameters())),
                                           lr=config.policy_learning_rate, eps=1e-5, weight_decay=0)
        self.value_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, list(agent.critic1.parameters()) + list(agent.critic2.parameters())),
            lr=config.value_learning_rate, eps=1e-5)

        self.target_entropy = - config.target_entropy_scale * torch.log(1 / torch.tensor(config.action_dim)).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_learning_rate, eps=1e-4)

        self.device = device
        self.dtype = dtype

        self.global_step = 0
        self.check_index = 0
        self.save_index = 0

        self.storage = self.set_storage()

        self.replay_buffer = ReplayBuffer(
            buffer_size=config.replay_buffer_size,
            transition=config.transition,
            transition_shape=config.transition_shape,
        )

        self.use_bc = self.config.use_bc

    def __str__(self):
        return f"SACTradingTrainer(task_type={self.config.task_type})"

    def __repr__(self):
        return self.__str__()

    def set_storage(self):

        transition = self.config.transition
        transition_shape = self.config.transition_shape

        storage = TensorDict({}, batch_size=self.config.num_steps).to(self.device)
        for name in transition:
            assert name in transition_shape
            shape = (self.config.num_steps, *transition_shape[name]["shape"])
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
                            init_state=None,
                            init_info=None,
                            reset=False):
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

            if self.global_step < self.config.learning_start:
                action = torch.randint(0, self.config.action_dim, (self.config.num_envs,)).to(self.device)
            else:
                # ALGO LOGIC: action logic
                with torch.no_grad():

                    # Construct the input tensor
                    input_tensor = TensorDict(
                        {
                            "dense": next_obs["features"],
                            "sparse": next_obs["times"].to(torch.int32),
                            "cashes": next_obs["policy_trading_cashes"],
                            "positions": next_obs["policy_trading_positions"].to(torch.int32),
                            "actions": next_obs["policy_trading_actions"].to(torch.int32),
                            "rets": next_obs["policy_trading_rets"],
                        }, batch_size=next_obs.batch_size
                    ).to(next_obs.device)

                    action, _, _ = self.agent.get_action_and_value(input_tensor)

            for key, value in next_obs.items():  # current state
                self.storage[key][step] = value
            self.storage["training_dones"][step] = next_done
            self.storage["training_actions"][step] = action

            if self.use_bc:
                expert_action = torch.tensor(info["expert_action"]).to(self.device)
                self.storage["training_expert_actions"][step] = expert_action

            next_obs, reward, done, truncted, info = self.train_environments.step(action.cpu().numpy())
            next_obs = TensorDict({key: torch.Tensor(value) for key, value in next_obs.items()},
                                  batch_size=self.config.num_envs).to(self.device)

            real_next_obs = TensorDict({
                "next_features": next_obs["features"],
                "next_times": next_obs["times"].to(torch.int32),
                "next_policy_trading_cashes": next_obs["policy_trading_cashes"],
                "next_policy_trading_positions": next_obs["policy_trading_positions"].to(torch.int32),
                "next_policy_trading_actions": next_obs["policy_trading_actions"].to(torch.int32),
                "next_policy_trading_rets": next_obs["policy_trading_rets"],
            }, batch_size=self.config.num_envs).to(self.device)

            for key, value in real_next_obs.items():
                self.storage[key][step] = value

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

        self.replay_buffer.update(self.storage)

    def update_value(self, flat_storage, b_inds, info):

        loss = info["v_loss"]

        # update value
        np.random.shuffle(b_inds)

        for start in range(0, self.config.batch_size, self.config.value_minibatch_size):
            end = start + self.config.value_minibatch_size
            mb_inds = b_inds[start:end]

            with torch.no_grad():
                input_tensor = TensorDict({
                    "dense": flat_storage["next_features"][mb_inds],
                    "sparse": flat_storage["next_times"][mb_inds].to(torch.int32),
                    "cashes": flat_storage["next_policy_trading_cashes"][mb_inds],
                    "positions": flat_storage["next_policy_trading_positions"][mb_inds].to(torch.int32),
                    "actions": flat_storage["next_policy_trading_actions"][mb_inds].to(torch.int32),
                    "rets": flat_storage["next_policy_trading_rets"][mb_inds],
                }, batch_size=self.config.value_minibatch_size).to(self.device)

                _, next_state_action_probs, next_state_log_prob = self.agent.get_action_and_value(input_tensor)
                next_target_q1, next_target_q2 = self.agent.get_value(input_tensor, use_target=True)

                min_target_q = next_state_action_probs * torch.min(next_target_q1,
                                                                   next_target_q2) - self.alpha * next_state_log_prob
                min_target_q = min_target_q.sum(dim=-1)

                next_target_q = flat_storage["training_rewards"][mb_inds] + \
                                (1.0 - flat_storage["training_dones"][mb_inds]) * self.config.gamma * min_target_q

            input_tensor = TensorDict({
                "dense": flat_storage["features"][mb_inds],
                "sparse": flat_storage["times"][mb_inds].to(torch.int32),
                "cashes": flat_storage["policy_trading_cashes"][mb_inds],
                "positions": flat_storage["policy_trading_positions"][mb_inds].to(torch.int32),
                "actions": flat_storage["policy_trading_actions"][mb_inds].to(torch.int32),
                "rets": flat_storage["policy_trading_rets"][mb_inds],
            }, batch_size=self.config.value_minibatch_size).to(self.device)

            q1, q2 = self.agent.get_value(input_tensor, use_target=False)

            q1_value = q1.gather(1, flat_storage["training_actions"][mb_inds].unsqueeze(-1).long()).view(-1)
            q2_value = q2.gather(1, flat_storage["training_actions"][mb_inds].unsqueeze(-1).long()).view(-1)

            q1_loss = nn.functional.mse_loss(q1_value, next_target_q)
            q2_loss = nn.functional.mse_loss(q2_value, next_target_q)

            loss = q1_loss + q2_loss

            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()

        res_info = {
            "v_loss": loss,
        }

        return res_info

    def update_policy(self,
                      flat_storage,
                      b_inds,
                      info):

        policy_update_steps = info["policy_update_steps"]
        if self.use_bc:
            bc_loss = info["bc_loss"]
        actor_loss = info["a_loss"]
        alpha_loss = info["alpha_loss"]

        # update policy
        for start in range(0, self.config.batch_size, self.config.policy_minibatch_size):

            policy_update_steps += 1
            end = start + self.config.policy_minibatch_size

            mb_inds = b_inds[start:end]
            input_tensor = TensorDict({
                "dense": flat_storage["features"][mb_inds],
                "sparse": flat_storage["times"][mb_inds].to(torch.int32),
                "cashes": flat_storage["policy_trading_cashes"][mb_inds],
                "positions": flat_storage["policy_trading_positions"][mb_inds].to(torch.int32),
                "actions": flat_storage["policy_trading_actions"][mb_inds].to(torch.int32),
                "rets": flat_storage["policy_trading_rets"][mb_inds],
            }, batch_size=self.config.policy_minibatch_size).to(self.device)
            _, log_prob, action_probs = self.agent.get_action_and_value(input_tensor)

            with torch.no_grad():
                q1, q2 = self.agent.get_value(input_tensor, use_target=False)
                min_q_value = torch.min(q1, q2)

            # no need for reparameterization, the expectation can be calculated for discrete actions
            actor_loss = (action_probs * ((self.alpha * log_prob) - min_q_value)).mean()

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
                expert_logits = self.agent.actor(input_tensor)
                bc_loss = nn.functional.cross_entropy(expert_logits, input_actions.flatten())

            loss = actor_loss

            if self.use_bc:
                loss = loss + self.config.bc_coef * bc_loss

            loss /= self.config.gradient_checkpointing_steps

            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

            # alpha loss
            alpha_loss = (action_probs.detach() * (
                        - self.log_alpha.exp() * (log_prob + self.target_entropy).detach())).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

            # update the target network
            for param, target_param in zip(self.agent.critic1.parameters(), self.agent.target_critic1.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
            for param, target_param in zip(self.agent.critic2.parameters(), self.agent.target_critic2.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        info = {
            "policy_update_steps": policy_update_steps,
            "a_loss": actor_loss,
            "alpha_loss": alpha_loss,
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
                self.alpha_optimizer.param_groups[0]["lr"] = frac * self.config.alpha_learning_rate

            # Explore the environment and collect data
            self.explore_environment(init_state=state, init_info=info, reset=False)

            if self.global_step > self.config.learning_start:
                # Flatten the storage
                flat_storage = self.replay_buffer.sample(self.config.batch_size).to(self.device)
                b_inds = np.arange(self.config.batch_size)

                trading_records = {
                    "policy_update_steps": 0,
                    "v_loss": torch.tensor(0),
                    "a_loss": torch.tensor(0),
                    "alpha_loss": torch.tensor(0),
                }
                if self.use_bc:
                    trading_records["bc_loss"] = torch.tensor(0)

                for epoch in range(self.config.update_epochs):
                    np.random.shuffle(b_inds)

                    # Update value
                    update_value_res_info = self.update_value(flat_storage,
                                                              b_inds,
                                                              trading_records)
                    trading_records.update(update_value_res_info)

                    if is_warmup:
                        continue

                    # Update policy
                    update_policy_info = self.update_policy(flat_storage,
                                                            b_inds,
                                                            trading_records)
                    trading_records.update(update_policy_info)

                metrics = {
                    f"{mode}/policy_learning_rate": self.policy_optimizer.param_groups[0]["lr"],
                    f"{mode}/value_learning_rate": self.value_optimizer.param_groups[0]["lr"],
                    f"{mode}/value_loss": trading_records["v_loss"].item(),
                    f"{mode}/policy_loss": trading_records["a_loss"].item(),
                    f"{mode}/alpha_loss": trading_records["alpha_loss"].item(),
                    f"{mode}/SPS": self.global_step / (time.time() - start_time),
                }

                if self.use_bc:
                    metrics[f"{mode}/bc_loss"] = trading_records["bc_loss"].item()
                logger.log_metric(metrics)

                logger.info(f"SPS: {self.global_step}, {(time.time() - start_time):.4f}")

                if self.global_step % self.config.check_steps >= self.check_index:
                    self.valid()
                    self.check_index += 1

                if self.global_step % self.config.save_steps >= self.save_index:
                    torch.save(self.agent.state_dict(), os.path.join(self.config.checkpoint_path, "{:08d}.pth".format(
                        self.global_step // self.config.save_steps)))
                    self.save_index += 1

        self.valid()

        torch.save(self.agent.state_dict(), os.path.join(self.config.checkpoint_path, "{:08d}.pth".format(self.global_step // self.config.save_steps + 1)))

        self.train_environments.close()
        self.valid_environments.close()
        self.test_environments.close()

    def valid(self):

        mode = "valid"

        trading_records = TradingRecords()

        # TRY NOT TO MODIFY: start the game
        state, info = self.valid_environments.reset()
        trading_records.add(
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
                        "cashes": next_obs["policy_trading_cashes"],
                        "positions": next_obs["policy_trading_positions"].to(torch.int32),
                        "actions": next_obs["policy_trading_actions"].to(torch.int32),
                        "rets": next_obs["policy_trading_rets"],
                    }, batch_size=next_obs.batch_size
                ).to(next_obs.device)

                action = self.agent.get_action(input_tensor)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncted, info = self.valid_environments.step(action.cpu().numpy())

            trading_records.add(
                dict(
                    action=info["action"][0],
                    action_label=info["action_label"][0],
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

            trading_records.add(
                dict(
                    action=info["action"][0],
                    action_label=info["action_label"][0],
                    ret=info["ret"][0],
                    total_profit=info["total_profit"][0],
                    timestamp=info["timestamp"][0],  # next timestamp
                    price=info["price"][0],  # next price
                    cash=info["cash"][0],  # next cash
                    position=info["position"][0],  # next position
                    value=info["value"][0],  # next value
                ),
            )

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
        trading_records.add(
            dict(
                action=info["action"][0],
                action_label=info["action_label"][0],
                ret=info["ret"][0],
                total_profit=info["total_profit"][0],
            )
        )

        rets = trading_records.data["ret"]
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
        records_df = trading_records.to_dataframe()
        records_df.to_json(os.path.join(self.config.exp_path, f"{mode}_records.jsonl"), orient="records", lines=True)