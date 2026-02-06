import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union

import sys

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from utils.vtt_latent import LatentModel
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    # ReplayBufferSamples,
    # RolloutBufferSamples,
)

from utils.pretrain_utils import vt_load

import copy


SelfPPO = TypeVar("SelfPPO", bound="PPO")

class PPO_VTT(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        n_envs: int = None,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        vtt_coef: float = 1e-3, # use vtt_coef to leverage scale of loss from vtt / poe
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        vtt_encoder = None,
        cnn_image = None,
        num_tactiles = None,
        cnn_tactile = None,
        separate_optimizer = False,
        _init_setup_model: bool = True,
        structure = None,
        contrastive_switch_limit = None,
        infonce = None,
        image_only = False,
        augment = False,
        env_name = None,
        contrastive_limit = None,
        alternation_loop = None,
        vtt_batch_size = 32,
        use_encoder = 'VTT'
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        
        
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.num_tactiles = num_tactiles
        self.env_name = env_name

        self.vtt_batch_size = vtt_batch_size
        self.separate_optimizer = separate_optimizer
        self.vtt_coef = vtt_coef
        self.structure = structure
        self.infonce = infonce
        self.contrastive_limit = contrastive_limit
        self.alternation_loop = alternation_loop
        self.lst_timestep = 0
        self.contrastive_switch_limit = contrastive_switch_limit # set 100 in correspondence to paper: Self-supervised Co-training for Video Representation Learning（Neurips’20）
        self.switch_tag = 0
        self.image_only = image_only
        self.augment = augment
        self.use_encoder = use_encoder

        if _init_setup_model:
            # self.vtt_encoder = vtt_encoder
            self.cnn_image = cnn_image
            self.cnn_tactile = cnn_tactile
            # self.vtt_encoder_optimizer = th.optim.Adam(self.vtt_encoder.parameters(), lr=1e-4)
            self.cnn_image_optimizer = th.optim.Adam(self.cnn_image.parameters(), lr=1e-4)
            self.cnn_tactile_optimizer = th.optim.Adam(self.cnn_tactile.parameters(), lr=1e-4)

            self._setup_model()
            
        self.latent = LatentModel((3,84,84), (3,), encoder=self.use_encoder).to('cuda')
        self.force_norm = th.tensor(50.0).to('cuda')
        # print('extractor: ', self.policy.features_extractor);exit(0)
        # self.features_extractor_optimizer = th.optim.Adam(self.policy.features_extractor.parameters(), lr=1e-4)
        self.policy_optimizer = th.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.latent_optimizer = th.optim.Adam(self.latent.parameters(), lr=1e-4)

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    def get_rollout_data(self, rollout_buffer, batch_size):
        # rewrite via /home/csq/miniconda3/envs/tactile_envs/lib/python3.11/site-packages/stable_baselines3/common/buffers.py:get()
        start_idx = 0
        buffer_size = self.n_steps
        buf = []
        indices = np.random.permutation(buffer_size)
        while start_idx < buffer_size:
            use_ind = indices[start_idx : start_idx + batch_size]
            # data = DictRolloutBufferSamples(
            #     observations={key: self.to_torch(obs[use_ind]) for (key, obs) in self.observations.items()},
            #     actions=self.to_torch(self.actions[use_ind]),
            #     old_values=self.to_torch(self.values[use_ind].flatten()),
            #     old_log_prob=self.to_torch(self.log_probs[use_ind].flatten()),
            #     advantages=self.to_torch(self.advantages[use_ind].flatten()),
            #     returns=self.to_torch(self.returns[use_ind].flatten()),
            # )
            data = {
                'observations': {key: self.to_torch(obs[use_ind]) for (key, obs) in rollout_buffer.observations.items()},
                'actions': self.to_torch(rollout_buffer.actions[use_ind]),
                'rewards': self.to_torch(rollout_buffer.rewards[use_ind]),
                'dones': th.zeros_like(self.to_torch(rollout_buffer.rewards[use_ind]))
            }
            buf.append(data)
            start_idx += batch_size
        return buf

    def truncate(self, x, env_name):
        if(env_name == 'tactile_envs/Insertion-v0'):
            truncate_length = 8
            x = x.unsqueeze(dim=1)
            obs = x
            for i in range(1, truncate_length):
                obs = th.cat([obs, x], dim=1)
            return obs
        elif env_name == 'Door':
            padding = (8, 9)
            x = F.pad(x, padding)
            x = x.reshape(x.shape[0], 8, 3)
            return x
        elif env_name in ['TwoArmPegInHole', 'TwoArmHandover']:
            # print('?? ', x.shape)
            padding = (5, 5)
            x = F.pad(x, padding)
            x = x.reshape(x.shape[0], 8, 3)
            return x
        elif env_name == 'Lift' or env_name == 'LiftCan' or env_name == 'LiftCap':
            padding = (9, 8)
            x = F.pad(x, padding)
            x = x.reshape(x.shape[0], 8, 3)
            return x
        else:
            padding = (2, 2)
            x = F.pad(x, padding)
            x = x.reshape(x.shape[0], 8, 3)
            return x
    
    def vtt_truncate(self, state_, tactile_, action_, reward_):
        indice = np.random.permutation(self.batch_size)
        indice = indice[0 : self.vtt_batch_size]
        state, tactile, action, reward = state_[indice], tactile_[indice], action_[indice], reward_[indice]
        return state, tactile, action, reward

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        
        CNNExtractor stored in self.policy.features_extractor
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        pg_losses = []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            """
            obses 的长度: ~ self.n_steps = config.rollout_length // config.n_envs
            其中, config.rollout_length 默认为 32768(now changed)
            """
            # print('in training, size of obses is: ', len(self.policy.features_extractor.obses_image))
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            # rollout_bufs = self.get_rollout_data(self.rollout_buffer, self.batch_size)
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                self.policy.optimizer.zero_grad()
                # self.vtt_encoder_optimizer.zero_grad()
                self.cnn_image_optimizer.zero_grad()
                self.cnn_tactile_optimizer.zero_grad()
                self.latent_optimizer.zero_grad()

                observations = rollout_data.observations
                
                obs_image = observations['image']
                obs_tactile = observations['tactile']
                # print("image shape: ", observations['image'].shape)
                # print("tactile shape: ", observations['tactile'].shape)
                if 'image' in observations and len(observations['image'].shape) == 5:
                    observations['image'] = observations['image'].permute(0, 2, 3, 1, 4) 
                    observations['image'] = observations['image'].reshape((observations['image'].shape[0], observations['image'].shape[1], observations['image'].shape[2], -1))
                if 'tactile' in observations and len(observations['tactile'].shape) == 5:
                    observations['tactile'] = observations['tactile'].reshape((observations['tactile'].shape[0], -1, observations['tactile'].shape[3], observations['tactile'].shape[4]))


                # x = vt_load(copy.deepcopy(observations), frame_stack=frame_stack)

                # print('?? rollout_data', rollout_data);exit(0)
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # [obses_image, obses_tactile] updated in evaluate_actions
                values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # print('shape: ',observations['image'], observations['tactile'])
                # print('extra', self.rollout_buffer.observations[idx]['image']);exit(0)

                observations['image'] = obs_image
                observations['tactile'] = obs_tactile
                if 'image' in observations and len(observations['image'].shape) == 5:
                    observations['image'] = observations['image'].permute(0, 1, 4, 2, 3)
                    padding = (10, 10, 10, 10)
                    observations['image'] = F.pad(observations['image'], padding)
                if 'tactile' in observations and len(observations['tactile'].shape) == 5:
                    observations['tactile'] = observations['tactile'].reshape((observations['tactile'].shape[0], observations['tactile'].shape[1], observations['tactile'].shape[2], -1))
                    observations['tactile'] = observations['tactile'].mean(dim = -1)
                # door- [32, 7] pen- [32,20] insertion- [32, 3] 

                actions = self.truncate(actions, self.env_name)
                # print('shapeeee: ', observations['image'].shape, observations['tactile'].shape,actions.shape)
                state_, tactile_, action_, reward_ = self.vtt_truncate(observations['image'], observations['tactile'], actions, rollout_data.old_values)
                if(self.num_tactiles == 1):tactile_ = th.cat([tactile_, tactile_], dim=-1)
                # print('shape: ', state_.shape, tactile_.shape, action_.shape, reward_.shape)
                loss_kld, loss_image, loss_reward, alignment_loss, contact_loss = self.latent.calculate_loss(state_, tactile_, action_, reward_, th.zeros_like(reward_), self.force_norm)
                # print('losses: ', loss_kld, loss_image, loss_reward, alignment_loss, contact_loss)

                # print('state: ', state_.shape, tactile_.shape, action_.shape, reward_.shape, done_.shape)

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                loss = policy_loss + loss_kld + loss_image + loss_reward + alignment_loss + contact_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # print('loss: ', loss, self.vtt_coef * (loss_kld + loss_image + loss_reward + alignment_loss + contact_loss))
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.policy.optimizer.step()
                # self.vtt_encoder_optimizer.zero_grad()
                self.cnn_tactile_optimizer.step()
                self.cnn_image_optimizer.step()
                self.latent_optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("loss/loss", loss.item())

        # self.logger.record("train/mae_loss", mae_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )