import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union

import sys
import cv2

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from utils.pretrain_utils import vt_load

import copy


SelfPPO = TypeVar("SelfPPO", bound="PPO")

class PPO_CNN(OnPolicyAlgorithm):
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
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        rollout_length = None,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vae_coef: float = 0.01,
        vf_coef: float = 0.5,
        contrastive_coef: float = 0.1,
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
        cnn_image = None,
        cnn_tactile = None,
        cnn_batch_size = 32,
        separate_optimizer = False,
        _init_setup_model: bool = True,
        structure = None,
        contrastive_switch_limit = None,
        infonce = None,
        image_only = False,
        augment = False,
        env_name = None,
        vae = None,
        use_vae = False,
        time_contrastive = False,
        contrastive_limit = None,
        alternation_loop = None
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
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.cnn_batch_size = cnn_batch_size
        self.separate_optimizer = separate_optimizer
        self.contrastive_coef = contrastive_coef
        self.structure = structure
        self.infonce = infonce
        self.infonce_optimizer = th.optim.Adam(self.infonce.parameters(), lr=1e-4)
        self.contrastive_limit = contrastive_limit
        self.alternation_loop = alternation_loop
        self.lst_timestep = 0
        self.contrastive_switch_limit = contrastive_switch_limit # set 100 in correspondence to paper: Self-supervised Co-training for Video Representation Learning（Neurips’20）
        self.switch_tag = 0
        self.image_only = image_only
        self.augment = augment
        self.rollout_length = rollout_length
        self.vae_coef = vae_coef
        self.use_vae = use_vae
        self.time_contrastive = time_contrastive
        self.policy_kwargs = policy_kwargs

        if _init_setup_model:
            self.cnn_image = cnn_image
            self.cnn_image_optimizer = th.optim.Adam(self.cnn_image.parameters(), lr=1e-4)
            if(not self.image_only):
                self.cnn_tactile = cnn_tactile
                self.cnn_tactile_optimizer = th.optim.Adam(self.cnn_tactile.parameters(), lr=1e-4)
                if self.use_vae:
                    self.vae = vae
                    self.vae_optimizer = th.optim.Adam(self.vae.parameters(), lr=1e-4)

            self._setup_model()

        ###
        # self.env.reset()
        # for i in range(3):
        #     action = self.env.action_space.sample()
        #     print('-- ', action)
            
        #     obs, reward, terminated, truncated, info = self.env.step(action)
        #     print('?? ', obs, reward, terminated)
        # exit(0)
        ###

        # print('extractor: ', self.policy.features_extractor);exit(0)
        # self.features_extractor_optimizer = th.optim.Adam(self.policy.features_extractor.parameters(), lr=1e-4)
        self.policy_optimizer = th.optim.Adam(self.policy.parameters(), lr=1e-4)

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def show_tactile(self, tactile, size=(480,480), max_shear=0.05, max_pressure=0.1, name='tactile'): # Note: default params work well for 16x16 or 32x32 tactile sensors, adjust for other sizes
        tactile = tactile.cpu()
        nx = tactile.shape[2]
        ny = tactile.shape[1]

        loc_x = np.linspace(0,size[1],nx)
        loc_y = np.linspace(size[0],0,ny)

        img = np.zeros((size[0],size[1],3))

        # print('tactile: ', tactile)
        max_shear, max_pressure = 1., 1.
        for i in range(0,len(loc_x),1):
            for j in range(0,len(loc_y),1):
                
                dir_x = np.clip(tactile[0,j,i]/max_shear,-1,1) * 20
                dir_y = np.clip(tactile[1,j,i]/max_shear,-1,1) * 20

                color = np.clip(tactile[2,j,i]/max_pressure,0,1)
                r, g = color, 1-color
                r = r.item()
                g = g.item()

                # print('col: ', color, r, g)
                cv2.arrowedLine(img, (int(loc_x[i]),int(loc_y[j])), (int(loc_x[i]+dir_x),int(loc_y[j]-dir_y)), (0.3,0.3,0.3), 4, tipLength=0.5)

        name="qwqtactile.png"
        cv2.imwrite(name, img)

        return img

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

        entropy_losses = []
        pg_losses, value_losses = [], []
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
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                self.policy.optimizer.zero_grad()
                self.cnn_image_optimizer.zero_grad()
                if(not self.image_only):
                    self.infonce_optimizer.zero_grad()
                    self.cnn_tactile_optimizer.zero_grad()
                    # self.features_extractor_optimizer.zero_grad()
                    if self.use_vae:
                        self.vae_optimizer.zero_grad()
                    self.policy_optimizer.zero_grad()
                # try:
                #     n_iter = rollout_data.observations['image'].shape[0] // self.cnn_batch_size
                # except:
                #     n_iter = rollout_data.observations['tactile'].shape[0] // self.cnn_batch_size

                observations = rollout_data.observations
                # rewards = rollout_data.values
                # print('tmp: ', observations['image'].shape, rewards.shape)
                # tac, obs = observations['tactile'][-1][0], observations['image']
                # self.show_tactile(tac[:3])
                # self.show_tactile(tac[3:])
                # torch.Size([512, 4, 64, 64, 3]) torch.Size([512, 4, 6, 32, 32])
                # print('?? ', observations['image'].shape, observations['tactile'].shape)
                # print("image shape: ", observations['image'].shape)
                # print("tactile shape: ", observations['tactile'].shape)
                frame_stack = 1
                if 'image' in observations and len(observations['image'].shape) == 5:
                    frame_stack = observations['image'].shape[1]
                    observations['image'] = observations['image'].permute(0, 2, 3, 1, 4) 
                    observations['image'] = observations['image'].reshape((observations['image'].shape[0], observations['image'].shape[1], observations['image'].shape[2], -1))
                if 'tactile' in observations and len(observations['tactile'].shape) == 5:
                    frame_stack = observations['tactile'].shape[1]
                    observations['tactile'] = observations['tactile'].reshape((observations['tactile'].shape[0], -1, observations['tactile'].shape[3], observations['tactile'].shape[4]))
                # print('in train: obs.shape= ', observations['image'].shape)

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # [obses_image, obses_tactile] updated in evaluate_actions
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
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

                # calculate contrastive loss 

                obses_image = th.cat(self.policy.features_extractor.obses_image, dim=0)
                if(not self.image_only):obses_tactile = th.cat(self.policy.features_extractor.obses_tactile, dim=0)
                contrastive_loss = th.tensor(0)
                vae_loss = th.tensor(0)
                if(obses_image.shape[0] > 1 and not self.image_only):
                    if(obses_image.shape[0] != 512):
                        print('obses shape: ', obses_image.shape)
                    if(self.num_timesteps <= self.contrastive_limit): # 只有将 obses 的长度不为 0 时才重新计算对比损失
                        contrastive_loss = th.tensor(0)
                        vae_loss = th.tensor(0)
                    else: # calculate contrastive loss alternately according to granularity
                       # TODO: not forget to switch image and tactile every (e.g. 100) times
                        obs1, obs2 = obses_image.clone(), obses_tactile.clone()
                        self.lst_timestep += 1
                        if(self.lst_timestep > self.contrastive_switch_limit):
                            self.switch_tag ^= 1
                            self.lst_timestep = 0
                        if(self.switch_tag == 1):
                           obs1, obs2 = obs2, obs1

                        contrastive_loss = self.infonce(obs1, obs2)
                        if self.use_vae:
                            # get embedding
                            # vt_torch: ["image": [512, 12, 64, 64], "tactile1": [512, 12, 32, 32]]
                            vt_torch = vt_load(observations, frame_stack=self.policy_kwargs['frame_stack'])
                            if th.cuda.is_available():
                                for key in vt_torch:
                                    vt_torch[key] = vt_torch[key].to('cuda')
                                    # print('key, shape: ', key, vt_torch[key].shape)
                            # exit(0)
                            encoded_image = self.cnn_image.get_embeddings(vt_torch, eval=False, use_tactile=not self.image_only, key='image')
                            encoded_image = encoded_image.reshape(encoded_image.shape[0], -1)
                            if(not self.image_only):
                                encoded_tactile = self.cnn_tactile.get_embeddings(vt_torch, eval=False, use_tactile=not self.image_only, key='tactile')
                                encoded_tactile = encoded_tactile.reshape(encoded_tactile.shape[0], -1)

                            recon, mu, log_var = self.vae(vt_torch, encoded_image, encoded_tactile, recon_target = "image")
                            vae_recon_loss, vae_kld_loss = \
                                self.vae.loss_function(recon, vt_torch["image"][:, 9:, :, :], mu, log_var)
                            vae_loss = vae_recon_loss + vae_kld_loss

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss_policy = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                # TODO: check contrastive loss, ablation study
                if(obses_image.shape[0] > 1 and self.num_timesteps > self.contrastive_limit and not self.image_only):
                    # print('use loss: policy loss + contrastive loss')                    # loss = loss_policy + self.contrastive_coef * contrastive_loss # we set contrastive_coef = 0.01
                    # print('fuck ',self.num_timesteps, self.contrastive_limit,contrastive_loss)
                    loss = loss_policy
                    loss = loss + self.contrastive_coef * contrastive_loss + self.vae_coef * vae_loss
                else:
                    # print('use loss: policy loss only')
                    loss = loss_policy
                    
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

                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.policy.optimizer.step()
                self.cnn_image_optimizer.step()
                if(not self.image_only and not self.time_contrastive): # TODO: change time_contrastive
                    self.infonce_optimizer.step()
                    self.cnn_tactile_optimizer.step()
                    # self.features_extractor_optimizer.step()
                    if self.use_vae:
                        self.vae_optimizer.step()
                        self.policy_optimizer.step()

                # net = self.policy.features_extractor
                # print('after: --------------------')
                # for name, para in net.named_parameters():
                #     if(name == 'cnn_tactile.convnet.6.weight'):
                #         print('name is: ', name)
                #         print(para)

                self.policy.features_extractor.obses_image = [] # TODO: check if we should clear obses every rollout
                if(not self.image_only):self.policy.features_extractor.obses_tactile = []

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        if(not self.image_only):
            save_mean_logits = {('image', 'weight'): 0, ('image', 'bias'): 0, ('tactile', 'weight'): 0, ('tactile', 'bias'): 0}
            save_mean_num = {('image', 'weight'): 0, ('image', 'bias'): 0, ('tactile', 'weight'): 0, ('tactile', 'bias'): 0}
            net = self.policy.features_extractor
            for name, parms in net.named_parameters():	
                if("convnet" in name):  # we focus on cnn_image & tactile.convnet.weight & bias
                    belong = 'image' if 'image' in name else 'tactile'
                    wb = 'weight' if 'weight' in name else 'bias'
                    mn = parms.grad.mean()
                    save_mean_logits[belong, wb] += mn
                    save_mean_num[belong, wb] += 1

                    # grad = grad.reshape(grad.shape[0], -1)
                    # print('-->name:', name, '-->grad_requirs:',parms.requires_grad)
                    # print('-->grad_value:',parms.grad.mean(-1))
            if(self.structure == 'drqv2'):
                net = self.policy
                for name, parms in net.named_parameters():	
                    # if(name == 'cnn_tactile.convnet.6.weight'):  # we focus on cnn_image & tactile.convnet.weight & bias
                    if('convnet' not in name):
                        print('name: ', name)
                        if(parms.grad != None):
                            grad = parms.grad.mean()
                            print('grad is: ', grad)
                for name1, name2 in save_mean_logits:
                    print('name: ', name1, name2, 'gradient_mean: ', save_mean_logits[name1, name2] / save_mean_num[name1, name2])
        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("loss/loss", loss.item())
        self.logger.record("loss/contrastive_loss", contrastive_loss.item())
        self.logger.record("loss/vae_loss", vae_loss.item())
        self.logger.record("train/explained_variance", explained_var)

        # self.logger.record("train/mae_loss", mae_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

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