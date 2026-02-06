# set cuda device command: MUJOCO_EGL_DEVICE_ID=
import argparse
import torch
import gym
# import minitouch.env
# import rl_zoo3.gym_patches
# import pybullet_envs

from stable_baselines3 import PPO, DDPG
from models.ppo_mae import PPO_MAE
from models.ppo_cnn import PPO_CNN
from models.ppo_vtt import PPO_VTT
from models.ppo_vtt_insertion import PPO_VTT_Insertion
from models.pretrain_models import CNNEncoder, InfoNCELoss, DrqPolicy
from models.pretrain_models import VTT, VTMAE, MAEPolicy # for mae reproduction
from models.cvae import CVAE

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

import tactile_envs
import envs
from utils.callbacks import create_callbacks
from models.pretrain_models import CNNPolicy, VTTPolicy, VTTInsertionPolicy

from utils.vtt_latent import VTT

def str2bool(v):
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    raise ValueError(f"boolean argument should be either True or False (got {v})")

def main():
    parser = argparse.ArgumentParser("M3L")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=int(5e5))
    parser.add_argument("--eval_every", type=int, default=int(4096)) # TODO: back to default as 2e5
    parser.add_argument("--total_timesteps", type=int, default=int(3e6)) # choices: 3e6, 6e6
    
    parser.add_argument("--wandb_dir", type=str, default="./wandb/")
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
 
    # Environment.
    task_set = ["HandManipulateBlockRotateZFixed-v1", "HandManipulateEggRotateFixed-v1", "HandManipulatePenRotateFixed-v1", "HandManipulatePenRotateFixedInst-v1", "HandManipulatePenRotate-v1"]
    parser.add_argument(
        "--env",
        type=str,
        default="tactile_envs/Insertion-v0",
        choices=[
            "tactile_envs/Insertion-v0",
            "Door",
            "Stack",
            "Lift",
            "LiftCan",
            "LiftCap",
            "Wipe",
            "TwoArmPegInHole", # Two Arm Lift
            "TrueTwoArmPegInHole",
            "PickPlace",
            "TwoArmHandover",
            "HandManipulateBlockRotateZFixed-v1",
            "HandManipulateEggRotateFixed-v1",
            "HandManipulatePenRotateFixed-v1",
            "HandManipulatePenRotateFixedInst-v1", # change setting path: /home/csq/tactile/tactile_envs/tactile_envs/envs/Gymnasium-Robotics/gymnasium_robotics/envs/shadow_dexterous_hand/manipulate.py
            "HandManipulatePenRotate-v1"
        ],
    )
    parser.add_argument("--n_envs", type=int, default=32)
    parser.add_argument(
        "--state_type",
        type=str,
        default="vision_and_touch",
        choices=["vision", "touch", "vision_and_touch"]
    )
    parser.add_argument("--norm_reward", type=str2bool, default=True)
    parser.add_argument("--use_latch", type=str2bool, default=True)
    
    parser.add_argument("--camera_idx", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--no_rotation", type=str2bool, default=True)

    # MAE.
    parser.add_argument("--representation", type=str2bool, default=True)
    parser.add_argument("--early_conv_masking", type=str2bool, default=True)
    
    parser.add_argument("--dim_embedding", type=int, default=64) # mlp extractor = dim_embedding
    parser.add_argument("--use_sincosmod_encodings", type=str2bool, default=True)
    parser.add_argument("--masking_ratio", type=float, default=0.95)
    
    parser.add_argument("--mae_batch_size", type=int, default=32)
    parser.add_argument("--train_mae_every", type=int, default=1)

    # PPO.
    parser.add_argument("--rollout_length", type=int, default=4096) # changed from 32768 to 131072
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--lr_ppo", type=float, default=1e-4)
    parser.add_argument("--vision_only_control", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=512)

    # PPO-MAE.
    parser.add_argument("--separate_optimizer", type=str2bool, default=False)

    # extra training settings
    parser.add_argument("--note", default="No special notes")
    parser.add_argument("--structure", default="drqv2", choices=["drqv2", "ppo"])

    # contrastive learning settings
    parser.add_argument("--contrastive_limit", type=int, default=(int)(0)) # 0 is way better for training
    parser.add_argument("--contrastive_switch_limit", type=int, default=(int)(100)) # 100 as default
    parser.add_argument("--alternation_loop", type=int, default=300000)
    parser.add_argument("--use_topk", type=bool, default=True)
    # parser.add_argument("--use_vae", type=bool, default=True)
    parser.add_argument("--use_vae", type=str, default='False', choices=['True', 'False'])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--image_only", type=bool, default=False)
    parser.add_argument("--augment", type=bool, default=False)
    parser.add_argument("--is_gen_shape", type=bool, default=False) # is testing zero shot transfer for different shapes of insertion
    parser.add_argument("--use_algo", type=str, default="cnn", choices=["drqv2", "cnn", "vtt", "mae", "poe", "concat"])
    parser.add_argument("--use_time", type=bool, default=False)

    config = parser.parse_args()
    if(config.use_vae == 'True'):config.use_vae = True
    else: config.use_vae = False

    set_random_seed(config.seed)

    if(config.structure == 'ppo'):
        config.dim_embedding = 256    
        infonce = InfoNCELoss(repr_dim=256*64, feature_dim=1024, use_topk=config.use_topk, topk=config.topk, use_time=config.use_time).to("cuda")
    else:    # drqv2
        infonce = InfoNCELoss(repr_dim=81*64, feature_dim=1024, use_topk=config.use_topk, topk=config.topk, use_time=config.use_time).to("cuda")

    num_tactiles = 0
    if config.state_type == "vision_and_touch" or config.state_type == "touch":
        num_tactiles = 2
        if config.env in task_set:
            num_tactiles = 1
        if(config.image_only):
            num_tactiles = 0
            
    env_config = {
        "use_latch": config.use_latch,
    }
    
    objects = [
        "square",
        "triangle",
        "horizontal",
        "vertical",
        "trapezoidal",
        "rhombus",
    ]
    objects_all = objects
    if(config.is_gen_shape):
        objects = ["triangle"]

    holders = ["holder1", "holder2", "holder3"]
    if(config.use_algo in ['vtt', 'poe', 'concat']):
        config.frame_stack = 9
        # config.dim_embedding = 288

    if config.use_algo == "drqv2":
        config.batch_size = 2
        config.n_envs = 1

    env_list = [
        envs.make_env(
            config.env,
            i,
            config.seed,
            config.state_type,
            objects=objects,
            holders=holders,
            camera_idx=config.camera_idx,
            frame_stack=config.frame_stack,
            no_rotation=config.no_rotation,
            **env_config,
        )
        for i in range(config.n_envs)
    ]

    if config.n_envs < 100:
        env = SubprocVecEnv(env_list)
    else:
        env = DummyVecEnv(env_list)
    env = VecNormalize(env, norm_obs=False, norm_reward=config.norm_reward)

    if(config.use_algo == 'cnn'):
        cnn_image = CNNEncoder(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='image', env_name=config.env, structure=config.structure)
        if(not config.image_only):cnn_tactile = CNNEncoder(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='tactile', env_name=config.env, structure=config.structure)
        else: cnn_tactile = None
        if config.use_vae: vae = CVAE(256, 3, 3)
        else: vae = None

        if torch.cuda.is_available():
            cnn_image.cuda()
            if(not config.image_only):cnn_tactile.cuda()
            if config.use_vae: vae.cuda()

        cnn_image.eval()
        if(not config.image_only):cnn_tactile.eval()
        if config.use_vae: vae.eval()

        cnn_image.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})
        if(not config.image_only):cnn_tactile.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})
        if config.use_vae: vae.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})

        policy = CNNPolicy
        policy_kwargs={
            "cnn_image": cnn_image,
            "cnn_tactile": cnn_tactile if not config.image_only else None,
            "vae": vae if config.use_vae else None,
            "dim_embeddings": config.dim_embedding,
            "vision_only_control": config.vision_only_control,
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "frame_stack": config.frame_stack,
            "obses": [],
            "image_only": config.image_only,
            "use_vae": config.use_vae,
            "augment": config.augment
        }

        # repr_dim is calculated via obs_image.shape (=[batch_size, 64, 256])

        model = PPO_CNN(
            policy,
            env,
            verbose=1,
            env_name=config.env,
            learning_rate=config.lr_ppo,
            tensorboard_log=config.wandb_dir+"tensorboard/",
            batch_size=config.batch_size,
            n_steps=config.rollout_length // config.n_envs,
            rollout_length=config.rollout_length,
            n_epochs=config.ppo_epochs,
            vae = vae,
            use_vae=config.use_vae,
            # mae_batch_size=config.mae_batch_size,
            separate_optimizer=config.separate_optimizer,
            policy_kwargs=policy_kwargs,
            cnn_image=cnn_image,
            cnn_tactile=cnn_tactile,
            structure=config.structure,
            infonce=infonce,
            image_only=config.image_only,
            augment=config.augment,
            contrastive_limit=config.contrastive_limit,
            contrastive_switch_limit=config.contrastive_switch_limit,
            time_contrastive=config.use_time,
            alternation_loop=config.alternation_loop
        )
    elif config.use_algo == 'vtt':   # VTT
        if(config.env == 'tactile_envs/Insertion-v0'):
            print('use: vtt insertion')
            cnn_image = CNNEncoder(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='image', env_name=config.env, structure=config.structure)
            if(not config.image_only):cnn_tactile = CNNEncoder(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='tactile', env_name=config.env, structure=config.structure)
            else: cnn_tactile = None
            if torch.cuda.is_available():
                cnn_image.cuda()
                if(not config.image_only):cnn_tactile.cuda()
            cnn_image.eval()
            if(not config.image_only):cnn_tactile.eval()

            cnn_image.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})
            if(not config.image_only):cnn_tactile.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})
            if config.use_vae: vae.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})

            policy = CNNPolicy
            policy_kwargs={
                "cnn_image": cnn_image,
                "cnn_tactile": cnn_tactile if not config.image_only else None,
                "dim_embeddings": config.dim_embedding,
                "vision_only_control": config.vision_only_control,
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
                "frame_stack": config.frame_stack,
                "obses": [],
                "image_only": config.image_only,
                "augment": config.augment
            }

            # repr_dim is calculated via obs_image.shape (=[batch_size, 64, 256])

            model = PPO_VTT_Insertion(
                policy,
                env,
                verbose=1,
                env_name=config.env,
                learning_rate=config.lr_ppo,
                tensorboard_log=config.wandb_dir+"tensorboard/",
                batch_size=config.batch_size,
                n_steps=config.rollout_length // config.n_envs,
                n_epochs=config.ppo_epochs,
                # mae_batch_size=config.mae_batch_size,
                separate_optimizer=config.separate_optimizer,
                policy_kwargs=policy_kwargs,
                cnn_image=cnn_image,
                cnn_tactile=cnn_tactile,
                structure=config.structure,
                infonce=infonce,
                image_only=config.image_only,
                augment=config.augment,
                contrastive_limit=config.contrastive_limit,
                contrastive_switch_limit=config.contrastive_switch_limit,
                alternation_loop=config.alternation_loop
            )
        else:
            config.batch_size = 512
            print('?')
            cnn_image = CNNEncoder(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='image', env_name=config.env, structure=config.structure)
            cnn_tactile = CNNEncoder(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='tactile', env_name=config.env, structure=config.structure)
            vtt_encoder = VTT(tactile_dim=6*32*32)
            # vtt_image = VTT(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='image', env_name=config.env, structure=config.structure)
            # if(not config.image_only):vtt_tactile = VTT(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='tactile', env_name=config.env, structure=config.structure)
            if torch.cuda.is_available():
                vtt_encoder.cuda()
            vtt_encoder.eval()

            vtt_encoder.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})
            
            policy = VTTPolicy
            policy_kwargs={
                "vtt_encoder": vtt_encoder,
                "cnn_image": cnn_image,
                "cnn_tactile": cnn_tactile,
                "dim_embeddings": config.dim_embedding,
                "vision_only_control": config.vision_only_control,
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
                "frame_stack": config.frame_stack,
                "obses": [],
                "image_only": config.image_only,
                "num_tactiles": num_tactiles,
                "augment": config.augment
            }

            # repr_dim is calculated via obs_image.shape (=[batch_size, 64, 256])

            model = PPO_VTT(
                policy,
                env,
                verbose=1,
                n_envs=config.n_envs,
                env_name=config.env,
                learning_rate=config.lr_ppo,
                tensorboard_log=config.wandb_dir+"tensorboard/",
                batch_size=config.batch_size,
                n_steps=config.rollout_length // config.n_envs,
                n_epochs=config.ppo_epochs,
                # mae_batch_size=config.mae_batch_size,
                separate_optimizer=config.separate_optimizer,
                policy_kwargs=policy_kwargs,
                vtt_encoder=vtt_encoder,
                cnn_image=cnn_image,
                cnn_tactile=cnn_tactile,
                structure=config.structure,
                infonce=infonce,
                image_only=config.image_only,
                num_tactiles=num_tactiles,
                augment=config.augment,
                contrastive_limit=config.contrastive_limit,
                contrastive_switch_limit=config.contrastive_switch_limit,
                alternation_loop=config.alternation_loop,
                use_encoder='VTT'
            )
    elif config.use_algo == 'poe':   # POE
        config.batch_size = 512
        print('?')
        cnn_image = CNNEncoder(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='image', env_name=config.env, structure=config.structure)
        cnn_tactile = CNNEncoder(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='tactile', env_name=config.env, structure=config.structure)
        vtt_encoder = VTT(tactile_dim=6*32*32)
        # vtt_image = VTT(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='image', env_name=config.env, structure=config.structure)
        # if(not config.image_only):vtt_tactile = VTT(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='tactile', env_name=config.env, structure=config.structure)
        if torch.cuda.is_available():
            vtt_encoder.cuda()
        vtt_encoder.eval()

        vtt_encoder.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})
        
        policy = VTTPolicy
        policy_kwargs={
            "vtt_encoder": vtt_encoder,
            "cnn_image": cnn_image,
            "cnn_tactile": cnn_tactile,
            "dim_embeddings": config.dim_embedding,
            "vision_only_control": config.vision_only_control,
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "frame_stack": config.frame_stack,
            "obses": [],
            "image_only": config.image_only,
            "num_tactiles": num_tactiles,
            "augment": config.augment
        }

        # repr_dim is calculated via obs_image.shape (=[batch_size, 64, 256])

        model = PPO_VTT(
            policy,
            env,
            verbose=1,
            n_envs=config.n_envs,
            env_name=config.env,
            learning_rate=config.lr_ppo,
            tensorboard_log=config.wandb_dir+"tensorboard/",
            batch_size=config.batch_size,
            n_steps=config.rollout_length // config.n_envs,
            n_epochs=config.ppo_epochs,
            # mae_batch_size=config.mae_batch_size,
            separate_optimizer=config.separate_optimizer,
            policy_kwargs=policy_kwargs,
            vtt_encoder=vtt_encoder,
            cnn_image=cnn_image,
            cnn_tactile=cnn_tactile,
            structure=config.structure,
            infonce=infonce,
            image_only=config.image_only,
            num_tactiles=num_tactiles,
            augment=config.augment,
            contrastive_limit=config.contrastive_limit,
            contrastive_switch_limit=config.contrastive_switch_limit,
            alternation_loop=config.alternation_loop,
            use_encoder='POE'
        )
    elif config.use_algo == 'concat':   # concatenation
        config.batch_size = 512
        print('?')
        cnn_image = CNNEncoder(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='image', env_name=config.env, structure=config.structure)
        cnn_tactile = CNNEncoder(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='tactile', env_name=config.env, structure=config.structure)
        vtt_encoder = VTT(tactile_dim=6*32*32)
        # vtt_image = VTT(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='image', env_name=config.env, structure=config.structure)
        # if(not config.image_only):vtt_tactile = VTT(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='tactile', env_name=config.env, structure=config.structure)
        if torch.cuda.is_available():
            vtt_encoder.cuda()
        vtt_encoder.eval()

        vtt_encoder.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})
        
        policy = VTTPolicy
        policy_kwargs={
            "vtt_encoder": vtt_encoder,
            "cnn_image": cnn_image,
            "cnn_tactile": cnn_tactile,
            "dim_embeddings": config.dim_embedding,
            "vision_only_control": config.vision_only_control,
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "frame_stack": config.frame_stack,
            "obses": [],
            "image_only": config.image_only,
            "num_tactiles": num_tactiles,
            "augment": config.augment
        }

        # repr_dim is calculated via obs_image.shape (=[batch_size, 64, 256])

        model = PPO_VTT(
            policy,
            env,
            verbose=1,
            n_envs=config.n_envs,
            env_name=config.env,
            learning_rate=config.lr_ppo,
            tensorboard_log=config.wandb_dir+"tensorboard/",
            batch_size=config.batch_size,
            n_steps=config.rollout_length // config.n_envs,
            n_epochs=config.ppo_epochs,
            # mae_batch_size=config.mae_batch_size,
            separate_optimizer=config.separate_optimizer,
            policy_kwargs=policy_kwargs,
            vtt_encoder=vtt_encoder,
            cnn_image=cnn_image,
            cnn_tactile=cnn_tactile,
            structure=config.structure,
            infonce=infonce,
            image_only=config.image_only,
            num_tactiles=num_tactiles,
            augment=config.augment,
            contrastive_limit=config.contrastive_limit,
            contrastive_switch_limit=config.contrastive_switch_limit,
            alternation_loop=config.alternation_loop,
            use_encoder='Concat'
        )
    elif config.use_algo == 'mae':
        import models.pretrain_models
        v = models.pretrain_models.VTT(
            image_size=(64, 64),
            tactile_size=(32, 32),
            image_patch_size=8,
            tactile_patch_size=4,
            dim=config.dim_embedding,
            depth=4,
            heads=4,
            mlp_dim=config.dim_embedding * 2,
            num_tactiles=num_tactiles,
            image_channels=3*config.frame_stack,
            tactile_channels=3*config.frame_stack,
            frame_stack=config.frame_stack,
        )

        mae = VTMAE(
            encoder=v,
            masking_ratio=config.masking_ratio, 
            decoder_dim=config.dim_embedding,  
            decoder_depth=3, 
            decoder_heads=4,
            num_tactiles=num_tactiles,
            early_conv_masking=config.early_conv_masking,
            use_sincosmod_encodings=config.use_sincosmod_encodings,
            frame_stack = config.frame_stack
        )
        if torch.cuda.is_available():
            mae.cuda()
        mae.eval()
        mae.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})
        
        policy = MAEPolicy
        policy_kwargs={
            "mae_model": mae,
            "dim_embeddings": config.dim_embedding,
            "vision_only_control": config.vision_only_control,
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "frame_stack": config.frame_stack,
        }

        model = PPO_MAE(
            policy,
            env,
            verbose=1,
            learning_rate=config.lr_ppo,
            tensorboard_log=config.wandb_dir+"tensorboard/",
            batch_size=config.batch_size,
            n_steps=config.rollout_length // config.n_envs,
            n_epochs=config.ppo_epochs,
            mae_batch_size=config.mae_batch_size,
            separate_optimizer=config.separate_optimizer,
            policy_kwargs=policy_kwargs,
            mae=mae,
        )
    
    elif config.use_algo == 'drqv2':
        cnn_image = CNNEncoder(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='image', env_name=config.env, structure=config.structure)
        if(not config.image_only):cnn_tactile = CNNEncoder(in_channels=3*config.frame_stack, num_tactiles=num_tactiles, encoder_dim=256, key='tactile', env_name=config.env, structure=config.structure)
        else: cnn_tactile = None
        if torch.cuda.is_available():
            cnn_image.cuda()
            if(not config.image_only):cnn_tactile.cuda()
        cnn_image.eval()
        if(not config.image_only):cnn_tactile.eval()

        cnn_image.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})
        if(not config.image_only):cnn_tactile.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})
        
        policy = DrqPolicy
        policy_kwargs={
            "cnn_image": cnn_image,
            "cnn_tactile": cnn_tactile if not config.image_only else None,
            "dim_embeddings": config.dim_embedding,
            "vision_only_control": config.vision_only_control,
            "net_arch": dict(pi=[256, 256], qf=[256, 256]),
            "frame_stack": config.frame_stack,
            "obses": [],
            "image_only": config.image_only,
            "augment": config.augment
        }
        
        # repr_dim is calculated via obs_image.shape (=[batch_size, 64, 256])

        model = DDPG(
            policy,
            # "MultiInputPolicy",
            env,
            verbose=1,
            buffer_size=100,
            # env_name=config.env,
            learning_rate=config.lr_ppo,
            tensorboard_log=config.wandb_dir+"tensorboard/",
            batch_size=config.batch_size,
            # n_steps=config.rollout_length // config.n_envs,
            # rollout_length=config.rollout_length,
            # n_epochs=config.ppo_epochs,
            # mae_batch_size=config.mae_batch_size,
            # separate_optimizer=config.separate_optimizer,
            policy_kwargs=policy_kwargs,
            # policy_kwargs=multi_policy_kwargs,
            # cnn_image=cnn_image,
            # cnn_tactile=cnn_tactile,
            # structure=config.structure,
            # infonce=infonce,
            # image_only=config.image_only,
            # augment=config.augment,
            # contrastive_limit=config.contrastive_limit,
            # contrastive_switch_limit=config.contrastive_switch_limit,
            # alternation_loop=config.alternation_loop
        )
    
    # def count_total_trainable_params(obj):
    #     total_params = 0
    #     for attribute_name in dir(obj):
    #         if not attribute_name.startswith("__"):
    #             try:
    #                 attribute = getattr(obj, attribute_name)
    #             except:
    #                 continue
    #             try:
    #                 if hasattr(attribute, "parameters"):
    #                     total_params += sum(p.numel() for p in attribute.parameters())
    #                     print(f"{attribute_name}: {total_params}")
    #             except:
    #                 continue
    #     return total_params
    # breakpoint()
    
    callbacks = create_callbacks(
        config, model, num_tactiles, objects_all, holders
    )
    model.learn(total_timesteps=config.total_timesteps, callback=callbacks)



if __name__ == "__main__":
    main()
