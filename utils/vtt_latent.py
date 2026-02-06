import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import warnings
import math
from torch.distributions import Normal
import torch.optim as optim
from collections import deque

def initialize_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def create_feature_actions(feature_, action_):
    N = feature_.size(0)
    # Flatten sequence of features.
    f = feature_[:, :-1].view(N, -1)
    n_f = feature_[:, 1:].view(N, -1)
    # Flatten sequence of actions.
    a = action_[:, :-1].view(N, -1)
    n_a = action_[:, 1:].view(N, -1)
    # Concatenate feature and action.
    fa = torch.cat([f, a], dim=-1)
    n_fa = torch.cat([n_f, n_a], dim=-1)
    return fa, n_fa


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


def build_mlp(
    input_dim,
    output_dim,
    hidden_units=[64, 64],
    hidden_activation=nn.Tanh(),
    output_activation=None,
):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def calculate_gaussian_log_prob(log_std, noise):
    return (-0.5 * noise.pow(2) - log_std).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_std.size(-1)


def calculate_log_pi(log_std, noise, action):
    gaussian_log_prob = calculate_gaussian_log_prob(log_std, noise)
    return gaussian_log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(mean, log_std):
    noise = torch.randn_like(mean)
    action = torch.tanh(mean + noise * log_std.exp())
    return action, calculate_log_pi(log_std, noise, action)


def calculate_kl_divergence(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) / q_std).pow_(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

class FixedGaussian(torch.jit.ScriptModule):
    """
    Fixed diagonal gaussian distribution.
    """
    def __init__(self, output_dim, std):
        super(FixedGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    # @torch.jit.script_method
    def forward(self, x):
        mean = torch.zeros(x.size(0), self.output_dim, device=x.device)
        std = torch.ones(x.size(0), self.output_dim, device=x.device).mul_(self.std)
        return mean, std


class Gaussian(torch.jit.ScriptModule):
    """
    Diagonal gaussian distribution with state dependent variances.
    """

    def __init__(self, input_dim, output_dim, hidden_units=(256, 256)):
        super(Gaussian, self).__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=2 * output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(0.2),
        ).apply(initialize_weight)

    # @torch.jit.script_method
    def forward(self, x):
        x = self.net(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 1e-5
        return mean, std


class LatentModel(nn.Module):
    """
    Stochastic latent variable model to estimate latent dynamics and the reward.
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        img_feature_dim=288,
        z1_dim=32,
        z2_dim=256,
        kld_coef=1e-4,
        reward_coef=1e-4,
        image_coef=1e-5,
        hidden_units=(256, 256),
        encoder='VTT'
    ):
        super(LatentModel, self).__init__()

        # p(z1(0)) = N(0, I)
        self.z1_prior_init = FixedGaussian(z1_dim, 1.0)
        # p(z2(0) | z1(0))
        self.z2_prior_init = Gaussian(z1_dim, z2_dim, hidden_units)
        # p(z1(t+1) | z2(t), a(t))
        self.z1_prior = Gaussian(
            z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        self.image_coef = image_coef
        self.kld_coef = kld_coef
        self.reward_coef = reward_coef
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_prior = Gaussian(
            z1_dim + z2_dim + action_shape[0],
            z2_dim,
            hidden_units,
        )

        # q(z1(0) | feat(0), tactile(0))
        self.z1_posterior_init = Gaussian(img_feature_dim, z1_dim, hidden_units)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), tactile(t+1), z2(t), a(t))
        self.z1_posterior = Gaussian(
            img_feature_dim  + z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_posterior = self.z2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward = Gaussian(
            2 * z1_dim + 2 * z2_dim + action_shape[0],
            1,
            hidden_units,
        )

        # feat(t) = Encoder(x(t))
        if encoder == 'VTT':
            self.encoder = VTT(tactile_dim=6*32*32)
        elif encoder == "POE":
            self.encoder = PoE_Encoder()
        else:
            self.encoder = Concatenation_Encoder()
        # p(x(t) | z1(t), z2(t))
        self.decoder = Decoder(
            z1_dim + z2_dim,
            state_shape[0],
            np.sqrt(0.1),
        )

        self.BCE_loss = nn.BCELoss()
        torch.manual_seed(0)
        self.apply(initialize_weight)

    # @torch.jit.script_method
    def sample_prior(self, actions_):
        z1_mean_ = []
        z1_std_ = []

        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.z1_prior_init(actions_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_prior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_mean_.append(z1_mean)
        z1_std_.append(z1_std)
        for t in range(1, actions_.size(1) + 1):
            # p(z1(t) | z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_prior(torch.cat([z2, actions_[:, t - 1]], dim=1))
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # p(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_prior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)

        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)

        return (z1_mean_, z1_std_)

    # @torch.jit.script_method
    def sample_posterior(self, features_, actions_):
        z1_mean_ = []
        z1_std_ = []
        z1_ = []
        z2_ = []

        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_mean_.append(z1_mean)
        z1_std_.append(z1_std)
        z1_.append(z1)
        z2_.append(z2)

        for t in range(1, actions_.size(1) + 1):
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2, actions_[:, t - 1]], dim=1))
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)
        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)
        return (z1_mean_, z1_std_, z1_, z2_)

    # @torch.jit.script_method
    def calculate_loss(self, state_, tactile_, action_, reward_, done_, force_norm):
        # action_ = self.truncate(action_)
        # Calculate the sequence of features.
        if(tactile_.shape[-1] > 6):
            tactile_ = tactile_[:, :, :6]
        feature_, aligment_check, contact_check = self.encoder(state_, tactile_)
        # print('after shape: ',feature_.shape, action_.shape)
        # Sample from latent variable model.
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_)

        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum() * self.kld_coef

        # Prediction loss of images.
        z_ = torch.cat([z1_, z2_], dim=-1)
        state_mean_, state_std_ = self.decoder(z_)
        state_noise_ = (state_ - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_ = ((-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)) * self.image_coef
        loss_image = -log_likelihood_.mean(dim=0).sum()

        # alignment check
        aligned = torch.ones_like(aligment_check)
        alignment_loss = self.BCE_loss(aligment_check, aligned)

        # contact check
        contact_mag = torch.linalg.norm(tactile_, dim=2).unsqueeze(dim=2)
        non_contact = torch.zeros_like(contact_mag)
        contact = torch.ones_like(contact_mag)
        contact_sign = torch.where(contact_mag > (15.0/force_norm), contact, non_contact)
        contact_loss = self.BCE_loss(contact_check, contact_sign)
        # Prediction loss of rewards.
        x = torch.cat([z_[:, :-1], action_, z_[:, 1:]], dim=-1)
        B, S, X = x.shape
        reward_mean_, reward_std_ = self.reward(x.view(B * S, X))
        reward_mean_ = reward_mean_.view(B, S, 1)
        reward_std_ = reward_std_.view(B, S, 1)
        reward_noise_ = (reward_ - reward_mean_) / (reward_std_ + 1e-8)
        log_likelihood_reward_ = ((-0.5 * reward_noise_.pow(2) - reward_std_.log()) - 0.5 * math.log(2 * math.pi)) * self.reward_coef
        loss_reward = -log_likelihood_reward_.mul_(1 - done_).mean(dim=0).sum()
        return loss_kld, loss_image, loss_reward, alignment_loss, contact_loss

    # @torch.jit.script_method
    def calculate_alignment_loss(self, state_, tactile_):
        feature_, aligment_check, contact_check = self.encoder(state_, tactile_)
        aligned = torch.zeros_like(aligment_check)
        alignment_loss = self.BCE_loss(aligment_check, aligned)
        return alignment_loss

class VTT(nn.Module):
    def __init__(self, tactile_dim, img_size=[84], img_patch_size=14, tactile_patches=2, in_chans=3, embed_dim=384, depth=6,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, tactile_to_dim=6, **kwargs):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], img_patch_size=img_patch_size, tactile_patch=tactile_patches,
            in_chan=in_chans, embeded_dim=embed_dim
        )
        img_patches = self.patch_embed.img_patches

        # contact embedding, alignment embedding and position embedding
        self.contact_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.align_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, img_patches + self.patch_embed.tactile_patch + 2, embed_dim))
        self.tactile_trunc = nn.Linear(tactile_dim, tactile_to_dim)
        self.raw = []

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.compress_patches = nn.Sequential(nn.Linear(embed_dim, embed_dim//4),
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(embed_dim//4, embed_dim//12))

        self.compress_layer = nn.Sequential(nn.Linear((img_patches + self.patch_embed.tactile_patch + 2)*embed_dim//12, 640),
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(640, 288))

        self.align_recognition = nn.Sequential(nn.Linear(embed_dim, 1),
                                               nn.Sigmoid())

        self.contact_recognition = nn.Sequential(nn.Linear(embed_dim, 1),
                                                 nn.Sigmoid())

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.align_embed, std=.02)
        trunc_normal_(self.contact_embed, std=.02)
    
    def initialize_training(self, train_args):

        # training parameters
        lr = train_args['lr']
        
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.batch_size = train_args['batch_size']

    def get_embeddings(self, x, eval=True, use_vision=True, use_tactile=True, key='image'):

        if eval:
            self.eval()
        else:
            self.train()

        """
        x.key: 'image' 'tactile1' 'tactile2'
        key = ['image', 'tactile']
        """

        assert(key in ['image', 'tactile'])
        if 'image' in x.keys():
            device = x['image'].device
        else:
            device = x['tactile'].device
            use_vision = False

        print('shape: ', x['image'].shape, x['tactile'].shape)
        x_image, x_tactile = self.get_truncate(x['image'], x['tactile'])

        print("qw image shape: ", x_image.shape)
        print("qw tactile shape: ", x_tactile.shape)     
        raw_res, _, _ = self.forward(x_image, x_tactile)
        # self.raw.append(raw_res)

        raw_res = raw_res.float().mean(dim=1)
        # print('after: ', raw_res.shape) [32, 288]

        return raw_res
    
    def get_truncate(self, x_image, x_tactile):
        padding = (10, 10, 10, 10)
        x_image = F.pad(x_image, padding)
        x_tactile = self.tactile_trunc(x_tactile)

        return x_image, x_tactile

    def interpolate_pos_encoding(self, x, w: int, h: int):
        npatch = x.shape[2] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        else:
            raise ValueError('Position Encoder does not match dimension')

    def prepare_tokens(self, x, tactile):
        B, S, nc, w, h = x.shape
        x, patched_tactile = self.patch_embed(x, tactile)
        x = torch.cat((x, patched_tactile),dim=2)
        alignment_embed = self.align_embed.expand(B, S, -1, -1)
        contact_embed = self.contact_embed.expand(B, S, -1, -1)
        # introduce contact embedding & alignment embedding
        x = torch.cat((contact_embed, x), dim=2)
        x = torch.cat((alignment_embed, x), dim=2)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def forward(self, x, tactile):
        x = self.prepare_tokens(x, tactile)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        img_tactile = self.compress_patches(x)
        B, S, patches, dim = img_tactile.size()
        img_tactile = img_tactile.view(B, S, -1)
        img_tactile = self.compress_layer(img_tactile)
        # print('after: ', img_tactile.shape,B,S,x.shape)
        return img_tactile, self.align_recognition(x[:, :, 0]), self.contact_recognition(x[:, :, 1])


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, S, N, C = x.shape
        qkv = self.qkv(x).reshape(B*S, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, S, N, C)
        attn = attn.view(B, S, -1, N, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=84, tactile_dim = 6, img_patch_size=14, tactile_patch=2, in_chan=3, embeded_dim=384):
        super().__init__()
        self.img_patches = int((img_size/img_patch_size)*(img_size/img_patch_size))
        self.img_size = img_size
        self.embeded_dim = embeded_dim
        self.proj = nn.Conv2d(in_chan, embeded_dim, kernel_size=img_patch_size, stride=img_patch_size)
        self.tactile_patch = tactile_patch
        self.decode_tactile = nn.Sequential(nn.Linear(tactile_dim, self.tactile_patch*embeded_dim))

    def forward(self, image, tactile):
        # Input shape batch, Sequence, in_Channels, H, W
        # Output shape batch, Sequence, patches & out_Channels
        B, S, C, H, W = image.shape
        image = image.view(B * S, C, H, W)
        pached_image = self.proj(image).flatten(2).transpose(1, 2).view(B, S, -1, self.embeded_dim)
        tactile = tactile.view(B*S, -1)
        decoded_tactile = self.decode_tactile(tactile).view(B, S, self.tactile_patch, -1)
        return pached_image, decoded_tactile


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, return_attention: bool = False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,)*(x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.MLP = nn.Sequential(nn.Linear(in_features, hidden_features),
                            act_layer(),
                            nn.Linear(hidden_features, out_features))
    def forward(self, x):
        x = self.MLP(x)
        return x


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class Concatenation_Encoder(nn.Module):
    """
    Concatenation
    """
    def __init__(self, input_dim=3, tactile_dim=6, img_dim=256, tactile_latent_dim=32):
        super(Concatenation_Encoder, self).__init__()

        self.img_net = nn.Sequential(
            # (3, 84, 84) -> (42, 42, 42)
            nn.Conv2d(input_dim, 16, 6, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 42, 42) -> (21, 21, 21)
            nn.Conv2d(16, 32, 6, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 21, 21) -> (128, 11, 21)
            nn.Conv2d(32, 64, 6, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 21, 21) -> (256, 11, 11)
            nn.Conv2d(64, 128, 6, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 11, 11) -> (256, 6,)
            nn.Conv2d(128, img_dim, 3, 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.tactile_net = nn.Sequential(
            nn.Tanh(),
            nn.Linear(tactile_dim, tactile_latent_dim),
            nn.LayerNorm(tactile_latent_dim),
            nn.LeakyReLU(0.2, inplace=True))

        self.tactile_recognize = nn.Sequential(nn.Linear(tactile_latent_dim + img_dim, 1),
                                               nn.Sigmoid())

        self.alignment_recognize = nn.Sequential(nn.Linear(tactile_latent_dim + img_dim, 1),
                                                 nn.Sigmoid())

        self.bottle_neck = nn.Sequential(nn.Linear(tactile_latent_dim + img_dim, tactile_latent_dim + img_dim))
        self.img_norm = nn.LayerNorm(img_dim)
        self.tactile_norm = nn.LayerNorm(tactile_latent_dim)
        self.layer_norm = nn.LayerNorm(tactile_latent_dim + img_dim)

    def forward(self, img, tactile):
        B, S, C, H, W = img.size()
        img = img.view(B * S, C, H, W)
        img_x = self.img_norm(self.img_net(img).view(B * S, -1))
        tactile = tactile.view(B * S, -1)
        tactile_x = self.tactile_norm(self.tactile_net(tactile))
        x = torch.cat((img_x, tactile_x), dim=1)
        x = self.layer_norm(x)
        x = x.view(B, S, -1)
        return self.bottle_neck(x), self.tactile_recognize(x), self.alignment_recognize(x)


class ImageEncoder(nn.Module):
    def __init__(self, input_dim=3, img_dim=256):
        super(ImageEncoder, self).__init__()
        self.img_net = nn.Sequential(
            # (3, 84, 84) -> (42, 42, 42)
            nn.Conv2d(input_dim, 16, 6, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 42, 42) -> (21, 21, 21)
            nn.Conv2d(16, 32, 6, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 21, 21) -> (128, 11, 21)
            nn.Conv2d(32, 64, 6, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 21, 21) -> (256, 11, 11)
            nn.Conv2d(64, 128, 6, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 11, 11) -> (256, 6,)
            nn.Conv2d(128, img_dim, 3, 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.img_norm = nn.LayerNorm(img_dim)

    def forward(self, x):
        B, S, C, H, W = x.size()
        x = x.view(B * S, C, H, W)
        img_x = self.img_norm(self.img_net(x).view(B * S, -1))
        return img_x


class TactileEncoder(nn.Module):
    def __init__(self, tactile_dim=6, tactile_latent_dim=32):
        super(TactileEncoder, self).__init__()
        self.tactile_net = nn.Sequential(
            nn.Tanh(),
            nn.Linear(tactile_dim, tactile_latent_dim),
            nn.LayerNorm(tactile_latent_dim),
            nn.LeakyReLU(0.2, inplace=True))
        self.tactile_norm = nn.LayerNorm(tactile_latent_dim)

    def forward(self, tactile):
        B, S, D = tactile.size()
        tactile = tactile.view(B * S, -1)
        tactile_x = self.tactile_norm(self.tactile_net(tactile))
        return tactile_x


class PoE_Encoder(nn.Module):
    def __init__(self, input_dim=3, tactile_dim=6, z_dim=288):
        super(PoE_Encoder, self).__init__()

        self.z_dim = z_dim
        self.img_encoder = ImageEncoder(input_dim, img_dim=z_dim * 2)
        self.tac_encoder = TactileEncoder(tactile_dim, tactile_latent_dim=z_dim * 2)

        self.z_prior_m = torch.nn.Parameter(
            torch.zeros(1, self.z_dim), requires_grad=False
        )

        self.z_prior_v = torch.nn.Parameter(
            torch.ones(1, self.z_dim), requires_grad=False
        )

        self.z_prior = (self.z_prior_m, self.z_prior_v)
        self.tactile_recognize = nn.Sequential(nn.Linear(z_dim, 1), nn.Sigmoid())
        self.alignment_recognize = nn.Sequential(nn.Linear(z_dim, 1), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(z_dim)

    def sample_gaussian(self, m, v, device):
        epsilon = Normal(0, 1).sample(m.size())
        z = m + torch.sqrt(v) * epsilon.to(device)
        return z

    def gaussian_parameters(self, h, dim: int = -1):
        m, h = torch.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        return m, v

    def product_of_experts(self, m_vect, v_vect):
        T_vect = 1.0 / v_vect

        mu = (m_vect * T_vect).sum(2) * (1 / T_vect.sum(2))
        var = 1 / T_vect.sum(2)

        return mu, var

    def duplicate(self, x, rep):
        return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])

    def forward(self, img, tac):
        batch_dim = img.size()[0]
        sequence_dim = img.size()[1]

        temp_dim = batch_dim
        temp_dim *= sequence_dim

        img_out = self.img_encoder(img).unsqueeze(2)
        tac_out = self.tac_encoder(tac).unsqueeze(2)

        # multimodal fusion
        mu_z_img, var_z_img = self.gaussian_parameters(img_out, dim=1)  # B*S, 128
        mu_z_frc, var_z_frc = self.gaussian_parameters(tac_out, dim=1)  # B*S, 128

        mu_prior, var_prior = self.z_prior  # 1, 128 for both

        # B*S, 128, 1
        mu_prior_resized = mu_prior.expand(temp_dim, *mu_prior.shape).reshape(-1, *mu_prior.shape[1:]).unsqueeze(2)
        var_prior_resized = var_prior.expand(temp_dim, *var_prior.shape).reshape(-1, *var_prior.shape[1:]).unsqueeze(2)

        m_vect = torch.cat([mu_z_img, mu_z_frc, mu_prior_resized], dim=2)
        var_vect = torch.cat([var_z_img, var_z_frc, var_prior_resized], dim=2)
        mu_z, var_z = self.product_of_experts(m_vect, var_vect)
        # Sample Gaussian to get latent
        z = self.sample_gaussian(mu_z, var_z, img.device)
        z = self.layer_norm(z)
        # z at this point has shape B*S, z_dim
        z = z.reshape(batch_dim, sequence_dim, -1)
        contact_binary, align_binary = self.tactile_recognize(z), self.alignment_recognize(z)
        return z, contact_binary, align_binary

class Decoder(torch.jit.ScriptModule):
    """
    Decoder.
    """
    def __init__(self, input_dim=288, output_dim=3, std=1.0):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 5, 5)
            nn.ConvTranspose2d(input_dim, 256, 5),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 5, 5) -> (128, 10, 10)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 10, 10) -> (64, 21, 21)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 21, 21) -> (32, 42, 42)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 42, 42) -> (3, 84, 84)
            nn.ConvTranspose2d(32, output_dim, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.std = std
    @torch.jit.script_method
    def forward(self, x):
        B, S, latent_dim = x.size()
        x = x.view(B * S, latent_dim, 1, 1)
        x = self.net(x)
        _, C, W, H = x.size()
        x = x.view(B, S, C, W, H)
        return x, torch.ones_like(x).mul_(self.std)

from collections import deque

import numpy as np
import torch


class LazyFrames:
    """
    Stacked frames which never allocate memory to the same frame.
    """

    def __init__(self, frames):
        self._frames = list(frames)

    def __array__(self, dtype):
        return np.array(self._frames, dtype=dtype)

    def __len__(self):
        return len(self._frames)


class SequenceBuffer:
    """
    Buffer for storing sequence data.
    """

    def __init__(self, num_sequences=8):
        self.num_sequences = num_sequences
        self._reset_episode = False
        self.state_ = deque(maxlen=self.num_sequences + 1)
        self.tactile_ = deque(maxlen=self.num_sequences + 1)
        self.action_ = deque(maxlen=self.num_sequences)
        self.reward_ = deque(maxlen=self.num_sequences)
        self.done_ = deque(maxlen=self.num_sequences)

    def reset(self):
        self._reset_episode = False
        self.state_.clear()
        self.tactile_.clear()
        self.action_.clear()
        self.reward_.clear()
        self.done_.clear()

    def reset_episode(self, state, tactile,):
        assert not self._reset_episode
        self._reset_episode = True
        self.state_.append(state)
        self.tactile_.append(tactile)

    def append(self, action, reward, done, next_state, next_tactile):
        assert self._reset_episode
        self.action_.append(action)
        self.reward_.append([reward])
        self.done_.append([done])
        self.state_.append(next_state)
        self.tactile_.append(next_tactile)

    def get(self):
        state_ = LazyFrames(self.state_)
        tactile_ = np.array(self.tactile_, dtype=np.float32)
        action_ = np.array(self.action_, dtype=np.float32)
        reward_ = np.array(self.reward_, dtype=np.float32)
        done_ = np.array(self.done_, dtype=np.float32)
        return state_, tactile_,  action_, reward_, done_

    def is_empty(self):
        return len(self.reward_) == 0

    def is_full(self):
        return len(self.reward_) == self.num_sequences

    def __len__(self):
        return len(self.reward_)
    