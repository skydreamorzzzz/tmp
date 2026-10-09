"""
This code is borrowed from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
"""

import enum
import math
import os
from datetime import datetime

import numpy as np
import torch as th

from abc import ABC, abstractmethod
import torch.distributed as dist

# This code is based on https://github.com/openai/guided-diffusion
import numpy as np
import torch as th

from utils.utils import *
from utils.plot_script import *
from utils.paramUtil import *

from models.losses import InterLoss, GeometricLoss
from models.GLI import TopologyBatch


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.
    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.
    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.
        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class UniformSamplerGeneric():
    def sample(self, lengths, device):
        B = len(lengths)
        numbers = []
        for i in range(B):
            number = th.randint(low=0, high=int(lengths[i].item()), size=[1])[0]
            numbers.append(number)
        return th.LongTensor(numbers).to(device)


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.
        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.
        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.
        Sub-classes should override this method to update the reweighting
        using losses from the model.
        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.
        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + th.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
            self,
            *,
            betas,
            model_mean_type,
            model_var_type,
            loss_type,
            rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, 2 * C, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
                p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def condition_score_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, t, p_mean_var, **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            pre_seq=None,
            transl_req=None,
            model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # concat seq
        if pre_seq is not None:
            T = pre_seq.shape[2]
            noise = th.randn_like(pre_seq)
            x_t = self.q_sample(pre_seq, t, noise=noise)
            x[:, :, :T] = x_t

        if transl_req is not None:
            for item in transl_req:
                noise = th.randn(2).type_as(x)
                transl = th.Tensor(item[1:]).type_as(x)
                x_t = self.q_sample(transl, t, noise=noise)
                x[:, :2, item[0]] = x_t

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            pre_seq=None,
            transl_req=None,
            progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                pre_seq=pre_seq,
                transl_req=transl_req,
                progress=progress,
        ):
            final = sample
            # final.append(sample["pred_xstart"])
        # return torch.cat(final, dim=0)
        return final["sample"]

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            pre_seq=None,
            transl_req=None,
            progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    pre_seq=pre_seq,
                    transl_req=transl_req
                )
                yield out
                img = out["sample"]

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"]}

    def ddim_sample_with_grad(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn, out_orig, x, t,
                                                     model_kwargs=model_kwargs)
            else:
                out = out_orig

        out["pred_xstart"] = out["pred_xstart"].detach()

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"].detach()}

    def ddim_reverse_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
                      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                      - out["pred_xstart"]
              ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_next)
                + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            skip_timesteps=0,
            init_image=None,
            randomize_class=False,
            cond_fn_with_grad=False,
            dump_steps=None,
            const_noise=False,
            x_start=None
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        # final = []
        for i, sample in enumerate(self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
                skip_timesteps=skip_timesteps,
                init_image=init_image,
                randomize_class=randomize_class,
                cond_fn_with_grad=cond_fn_with_grad,
                x_start=x_start
        )):
            pass
            # final.append(sample["pred_xstart"])
            # return torch.cat(final, dim=0)
        return sample["pred_xstart"]

    def ddim_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            skip_timesteps=0,
            init_image=None,
            randomize_class=False,
            cond_fn_with_grad=False,
            x_start=None
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            if x_start is not None:
                B, T, D = img.shape
                img[:, :, [0, 2]] = x_start[:, :T, [0, 2]]
                img[:, :, [262, 264]] = x_start[:, :T, [262, 264]]
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, control=False):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        ######################### fine-tuning for control #################################
        if control:
            B, T, D = x_t.shape
            x_t[:, :, [0, 2]] = x_start[:, :T, [0, 2]]
            x_t[:, :, [262, 264]] = x_start[:, :T, [262, 264]]
        ###################################################################################

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2).view(-1, 1).mean(-1)
            # if "vb" in terms:
            #     terms["loss"] = terms["mse"] + terms["vb"]
            # else:
            #     terms["loss"] = terms["mse"]
            terms["target"] = target
            terms["pred"] = model_output
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class MotionDiffusion(GaussianDiffusion):

    def __init__(self, use_timesteps, motion_rep, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.motion_rep = motion_rep
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        # print(self.timestep_map)
        kwargs["betas"] = np.array(new_betas)

        # 调用父类初始化，这会设置 num_timesteps 等属性
        super().__init__(**kwargs)

        self.normalizer = MotionNormalizerTorch()
        self.topology_loss = TopologyBatch(t2m_kinematic_chain, 22)
        self.t_bar = 900
        self.t_bar_gli = 100

    def p_mean_variance(
            self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def p_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        sampling_info = None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            pred_x_start = out["pred_xstart"]

            noise = th.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            if cond_fn is not None:
                out["mean"] = self.condition_score_with_grad(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        sampling_info['t'] = t
        # 调用 gradients 的条件：
        # 1. (t <= 700 and unrelated存在) 或 t <= 100: 触发 Unrelated Loss 和 GLI Loss
        # 2. t <= 900: 触发 Group/Person Distance Loss（在函数内部有独立检查）
        need_unrelated_gli = (t <= self.t_bar and sampling_info.get('unrelated')) or t <= self.t_bar_gli
        # 检查是否有inter_graph和Interaction_Definitions（新系统）
        has_new_system = 'inter_graph' in sampling_info and sampling_info.get('inter_graph') is not None and 'Interaction_Definitions' in sampling_info.get('inter_graph', {})
        need_distance_direction = t <= 900 and has_new_system
        
        if need_unrelated_gli or need_distance_direction:
            # 使用未detach的pred_x_start计算梯度
            loss, gradient = self.gradients(pred_x_start, x, sampling_info)
            # 梯度缩放：使用更大的缩放因子来确保loss有效下降
            # 根据时间步动态调整：早期使用较大梯度进行粗调，后期保持稳定精调
            t_normalized = t / 1000.0  # 归一化到[0, 1]
            # 早期(t>500): scale=1.0, 后期(t<500): scale逐渐减小到0.5
            gradient_scale = 5 + t_normalized * 5
            sample = sample - gradient_scale * gradient
            # 计算完梯度后再detach，避免内存泄漏
            pred_x_start = pred_x_start.detach()

        return {"sample": sample, "pred_xstart": pred_x_start if need_unrelated_gli or need_distance_direction else out["pred_xstart"].detach()}

    def ddim_sample_control_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            skip_timesteps=0,
            init_image=None,
            randomize_class=False,
            cond_fn_with_grad=False,
            dump_steps=None,
            const_noise=False,
            x_start=None
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        # final = []
        for i, sample in enumerate(self.ddim_sample_control_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
                skip_timesteps=skip_timesteps,
                init_image=init_image,
                randomize_class=randomize_class,
                cond_fn_with_grad=cond_fn_with_grad,
                x_start=x_start
        )):
            pass
            # final.append(sample["pred_xstart"])
            # return torch.cat(final, dim=0)
        return sample["pred_xstart"]

    def ddim_sample_control_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            skip_timesteps=0,
            init_image=None,
            randomize_class=False,
            cond_fn_with_grad=False,
            x_start=None
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            if x_start is not None:
                B, T, D = img.shape
                img[:, :, [0, 2]] = x_start[:, :T, [0, 2]]
                img[:, :, [262, 264]] = x_start[:, :T, [262, 264]]
            with th.no_grad():
                sample_fn = self.ddim_sample_control
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def ddim_sample_control(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn, out_orig, x, t,
                                                     model_kwargs=model_kwargs)
            else:
                out = out_orig

        pred_x_start = out["pred_xstart"]
        out["pred_xstart"] = out["pred_xstart"].detach()

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        loss, gradient = self.gradients_ddim(pred_x_start, x)
        sample = sample - gradient

        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"].detach()}


    def p_sample_loop_multi(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        inter_graph = None,
        distance_weight=1.0,
        angle_weight=0.01,
        motion_text_map=None,
        ablation=None,
        person_model=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        denoise = []
        if dump_steps is not None:
            dump = []

        for i, sample in enumerate(self.p_sample_loop_progressive_multi(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
            inter_graph = inter_graph,
            distance_weight = distance_weight,
            angle_weight = angle_weight,
            motion_text_map = motion_text_map,
            ablation = ablation,
            person_model = person_model,
        )):
            final = sample
        return final

    def p_sample_loop_progressive_multi(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
        inter_graph=None,
        distance_weight=1.0,
        angle_weight=0.01,
        motion_text_map=None,
        ablation=None,
        person_model=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().

        In one timestep, we sample each character separately, and do sampling guidance in p_sample_with_grad().
        Different texts can be assigned to different characters.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        # 获取总人数（从idx_to_char获取）
        if 'idx_to_char' not in inter_graph:
            raise ValueError("inter_graph中缺少'idx_to_char'字段，请使用新的Interaction_Definitions格式")
        n_person = len(inter_graph['idx_to_char'])

        sampling_info = {}  # Dictionary to hold additional sampling context/info
        
        # 存储引导头权重
        sampling_info['distance_weight'] = distance_weight
        sampling_info['angle_weight'] = angle_weight
        
        # 存储消融选项
        if ablation is not None:
            sampling_info['ablation'] = ablation

        imgs = [th.randn(*shape, device=device) for i in range(n_person)]
        starts = [th.randn(*shape, device=device) for i in range(n_person)]  # Store predicted x_start for each person

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        # original_cond = model_kwargs.get('cond', None)
        
        # 使用新的Group/Person系统
        if 'Interaction_Definitions' not in inter_graph:
            raise ValueError("inter_graph中缺少'Interaction_Definitions'字段，请使用新的Group/Person交互格式")
        
        # ========== Group/Person系统 ==========
        # 根据Interaction_Definitions顺序生成
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            sampling_info['t'] = t
            sampling_info['inter_graph'] = inter_graph
            
            # 按照Interaction_Definitions顺序处理每个交互
            for interaction_def in inter_graph['Interaction_Definitions']:
                # 每次交互前更新 all_starts（确保使用最新的 starts）
                sampling_info['all_starts'] = starts
                
                interaction_type = interaction_def['interaction_type']
                interaction_id = interaction_def['Interaction_id']
                
                if interaction_type == 'Group':
                    # Group交互：拼接两人，使用forward
                    group_name = interaction_id
                    if group_name not in inter_graph['groups']:
                        continue
                    
                    idx1, idx2 = inter_graph['groups'][group_name]
                    
                    # 拼接两人的噪声 (B, T, 2D)
                    x_concat = th.cat([imgs[idx1], imgs[idx2]], dim=-1)
                    
                    # 准备model_kwargs（按交互切换文本条件）
                    # motion_text_map 现在可传入 {key: cond_embedding}：
                    # - Group: key=group_name（如"Group1"）
                    # - Person: key=person_idx（int）
                    cond_group = model_kwargs.get('cond')
                    if isinstance(motion_text_map, dict) and group_name in motion_text_map:
                        _v = motion_text_map[group_name]
                        # 兼容：只有当传入的是预编码的 embedding（Tensor）时才覆盖cond
                        if isinstance(_v, th.Tensor):
                            cond_group = _v
                    model_kwargs_group = {
                        'mask': model_kwargs.get('mask'),
                        'cond': cond_group,
                        'b': None  # forward不需要b
                    }
                    
                    # 准备sampling_info
                    sampling_info['cond'] = None
                    sampling_info['unrelated'] = []
                    sampling_info['out_unrelated'] = []
                    sampling_info['current_indices'] = [idx1, idx2]  # 传递当前处理的索引
                    
                    # 调用p_sample_with_grad
                    out = self.p_sample_with_grad(
                        model,
                        x_concat,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs_group,
                        sampling_info=sampling_info,
                    )
                    
                    # 分割输出
                    sample = out["sample"]
                    pred_xstart = out.get("pred_xstart", sample)
                    D = shape[2]
                    
                    imgs[idx1] = sample[..., :D]
                    imgs[idx2] = sample[..., D:]
                    # 分割pred_xstart并保留梯度（如果pred_xstart有梯度的话）
                    starts[idx1] = pred_xstart[..., :D]
                    starts[idx2] = pred_xstart[..., D:]
                    
                elif interaction_type == 'Person':
                    # Person交互：单人以目标Group的平均动作为条件（b_avg）进行生成
                    person_char_ids = [obj for obj in interaction_def['Interaction_objects'] if obj.startswith('P')]
                    if not person_char_ids:
                        continue
                    person_id = person_char_ids[0]
                    person_idx = inter_graph['char_to_idx'][person_id]
                    
                    target_group_name = inter_graph['person_to_group'].get(person_idx)
                    if not target_group_name:
                        continue

                    # 计算目标Group的平均（在当前扩散状态 x_t 空间）
                    target_members = inter_graph['groups'].get(target_group_name, [])
                    if not target_members:
                        continue
                    group_motions = [imgs[idx] for idx in target_members]
                    b_avg = th.mean(th.stack(group_motions, dim=0), dim=0)
                    
                    # 准备model_kwargs（按交互切换文本条件）
                    cond_person = model_kwargs.get('cond')
                    if isinstance(motion_text_map, dict) and person_idx in motion_text_map:
                        _v = motion_text_map[person_idx]
                        # 兼容：只有当传入的是预编码的 embedding（Tensor）时才覆盖cond
                        if isinstance(_v, th.Tensor):
                            cond_person = _v
                    model_kwargs_person = {
                        'mask': model_kwargs.get('mask'),
                        'cond': cond_person,
                        'b': b_avg  # 使用Group平均作为条件，触发 single_forward
                    }
                    
                    # 准备sampling_info
                    # 注意：新系统不需要unrelated节点，设置为空列表
                    sampling_info['cond'] = None
                    sampling_info['unrelated'] = []
                    sampling_info['out_unrelated'] = []
                    sampling_info['current_indices'] = [person_idx]

                    # Person 交互优先使用 person_model（FreeMotion 单人去噪器），
                    # 若未提供则回退到默认的 Group 去噪器
                    _person_denoiser = person_model if person_model is not None else model

                    # 调用p_sample_with_grad
                    out = self.p_sample_with_grad(
                        _person_denoiser,
                        imgs[person_idx],
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs_person,
                        sampling_info=sampling_info,
                    )
                    
                    imgs[person_idx] = out["sample"]
                    starts[person_idx] = out.get("pred_xstart", out["sample"])
                    
                yield imgs

    # =========================================================
    # FreeMotion 原始 in/out 格式采样循环
    # =========================================================

    def p_sample_loop_freemotion(
        self,
        person_model,
        shape,
        inter_graph,
        motion_text_map=None,
        pair_constraints=None,
        distance_weight=1.0,
        device=None,
        progress=True,
    ):
        """
        FreeMotion 风格的多人采样：所有人均使用 single_forward()，
        条件为 in_nodes 邻居的前一步均值。

        Args:
            person_model:   FreeMotion 去噪器（CFGAdaptiveModel）
            shape:          (B, T, D) 单人噪声形状
            inter_graph:    {'in': [[...], ...], 'out': [[...], ...]}
            motion_text_map:{person_idx: cond_embedding (Tensor, shape [1, 768])}
            pair_constraints:{(i, j): (lower, upper)} 所有人对的距离约束
                              - 连通对  → LLM interactive_distances 对应的 (lower, upper)
                              - 非连通对 → (non_interactive_lower, inf)
            distance_weight:梯度引导强度
            device:         目标设备
        """
        final = None
        for final in self.p_sample_loop_freemotion_progressive(
            person_model=person_model,
            shape=shape,
            inter_graph=inter_graph,
            motion_text_map=motion_text_map or {},
            pair_constraints=pair_constraints or {},
            distance_weight=distance_weight,
            device=device,
            progress=progress,
        ):
            pass
        return final

    def p_sample_loop_freemotion_progressive(
        self,
        person_model,
        shape,
        inter_graph,
        motion_text_map=None,
        pair_constraints=None,
        distance_weight=1.0,
        device=None,
        progress=True,
    ):
        """
        FreeMotion 原始采样循环（生成器）。

        每个时间步：
          1. 保存本步开始时所有人的估计（prev_imgs），避免顺序依赖
          2. 对每个人 i：b_avg = mean(prev_imgs[j] for j in in_nodes[i])
          3. 调用 person_model.single_forward()（通过 b 参数触发 CFGAdaptiveModel）
          4. 若 t ≤ 900，施加距离梯度引导

        Yields:
            List[Tensor] — 每步所有人的当前噪声估计
        """
        if device is None:
            device = next(person_model.parameters()).device

        n_person = len(inter_graph["in"])
        in_nodes  = inter_graph["in"]   # list of lists

        imgs   = [th.randn(*shape, device=device) for _ in range(n_person)]
        starts = [th.zeros(*shape, device=device)  for _ in range(n_person)]

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices, desc="FreeMotion sampling")

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            # 保存上一步估计（所有人同时使用，避免顺序更新偏差）
            prev_imgs = [img.clone().detach() for img in imgs]

            for person_idx in range(n_person):
                # 条件 b：in_nodes 邻居的均值
                nbrs = in_nodes[person_idx]
                if nbrs:
                    b_avg = th.mean(
                        th.stack([prev_imgs[j] for j in nbrs], dim=0), dim=0
                    )
                else:
                    b_avg = th.zeros_like(prev_imgs[0])

                # 文本条件
                cond = (motion_text_map or {}).get(person_idx, None)
                model_kwargs = {"mask": None, "cond": cond, "b": b_avg}

                # ── 带梯度的采样步 ──────────────────────────────
                with th.enable_grad():
                    x_g = imgs[person_idx].detach().requires_grad_(True)
                    out = self.p_mean_variance(
                        person_model, x_g, t,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                    )
                    pred_xstart = out["pred_xstart"]

                    noise = th.randn_like(x_g)
                    nonzero_mask = (t != 0).float().view(
                        -1, *([1] * (len(x_g.shape) - 1))
                    )
                sample = (
                    out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
                )

                # ── 距离梯度引导（t ≤ 900）──────────────────────
                if i <= 900 and pair_constraints:
                    sample = self._apply_freemotion_dist_guidance(
                        sample=sample,
                        pred_xstart=pred_xstart,
                        x_g=x_g,
                        person_idx=person_idx,
                        all_starts=starts,
                        pair_constraints=pair_constraints,
                        distance_weight=distance_weight,
                        t_val=i,
                    )

                imgs[person_idx]   = sample.detach()
                starts[person_idx] = pred_xstart.detach()

            yield imgs

    def _apply_freemotion_dist_guidance(
        self,
        sample,
        pred_xstart,
        x_g,
        person_idx,
        all_starts,
        pair_constraints,
        distance_weight,
        t_val,
    ):
        """
        对当前人（person_idx）施加与其他人的距离梯度引导。

        pair_constraints[(i, j)] = (lower, upper)
          lower/upper 单位：米；upper=inf 表示只有下界约束。
        """
        try:
            from models.group_person_losses import soft_range_loss
        except ImportError:
            return sample

        device = pred_xstart.device
        motion_std  = self.normalizer.motion_std.to(device)
        motion_mean = self.normalizer.motion_mean.to(device)

        # 反归一化当前人的 pred_xstart（保留梯度）
        cur_denorm = pred_xstart * motion_std + motion_mean  # (B, T, 262)

        total_loss = th.tensor(0.0, device=device, requires_grad=False)

        for (i, j), (lower, upper) in pair_constraints.items():
            if i != person_idx:
                continue
            if j >= len(all_starts):
                continue
            if lower is None:
                continue

            # 对方的反归一化预测（detach，无梯度）
            other_denorm = (
                all_starts[j].detach() * motion_std + motion_mean
            )

            # Root 位置（第 0 关节，XZ 平面距离）
            # 262 维 = 22*3(pos) + 22*3(vel) + ...  → [0:3] 是 root XYZ
            cur_root   = cur_denorm[..., :3][..., [0, 2]]   # (B, T, 2) XZ
            other_root = other_denorm[..., :3][..., [0, 2]] # (B, T, 2) XZ

            dist = th.norm(cur_root - other_root, dim=-1)   # (B, T)
            loss = soft_range_loss(dist, lower, upper)
            total_loss = total_loss + loss

        if total_loss.item() == 0.0:
            return sample

        # 动态权重
        dynamic_scale = 1000.0 / (t_val + 100.0)
        effective_weight = distance_weight * dynamic_scale

        grad = th.autograd.grad(
            effective_weight * total_loss,
            x_g,
            create_graph=False,
            allow_unused=True,
        )[0]

        if grad is not None:
            t_norm = t_val / 1000.0
            gradient_scale = 5.0 + t_norm * 5.0
            sample = sample - gradient_scale * grad

        return sample

    def parse_range(self, constraint_str):
        """
        解析约束字符串，返回 (lower, upper) 作为目标区间边界
        
        支持格式:
        - "<=0.3m" -> (0, 0.3)
        - ">=0.5m and <1.5m" -> (0.5, 1.5)
        - ">=1.5m and <=2.5m" -> (1.5, 2.5)
        - ">3m" -> (3, inf)
        """
        import re
        
        if constraint_str is None:
            return None, None
        
        # 移除空格并转换为小写
        constraint_str = constraint_str.replace(" ", "").lower()
        
        # 提取所有数字
        numbers = re.findall(r'\d+\.?\d*', constraint_str)
        numbers = [float(n) for n in numbers]
        
        if not numbers:
            return None, None
        
        # 检查是否有 "and" (表示范围)
        if "and" in constraint_str:
            # 范围约束
            lower = numbers[0] if ">=" in constraint_str.split("and")[0] or ">" in constraint_str.split("and")[0] else 0
            upper = numbers[1] if len(numbers) > 1 else float('inf')
            return lower, upper
        else:
            # 单边约束
            if "<=" in constraint_str or "<" in constraint_str:
                return 0, numbers[0]
            elif ">=" in constraint_str or ">" in constraint_str:
                return numbers[0], float('inf')
            else:
                return None, None
    
    def interval_violation(self, x, lower, upper, eps=1e-6):
        """
        计算区间违规程度（局部归一化形式）
        
        Args:
            x: 实际值 (B, T) 或任意形状的张量
            lower: 区间下界
            upper: 区间上界
            eps: 防止除零的小值
            
        Returns:
            violation: 违规程度，0表示在区间内，>0表示违规（相对于区间宽度的倍数）
        """
        # 处理只有下界约束的情况（upper为无穷大）：将上界设为10米
        if upper == float('inf'):
            upper = 10.0  # 将无穷大上界替换为10米
        
        # 有上下界的完整区间：局部归一化
        width = max(upper - lower, eps)
        below = (lower - x).clamp(min=0.0)   # x < lower 时为正
        above = (x - upper).clamp(min=0.0)   # x > upper 时为正
        violation = (below + above) / width  # 超出多少个"区间宽度"
        return violation  # 0 表示在区间里，>0 表示违规
    
    def range_soft_loss(self, x, lower, upper):
        """
        软区间损失：基于局部归一化的违规程度计算
        
        Args:
            x: 实际值 (B, T) 或任意形状的张量
            lower: 区间下界
            upper: 区间上界
            
        Returns:
            loss: 软损失值（标量）
        """
        violation = self.interval_violation(x, lower, upper)
        # 对违规程度进行平方惩罚
        return (violation ** 2).mean()
    
    def temporal_window_constraint_loss(self, values, lower, upper, 
                                       window_size=10, satisfy_ratio=0.7,
                                       tolerance_margin=0.2):
        """
        时序窗口约束损失：在滑动窗口内统计满足率，而非逐帧强制约束
        
        核心思想：
        1. 允许短期违反约束（如转身过程）
        2. 保证长期统计满足约束
        3. 分层惩罚：核心区间 vs 宽容区间
        
        Args:
            values: (B, T) 实际值
            lower: 核心约束下界
            upper: 核心约束上界
            window_size: 滑动窗口大小（帧数），默认10帧≈0.33秒
            satisfy_ratio: 窗口内要求的满足率，默认0.7（70%）
            tolerance_margin: 宽容区间边距（相对值或绝对值）
            
        Returns:
            loss: 综合约束损失
        """
        B, T = values.shape
        
        # 处理无穷大边界
        if upper == float('inf'):
            # 只有下界约束，使用简化处理
            core_lower = lower
            tolerance_lower = lower - tolerance_margin * abs(lower) if lower > 0 else lower - tolerance_margin
            
            # 核心区间外的严重违反
            severe_violation = th.maximum(th.zeros_like(values), core_lower - values)
            severe_loss = (severe_violation ** 2).mean()
            
            # 宽容区间内的轻微违反
            mild_violation = th.maximum(th.zeros_like(values), 
                                       th.minimum(severe_violation, tolerance_lower - values))
            mild_loss = 0.3 * (mild_violation ** 2).mean()
            
            return severe_loss + mild_loss
        
        # 定义核心区间和宽容区间
        core_mid = (lower + upper) / 2.0
        core_margin = (upper - lower) / 2.0
        
        # 宽容区间比核心区间宽
        tolerance_lower = lower - tolerance_margin * (upper - lower)
        tolerance_upper = upper + tolerance_margin * (upper - lower)
        tolerance_mid = (tolerance_lower + tolerance_upper) / 2.0
        tolerance_margin_val = (tolerance_upper - tolerance_lower) / 2.0
        
        # 1. 时序窗口满足率约束
        in_core_range = (values >= lower) & (values <= upper)  # (B, T)
        
        window_loss = 0
        if T >= window_size:
            # 滑动窗口计算满足率
            for t in range(T - window_size + 1):
                window_satisfy = in_core_range[:, t:t+window_size].float().mean()
                if window_satisfy < satisfy_ratio:
                    # 未达到满足率，产生损失
                    shortage = satisfy_ratio - window_satisfy
                    window_loss += (shortage ** 2)
            
            window_loss = window_loss / (T - window_size + 1)  # 平均
        else:
            # 序列太短，直接计算整体满足率
            overall_satisfy = in_core_range.float().mean()
            if overall_satisfy < satisfy_ratio:
                shortage = satisfy_ratio - overall_satisfy
                window_loss = (shortage ** 2)
        
        # 2. 分层约束损失：差异化惩罚
        diff = values - core_mid
        
        # 核心区间内：无损失
        # 宽容区间内：小损失
        in_tolerance = (diff.abs() > core_margin) & (diff.abs() <= tolerance_margin_val)
        tolerance_excess = (diff.abs() - core_margin) * in_tolerance.float()
        tolerance_loss = 0.3 * (tolerance_excess ** 2).mean()
        
        # 宽容区间外：大损失
        out_tolerance = diff.abs() > tolerance_margin_val
        severe_excess = (diff.abs() - tolerance_margin_val) * out_tolerance.float()
        severe_loss = (severe_excess ** 2).mean()
        
        # 3. 时序平滑度约束：防止剧烈抖动
        if T > 1:
            temporal_diff = values[:, 1:] - values[:, :-1]
            smoothness_loss = 0.05 * (temporal_diff ** 2).mean()
        else:
            smoothness_loss = 0
        
        # 组合损失
        total_loss = window_loss + tolerance_loss + severe_loss + smoothness_loss
        
        return total_loss
    
    def compute_distance(self, x_start, cond_start):
        """
        计算两个人物之间的3D欧氏距离
        
        Args:
            x_start: 被生成人物的 x_start (已反归一化)
            cond_start: 条件人物的 x_start (已反归一化)
            
        Returns:
            distances: 3D空间中的距离 (B, T)
        """
        B, T = x_start.shape[:2]
        
        # 提取根节点位置
        pred_g_joints1 = x_start[..., :22 * 3].reshape(B, T, 22, 3)
        pred_g_joints2 = cond_start[..., :22 * 3].reshape(B, T, 22, 3)
        
        root_pos1 = pred_g_joints1[..., 0, :]
        root_pos2 = pred_g_joints2[..., 0, :]
        
        # 计算3D空间中的欧氏距离
        distances = th.norm((root_pos1 - root_pos2), dim=-1)  # (B, T)
        
        return distances
    
    
    def parse_distance_value(self, distance_str):
        """
        解析距离值字符串，返回浮点数
        
        支持格式:
        - "0.5m" -> 0.5
        - "1.5m" -> 1.5
        """
        import re
        if distance_str is None:
            return None
        
        match = re.search(r'(\d+\.?\d*)m?', str(distance_str))
        return float(match.group(1)) if match else None
    
    def group_separation_loss(self, starts, inter_graph, min_group_distance=4.0, weight=1.0):
        """
        组间距离约束：确保不同交互组之间的成员保持最小距离
        
        Args:
            starts: 所有人物的x_start列表 [x_start_0, x_start_1, ...]
            inter_graph: 交互图，包含 'groups' 字段
                - groups: 列表，每个元素是一个组的成员索引列表
                  例如: [[0, 1], [2, 3]] 表示P1-P2是一组，P3-P4是另一组
            min_group_distance: 不同组成员之间的最小距离（米）
            weight: 损失权重
            
        Returns:
            loss: 组间距离约束损失
        """
        if 'groups' not in inter_graph or len(inter_graph['groups']) <= 1:
            # 如果没有组信息或只有一个组，不需要组间约束
            return th.tensor(0.0, device=starts[0].device, dtype=starts[0].dtype)
        
        groups = inter_graph['groups']
        total_loss = 0.0
        num_pairs = 0
        
        # 遍历所有组对
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if i >= j:  # 只计算一次每对组
                    continue
                
                # 计算group1和group2之间所有成员对的距离
                for idx1 in group1:
                    for idx2 in group2:
                        # 提取root位置
                        x_start1 = starts[idx1]
                        x_start2 = starts[idx2]
                        
                        B, T = x_start1.shape[:2]
                        
                        joints1 = x_start1[..., :22 * 3].reshape(B, T, 22, 3)
                        joints2 = x_start2[..., :22 * 3].reshape(B, T, 22, 3)
                        
                        root1 = joints1[..., 0, :]  # (B, T, 3)
                        root2 = joints2[..., 0, :]  # (B, T, 3)
                        
                        # 计算XZ平面距离
                        rel_pos = root1 - root2  # (B, T, 3)
                        rel_pos_xz = rel_pos[..., [0, 2]]  # (B, T, 2)
                        distances = th.norm(rel_pos_xz, dim=-1)  # (B, T)
                        
                        # 如果距离小于最小距离，添加惩罚
                        violation = th.clamp(min_group_distance - distances, min=0.0)
                        total_loss += (violation ** 2).mean()
                        num_pairs += 1
        
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
        
        return weight * total_loss
    
    def position_guidance_loss(self, x_start, cond_start, relative_position, distance_constraint, weight=1.0):
        """
        距离引导头：约束双人交互中的距离关系
        
        Args:
            x_start: 被生成人物(generated)的预测 x_start (已反归一化)
            cond_start: 条件人物(condition)的 x_start (已反归一化)
            relative_position: 相对位置字典（已弃用，保留用于兼容）
            distance_constraint: 距离约束字典，必须包含 relative_distance 字段
                - relative_distance: 距离范围字符串，如 ">= 4.51m and <= 4.81m"
                  直接指定两个人之间允许的root距离范围
            weight: 损失权重
            
        Returns:
            loss: 距离约束损失
            position_info: 位置信息字典，如果约束无效则返回 None
        """
        if distance_constraint is None or 'relative_distance' not in distance_constraint:
            return 0, None
        
        B, T = x_start.shape[:2]
        total_loss = 0
        position_info = {}
        
        # 提取root位置 (joint 0)
        # 需要先reshape成 (B, T, 22, 3) 才能正确提取关节位置
        pred_g_joints = x_start[..., :22 * 3].reshape(B, T, 22, 3)
        cond_g_joints = cond_start[..., :22 * 3].reshape(B, T, 22, 3)
        
        pred_root = pred_g_joints[..., 0, :]  # (B, T, 3) - joint 0 是 root
        cond_root = cond_g_joints[..., 0, :]  # (B, T, 3)
        
        # 计算实际相对位置（XZ平面）
        rel_pos_3d = pred_root - cond_root  # (B, T, 3)
        rel_pos_xz = rel_pos_3d[..., [0, 2]]  # (B, T, 2) - 只取XZ平面
        
        # 距离范围约束：直接使用JSON中的relative_distance字段
        # 格式如: ">= 4.51m and <= 4.81m"
        relative_distance_str = distance_constraint.get('relative_distance')
        
        if relative_distance_str is not None:
            # 解析relative_distance字符串，获取min和max距离
            min_distance, max_distance = self.parse_range(relative_distance_str)
            
            if min_distance is not None and max_distance is not None:
                # 计算当前实际距离（XZ平面上的欧氏距离）
                current_distances = th.norm(rel_pos_xz, dim=-1)  # (B, T)
                
                # 使用软区间损失
                distance_loss = self.range_soft_loss(current_distances, min_distance, max_distance)
                total_loss += distance_loss
                
                position_info['distance_loss'] = distance_loss.item()
                position_info['min_distance'] = min_distance
                position_info['max_distance'] = max_distance
                position_info['current_distance_mean'] = current_distances.mean().item()
                position_info['current_distance_min'] = current_distances.min().item()
                position_info['current_distance_max'] = current_distances.max().item()
        
        return weight * total_loss, position_info

    def parse_angle_constraint(self, angle_str):
        """
        解析朝向角度约束字符串，返回 (min_angle, max_angle)
        
        支持格式:
        - "<=30deg" -> (0, 30)
        - ">=120deg and <=180deg" -> (120, 180)
        - ">=0deg and <=45deg" -> (0, 45)
        """
        import re
        
        if angle_str is None:
            return None, None
        
        angle_str = angle_str.replace(" ", "").lower()
        
        # 提取所有数字
        numbers = re.findall(r'\d+\.?\d*', angle_str)
        numbers = [float(n) for n in numbers]
        
        if not numbers:
            return None, None
        
        # 检查是否有 "and" (表示范围)
        if "and" in angle_str:
            min_angle = numbers[0] if ">=" in angle_str.split("and")[0] or ">" in angle_str.split("and")[0] else 0
            max_angle = numbers[1] if len(numbers) > 1 else 180.0
            # 确保角度在0-180度范围内
            min_angle = max(0.0, min(180.0, min_angle))
            max_angle = max(0.0, min(180.0, max_angle))
            return min_angle, max_angle
        else:
            # 单边约束
            if "<=" in angle_str or "<" in angle_str:
                return 0.0, min(180.0, numbers[0])
            elif ">=" in angle_str or ">" in angle_str:
                return max(0.0, numbers[0]), 180.0
            else:
                return None, None
    
    def compute_direction(self, x_start, cond_start):
        """
        计算generated人的head方向与generated指向condition的方向之间的夹角（-180到180度）
        
        Args:
            x_start: 被生成人物的 x_start (已反归一化)
            cond_start: 条件人物的 x_start (已反归一化)
            
        Returns:
            angles: 头部朝向与人物间方向的夹角 (B, T)，范围 -180到180度
                   0度表示直视对方（强交互），正角度表示右侧，负角度表示左侧
        """
        B, T = x_start.shape[:2]
        
        # 提取关节位置
        pred_g_joints_generated = x_start[..., :22 * 3].reshape(B, T, 22, 3)
        pred_g_joints_condition = cond_start[..., :22 * 3].reshape(B, T, 22, 3)
        
        # 计算generated人的头部朝向（从脖子到头部：关节9→13）
        joint_9_gen = pred_g_joints_generated[..., 9, :]  # (B, T, 3) 脖子
        joint_13_gen = pred_g_joints_generated[..., 13, :]  # (B, T, 3) 头部
        
        # generated的头部朝向向量：从脖子指向头部
        head_dir_generated = joint_13_gen - joint_9_gen  # (B, T, 3)
        
        # 计算generated指向condition的方向向量
        # 使用root位置（关节0）来表示人物的中心位置
        root_generated = pred_g_joints_generated[..., 0, :]  # (B, T, 3)
        root_condition = pred_g_joints_condition[..., 0, :]  # (B, T, 3)
        
        # generated指向condition的方向向量
        direction_to_condition = root_condition - root_generated  # (B, T, 3)
        
        # 投影到XZ平面并归一化
        head_dir_xz = head_dir_generated[..., [0, 2]]  # (B, T, 2) 只取XZ平面 [x, z]
        head_dir_xz_norm = th.norm(head_dir_xz, dim=-1, keepdim=True) + 1e-8
        head_dir_xz = head_dir_xz / head_dir_xz_norm  # 归一化
        
        direction_to_cond_xz = direction_to_condition[..., [0, 2]]  # (B, T, 2) [x, z]
        direction_to_cond_xz_norm = th.norm(direction_to_cond_xz, dim=-1, keepdim=True) + 1e-8
        direction_to_cond_xz = direction_to_cond_xz / direction_to_cond_xz_norm  # 归一化
        
        # 计算两个向量的角度（使用atan2得到有符号角度）
        # atan2(y, x) 返回从x轴正方向到点(x,y)的角度，范围[-π, π]
        # 在XZ平面中，x是水平方向，z是深度方向（可以看作y）
        # 计算head_dir的角度（相对于X轴正方向）
        head_angle = th.atan2(head_dir_xz[..., 1], head_dir_xz[..., 0])  # (B, T) atan2(z, x)
        
        # 计算direction_to_cond的角度
        dir_angle = th.atan2(direction_to_cond_xz[..., 1], direction_to_cond_xz[..., 0])  # (B, T) atan2(z, x)
        
        # 计算角度差（head相对于direction的角度）
        angle_diff = head_angle - dir_angle  # (B, T) 范围 -2π 到 2π
        
        # 归一化到 -180° 到 180° 范围
        angles = angle_diff * 180.0 / th.tensor(np.pi, device=angle_diff.device, dtype=angle_diff.dtype)  # 转换为度数
        # 将角度归一化到 [-180, 180] 范围
        angles = ((angles + 180.0) % 360.0) - 180.0  # 确保在 -180 到 180 之间
        
        return angles
    
    def direction_guidance_loss(self, x_start, cond_start, interaction_type='side_interaction', weight=0.01,
                                generated_idx=None, condition_idx=None, inter_graph=None):
        """
        头部朝向角度引导：约束generated人的head方向与generated指向condition的方向之间的夹角
        
        交互类型定义：
        - face_to_face（正对交互）：人大致正对对方，朝向与指向对方的方向夹角在 -15° 到 15° 之间
          约束范围：-15° 到 15°（直视对方）
        - side_interaction（侧向交互）：对方位于侧向范围，适合表示并排、肩并肩walk等交互
          约束范围：80° 到 100° 和 -100° 到 -80°（两个区间，侧视）
        - unlimited（不限制方向）：不约束头部朝向，适合观察、路过等弱交互
          无约束
        
        Args:
            x_start: 被生成人物的预测 x_start (已反归一化)
            cond_start: 条件人物的 x_start (已反归一化)
            interaction_type: 交互类型，'face_to_face'、'side_interaction' 或 'unlimited'，决定约束范围
            weight: 损失权重
            generated_idx: 生成人物的索引（可选，用于输出）
            condition_idx: 条件人物的索引（可选，用于输出）
            inter_graph: 交互图（可选，用于获取人物ID映射）
            
        Returns:
            loss: 角度约束损失
            angles: 头部朝向与人物间方向的夹角 (B, T)，范围 -180° 到 180°
                   0度表示直视对方，正角度表示右侧，负角度表示左侧
            angle_range: 约束范围字符串，用于输出显示
        """
        # 计算角度
        angles = self.compute_direction(x_start, cond_start)  # (B, T) 范围 -180° 到 180°
        
        # 根据交互类型确定约束范围
        if interaction_type == 'unlimited':
            # 不限制方向，返回零损失（tensor）
            zero_loss = th.tensor(0.0, device=angles.device, dtype=angles.dtype)
            angle_range = "unlimited"
            return zero_loss, angles, angle_range
        elif interaction_type == 'side_interaction':
            # 侧向交互：80° 到 100° 和 -100° 到 -80°（两个区间）
            # 左侧和右侧应该只有一个loss生效
            violation1 = self.interval_violation(angles, 80.0, 100.0)  # 右侧区间 [80°, 100°]
            violation2 = self.interval_violation(angles, -100.0, -80.0)  # 左侧区间 [-100°, -80°]
            
            # 判断角度在哪个区间或更接近哪个区间
            in_right_range = (angles >= 80.0) & (angles <= 100.0)  # 在右侧区间内
            in_left_range = (angles >= -100.0) & (angles <= -80.0)  # 在左侧区间内
            
            # 如果角度在右侧区间内，使用右侧loss；如果在左侧区间内，使用左侧loss
            # 如果不在任何区间内，根据角度正负判断更接近哪个区间
            # 角度 >= 0 时更接近右侧区间，角度 < 0 时更接近左侧区间
            violation = th.where(in_right_range,
                                violation1,  # 在右侧区间内，使用右侧loss
                                th.where(in_left_range,
                                        violation2,  # 在左侧区间内，使用左侧loss
                                        th.where(angles >= 0.0,
                                                violation1,  # 不在区间内但角度>=0，使用右侧loss
                                                violation2)))  # 不在区间内但角度<0，使用左侧loss
            angle_range = "[80°-100°] and [-100°--80°]"
        elif interaction_type == 'face_to_face':
            # 正对交互：-15° 到 15°（直视对方）
            # 使用绝对值避免 ±180° 边界问题（-179° 和 179° 应该被视为接近，而不是相差 358°）
            abs_angles = th.abs(angles)
            violation = self.interval_violation(abs_angles, 0.0, 10.0)
            angle_range = "[0°-10°]"
        else:
            # 默认使用 unlimited，不限制
            zero_loss = th.tensor(0.0, device=angles.device, dtype=angles.dtype)
            angle_range = "unlimited"
            return zero_loss, angles, angle_range
        
        # 如果超出范围，添加惩罚
        range_penalty = (violation ** 2).mean()
        loss = range_penalty
        
        return weight * loss, angles, angle_range

    def gradients(self, x_start, x, sampling_info):
        with th.enable_grad():
            losses = 0
            
            # 检查是否是Group交互（拼接数据，维度是2倍）
            is_group_interaction = False
            single_person_dim = None
            if 'all_starts' in sampling_info and len(sampling_info['all_starts']) > 0:
                single_person_dim = sampling_info['all_starts'][0].shape[-1]
                if x_start.shape[-1] == 2 * single_person_dim:
                    is_group_interaction = True
            
            if is_group_interaction:
                # Group交互：先分割拼接数据
                D = single_person_dim
                x_start_person1 = x_start[..., :D]
                x_start_person2 = x_start[..., D:]
                
                # x_start_person1/2 和 x_start 此时是未反归一化的预测值
            else:
                # Person交互或普通情况：x_start 此时是未反归一化的预测值
                pass 
            
            B, T = x_start.shape[:2] if not is_group_interaction else x_start_person1.shape[:2]
            
            # Group/Person距离损失系统
            if sampling_info['t'] <= 900 and 'inter_graph' in sampling_info and 'all_starts' in sampling_info:
                inter_graph = sampling_info['inter_graph']
                
                # 准备所有人的start（反归一化后），并确保使用当前最新的带梯度的预测值
                all_starts_normalized = []
                
                current_indices = sampling_info.get('current_indices', [])
                
                for i, start in enumerate(sampling_info['all_starts']):
                    # 如果是当前正在优化的人，使用 x_start (带梯度)
                    target_start = start
                    if is_group_interaction:
                        if len(current_indices) == 2:
                            idx1, idx2 = current_indices
                            if i == idx1:
                                target_start = x_start_person1
                            elif i == idx2:
                                target_start = x_start_person2
                        elif len(current_indices) == 1:
                            # Person交互但输入为(A,B)拼接：只用A半边参与loss/梯度替换
                            idx = current_indices[0]
                            if i == idx:
                                target_start = x_start_person1
                    else:
                        if len(current_indices) > 0 and i == current_indices[0]:
                            target_start = x_start
                    
                    # 反归一化（直接计算以保留梯度）
                    device = target_start.device
                    motion_std = self.normalizer.motion_std.to(device)
                    motion_mean = self.normalizer.motion_mean.to(device)
                    
                    # 统一进行反归一化
                    start_norm = target_start * motion_std + motion_mean
                    all_starts_normalized.append(start_norm)
                
                # 导入Group/Person损失函数
                # 检查是否有消融选项
                ablation = sampling_info.get('ablation', {})
                if ablation:
                    from models.group_person_losses_ablation import compute_group_person_losses_ablation as compute_gp_losses
                else:
                    from models.group_person_losses import compute_group_person_losses as compute_gp_losses
                
                # 计算所有距离损失
                distance_weight = sampling_info.get('distance_weight', 1.0)
                
                # 动态权重：在早期（t大）时给予更大的引导力，后期（t小）时逐渐减小
                # scale = 1000 / (t + 100)
                # 例如：t=1000 -> scale≈0.91, t=500 -> scale≈1.67, t=100 -> scale=5.0, t=0 -> scale=10.0
                # 注意：这个公式在 t 小时 scale 更大，符合"后期需要更强引导"的需求
                t_val_for_scale = sampling_info['t'].item() if isinstance(sampling_info['t'], th.Tensor) else sampling_info['t']
                dynamic_scale = 1000.0 / (t_val_for_scale + 100.0)
                
                # 应用动态权重到基础权重
                effective_weight = distance_weight * dynamic_scale
                
                # 根据消融选项调用不同的损失函数
                if ablation:
                    enable_orientation = ablation.get('enable_orientation', True)
                    enable_person = ablation.get('enable_person', True)
                    enable_group = ablation.get('enable_group', True)
                    enable_non_interaction = ablation.get('enable_non_interaction', True)
                    
                    group_person_loss, loss_dict = compute_gp_losses(
                        all_starts_normalized, 
                        inter_graph, 
                        weight=effective_weight,
                        enable_group_intra=enable_group,
                        enable_group_inter=enable_non_interaction,  # 修复: inter_group 只依赖 non_interaction
                        enable_person_target=enable_person,
                        enable_person_orientation=enable_orientation and enable_person,
                        enable_person_other=enable_non_interaction and enable_person,
                        enable_person_person=enable_non_interaction and enable_person,
                    )
                else:
                    group_person_loss, loss_dict = compute_gp_losses(
                    all_starts_normalized, 
                    inter_graph, 
                    weight=effective_weight
                    )
                losses += group_person_loss

            # ===========================
            # GLI Loss（TopologyBatch）
            t_val_gli = sampling_info['t'].item() if isinstance(sampling_info.get('t'), th.Tensor) else int(sampling_info.get('t', 0))
            current_indices = sampling_info.get('current_indices', [])
            if t_val_gli <= self.t_bar_gli and is_group_interaction and len(current_indices) == 2:
                device_for_norm = x_start_person1.device
                motion_std = self.normalizer.motion_std.to(device_for_norm)
                motion_mean = self.normalizer.motion_mean.to(device_for_norm)
                start1 = x_start_person1 * motion_std + motion_mean
                start2 = x_start_person2 * motion_std + motion_mean
                motion1 = start1[..., :22 * 3].reshape(B, T, 22, 3)
                motion2 = start2[..., :22 * 3].reshape(B, T, 22, 3)
                GLI_raw = self.topology_loss(motion1, motion2)
                losses += 0.6 * GLI_raw.sum()

            if losses.item() == 0:
                return 0, 0

            # allow_unused=True: 允许某些输入张量未参与计算（例如Group交互时x是拼接的）
            grad_outputs = torch.autograd.grad(losses, inputs=x, create_graph=False, allow_unused=True)
            if grad_outputs[0] is None:
                # 如果梯度为None，返回零梯度
                gradient = torch.zeros_like(x)
            else:
                gradient = grad_outputs[0]
            
            # Person交互的(A,B)拼接输入：只对A应用guidance梯度，B不参与引导（保持透明）
            if is_group_interaction and single_person_dim is not None:
                current_indices = sampling_info.get('current_indices', [])
                if len(current_indices) == 1:
                    gradient[..., single_person_dim:] = 0.0
            
            # Detach操作（避免内存泄漏）
            if is_group_interaction:
                x_start_person1.detach()
                x_start_person2.detach()
            else:
                x_start.detach()
                x.detach()
        return losses.to(x.device), gradient.to(x.device)


    def gradients_ddim(self, x_start, x):
        with th.enable_grad():
            losses = 0
            B, T = x_start.shape[:2]
            x_start = x_start.reshape(B, T, 2, -1)
            x_start = self.normalizer.backward(x_start, global_rt=True)

            pred_g_joints = x_start[...,: 22 * 3].reshape(B, T, -1, 22, 3)
            motion1 = pred_g_joints[:, :, 0]
            motion2 = pred_g_joints[:, :, 1]

            # GLI（TopologyBatch）：恢复拓扑/几何约束梯度
            GLI_raw = self.topology_loss(motion1, motion2)
            losses += 0.6 * GLI_raw.sum()
            x_start.detach()
            x.detach()

            if losses.item() == 0:
                return 0, 0

            gradient = torch.autograd.grad(losses, inputs=x, create_graph=False)[0]
            x_start.detach()
            x.detach()
        return losses.to(x.device), gradient.to(x.device)


    def training_losses(self, model, mask, t_bar, cond_mask, *args, **kwargs):
        target = kwargs["x_start"]
        B, T = target.shape[:-1]

        target = target.reshape(B, T, 2, -1)
        mask = mask.reshape(B, T, -1, 1)

        target = self.normalizer.forward(target)

        kwargs["x_start"] = target.reshape(B, T, -1)
        items = super().training_losses(self._wrap_model(model), *args, **kwargs)
        prediction = items["pred"].reshape(B, T, 2, -1)
        target = items["target"].reshape(B, T, 2, -1)

        timestep_mask = (kwargs["t"] <= t_bar).float()  # * cond_mask[0, 0]

        interloss_manager = InterLoss("l2", 22)
        interloss_manager.forward(prediction, target, mask, timestep_mask)

        loss_a_manager = GeometricLoss("l2", 22, "A")
        loss_a_manager.forward(prediction[...,0,:], target[...,0,:], mask[...,0,:], timestep_mask)

        loss_b_manager = GeometricLoss("l2", 22, "B")
        loss_b_manager.forward(prediction[...,1,:], target[...,1,:], mask[...,0,:], timestep_mask)

        losses = {}
        losses.update(loss_a_manager.losses)
        losses.update(loss_b_manager.losses)
        losses.update(interloss_manager.losses)
        losses["total"] = loss_a_manager.losses["A"] + loss_b_manager.losses["B"] + interloss_manager.losses["total"]

        return losses

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        self.mse_criterion = th.nn.MSELoss(reduction='none')
        super().__init__(**kwargs)

    def p_mean_variance(
            self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
            self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        items = super().training_losses(self._wrap_model(model), *args, **kwargs)
        return items

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
