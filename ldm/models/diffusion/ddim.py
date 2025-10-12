"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

import os
from einops import rearrange
from PIL import Image

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def ddim_inversion(self, x_latent, cond=None, t_start=0,
                       unconditional_guidance_scale=1.0, unconditional_conditioning=None,
                       use_original_steps=False, verbose=True):
        """
        Deterministic DDIM forward: starting from x_0 (latent), run forward-DDIM steps
        to obtain x_t (the 'noised' latent). This is a deterministic 'inversion' path
        following the DDIM equations (eta=0 style).

        Args:
            x_latent: (b, C, H, W) tensor, the latent of the input image (z0).
            cond: conditioning for model (text embeddings etc.), same as decode() uses.
            t_start: int, number of DDIM steps to run (this mirrors how img2img maps strength -> t_enc).
            unconditional_guidance_scale: guidance scale (for classifier-free guidance).
            unconditional_conditioning: unconditional conditioning (empty prompts).
            use_original_steps: whether to use original ddpm steps or the ddim_timesteps.
        Returns:
            x_enc: tensor, the latent after forward-DDIM (z_t).
        """
        device = x_latent.device
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        t_start = int(t_start)
        if t_start <= 0:
            return x_latent
        if t_start > timesteps.shape[0]:
            if verbose:
                print(f"Warning: t_start ({t_start}) > available ddim timesteps ({timesteps.shape[0]}). Clamping.")
            t_start = timesteps.shape[0]

        # use the first t_start timesteps in ascending order to step forward
        forward_timesteps = timesteps[:t_start]
        total_steps = forward_timesteps.shape[0]
        if verbose:
            print(f"Running DDIM inversion/encoding with {total_steps} forward timesteps")

        noised_latents = []
        x_enc = x_latent
        b = x_latent.shape[0]

        # For indexing into precomputed buffers (ddim_alphas, etc.), we map each timestep value
        # to its index inside self.ddim_timesteps.
        # Note: self.ddim_timesteps is typically the same as `timesteps` used above when use_original_steps=False
        # so indices will be 0..N-1; this mapping is robust in either case.
        # Convert to python list for iteration
        steps_list = list(forward_timesteps)

        # build a mapping from timestep value -> index in buffers (int)
        # If self.ddim_timesteps is numpy array, use np.where; fall back to matching by position.
        try:
            base_timesteps = list(self.ddim_timesteps)
        except Exception:
            base_timesteps = list(forward_timesteps)

        idx_map = {}
        for idx, tv in enumerate(base_timesteps):
            idx_map[int(tv)] = idx

        for i, step in enumerate(forward_timesteps):
            step_i = int(step)
            index = idx_map.get(step_i, None)
            if index is None:
                # fallback: use i
                index = i

            ts = torch.full((b,), step_i, device=device, dtype=torch.long)

            # predict e_t (with classifier-free guidance if applicable)
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x_enc, ts, cond)
            else:
                x_in = torch.cat([x_enc] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([unconditional_conditioning, cond])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            # compute pred_x0 (same formula as in p_sample_ddim)
            # gather alpha_t and sqrt(1-alpha_t) using index mapping
            a_t = self.ddim_alphas[index]                # alpha_t (tensor scalar)
            sqrt_one_minus_at = self.ddim_sqrt_one_minus_alphas[index]  # sqrt(1-alpha_t)
            # expand to match batch/shape
            a_t_sqrt = torch.sqrt(a_t).to(device)
            # ensure shapes broadcast
            a_t_sqrt = a_t_sqrt.view(1, 1, 1, 1)
            sqrt_one_minus_at = torch.tensor(sqrt_one_minus_at, device=device).view(1, 1, 1, 1)

            pred_x0 = (x_enc - sqrt_one_minus_at * e_t) / a_t_sqrt

            # determine alpha_{t_next} (index_next)
            # find next index in base_timesteps: pick index_next corresponding to next forward timestep value,
            # if exists; otherwise fallback to current index (safety).
            if i + 1 < total_steps:
                next_step = int(forward_timesteps[i + 1])
                index_next = idx_map.get(next_step, index + 1 if (index + 1) < len(self.ddim_alphas) else index)
            else:
                # compute index_next as index+1 if possible; if not, just reuse index
                index_next = index + 1 if (index + 1) < len(self.ddim_alphas) else index

            a_next = self.ddim_alphas[index_next]
            sqrt_one_minus_anext = self.ddim_sqrt_one_minus_alphas[index_next]

            # build x_next using deterministic DDIM step (eta=0)
            a_next_sqrt = torch.sqrt(a_next).to(device).view(1, 1, 1, 1)
            sqrt_one_minus_anext = torch.tensor(sqrt_one_minus_anext, device=device).view(1, 1, 1, 1)

            x_next = a_next_sqrt * pred_x0 + sqrt_one_minus_anext * e_t

            noised_latents.append(x_enc.detach())
            x_enc = x_next

        noised_latents.reverse()
        return x_enc, noised_latents


    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec


    @torch.no_grad()
    def decode_with_fourier(self,
                            source_latent,
                            source_cond,
                            source_unconditional_guidance_scale,
                            source_unconditional_conditioning,
                            noised_latents,
                            t_start,
                            # target_latent=None,
                            # target_cond=None,
                            # target_unconditional_guidance_scale=None,
                            # target_unconditional_conditioning=None,
                            use_original_steps=False):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)

        source_dec = source_latent
        # target_dec = target_latent

        for i, (step, noised_latent) in enumerate(zip(iterator, noised_latents)):
            index = total_steps - i - 1

            source_ts = torch.full((source_latent.shape[0],), step, device=source_latent.device, dtype=torch.long)
            source_dec, _ = self.p_sample_ddim(source_dec, source_cond, source_ts, index=index, use_original_steps=use_original_steps,
                                               unconditional_guidance_scale=source_unconditional_guidance_scale,
                                               unconditional_conditioning=source_unconditional_conditioning)

            # target_ts = torch.full((target_latent.shape[0],), step, device=target_latent.device, dtype=torch.long)
            # target_dec, _ = self.p_sample_ddim(target_dec, target_cond, target_ts, index=index, use_original_steps=use_original_steps,
            #                                    unconditional_guidance_scale=target_unconditional_guidance_scale,
            #                                    unconditional_conditioning=target_unconditional_conditioning)
            
            # 2D FFT
            source_fft = torch.fft.fft2(source_dec.to(torch.float32), dim=(-2, -1))
            target_fft = torch.fft.fft2(noised_latent.to(torch.float32), dim=(-2, -1))

            source_fft_shift = torch.fft.fftshift(source_fft, dim=(-2, -1))
            target_fft_shift = torch.fft.fftshift(target_fft, dim=(-2, -1))

            B, C, H, W = source_latent.shape
            cutoff_ratio = 0.0

            # circular mask
            y, x = torch.meshgrid(
                torch.arange(H, device=source_latent.device),
                torch.arange(W, device=source_latent.device),
                indexing='ij'
            )

            cy, cx = H//2, W//2
            r = torch.sqrt((x - cx)**2 + (y - cy)**2)
            r_norm = r / r.max()
            # mask_high = (r_norm >= cutoff_ratio).float()
            # mask = mask_high[None, None, :, :]
            mask_low = (r_norm <= cutoff_ratio).float()
            mask = mask_low[None, None, :, :]

            # phase swap
            source_mag = source_fft_shift.abs()
            source_phase = torch.angle(source_fft_shift)
            target_phase = torch.angle(target_fft_shift)
            swap_phase = source_phase * (1 - mask) + target_phase * mask
            swapped_fft_shift = torch.polar(source_mag, swap_phase)

            swapped_fft = torch.fft.ifftshift(swapped_fft_shift, dim=(-2, -1))
            swapped_dec = torch.fft.ifft2(swapped_fft, dim=(-2, -1)).real

            source_dec = swapped_dec

            # test reconstruction
            # test_reconstruction_path = "_outputs/reconstruction/"
            # if not os.path.exists(test_reconstruction_path):
            #     os.makedirs(test_reconstruction_path, exist_ok=True)
            # imgs = self.model.decode_first_stage(source_dec) # or target_dec
            # imgs = torch.clamp((imgs + 1.0) / 2.0, min=0.0, max=1.0)
            # for j, img in enumerate(imgs):
            #     img = 255. * rearrange(img.cpu().numpy(), 'c h w -> h w c')
            #     Image.fromarray(img.astype(np.uint8)).save(os.path.join(test_reconstruction_path, f"{i}_{j}.png"))

        return source_dec