"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def adain(source_feat, target_feat):
    source_feat_mean = source_feat.mean(dim=[0, 2, 3], keepdim=True)
    source_feat_std = source_feat.std(dim=[0, 2, 3], keepdim=True)
    target_feat_mean = target_feat.mean(dim=[0, 2, 3], keepdim=True)
    target_feat_std = target_feat.std(dim=[0, 2, 3], keepdim=True)
    return ((source_feat - source_feat_mean) / source_feat_std) * target_feat_std + target_feat_mean


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        default=False,
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument("--sty-img", type=str, default="_data/sty/sty1.png")

    parser.add_argument("--cutoff_ratio", type=float, default=0.2)

    parser.add_argument("--use_gaussian_filter", action="store_true")

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    # sample_path = os.path.join(outpath, "samples")
    # os.makedirs(sample_path, exist_ok=True)
    # base_count = len(os.listdir(sample_path))
    # grid_count = len(os.listdir(outpath)) - 1
    sample_path = outpath
    os.makedirs(sample_path, exist_ok=True)

    assert os.path.isfile(opt.init_img)
    init_image = load_img(opt.init_img).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    input_basename = os.path.splitext(os.path.basename(opt.init_img))[0]

    ##### sty image
    assert os.path.isfile(opt.sty_img)
    sty_image = load_img(opt.sty_img).to(device)
    sty_image = repeat(sty_image, '1 ... -> b ...', b=batch_size)
    sty_latent = model.get_first_stage_encoding(model.encode_first_stage(sty_image))

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    # ensure t_enc within schedule bounds
    max_steps = sampler.ddim_timesteps.shape[0] if hasattr(sampler, 'ddim_timesteps') else opt.ddim_steps
    t_enc = min(t_enc, max_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        """
                        <기존 코드>

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # z_enc = sampler.ddim_inversion(init_latent,
                        #                                        cond=model.get_learned_conditioning(batch_size * [""]),
                        #                                        t_start=t_enc,
                        #                                        unconditional_guidance_scale=opt.scale,
                        #                                        unconditional_conditioning=model.get_learned_conditioning(batch_size * [""]))

                        ##### sty image encode
                        # z_enc_style = sampler.stochastic_encode(sty_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # z_enc_style = sampler.ddim_inversion(sty_latent,
                        #                                      cond=model.get_learned_conditioning(batch_size * [""]),
                        #                                      t_start=t_enc,
                        #                                      unconditional_guidance_scale=opt.scale,
                        #                                      unconditional_conditioning=model.get_learned_conditioning(batch_size * [""]))
                        
                        # adain (ddim inversion과 더불어 보완하기)
                        # z_enc = adain(z_enc, z_enc_style)

                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,)
                        """


                        ##### new version
                        empty_cond = model.get_learned_conditioning(batch_size * [""])
                        empty_uncond = None

                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))

                        z_enc_content, noised_latents = sampler.ddim_inversion(
                            init_latent,
                            cond=empty_cond,
                            t_start=t_enc,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=empty_uncond
                        )

                        z_enc_style, _ = sampler.ddim_inversion(
                            sty_latent,
                            cond=empty_cond,
                            t_start=t_enc,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=empty_uncond
                        )

                        # z_enc = adain(z_enc_content, z_enc_style) # use adain for making initial noise
                        # z_enc = z_enc_content                     # use ddim inversion for making initial noise

                        # noised latents visualize
                        # ddim_inversion_path = "_outputs/ddim_inversion"
                        # if not os.path.exists(ddim_inversion_path):
                        #     os.makedirs(ddim_inversion_path, exist_ok=True)
                        # for i, noised_latent in enumerate(noised_latents):
                        #     img = model.decode_first_stage(noised_latent)
                        #     img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
                        #     img = img.squeeze(dim=0)
                        #     img = 255. * rearrange(img.cpu().numpy(), 'c h w -> h w c')
                        #     Image.fromarray(img.astype(np.uint8)).save(os.path.join(ddim_inversion_path, f"{i}.png"))
                        # return

                        samples = sampler.decode_with_fourier(
                            source_latent=z_enc,
                            source_cond=c,
                            source_unconditional_guidance_scale=opt.scale,
                            source_unconditional_conditioning=uc,
                            noised_latents=noised_latents,
                            t_start=t_enc,
                            use_gaussian_filter=opt.use_gaussian_filter,
                            cutoff_ratio=opt.cutoff_ratio
                        )


                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            # for x_sample in x_samples:
                            #     x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            #     Image.fromarray(x_sample.astype(np.uint8)).save(
                            #         os.path.join(sample_path, f"{base_count:05}.png"))
                            #     base_count += 1
                            for i, x_sample in enumerate(x_samples):
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                if isinstance(prompts, (list, tuple)):
                                    prompt_i = prompts[i]
                                else:
                                    prompt_i = prompts
                                filename = f"{input_basename}_{prompt_i}"
                                if batch_size > 1:
                                    filename = f"{filename}_{i}"
                                filename = filename + ".png"
                                Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, filename))

                        all_samples.append(x_samples)

                # if not opt.skip_grid:
                #     # additionally, save as grid
                #     grid = torch.stack(all_samples, 0)
                #     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                #     grid = make_grid(grid, nrow=n_rows)
 
                #     # to image
                #     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                #     Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                #     grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()



"""
CUDA_VISIBLE_DEVICES=0 python scripts/img2img.py \
--prompt "Oil Panting" \
--init-img "_data/cnt/woman.png" \
--outdir "_outputs/" \
--cutoff_ratio 0.2 \
--strength 0.8 \
--use_gaussian_filter # cutoff_ratio of gaussian_filter * 2 ~= cutoff_ratio of ideal filter

style keyword: Pixel, Van Gogh Style, Monochrome Sketching, Cyberpunk, Chinese Ink, Oil Panting, Studio Ghibli, Crayon Painting, LEGO Toy
"""