from tqdm import tqdm
from DWTDiff.ddim_sampler import DDIM_Sampler
from DWTDiff.DWT_util import *


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class DWT_Sampler(DDIM_Sampler):

    def __init__(self, model, schedule="linear", **kwargs):
        super(DWT_Sampler, self).__init__(model, schedule, **kwargs)

    @torch.no_grad()
    def decode_with_low_FBS(self, ref_zT, cond, t_dec, unconditional_guidance_scale,
                            unconditional_conditioning, use_original_steps=False, callback=None,
                            dwt_level=1, high_freq_transfer_ratio=0.0, end_step=500):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        ref_z = ref_zT
        gen_z = torch.randn_like(ref_zT)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((ref_zT.shape[0],), step, device=ref_zT.device, dtype=torch.long)
            if step >= end_step:

                ref_z, _, _ = self.p_sample_ddim(ref_z, unconditional_conditioning,
                                                 ts,
                                                 index=index,
                                                 use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=1.0,
                                                 unconditional_conditioning=None)

                gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index,
                                                 use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)

                gen_z = low_freq_transfer(ref=ref_z, src=gen_z, dwt_level=dwt_level, high_freq_transfer_ratio=high_freq_transfer_ratio)

            else:
                gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return gen_z

    @torch.no_grad()
    def decode_with_masked_low_FBS(self, ref_zT, mask, cond, t_dec,
                                   unconditional_guidance_scale,
                                   unconditional_conditioning, use_original_steps=False,
                                   callback=None,
                                   dwt_level=1, high_freq_transfer_ratio=0.0, end_step=500):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        ref_z = ref_zT
        gen_z = torch.randn_like(ref_zT)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((ref_zT.shape[0],), step, device=ref_zT.device, dtype=torch.long)
            if step >= end_step:
                ref_z, _, _ = self.p_sample_ddim(ref_z, unconditional_conditioning,
                                                 ts,
                                                 index=index,
                                                 use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=1.0,
                                                 unconditional_conditioning=None)
                gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index,
                                                 use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
                gen_z = low_freq_transfer(ref=ref_z, src=gen_z, dwt_level=dwt_level,
                                          high_freq_transfer_ratio=high_freq_transfer_ratio)
                gen_z = mask * gen_z + (1 - mask) * ref_z
            else:
                ref_z, _, _ = self.p_sample_ddim(ref_z, unconditional_conditioning,
                                                 ts,
                                                 index=index,
                                                 use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=1.0,
                                                 unconditional_conditioning=None)
                gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
                gen_z = mask * gen_z + (1 - mask) * ref_z
            if callback: callback(i)
        return gen_z

    @torch.no_grad()
    def decode_with_high_FBS(self, ref_latent, cond, t_dec, unconditional_guidance_scale,
                             unconditional_conditioning, use_original_steps=False, callback=None,
                             dwt_level=1, end_step=500, inner_loop=1):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = torch.randn_like(ref_latent)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((ref_latent.shape[0],), step, device=ref_latent.device, dtype=torch.long)
            b, _, _, _ = ref_latent.shape
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=ref_latent.device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=ref_latent.device)
            if step >= end_step:
                ref_latent_prev, _, _ = self.p_sample_ddim(ref_latent, unconditional_conditioning, ts,
                                                           index=index,
                                                           use_original_steps=use_original_steps,
                                                           unconditional_guidance_scale=1.0,
                                                           unconditional_conditioning=None)
                for u in range(inner_loop):
                    x_dec_prev, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index,
                                                          use_original_steps=use_original_steps,
                                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                                          unconditional_conditioning=unconditional_conditioning)
                    x_dec_prev = multi_level_high_freq_transfer(ref=ref_latent_prev, src=x_dec_prev,
                                                                dwt_level=dwt_level)
                    x_dec = torch.sqrt(a_t / a_prev) * x_dec_prev + torch.sqrt(1 - a_t / a_prev) * torch.randn_like(
                        x_dec)

                ref_latent = ref_latent_prev
                x_dec = x_dec_prev
                # x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index,
                #                                  use_original_steps=use_original_steps,
                #                                  unconditional_guidance_scale=unconditional_guidance_scale,
                #                                  unconditional_conditioning=unconditional_conditioning)
            else:
                x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
