import torch
import numpy as np
from tqdm import tqdm
from DCTDiff.dct_util import dct_2d, idct_2d, dct, idct, low_pass, high_pass, dct_coefficients_swap
from DCTDiff.ddim_sampler import DDIM_Sampler


def calc_mean_std(feat, dim, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    feat_var = torch.var(feat, dim=dim, keepdim=True) + eps
    feat_std = feat_var.sqrt()
    feat_mean = torch.mean(feat, dim=dim, keepdim=True)
    return feat_mean, feat_std


def equifrequency_instance_normalization(content_feat, style_feat, dim):
    content_mean, content_std = calc_mean_std(content_feat, dim)
    style_mean, style_std = calc_mean_std(style_feat, dim)

    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean

# def calc_mean_std(feat, eps=1e-5):
#     # eps is a small value added to the variance to avoid divide-by-zero.
#     size = feat.size()
#     assert (len(size) == 4)
#     N, C = size[:2]
#     feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
#     feat_std = feat_var.sqrt().view(N, C, 1, 1)
#     feat_mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)
#     return feat_mean, feat_std
#
#
# def adaptive_instance_normalization(content_feat, style_feat):
#     assert (content_feat.size()[:2] == style_feat.size()[:2])
#     size = content_feat.size()
#     style_mean, style_std = calc_mean_std(style_feat)
#     content_mean, content_std = calc_mean_std(content_feat)
#
#     normalized_feat = (content_feat - content_mean.expand(
#         size)) / content_std.expand(size)
#     return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class DCT_Sampler(DDIM_Sampler):

    def __init__(self, model, schedule="linear", **kwargs):
        super(DCT_Sampler, self).__init__(model, schedule, **kwargs)

    def window_partition(self, feature, patch_num):
        # feature: b, c, h, w
        b, c, h, w = feature.shape
        assert h % patch_num == 0 and w % patch_num == 0
        feature = torch.nn.Unfold(kernel_size=(h // patch_num, w // patch_num),
                                  stride=(h // patch_num, w // patch_num))(feature)
        feature = feature.squeeze().permute(1, 0).view(-1, c, h // patch_num, w // patch_num)
        # n, c, h // patch_num, w // patch_num
        return feature

    def window_merge(self, feature, h, w):
        n, c, p_h, p_w = feature.shape
        feature = torch.reshape(feature, shape=(1, n, -1)).permute(0, 2, 1)
        feature = torch.nn.functional.fold(feature, kernel_size=(p_h, p_w), stride=(p_h, p_w),
                                           output_size=(h, w))
        return feature

    @torch.no_grad()
    def decode_with_low_FBS(self, ref_guidance, cond, t_dec, unconditional_guidance_scale,
                            unconditional_conditioning, use_original_steps=False, callback=None,
                            lp_percentile=50, dct_calibration_ratio=0.45):

        def low_FBS(ref_z, gen_z, lp_percentile):
            _, c, h, w = gen_z.shape
            threshold_w = int(w * lp_percentile / 100)
            threshold_h = int(h * lp_percentile / 100)

            gen_dct_w = dct(gen_z, norm='ortho')
            ref_dct_w = dct(ref_z, norm='ortho')
            mask = torch.range(1, w).cuda()
            mask = torch.where(mask <= threshold_w, torch.ones_like(mask), torch.zeros_like(mask))
            mask = torch.reshape(mask, shape=(1, w)).repeat((h, 1))
            gen_dct_w = ref_dct_w * mask + gen_dct_w * (1 - mask)
            gen_z = idct(gen_dct_w, norm='ortho')

            gen_z_trans = gen_z.permute((0, 1, 3, 2))
            ref_z_trans = ref_z.permute((0, 1, 3, 2))
            gen_dct_h = dct(gen_z_trans, norm='ortho')
            ref_dct_h = dct(ref_z_trans, norm='ortho')
            mask = torch.range(1, h).cuda()
            mask = torch.where(mask <= threshold_h, torch.ones_like(mask), torch.zeros_like(mask))
            mask = torch.reshape(mask, shape=(1, h)).repeat((w, 1)).cuda()
            gen_dct_h = ref_dct_h * mask + gen_dct_h * (1 - mask)
            gen_z_trans = idct(gen_dct_h, norm='ortho')
            gen_z = gen_z_trans.permute(0, 1, 3, 2)
            return gen_z

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Sampling image', total=total_steps)
        zT = torch.randn_like(ref_guidance[0]).cuda()
        end_step = self.ddpm_num_timesteps - self.ddpm_num_timesteps * dct_calibration_ratio

        gen_z = zT
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((zT.shape[0],), step, device=zT.device, dtype=torch.long)
            if step >= end_step:
                gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
                gen_z = low_FBS(ref_guidance[i], gen_z, lp_percentile)
            else:
                gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return gen_z

    @torch.no_grad()
    def decode(self, zT, cond, t_dec, unconditional_guidance_scale,
               unconditional_conditioning, use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Sampling image', total=total_steps)

        gen_z = zT
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((zT.shape[0],), step, device=zT.device, dtype=torch.long)
            gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index,
                                             use_original_steps=use_original_steps,
                                             unconditional_guidance_scale=1,
                                             unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return gen_z


    @torch.no_grad()
    def decode_with_masked_low_FBS(self, ref_guidance, mask, cond, t_dec, unconditional_guidance_scale,
                            unconditional_conditioning, use_original_steps=False, callback=None,
                            lp_percentile=50, dct_calibration_ratio=0.45):

        def low_FBS(ref_z, gen_z, lp_percentile):
            _, c, h, w = gen_z.shape
            threshold_w = int(w * lp_percentile / 100)
            threshold_h = int(h * lp_percentile / 100)

            gen_dct_w = dct(gen_z, norm='ortho')
            ref_dct_w = dct(ref_z, norm='ortho')
            mask = torch.range(1, w).cuda()
            mask = torch.where(mask <= threshold_w, torch.ones_like(mask), torch.zeros_like(mask))
            mask = torch.reshape(mask, shape=(1, w)).repeat((h, 1))
            gen_dct_w = ref_dct_w * mask + gen_dct_w * (1 - mask)
            gen_z = idct(gen_dct_w, norm='ortho')

            gen_z_trans = gen_z.permute((0, 1, 3, 2))
            ref_z_trans = ref_z.permute((0, 1, 3, 2))
            gen_dct_h = dct(gen_z_trans, norm='ortho')
            ref_dct_h = dct(ref_z_trans, norm='ortho')
            mask = torch.range(1, h).cuda()
            mask = torch.where(mask <= threshold_h, torch.ones_like(mask), torch.zeros_like(mask))
            mask = torch.reshape(mask, shape=(1, h)).repeat((w, 1)).cuda()
            gen_dct_h = ref_dct_h * mask + gen_dct_h * (1 - mask)
            gen_z_trans = idct(gen_dct_h, norm='ortho')
            gen_z = gen_z_trans.permute(0, 1, 3, 2)
            return gen_z

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Sampling image', total=total_steps)
        zT = torch.randn_like(ref_guidance[0]).cuda()
        end_step = self.ddpm_num_timesteps - self.ddpm_num_timesteps * dct_calibration_ratio

        gen_z = zT
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((zT.shape[0],), step, device=zT.device, dtype=torch.long)
            if step >= end_step:
                gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
                gen_z = low_FBS(ref_guidance[i], gen_z, lp_percentile)
                gen_z = mask * gen_z + (1 - mask) * ref_guidance[i]
            else:
                gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
                gen_z = mask * gen_z + (1 - mask) * ref_guidance[i]
            if callback: callback(i)
        return gen_z

    @torch.no_grad()
    def decode_with_high_FBS(self, ref_guidance, cond, t_dec, unconditional_guidance_scale,
                             unconditional_conditioning, use_original_steps=False, callback=None,
                             hp_percentile=10, dct_calibration_ratio=0.45):

        def high_FBS(ref_z, gen_z, hp_percentile):
            _, c, h, w = gen_z.shape
            threshold_w = int(w * hp_percentile / 100)
            threshold_h = int(h * hp_percentile / 100)

            gen_dct_w = dct(gen_z, norm='ortho')
            ref_dct_w = dct(ref_z, norm='ortho')
            # ref_dct_w = equifrequency_instance_normalization(content_feat=ref_dct_w, style_feat=gen_dct_w, dim=2)

            mask = torch.range(1, w).cuda()
            mask = torch.where(mask >= threshold_w, torch.ones_like(mask), torch.zeros_like(mask))
            mask = torch.reshape(mask, shape=(1, w)).repeat((h, 1)).cuda()
            gen_dct_w = ref_dct_w * mask + gen_dct_w * (1 - mask)
            gen_z = idct(gen_dct_w, norm='ortho')

            gen_z_trans = gen_z.permute((0, 1, 3, 2))
            ref_z_trans = ref_z.permute((0, 1, 3, 2))
            gen_dct_h = dct(gen_z_trans, norm='ortho')
            ref_dct_h = dct(ref_z_trans, norm='ortho')
            # ref_dct_h = equifrequency_instance_normalization(content_feat=ref_dct_h, style_feat=gen_dct_h, dim=2)

            mask = torch.range(1, h).cuda()
            mask = torch.where(mask >= threshold_h, torch.ones_like(mask), torch.zeros_like(mask))
            mask = torch.reshape(mask, shape=(1, h)).repeat((w, 1)).cuda()
            gen_dct_h = ref_dct_h * mask + gen_dct_h * (1 - mask)
            gen_z_trans = idct(gen_dct_h, norm='ortho')
            gen_z = gen_z_trans.permute(0, 1, 3, 2)
            return gen_z

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        zT = torch.randn_like(ref_guidance[0]).cuda()
        end_step = self.ddpm_num_timesteps - self.ddpm_num_timesteps * dct_calibration_ratio

        gen_z = zT
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((zT.shape[0],), step, device=zT.device, dtype=torch.long)
            if step >= end_step:
                gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
                gen_z = high_FBS(ref_guidance[i], gen_z, hp_percentile)
            else:
                gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return gen_z

    @torch.no_grad()
    def decode_with_mid_FBS(self, ref_guidance, cond, t_dec, unconditional_guidance_scale,
                            unconditional_conditioning, use_original_steps=False, callback=None,
                            mp_percentiles=(10, 70), end_step=500):

        def mid_FBS(ref_z, gen_z, mp_percentiles):
            _, _, h, w = gen_z.shape
            gen_dct_w = dct(gen_z, norm='ortho')
            ref_dct_w = dct(ref_z, norm='ortho')
            # ref_dct_w = equifrequency_instance_normalization(content_feat=ref_dct_w, style_feat=gen_dct_w, dim=2)
            lower_bound_w = int(w * mp_percentiles[0] / 100)
            upper_bound_w = int(w * mp_percentiles[1] / 100)
            mask = torch.range(1, w).cuda()
            mask = torch.where(torch.logical_and((mask >= lower_bound_w), (mask <= upper_bound_w)),
                               torch.ones_like(mask), torch.zeros_like(mask))
            mask = torch.reshape(mask, shape=(1, w)).repeat((h, 1)).cuda()
            gen_dct_w = ref_dct_w * mask + gen_dct_w * (1 - mask)
            gen_z = idct(gen_dct_w, norm='ortho')

            gen_z_trans = gen_z.permute((0, 1, 3, 2))
            ref_z_trans = ref_z.permute((0, 1, 3, 2))
            gen_dct_h = dct(gen_z_trans, norm='ortho')
            ref_dct_h = dct(ref_z_trans, norm='ortho')
            # ref_dct_h = equifrequency_instance_normalization(content_feat=ref_dct_h, style_feat=gen_dct_h, dim=2)
            lower_bound_h = int(h * mp_percentiles[0] / 100)
            upper_bound_h = int(h * mp_percentiles[1] / 100)
            mask = torch.range(1, h).cuda()
            mask = torch.where(torch.logical_and((mask >= lower_bound_h), (mask <= upper_bound_h)),
                               torch.ones_like(mask), torch.zeros_like(mask))
            mask = torch.reshape(mask, shape=(1, h)).repeat((w, 1)).cuda()
            gen_dct_h = ref_dct_h * mask + gen_dct_h * (1 - mask)
            gen_z_trans = idct(gen_dct_h, norm='ortho')
            gen_z = gen_z_trans.permute(0, 1, 3, 2)
            return gen_z

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        gen_zT = torch.randn_like(ref_guidance[0]).cuda()

        gen_z = gen_zT
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((gen_zT.shape[0],), step, device=gen_zT.device, dtype=torch.long)
            if step >= end_step:
                gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
                gen_z = mid_FBS(ref_guidance[i], gen_z, mp_percentiles)
            else:
                gen_z, _, _ = self.p_sample_ddim(gen_z, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return gen_z

    @torch.no_grad()
    def decode_with_low_pass_FBS_new(self, z0, cond, t_dec, unconditional_guidance_scale,
                                     unconditional_conditioning, use_original_steps=False, callback=None,
                                     lp_percentile=50, end_step=500):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = torch.randn_like(z0)
        b, c, h, w = x_dec.shape
        threshold_w = int(w * lp_percentile / 100)
        threshold_h = int(h * lp_percentile / 100)
        calibration_steps = sum(time_range >= end_step)
        weights = 1 - torch.sigmoid(torch.linspace(-10, 10, calibration_steps)) ** 2
        noise = torch.randn_like(z0)
        for i, step in enumerate(iterator):  # step: 990, 980, 970, ..., 0
            index = total_steps - i - 1  # 99, 98, 97, ... , 0

            a_t = torch.full((b, 1, 1, 1), alphas[index], device=z0.device)
            ref_latent = torch.sqrt(a_t) * z0 + torch.sqrt(1 - a_t) * noise

            ts = torch.full((z0.shape[0],), step, device=z0.device, dtype=torch.long)
            if step >= end_step:
                x_dec_dct_w = dct(x_dec, norm='ortho')
                ref_latent_dct_w = dct(ref_latent, norm='ortho')
                mask = torch.range(1, w)
                mask = torch.where(mask <= threshold_w, torch.ones_like(mask), torch.zeros_like(mask))
                mask = torch.reshape(mask, shape=(1, w)).repeat((h, 1)).cuda()
                x_dec_dct_w = (ref_latent_dct_w * weights[i] + x_dec_dct_w * (1 - weights[i])) * mask + x_dec_dct_w * (
                        1 - mask)
                x_dec = idct(x_dec_dct_w, norm='ortho')

                x_dec_trans = x_dec.permute((0, 1, 3, 2))
                ref_latent_trans = ref_latent.permute((0, 1, 3, 2))
                x_dec_dct_h = dct(x_dec_trans, norm='ortho')
                ref_latent_dct_h = dct(ref_latent_trans, norm='ortho')
                mask = torch.range(1, h)
                mask = torch.where(mask <= threshold_h, torch.ones_like(mask), torch.zeros_like(mask))
                mask = torch.reshape(mask, shape=(1, h)).repeat((w, 1)).cuda()
                x_dec_dct_h = (ref_latent_dct_h * weights[i] + x_dec_dct_h * (1 - weights[i])) * mask + x_dec_dct_h * (
                        1 - mask)
                x_dec_trans = idct(x_dec_dct_h, norm='ortho')
                x_dec = x_dec_trans.permute(0, 1, 3, 2)

            x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

    @torch.no_grad()
    def decode_with_mid_pass_FBS_new(self, z0, cond, t_dec, unconditional_guidance_scale,
                                     unconditional_conditioning, use_original_steps=False, callback=None,
                                     mp_percentiles=(10, 70), end_step=500):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = torch.randn_like(z0)
        b, c, h, w = x_dec.shape
        lower_bound_w = int(w * mp_percentiles[0] / 100)
        upper_bound_w = int(w * mp_percentiles[1] / 100)
        lower_bound_h = int(h * mp_percentiles[0] / 100)
        upper_bound_h = int(h * mp_percentiles[1] / 100)
        calibration_steps = sum(time_range >= end_step)
        weights = 1 - torch.sigmoid(torch.linspace(-20, 20, calibration_steps)) ** 1
        noise = torch.randn_like(z0)
        for i, step in enumerate(iterator):  # step: 990, 980, 970, ..., 0
            index = total_steps - i - 1  # 99, 98, 97, ... , 0

            a_t = torch.full((b, 1, 1, 1), alphas[index], device=z0.device)
            ref_latent = torch.sqrt(a_t) * z0 + torch.sqrt(1 - a_t) * noise

            ts = torch.full((z0.shape[0],), step, device=z0.device, dtype=torch.long)
            if step >= end_step:
                x_dec_dct_w = dct(x_dec, norm='ortho')
                ref_latent_dct_w = dct(ref_latent, norm='ortho')
                mask = torch.range(1, w)
                mask = torch.where(torch.logical_and((mask >= lower_bound_w), (mask <= upper_bound_w)),
                                   torch.ones_like(mask), torch.zeros_like(mask))
                mask = torch.reshape(mask, shape=(1, w)).repeat((h, 1)).cuda()
                x_dec_dct_w = (ref_latent_dct_w * weights[i] + x_dec_dct_w * (1 - weights[i])) * mask + x_dec_dct_w * (
                        1 - mask)
                x_dec = idct(x_dec_dct_w, norm='ortho')

                x_dec_trans = x_dec.permute((0, 1, 3, 2))
                ref_latent_trans = ref_latent.permute((0, 1, 3, 2))
                x_dec_dct_h = dct(x_dec_trans, norm='ortho')
                ref_latent_dct_h = dct(ref_latent_trans, norm='ortho')
                mask = torch.range(1, h)
                mask = torch.where(torch.logical_and((mask >= lower_bound_h), (mask <= upper_bound_h)),
                                   torch.ones_like(mask), torch.zeros_like(mask))
                mask = torch.reshape(mask, shape=(1, h)).repeat((w, 1)).cuda()
                x_dec_dct_h = (ref_latent_dct_h * weights[i] + x_dec_dct_h * (1 - weights[i])) * mask + x_dec_dct_h * (
                        1 - mask)
                x_dec_trans = idct(x_dec_dct_h, norm='ortho')
                x_dec = x_dec_trans.permute(0, 1, 3, 2)

            x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

    @torch.no_grad()
    def decode_with_style_guidance(self, content_latent, style_latent, t_dec,
                                   unconditional_conditioning, use_original_steps=False, callback=None, end_step=0):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        _, c, h, w = content_latent.shape

        content_latent = torch.randn_like(content_latent)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((content_latent.shape[0],), step, device=content_latent.device, dtype=torch.long)
            if step > end_step:

                content_latent = adaptive_instance_normalization(content_feat=content_latent, style_feat=style_latent)

                style_latent, _, _, extracted_list = self.p_sample_ddim(style_latent, unconditional_conditioning, ts,
                                                                        index=index,
                                                                        use_original_steps=use_original_steps,
                                                                        unconditional_guidance_scale=1.0,
                                                                        unconditional_conditioning=None, extract=True)
                content_latent, _, _ = self.p_sample_ddim(content_latent, unconditional_conditioning, ts,
                                                          index=index,
                                                          use_original_steps=use_original_steps,
                                                          unconditional_guidance_scale=1.0,
                                                          unconditional_conditioning=None, inject=True,
                                                          mean_std_list=extracted_list)
            else:
                content_latent, _, _ = self.p_sample_ddim(content_latent, unconditional_conditioning, ts,
                                                          index=index,
                                                          use_original_steps=use_original_steps,
                                                          unconditional_guidance_scale=1.0,
                                                          unconditional_conditioning=None)

            if callback: callback(i)
        return content_latent

    @torch.no_grad()
    def decode_with_low_pass_FBS_z0(self, ref_latent, cond, t_dec, unconditional_guidance_scale,
                                    unconditional_conditioning, use_original_steps=False, callback=None,
                                    lp_percentile=50, end_step=500):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = torch.randn_like(ref_latent)
        b, c, h, w = x_dec.shape
        threshold_w = int(w * lp_percentile / 100)
        threshold_h = int(h * lp_percentile / 100)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((ref_latent.shape[0],), step, device=ref_latent.device, dtype=torch.long)
            if step >= end_step:
                x_dec_dct_w = dct(x_dec, norm='ortho')
                ref_latent_dct_w = dct(ref_latent, norm='ortho')
                mask = torch.range(1, w)
                mask = torch.where(mask <= threshold_w, torch.ones_like(mask), torch.zeros_like(mask))
                mask = torch.reshape(mask, shape=(1, w)).repeat((h, 1)).cuda()
                x_dec_dct_w = ref_latent_dct_w * mask + x_dec_dct_w * (1 - mask)
                x_dec = idct(x_dec_dct_w, norm='ortho')

                x_dec_trans = x_dec.permute((0, 1, 3, 2))
                ref_latent_trans = ref_latent.permute((0, 1, 3, 2))
                x_dec_dct_h = dct(x_dec_trans, norm='ortho')
                ref_latent_dct_h = dct(ref_latent_trans, norm='ortho')
                mask = torch.range(1, h)
                mask = torch.where(mask <= threshold_h, torch.ones_like(mask), torch.zeros_like(mask))
                mask = torch.reshape(mask, shape=(1, h)).repeat((w, 1)).cuda()
                x_dec_dct_h = ref_latent_dct_h * mask + x_dec_dct_h * (1 - mask)
                x_dec_trans = idct(x_dec_dct_h, norm='ortho')
                x_dec = x_dec_trans.permute(0, 1, 3, 2)

                ref_latent, _, _ = self.p_sample_ddim(ref_latent, unconditional_conditioning, ts,
                                                      index=index,
                                                      use_original_steps=use_original_steps,
                                                      unconditional_guidance_scale=1.0,
                                                      unconditional_conditioning=None)

                x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index,
                                                 use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)

            else:
                x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)

            if callback: callback(i)
        return x_dec