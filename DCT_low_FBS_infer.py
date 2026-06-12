import einops
import numpy as np
import random
import torch
from PIL import Image
import os
from pytorch_lightning import seed_everything
from DCTDiff.tools import create_model, load_state_dict
from DCTDiff.DCT_sampler import DCT_Sampler

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# random seed
seed = -1

# set the total steps of the sampling trajectory
decode_steps = 100

# set the ratio of the calibration phase to the whole sampling trajectory
dct_calibration_ratio = 0.5

unconditional_guidance_scale = 7.5

# set the image path of the reference image
# img_path = 'test1.jpg'
img_path = 'test2.jpg'

# resolution of the generated image
W, H = Image.open(img_path).size

# set the target text prompt
# target_prompt = 'photo of a young man, highly detailed'
target_prompt = 'oil painting of a dog, highly detailed'

model = create_model('models/model_ldm_v15.yaml').cuda()
model.load_state_dict(load_state_dict('models/v1-5-pruned-emaonly.ckpt', location='cuda'), strict=False)
sampler = DCT_Sampler(model)
sampler.make_schedule(ddim_num_steps=decode_steps)

img = np.array(Image.open(img_path).resize((W, H)), dtype=np.float32)
img = img / 127.5 - 1.0  # -1 ~ 1
img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].cuda()  # 1, 3, h, w

if seed == -1:
    seed = random.randint(0, 65535)
seed_everything(seed)

un_cond = {"c_crossattn": [model.get_learned_conditioning([''])]}
cond = {"c_crossattn": [model.get_learned_conditioning([target_prompt])]}
shape = (4, H // 8, W // 8)

encoder_posterior = model.encode_first_stage(img_tensor)
z0 = model.get_first_stage_encoding(encoder_posterior).detach()

if not os.path.exists('inversion_trajectory.pt'):
    ref_zT, intermediates = sampler.inversion(x0=z0, cond=un_cond, t_inv=decode_steps, return_intermediate=True)
    inversion_trajectory = torch.concat(intermediates, dim=0)
    torch.save(inversion_trajectory, 'inversion_trajectory.pt')

inversion_trajectory = torch.load('inversion_trajectory.pt').cuda()
intermediates = list(torch.split(inversion_trajectory, 1, dim=0))
intermediates = intermediates[::-1]

gen_z0 = sampler.decode_with_low_FBS(ref_guidance=intermediates, cond=cond, t_dec=decode_steps,
                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                     unconditional_conditioning=un_cond, lp_percentile=70,
                                     dct_calibration_ratio=dct_calibration_ratio)

gen_x0 = torch.clip(model.decode_first_stage(gen_z0), min=-1, max=1).squeeze()
gen_x0 = (einops.rearrange(gen_x0, 'c h w -> h w c') * 127.5 + 127.5).cpu().numpy().astype(np.uint8)
# Image.fromarray(gen_x0).save('test1_low-FBS_res.jpg')
Image.fromarray(gen_x0).save('test2_low-FBS_res.jpg')