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

# resolution of the generated image
H, W = 512, 512

# random seed
seed = -1

inversion_steps = 500
decode_steps = 100

unconditional_guidance_scale = 1.0

# set the image path of the reference image
img_path = 'girl_src.jpg'

model = create_model('models/model_ldm_v15.yaml').cuda()
model.load_state_dict(load_state_dict('models/v1-5-pruned-emaonly.ckpt', location='cuda'), strict=False)
sampler = DCT_Sampler(model)

img = np.array(Image.open(img_path).resize((W, H)), dtype=np.float32)
img = img / 127.5 - 1.0  # -1 ~ 1
img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].cuda()  # 1, 3, h, w

if seed == -1:
    seed = random.randint(0, 65535)
seed_everything(seed)

un_cond = {"c_crossattn": [model.get_learned_conditioning([''])]}
shape = (4, H // 8, W // 8)

encoder_posterior = model.encode_first_stage(img_tensor)
z0 = model.get_first_stage_encoding(encoder_posterior).detach()
sampler.make_schedule(ddim_num_steps=inversion_steps)
zT = sampler.inversion(x0=z0, cond=un_cond, t_inv=inversion_steps, return_intermediate=False)

sampler.make_schedule(ddim_num_steps=decode_steps)
gen_z0 = sampler.decode(zT, cond=un_cond, t_dec=decode_steps,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=un_cond)

gen_x0 = torch.clip(model.decode_first_stage(gen_z0), min=-1, max=1).squeeze()
gen_x0 = (einops.rearrange(gen_x0, 'c h w -> h w c') * 127.5 + 127.5).cpu().numpy().astype(np.uint8)
Image.fromarray(gen_x0).save('inv_res.jpg')
