import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import numpy as np


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


def dwt(x):
    x01 = x[:, :, 0::2, :] / 4
    x02 = x[:, :, 1::2, :] / 4
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def idwt(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :]
    x2 = x[:, out_channel:out_channel * 2, :, :]
    x3 = x[:, out_channel * 2:out_channel * 3, :, :]
    x4 = x[:, out_channel * 3:out_channel * 4, :, :]

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


def multi_level_DWT(x, dwt_level):
    DWT_list = []
    for _ in range(dwt_level):
        res = dwt(x)
        DWT_list.append(res)
        _, c, _, _ = res.shape
        x = res[:, :c//4, :, :]
    return DWT_list


def dwt_res_grid(x):
    dwt_res = dwt(x)
    _, c, _, _ = dwt_res.shape
    x1 = dwt_res[:, 0:c//4, :, :]
    x2 = dwt_res[:, c//4:c//4 * 2, :, :]
    x3 = dwt_res[:, c//4 * 2:c//4 * 3, :, :]
    x4 = dwt_res[:, c//4 * 3:c//4 * 4, :, :]
    dwt_res = torch.cat([x1, x2, x3, x4], dim=0)
    grid = torchvision.utils.make_grid(dwt_res, nrow=2, padding=0).permute(1, 2, 0)
    # grid = grid * 255
    # Image.fromarray(grid.cpu().numpy().astype(np.uint8)).show()
    return grid


def multi_level_dwt_res_show(x, dwt_level, ratio=1):
    DWT_list = multi_level_DWT(x, dwt_level)
    temp = None
    for i in range(len(DWT_list)):
        DWT = DWT_list.pop()
        _, c, _, _ = DWT.shape
        x1 = DWT[:, 0:c // 4, :, :]
        x2 = DWT[:, c // 4:c // 4 * 2, :, :] * ratio
        x3 = DWT[:, c // 4 * 2:c // 4 * 3, :, :] * ratio
        x4 = DWT[:, c // 4 * 3:c // 4 * 4, :, :] * ratio
        if i > 0:
            x1 = temp
        DWT = torch.cat([x1, x2, x3, x4], dim=0)
        grid = torchvision.utils.make_grid(DWT, nrow=2, padding=0)
        temp = grid.unsqueeze(0)
    grid = grid.permute(1, 2, 0)  # h, w, c
    return grid


def high_freq_enhance(x, ratio):
    dwt_res = dwt(x)
    _, c, _, _ = dwt_res.shape
    x1 = dwt_res[:, 0:c // 4, :, :]
    x2 = dwt_res[:, c // 4:c // 4 * 2, :, :] * ratio
    x3 = dwt_res[:, c // 4 * 2:c // 4 * 3, :, :] * ratio
    x4 = dwt_res[:, c // 4 * 3:c // 4 * 4, :, :] * ratio
    dwt_res = torch.cat([x1, x2, x3, x4], dim=1)
    enhanced = idwt(dwt_res)
    return enhanced


def low_freq_enhance(x, ratio):
    dwt_res = dwt(x)
    _, c, _, _ = dwt_res.shape
    x1 = dwt_res[:, 0:c // 4, :, :] * ratio
    x2 = dwt_res[:, c // 4:c // 4 * 2, :, :]
    x3 = dwt_res[:, c // 4 * 2:c // 4 * 3, :, :]
    x4 = dwt_res[:, c // 4 * 3:c // 4 * 4, :, :]
    dwt_res = torch.cat([x1, x2, x3, x4], dim=1)
    enhanced = idwt(dwt_res)
    return enhanced


def contrast_enhance(x, ratio):
    dwt_res = dwt(x)
    _, c, _, _ = dwt_res.shape
    x1 = dwt_res[:, 0:c // 4, :, :]
    x2 = dwt_res[:, c // 4:c // 4 * 2, :, :]
    x3 = dwt_res[:, c // 4 * 2:c // 4 * 3, :, :]
    x4 = dwt_res[:, c // 4 * 3:c // 4 * 4, :, :]

    x1 = transforms.ColorJitter(contrast=(ratio, ratio))(x1)
    dwt_res = torch.cat([x1, x2, x3, x4], dim=1)
    enhanced = idwt(dwt_res)
    return enhanced


def high_freq_transfer(ref, src, ratio):
    ref_dwt = dwt(ref)
    src_dwt = dwt(src)
    _, c, _, _ = ref_dwt.shape
    src_dwt[:, c // 4:c // 4 * 2, :, :] = ref_dwt[:, c // 4:c // 4 * 2, :, :] * ratio
    src_dwt[:, c // 4 * 2:c // 4 * 3, :, :] = ref_dwt[:, c // 4 * 2:c // 4 * 3, :, :] * ratio
    src_dwt[:, c // 4 * 3:c // 4 * 4, :, :] = ref_dwt[:, c // 4 * 3:c // 4 * 4, :, :] * ratio
    src = idwt(src_dwt)
    return src


def multi_level_high_freq_transfer(ref, src, dwt_level):
    ref = adaptive_instance_normalization(content_feat=ref, style_feat=src)
    ref_dwt_list = multi_level_DWT(ref, dwt_level)
    src_dwt_list = multi_level_DWT(src, dwt_level)
    temp = None
    for i in range(dwt_level):
        ref_dwt = ref_dwt_list.pop(-1)
        src_dwt = src_dwt_list.pop(-1)
        ref_dwt = adaptive_instance_normalization(content_feat=ref_dwt, style_feat=src_dwt)
        _, c, _, _ = ref_dwt.shape
        if i > 0:
            src_dwt[:, :c//4, :, :] = temp

        src_dwt[:, c // 4:c // 4 * 2, :, :] = ref_dwt[:, c // 4:c // 4 * 2, :, :]
        src_dwt[:, c // 4 * 2:c // 4 * 3, :, :] = ref_dwt[:, c // 4 * 2:c // 4 * 3, :, :]
        src_dwt[:, c // 4 * 3:c // 4 * 4, :, :] = ref_dwt[:, c // 4 * 3:c // 4 * 4, :, :]
        temp = idwt(src_dwt)
    return temp


def low_freq_transfer(ref, src, dwt_level, high_freq_transfer_ratio=0.0):
    ref_dwt_list = multi_level_DWT(ref, dwt_level)
    src_dwt_list = multi_level_DWT(src, dwt_level)
    temp = None
    for i in range(dwt_level):
        ref_dwt = ref_dwt_list.pop(-1)
        src_dwt = src_dwt_list.pop(-1)
        _, c, _, _ = ref_dwt.shape
        if i == 0:
            src_dwt[:, :c // 4, :, :] = ref_dwt[:, :c // 4, :, :]
        if i > 0:
            src_dwt[:, :c // 4, :, :] = temp
        src_dwt[:, c // 4:c // 4 * 2, :, :] = src_dwt[:, c // 4:c // 4 * 2, :, :] * (1 - high_freq_transfer_ratio) + \
                                              ref_dwt[:, c // 4:c // 4 * 2, :, :] * high_freq_transfer_ratio
        src_dwt[:, c // 4 * 2:c // 4 * 3, :, :] = src_dwt[:, c // 4 * 2:c // 4 * 3, :, :] * (1 - high_freq_transfer_ratio) + \
                                                  ref_dwt[:, c // 4 * 2:c // 4 * 3, :, :] * high_freq_transfer_ratio
        src_dwt[:, c // 4 * 3:c // 4 * 4, :, :] = src_dwt[:, c // 4 * 3:c // 4 * 4, :, :] * (1 - high_freq_transfer_ratio) + \
                                                  ref_dwt[:, c // 4 * 3:c // 4 * 4, :, :] * high_freq_transfer_ratio
        temp = idwt(src_dwt)
    return temp


def SR_dwt_transfer(lr, hr, dwt_level):
    _, _, lr_h, lr_w = lr.shape
    _, _, hr_h, hr_w = hr.shape
    assert (hr_h / lr_h == 2 ** dwt_level) and (hr_w / lr_w == 2 ** dwt_level), 'spatial dimension not match'
    hr_dwt_list = multi_level_DWT(hr, dwt_level)
    temp = None
    for i in range(dwt_level):
        hr_dwt = hr_dwt_list.pop(-1)
        _, c, _, _ = hr_dwt.shape
        if i == 0:
            hr_dwt[:, :c // 4, :, :] = lr
        if i > 0:
            hr_dwt[:, :c // 4, :, :] = temp
        temp = idwt(hr_dwt)
    return temp


if __name__ == "__main__":

    x = Image.open('../test1.jpg')
    x = transforms.ToTensor()(x)   # 0 ~ 1
    x = torch.unsqueeze(x, 0)
    x = transforms.Resize(size=(512, 512))(x)

    # subbands = dwt(x)
    # reconstruct = idwt(subbands) * 255
    # reconstruct = reconstruct.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # Image.fromarray(reconstruct).show()

    # DWT_list = multi_level_DWT(x, 3)
    # IDWT_list = multi_level_IDWT(DWT_list)
    # reconstruct = IDWT_list[-1] * 255
    # reconstruct = reconstruct.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # Image.fromarray(reconstruct).show()

    # grid = multi_level_dwt_res_show(x, 2)
    # print(grid.shape)
    # grid = grid * 255
    # grid = grid.clip(0, 255)
    # Image.fromarray(grid.cpu().numpy().astype(np.uint8)).show()

    # enhanced = high_freq_enhance(x, 10) * 255
    # enhanced = enhanced.clip(0, 255)
    # enhanced = enhanced.squeeze().permute(1, 2, 0)
    # Image.fromarray(enhanced.cpu().numpy().astype(np.uint8)).show()

    # enhanced = low_freq_enhance(x, 2) * 255
    # enhanced = enhanced.clip(0, 255)
    # enhanced = enhanced.squeeze().permute(1, 2, 0)
    # Image.fromarray(enhanced.cpu().numpy().astype(np.uint8)).show()

    # enhanced = contrast_enhance(x, 1) * 255
    # enhanced = enhanced.clip(0, 255)
    # enhanced = enhanced.squeeze().permute(1, 2, 0)
    # Image.fromarray(enhanced.cpu().numpy().astype(np.uint8)).show()

    # src = torch.randn_like(x)
    # ref = x
    # src = high_freq_transfer(ref=ref, src=src, ratio=1)
    # grid = multi_level_dwt_res_show(src, 1, 20)
    # print(grid.shape)
    # grid = grid * 255
    # grid = grid.clip(0, 255)

    # src = torch.randn_like(x)
    # ref = x
    # src = multi_level_high_freq_transfer(ref=ref, src=src, dwt_level=8, ratio=1)
    # grid = multi_level_dwt_res_show(src, 1, 20)
    # print(grid.shape)
    # grid = grid * 255
    # grid = grid.clip(0, 255)
    # Image.fromarray(grid.cpu().numpy().astype(np.uint8)).show()

    # src = torch.randn_like(x)
    # ref = x
    # src = low_freq_transfer(ref=ref, src=src, dwt_level=2, high_freq_ratio=0)
    # grid = multi_level_dwt_res_show(src, dwt_level=1, ratio=100)
    # print(grid.shape)
    # grid = grid * 255
    # grid = grid.clip(0, 255)
    # Image.fromarray(grid.cpu().numpy().astype(np.uint8)).show()

    # src = torch.randn_like(x)
    # ref = x
    # src = low_freq_transfer(ref=ref, src=src, dwt_level=3, high_freq_transfer_ratio=0.8)
    # src = src.squeeze().permute(1, 2, 0)
    # src = src * 255
    # src = src.clip(0, 255)
    # Image.fromarray(src.cpu().numpy().astype(np.uint8)).show()

    lr = x
    hr = torch.zeros(size=(1, 3, 512*4, 512*4))
    res = SR_dwt_transfer(lr, hr, 2)
    # grid = multi_level_dwt_res_show(res, dwt_level=2, ratio=100)
    # print(grid.shape)
    # grid = grid * 255
    # grid = grid.clip(0, 255)
    # Image.fromarray(grid.cpu().numpy().astype(np.uint8)).show()
    print(res.shape)
    res = res.squeeze().permute(1, 2, 0) * 255
    res = res.clip(0, 255)
    Image.fromarray(res.cpu().numpy().astype(np.uint8)).show()





