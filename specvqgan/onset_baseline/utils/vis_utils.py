import copy
import cv2
import itertools as itl
import json
import kornia as K
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import PIL
from PIL import Image, ImageDraw, ImageFont
import pylab
import random

import torch

import pdb

def clip_rescale(x, lo = None, hi = None):
    if lo is None:
        lo = np.min(x)
    if hi is None:
        hi = np.max(x)
    return np.clip((x - lo)/(hi - lo), 0., 1.)

def apply_cmap(im, cmap = pylab.cm.jet, lo = None, hi = None):
    return cmap(clip_rescale(im, lo, hi).flatten()).reshape(im.shape[:2] + (-1,))[:, :, :3]

def cmap_im(cmap, im, lo = None, hi = None):
    return np.uint8(255*apply_cmap(im, cmap, lo, hi))

def calc_acc(prob, labels, k=1):
    thred = 0.5
    pred = torch.argsort(prob, dim=-1, descending=True)[..., :k]
    corr = (pred.view(-1) == labels).cpu().numpy()
    corr = corr.reshape((-1, resol*resol))
    acc = corr.sum(1) / (resol*resol)  # compute rate of successful patch for each image
    corr_index = np.where((acc > thred) == True)[0]
    return corr_index    

# def compute_acc_list(A_IS, k=0): 
#     criterion = nn.NLLLoss()
#     M, N = A_IS.size()
#     target = torch.from_numpy(np.repeat(np.eye(N), M // N, axis=0)).to(DEVICE)
#     _, labels = target.max(dim=1)
#     loss = criterion(torch.log(A_IS), labels.long())
#     acc = None
#     if k > 0:
#         corr_index = calc_acc(A_IS, labels, k)
#     return corr_index

def get_fcn_sim(full_img, feat_audio, net, B, resol, norm=True):
    feat_img = net.forward_fcn(full_img)
    feat_img = feat_img.permute(0, 2,3,1).reshape(-1, 128)
    A_II, A_IS, A_SI = net.GetAMatrix(feat_img, feat_audio, norm=norm)
    A_IS_ = A_IS.reshape((B, resol*resol, B))
    A_IIS_ = (A_II @ A_IS).reshape((B, resol*resol, B))
    A_II_ = A_II.reshape((B, resol*resol, B*resol*resol))

    return A_IS_, A_IIS_, A_II_

def upsample_lowest(sim, im_h, im_w, pr): 
    sim_h, sim_w = sim.shape
    prob_map_per_patch = np.zeros((im_h, im_w, pr.resol*pr.resol))
    # pdb.set_trace()
    for i in range(pr.resol): 
        for j in range(pr.resol): 
            y1 = pr.patch_stride * i 
            y2 = pr.patch_stride * i + pr.psize
            x1 = pr.patch_stride * j
            x2 = pr.patch_stride * j + pr.psize
            prob_map_per_patch[y1:y2, x1:x2, i * pr.resol + j] = sim[i, j]
    # pdb.set_trace()
    upsampled = np.sum(prob_map_per_patch, axis=-1) / np.sum(prob_map_per_patch > 0, axis=-1)

    return upsampled


def grid_interp(pr, input, output_size, mode='bilinear'):
    # import pdb; pdb.set_trace()
    n = 1
    c = 1
    ih, iw = input.shape
    input = input.view(n, c, ih, iw)
    oh, ow = output_size

    pad = (pr.psize - pr.patch_stride) // 2 
    ch = oh - pad * 2 
    cw = ow - pad * 2
    # normalize to [-1, 1]
    h = (torch.arange(0, oh) - pad) / (ch-1) * 2 - 1
    w = (torch.arange(0, ow) - pad) / (cw-1) * 2 - 1

    grid = torch.zeros(oh, ow, 2)
    grid[:, :, 0] = w.unsqueeze(0).repeat(oh, 1)
    grid[:, :, 1] = h.unsqueeze(0).repeat(ow, 1).transpose(0, 1)
    grid = grid.unsqueeze(0).repeat(n, 1, 1, 1) # grid.shape: [n, oh, ow, 2]
    grid = grid.to(input.device)
    res = torch.nn.functional.grid_sample(input, grid, mode=mode, padding_mode="border", align_corners=False).squeeze()
    return res 


def upsample_lowest_torch(sim, im_h, im_w, pr): 
    sim = sim.reshape(pr.resol*pr.resol)
    # precompute the temeplate
    prob_map_per_patch = torch.from_numpy(pr.template).to('cuda')
    prob_map_per_patch = prob_map_per_patch * sim.reshape(1,1,-1)
    upsampled = torch.sum(prob_map_per_patch, dim=-1) / torch.sum(prob_map_per_patch > 0, dim=-1)

    return upsampled


def gen_vis_map(prob, im_h, im_w, pr, bound=False, lo=0, hi=0.3, mode='nearest'): 
    """
    prob: probability map for patches
    im_h, im_w: original image size
    resol: resolution of patches
    bound: whether to give low and high bound for probability
    lo: 
    hi: 
    mode: upsample method for probability
    """
    resol = pr.resol
    if mode == 'nearest': 
        resample = PIL.Image.NEAREST
    elif mode == 'bilinear': 
        resample = PIL.Image.BILINEAR
    sim = prob.reshape((resol, resol))
    # pdb.set_trace()
    # updample similarity
    if mode in ['nearest', 'bilinear']: 
        if torch.is_tensor(sim): 
            sim = sim.cpu().numpy()
        sim_up = np.array(Image.fromarray(sim).resize((im_w, im_h), resample=resample))
    elif mode == 'lowest': 
        sim_up = upsample_lowest_torch(sim, im_w, im_h, pr)
        sim_up = sim_up.detach().cpu().numpy()
    elif mode == 'grid': 
        sim_up = grid_interp(pr, sim, (im_h, im_w), 'bilinear')
        sim_up = sim_up.detach().cpu().numpy()

    if not bound: 
        lo = None
        hi = None
    # generate heat map
    # pdb.set_trace()
    vis = cmap_im(pylab.cm.jet, sim_up, lo=lo, hi=hi)

    # p weights the cmap on original image
    p = sim_up / sim_up.max() * 0.3 + 0.3
    p = p[..., None]
    
    return p, vis


def gen_upsampled_prob(prob, im_h, im_w, pr, bound=False, lo=0, hi=0.3, mode='nearest'): 
    """
    prob: probability map for patches
    im_h, im_w: original image size
    resol: resolution of patches
    bound: whether to give low and high bound for probability
    lo: 
    hi: 
    mode: upsample method for probability
    """
    resol = pr.resol
    if mode == 'nearest': 
        resample = PIL.Image.NEAREST
    elif mode == 'bilinear': 
        resample = PIL.Image.BILINEAR
    sim = prob.reshape((resol, resol))
    # pdb.set_trace()
    # updample similarity
    if mode in ['nearest', 'bilinear']: 
        if torch.is_tensor(sim): 
            sim = sim.cpu().numpy()
        sim_up = np.array(Image.fromarray(sim).resize((im_w, im_h), resample=resample))
    elif mode == 'lowest': 
        sim_up = upsample_lowest_torch(sim, im_w, im_h, pr)
        sim_up = sim_up.cpu().numpy()
    sim_up = sim_up / sim_up.max()
    return sim_up


def gen_vis_map_probmap_up(prob_up, bound=False, lo=0, hi=0.3, mode='nearest'): 
    if mode == 'nearest': 
        resample = PIL.Image.NEAREST
    elif mode == 'bilinear': 
        resample = PIL.Image.BILINEAR
    if not bound: 
        lo = None
        hi = None
    vis = cmap_im(pylab.cm.jet, prob_up, lo=None, hi=None)
    if bound: 
        # when hi gets larger, cmap becomes less visibal
        p = prob_up / prob_up.max() * (0.3+0.4*(1-hi)) + 0.3
    else: 
        # if not bound, cmap always weights 0.3 on original image
        p = prob_up / prob_up.max() * 0.3 + 0.3
    p = p[..., None]
    
    return p, vis


def rgb2bgr(im): 
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def gen_bbox_patches(im, patch_ind, resol, patch_size=64, lin_w=3, lin_color=np.array([255,0,0])): 
    # TODO: make it work for different image size
    stride = int((256-patch_size)/(resol-1))
    
    im_w, im_h = im.shape[1], im.shape[0]

    r_ind = patch_ind // resol
    c_ind = patch_ind % resol
    y1 = r_ind * stride
    y2 = r_ind * stride + patch_size
    x1 = c_ind * stride
    x2 = c_ind * stride + patch_size

    im_bbox = copy.deepcopy(im)
    im_bbox[y1:y1+lin_w, x1:x2, :] = lin_color
    im_bbox[y2-lin_w:y2, x1:x2, :] = lin_color
    im_bbox[y1:y2, x1:x1+lin_w, :] = lin_color
    im_bbox[y1:y2, x2-lin_w:x2, :] = lin_color
    
    return (x1, y1, x2-x1, y2-y1), im_bbox 

def get_fcn_sim(full_img, feat_audio, net, B, resol, norm=True):
    feat_img = net.forward_fcn(full_img)
    feat_img = feat_img.permute(0, 2,3,1).reshape(-1, 128)
    A_II, A_IS, A_SI = net.GetAMatrix(feat_img, feat_audio, norm=norm)
    A_IS_ = A_IS.reshape((B, resol*resol, B))
    A_IIS_ = (A_II @ A_IS).reshape((B, resol*resol, B))
    A_II_ = A_II.reshape((B, resol*resol, B, resol*resol))
    return A_IS_, A_IIS_, A_II_

def put_text(im, text, loc, font_scale=4): 
    fontScale = font_scale
    thickness = int(fontScale / 4)
    fontColor = (0,255,255)
    lineType = 4
    im = cv2.putText(im, text, loc, cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor, thickness, lineType)
    return im

def im2video(save_path, frame_list, fps=5): 
    height, width, _ = frame_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    for frame in frame_list: 
        video.write(rgb2bgr(frame))

    cv2.destroyAllWindows()
    video.release()
    new_name = "{}_new{}".format(save_path[:-4], save_path[-4:])
    os.system("ffmpeg -v quiet -y -i \"{}\" -pix_fmt yuv420p -vcodec h264 -strict -2 -acodec aac \"{}\"".format(save_path, new_name))
    os.system("rm -rf \"{}\"".format(save_path))

def get_face_landmark(frame_path_): 
    video_folder = Path(frame_path_).parent.parent
    frame_name = frame_path_.split('/')[-1]
    face_landmark_path = os.path.join(video_folder, "face_bbox_landmark.json")
    if not os.path.exists(face_landmark_path): 
        return None
    with open(face_landmark_path, 'r') as f:
        face_landmark = json.load(f)
    if len(face_landmark[frame_name]) == 0: 
        return None
    b = face_landmark[frame_name][0]
    return b

def make_color_wheel():
    # same source as color_flow

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    #colorwheel = zeros(ncols, 3) # r g b
    # matlab correction
    colorwheel = np.zeros((1+ncols, 4)) # r g b

    col = 0
    #RY
    colorwheel[1:1+RY, 1] = 255
    colorwheel[1:1+RY, 2] = np.floor(255*np.arange(0, 1+RY-1)/RY).T
    col = col+RY

    #YG
    colorwheel[col+1:col+1+YG, 1] = 255 - np.floor(255*np.arange(0,1+YG-1)/YG).T
    colorwheel[col+1:col+1+YG, 2] = 255
    col = col+YG

    #GC
    colorwheel[col+1:col+1+GC, 2] = 255
    colorwheel[col+1:col+1+GC, 3] = np.floor(255*np.arange(0,1+GC-1)/GC).T
    col = col+GC

    #CB
    colorwheel[col+1:col+1+CB, 2] = 255 - np.floor(255*np.arange(0,1+CB-1)/CB).T
    colorwheel[col+1:col+1+CB, 3] = 255
    col = col+CB

    #BM
    colorwheel[col+1:col+1+BM, 3] = 255
    colorwheel[col+1:col+1+BM, 1] = np.floor(255*np.arange(0,1+BM-1)/BM).T
    col = col+BM

    #MR
    colorwheel[col+1:col+1+MR, 3] = 255 - np.floor(255*np.arange(0,1+MR-1)/MR).T
    colorwheel[col+1:col+1+MR, 1] = 255  

    # 1-based to 0-based indices
    return colorwheel[1:, 1:]

def warp(im, flow): 
    # im : C x H x W
    # flow : 2 x H x W, such that flow[dst_y, dst_x] = (src_x, src_y),
    #     where (src_x, src_y) is the pixel location we want to sample from.

    # grid_sample the grid is in the range in [-1, 1] 
    grid =  -1. + 2. * flow/(-1 + np.array([im.shape[2], im.shape[1]], np.float32))[:, None, None]

    # print('grid range =', grid.min(), grid.max())
    ft = torch.FloatTensor
    warped = torch.nn.functional.grid_sample(
        ft(im[None].astype(np.float32)), 
        ft(grid.transpose((1, 2, 0))[None]), 
        mode = 'bilinear', padding_mode = 'zeros', align_corners=True)
    return warped.cpu().numpy()[0].astype(im.dtype)

def compute_color(u, v):
    # from same source as color_flow; please see above comment
    # nan_idx = ut.lor(np.isnan(u), np.isnan(v))
    nan_idx = np.logical_or(np.isnan(u), np.isnan(v))
    u[nan_idx] = 0
    v[nan_idx] = 0
    colorwheel = make_color_wheel()
    ncols = colorwheel.shape[0]
    
    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u)/np.pi
    
    #fk = (a + 1)/2. * (ncols-1) + 1
    fk = (a + 1)/2. * (ncols-1)

    k0 = np.array(np.floor(fk), 'l')

    k1 = k0 + 1
    k1[k1 == ncols] = 1

    f = fk - k0

    im = np.zeros(u.shape + (3,))
    
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0]/255.
        col1 = tmp[k1]/255.
        col = (1-f)*col0 + f*col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx]*(1-col[idx])
        col[np.logical_not(idx)] *= 0.75
        im[:, :, i] = np.uint8(np.floor(255*col*(1-nan_idx)))

    return im

def color_flow(flow, max_flow = None):
    flow = flow.copy()
    # based on flowToColor.m by Deqing Sun, orignally based on code by Daniel Scharstein
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10
    height, width, nbands = flow.shape
    assert nbands == 2
    u, v = flow[:,:,0], flow[:,:,1]
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1.

    idx_unknown = np.logical_or(np.abs(u) > UNKNOWN_FLOW_THRESH,  np.abs(v) > UNKNOWN_FLOW_THRESH)
    u[idx_unknown] = 0
    v[idx_unknown] = 0
    
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(maxrad, np.max(rad))

    if max_flow > 0:
        maxrad = max_flow

    u = u/(maxrad + np.spacing(1))
    v = v/(maxrad + np.spacing(1))
    
    im = compute_color(u, v)
    im[idx_unknown] = 0
    return im

def plt_fig_to_np_img(fig): 
    canvas = FigureCanvas(fig)  # draw the canvas, cache the renderer
    canvas.draw() 
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(int(height), int(width), 3)

    return image

def save_np_img(image, path): 
    cv2.imwrite(path, rgb2bgr(image))

def find_patch_topk_aud(mat, top_k): 
    top_k_ind = torch.argsort(mat, dim=-1, descending=True)[..., :top_k].squeeze()
    top_k_ind = top_k_ind.reshape(-1).cpu().numpy()
    return top_k_ind

def find_patch_pred_topk(mat, top_k, target): 
    M, N = mat.size()
    labels = torch.from_numpy(target * np.ones(M)).to('cuda')
    top_k_ind = torch.sum(torch.argsort(mat, dim=-1, descending=True)[..., :top_k] == labels.view(-1, 1), dim=-1).nonzero().reshape(-1)
    top_k_ind  = top_k_ind.reshape(-1).cpu().numpy()
    return top_k_ind

def gen_masked_img(mask_ind, resol, img): 
    mask = torch.zeros(resol*resol)
    mask = mask.scatter_(0, torch.from_numpy(mask_ind), 1.)
    mask = mask.reshape(resol, resol).numpy()
    img_h = img.shape[1]
    img_w = img.shape[0]
    mask_up = np.array(Image.fromarray(mask*255).resize((img_h, img_w), resample=PIL.Image.NEAREST))
    mask_up = mask_up[..., None]
    image_seg = np.uint8(img * 0.7 + mask_up * 0.3)
    
    return image_seg

def drop_2rand_ch(patch, remain_c=0): 
    B, P, C, H, W = patch.shape
    patch_c = patch[:, :, remain_c, :, :].unsqueeze(2)
    # patch_droped = torch.zeros_like(patch)
    # patch_droped[:, :, remain_c, :, :] = patch_c
    c_std = torch.std(patch_c, dim=(3,4))
    gauss_n = 0.5 + (0.01 * c_std.reshape(B, P, 1, 1, 1) * torch.randn(B, P, 2, H, W).to('cuda'))
    
    patch_dropped = torch.cat([gauss_n[:, :, :remain_c], patch_c, gauss_n[:, :, remain_c:]], dim=2)
    
    return patch_dropped
    # pdb.set_trace()

def vis_patch(patch, exp_path, resol, b_step): 
    B, P, C, H, W = patch.shape
    for i in range(B): 
        patch_i = patch[i].reshape(resol, resol, C, H, W)
        patch_i = patch_i.permute(2, 0, 3, 1, 4)
        patch_folded_i = patch_i.reshape(C, resol*H, resol*W)
        patch_folded_i = (patch_folded_i * 255).cpu().numpy().astype(np.uint8).transpose(1,2,0)
        cv2.imwrite('{}/{}_{}_patch_folded.jpg'.format(exp_path, str(b_step).zfill(4), str(i).zfill(4)), rgb2bgr(patch_folded_i))

def blur_patch(patch, k_size=3, sigma=0.5): 
    B, P, C, H, W = patch.shape
    gauss = K.filters.GaussianBlur2d((k_size, k_size), (sigma, sigma))
    patch = patch.reshape(B*P, C, H, W)
    blur_patch = gauss(patch).reshape(B, P, C, H, W)
    return blur_patch

def gray_project_patch(patch, device):
    N, P, C, H, W = patch.size()
    a = torch.tensor([[-1, 2, -1]]).float()
    B = (torch.eye(3) - (a.T @ a) / (a @ a.T)).to(device)
    patch = patch.permute(0, 1, 3, 4, 2)
    patch = (patch @ B).permute(0, 1, 4, 2, 3)
    return patch

def parse_color(c):
    if type(c) == type((0,)) or type(c) == type(np.array([1])):
        return c
    elif type(c) == type(''):
        return color_from_string(c)

def colors_from_input(color_input, default, n):
    """ Parse color given as input argument; gives user several options """
    # todo: generalize this to non-colors
    expanded = None
    if color_input is None:
        expanded = [default] * n
    elif (type(color_input) == type((1,))) and map(type, color_input) == [int, int, int]:
        # expand (r, g, b) -> [(r, g, b), (r, g, b), ..]
        expanded = [color_input] * n
    else:
        # general case: [(r1, g1, b1), (r2, g2, b2), ...]
        expanded = color_input

    expanded = map(parse_color, expanded)
    return expanded

def draw_pts(im, points, colors = None, width = 1, texts = None):
    # ut.check(colors is None or len(colors) == len(points))
    points = list(points)
    colors = colors_from_input(colors, (255, 0, 0), len(points))
    rects = [(p[0] - width/2, p[1] - width/2, width, width) for p in points]
    return draw_rects(im, rects, fills = colors, outlines = [None]*len(points), texts = texts)

def to_pil(im): 
    #print im.dtype
    return Image.fromarray(np.uint8(im))

def from_pil(pil): 
  #print pil
  return np.array(pil)

def draw_on(f, im):
    pil = to_pil(im)
    draw = ImageDraw.ImageDraw(pil)
    f(draw)
    return from_pil(pil)

def fail(s = ''): raise RuntimeError(s)

def check(cond, str = 'Check failed!'):
    if not cond:
        fail(str)

def draw_rects(im, rects, outlines = None, fills = None, texts = None, text_colors = None, line_widths = None, as_oval = False):
    rects = list(rects)
    outlines = colors_from_input(outlines, (0, 0, 255), len(rects))
    outlines = list(outlines)
    text_colors = colors_from_input(text_colors, (255, 255, 255), len(rects))
    text_colors = list(text_colors)
    fills = colors_from_input(fills, None, len(rects))
    fills = list(fills)
    
    if texts is None: texts = [None] * len(rects)
    if line_widths is None: line_widths = [None] * len(rects)
    
    def check_size(x, s): 
        check(x is None or len(list(x)) == len(rects), "%s different size from rects" % s)
    check_size(outlines, 'outlines')
    check_size(fills, 'fills')
    check_size(texts, 'texts')
    check_size(text_colors, 'texts')
    
    def f(draw):
        for (x, y, w, h), outline, fill, text, text_color, lw in zip(rects, outlines, fills, texts, text_colors, line_widths):
            if lw is None:
                if as_oval:
                    draw.ellipse((x, y, x + w, y + h), outline = outline, fill = fill)
                else:
                    draw.rectangle((x, y, x + w, y + h), outline = outline, fill = fill)
            else:
                d = int(np.ceil(lw/2))
                draw.rectangle((x-d, y-d, x+w+d, y+d), fill = outline)
                draw.rectangle((x-d, y-d, x+d, y+h+d), fill = outline)
                
                draw.rectangle((x+w+d, y+h+d, x-d, y+h-d), fill = outline)
                draw.rectangle((x+w+d, y+h+d, x+w-d, y-d), fill = outline)
                
            if text is not None:
                # draw text inside rectangle outline
                border_width = 2
                draw.text((border_width + x, y), text, fill = text_color)
    return draw_on(f, im)

def rand_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def int_tuple(x): 
    return tuple([int(v) for v in x])

itup = int_tuple

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)
purple = (255, 0, 255)
cyan = (0, 255, 255)


def stash_seed(new_seed = 0):
    """ Sets the random seed to new_seed. Returns the old seed. """
    if type(new_seed) == type(''):
        new_seed = hash(new_seed) % 2**32

    py_state = random.getstate()
    random.seed(new_seed)

    np_state = np.random.get_state()
    np.random.seed(new_seed)
    return (py_state, np_state)


def do_with_seed(f, seed = 0):
    old_seed = stash_seed(seed)
    res = f()
    unstash_seed(old_seed[0], old_seed[1])
    return res

def sample_at_most(xs, bound):
    return random.sample(xs, min(bound, len(xs)))

class ColorChooser:
    def __init__(self, dist_thresh = 500, attempts = 500, init_colors = [], init_pts = []):
        self.pts = init_pts
        self.colors = init_colors
        self.attempts = attempts
        self.dist_thresh = dist_thresh

    def choose(self, new_pt = (0, 0)):
        new_pt = np.array(new_pt)
        nearby_colors = []
        for pt, c in zip(self.pts, self.colors):
            if np.sum((pt - new_pt)**2) <= self.dist_thresh**2:
                nearby_colors.append(c)

        if len(nearby_colors) == 0:
            color_best = rand_color()
        else:
            nearby_colors = np.array(sample_at_most(nearby_colors, 100), 'l')
            choices = np.array(np.random.rand(self.attempts, 3)*256, 'l')
            dists = np.sqrt(np.sum((choices[:, np.newaxis, :] - nearby_colors[np.newaxis, :, :])**2, axis = 2))
            costs = np.min(dists, axis = 1)
        assert costs.shape == (len(choices),)
        color_best = itup(choices[np.argmax(costs)])

        self.pts.append(new_pt)
        self.colors.append(color_best)
        return color_best

def unstash_seed(py_state, np_state):
    random.setstate(py_state)
    np.random.set_state(np_state)

def distinct_colors(n):
    #cc = ColorChooser(attempts = 10, init_colors = [red, green, blue, yellow, purple, cyan], init_pts = [(0, 0)]*6)
    cc = ColorChooser(attempts = 100, init_colors = [red, green, blue, yellow, purple, cyan], init_pts = [(0, 0)]*6)
    do_with_seed(lambda : [cc.choose((0,0)) for x in range(n)])
    return cc.colors[:n]

def make(w, h, fill = (0,0,0)):
    return np.uint8(np.tile([[fill]], (h, w, 1)))

def rgb_from_gray(img, copy = True, remove_alpha = True):
    if img.ndim == 3 and img.shape[2] == 3:
        return img.copy() if copy else img
    elif img.ndim == 3 and img.shape[2] == 4:
        return (img.copy() if copy else img)[..., :3]
    elif img.ndim == 3 and img.shape[2] == 1:
        return np.tile(img, (1,1,3))
    elif img.ndim == 2:
        return np.tile(img[:,:,np.newaxis], (1,1,3))
    else:
        raise RuntimeError('Cannot convert to rgb. Shape: ' + str(img.shape))

def hstack_ims(ims, bg_color = (0, 0, 0)):
    max_h = max([im.shape[0] for im in ims])
    result = []
    for im in ims:
        #frame = np.zeros((max_h, im.shape[1], 3))
        frame = make(im.shape[1], max_h, bg_color)
        frame[:im.shape[0],:im.shape[1]] = rgb_from_gray(im)
        result.append(frame)
    return np.hstack(result)

def gen_ranked_prob_map(prob_map): 
    prob_ranked = torch.zeros_like(prob_map)
    _, index = torch.topk(prob_map, len(prob_map), largest=False)
    prob_ranked[index] = torch.arange(len(prob_map)).float().cuda()
    prob_ranked = prob_ranked.float() / torch.max(prob_ranked)
    return prob_ranked

def get_topk_patch_mask(prob_map): 
    # _, index = 
    pass

def load_img(frame_path): 
    image = Image.open(frame_path).convert('RGB')
    image = image.resize((256, 256), resample=PIL.Image.BILINEAR)
    image = np.array(image)

    img_h = image.shape[0]
    img_w = image.shape[1]

    return image, img_h, img_w

def plt_subp_show_img(fig, img, cols, rows, subp_index, interpolation='bilinear', aspect='auto'): 
    fig.add_subplot(rows, cols, subp_index)
    plt.cla()
    plt.axis('off')
    plt.imshow(img, interpolation=interpolation, aspect=aspect)
    return fig

 
    