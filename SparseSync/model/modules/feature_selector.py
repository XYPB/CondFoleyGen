import einops
import math
import torch
import sys
import logging

sys.path.insert(0, '.')  # nopep8
from utils.utils import get_obj_from_str, instantiate_from_config
from model.modules.transformer import Block, Config, SelfAttention

logger = logging.getLogger(f'main.{__name__}')

class SparseSync(torch.nn.Module):

    def __init__(
        self, vis_pos_emb_module, aud_pos_emb_module, num_offset_cls,
        visual_block_shape, audio_block_shape, pre_norm_cfg, v_selector_cfg, a_selector_cfg, global_transformer_cfg,
        n_layer=12, n_head=8, n_embd=256, tok_pdrop=0., embd_pdrop=0., resid_pdrop=0., attn_pdrop=0.,
        n_unmasked=0
    ):
        super().__init__()
        self.config = Config(
            num_offset_cls=num_offset_cls, audio_block_shape=audio_block_shape,
            visual_block_shape=visual_block_shape, embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop, tok_pdrop=tok_pdrop, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_unmasked=n_unmasked
        )
        super().__init__()
        # input norm
        self.pre_lnorm_vis = instantiate_from_config(pre_norm_cfg)
        self.pre_lnorm_aud = instantiate_from_config(pre_norm_cfg)
        # pos embeddings on features
        self.vis_pos_emb = instantiate_from_config(vis_pos_emb_module)
        self.aud_pos_emb = instantiate_from_config(aud_pos_emb_module)
        # selector transformers
        self.v_selector = instantiate_from_config(v_selector_cfg)  # FeatureSelectorTransformer
        self.a_selector = instantiate_from_config(a_selector_cfg)  # FeatureSelectorTransformer
        # aggregation transformer
        self.global_transformer = instantiate_from_config(global_transformer_cfg)  # GlobalTransformer
        # head
        if isinstance(self.global_transformer, GlobalMLP):
            N_av = a_selector_cfg.params.num_selectors + v_selector_cfg.params.num_selectors
            self.off_head = torch.nn.Linear(N_av*self.config.n_embd, self.config.num_offset_cls)
        else:
            self.off_head = torch.nn.Linear(self.config.n_embd, self.config.num_offset_cls, bias=False)

        self.apply(init_weights)

    def forward(self, vis: torch.Tensor, aud: torch.Tensor):
        B, Tv, Dv, H, W = vis.shape
        B, Da, F, Ta = aud.shape
        assert Da == Dv, f'Please define a bridge or fix {Da} vs {Dv}'
        # (B, Tv, H, W, Dv), (B, F, Ta, Da) <-
        vis, aud = vis.permute(0, 1, 3, 4, 2).contiguous(), aud.permute(0, 2, 3, 1).contiguous()
        # making sure that both embeddings are normalized to the same base (can be configured as identity tho)
        vis, aud = self.pre_lnorm_vis(vis), self.pre_lnorm_aud(aud)
        # apply individual pos embeddings
        vis, aud = self.vis_pos_emb(vis), self.aud_pos_emb(aud)
        # narrow down the dimension with selectors
        vis, aud = self.v_selector(vis), self.a_selector(aud)
        # aggregate infomation
        off_logits = self.global_transformer(vis, aud)
        # picking the first token which correspond to the `off_tok` as the prediction token
        off_logits = self.off_head(off_logits[:, 0, :])
        return off_logits


class FeatureSelectorTransformer(torch.nn.Module):

    def __init__(self, num_selectors, n_layer, n_head, n_embd, embd_pdrop, resid_pdrop, attn_pdrop,
                 pos_emb_cfg=None) -> None:
        super().__init__()
        config = Config(embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                        n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        self.num_selectors = num_selectors
        self.selectors = torch.nn.Parameter(torch.randn(num_selectors, n_embd))

        self.pos_emb_cfg = pos_emb_cfg
        if self.pos_emb_cfg is not None:
            self.selectors_pos_emb = instantiate_from_config(pos_emb_cfg)
            logger.info(f'Positional embedding module for selectors: {pos_emb_cfg.target.split(".")[-1]}')

        self.selectors_drop = torch.nn.Dropout(config.embd_pdrop)
        self.context_drop = torch.nn.Dropout(config.embd_pdrop)
        # stack of decoder-like layers
        self.blocks = torch.nn.Sequential(*[CrossBlock(config) for _ in range(config.n_layer)])

        self.apply(init_weights)
        logger.info(f'Selector has {num_selectors} selectors')

    def forward(self, context):
        '''takes in a set of features (context): (B, Tv, H, W, Dv) or (B, F, Ta, Da)'''
        B, D = context.shape[0], context.shape[-1]
        # (B, T, D) <= flattening the context
        context = context.view(B, -1, D)
        # broadcast the trainable selectors
        selectors = einops.repeat(self.selectors, 's d -> b s d', b=B)
        # maybe add pos emb
        selectors = selectors if self.pos_emb_cfg is None else self.selectors_pos_emb(selectors)
        # dropout
        selectors, context = self.selectors_drop(selectors), self.context_drop(context)
        # apply a stack of decoder-like blocks
        selectors, context = self.blocks((selectors, context))

        # (B, num_selectors, D)
        return selectors


class GlobalTransformer(torch.nn.Module):

    def __init__(self, tok_pdrop, embd_pdrop, resid_pdrop, attn_pdrop, n_layer, n_head, n_embd) -> None:
        super().__init__()
        self.config = Config(embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                             n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        # input norm
        self.vis_in_lnorm = torch.nn.LayerNorm(self.config.n_embd)
        self.aud_in_lnorm = torch.nn.LayerNorm(self.config.n_embd)
        # aux tokens
        self.OFF_tok = torch.nn.Parameter(torch.randn(1, 1, n_embd))
        self.MOD_tok = torch.nn.Parameter(torch.randn(1, 1, n_embd))
        # whole token dropout
        self.tok_drop_vis = torch.nn.Dropout2d(tok_pdrop)
        self.tok_drop_aud = torch.nn.Dropout2d(tok_pdrop)
        # the stem
        self.drop = torch.nn.Dropout(embd_pdrop)
        self.blocks = torch.nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        # pre-output norm
        self.ln_f = torch.nn.LayerNorm(self.config.n_embd)

        self.apply(init_weights)

    def forward(self, vis_selectors: torch.Tensor, aud_selectors: torch.Tensor, targets=None):
        B, Sv, D = vis_selectors.shape
        B, Sa, D = aud_selectors.shape
        # broadcasting special tokens to the batch size
        off_tok = einops.repeat(self.OFF_tok, '1 1 d -> b 1 d', b=B)
        mod_tok = einops.repeat(self.MOD_tok, '1 1 d -> b 1 d', b=B)
        # norm
        vis_selectors, aud_selectors = self.vis_in_lnorm(vis_selectors), self.aud_in_lnorm(aud_selectors)
        # maybe whole token dropout
        vis_selectors, aud_selectors = self.tok_drop_vis(vis_selectors), self.tok_drop_aud(aud_selectors)
        # (B, 2+Sv+1+Sa, D)
        x = torch.cat((off_tok, vis_selectors, mod_tok, aud_selectors), dim=1)
        # dropout -> stem -> norm
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return x


class GlobalMLP(torch.nn.Module):

    def __init__(self, tok_pdrop, embd_pdrop, resid_pdrop, attn_pdrop, n_layer, n_head, n_embd) -> None:
        super().__init__()
        self.config = Config(embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                             n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        # input norm
        self.vis_in_lnorm = torch.nn.LayerNorm(self.config.n_embd)
        self.aud_in_lnorm = torch.nn.LayerNorm(self.config.n_embd)
        # whole token dropout
        self.tok_drop_vis = torch.nn.Dropout2d(tok_pdrop)
        self.tok_drop_aud = torch.nn.Dropout2d(tok_pdrop)
        # the stem
        self.drop = torch.nn.Dropout(embd_pdrop)

        self.apply(init_weights)

    def forward(self, vis_selectors: torch.Tensor, aud_selectors: torch.Tensor, targets=None):
        B, Sv, D = vis_selectors.shape
        B, Sa, D = aud_selectors.shape
        # norm
        vis_selectors, aud_selectors = self.vis_in_lnorm(vis_selectors), self.aud_in_lnorm(aud_selectors)
        # maybe whole token dropout
        vis_selectors, aud_selectors = self.tok_drop_vis(vis_selectors), self.tok_drop_aud(aud_selectors)
        # cat and flatten
        x = torch.cat([vis_selectors, aud_selectors], dim=1)
        x = x.view(B, (Sv+Sa)*D)
        # dropout -> stem
        x = self.drop(x)
        # adding sequence dimension just for compatibility with the parent class
        x = x.unsqueeze(1)
        return x


def init_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class PositionEmbeddingSelectors(torch.nn.Module):

    def __init__(self, max_pos, n_embd) -> None:
        super().__init__()
        self.max_t = max_pos
        self.n_embd = n_embd
        self.time_embed = torch.nn.Embedding(self.max_t, self.n_embd)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.time_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): a batch of selectors (B, max_t, d)
        Returns:
            torch.Tensor: x + pos
        '''
        return x + self.make_pos_emb(x)

    def make_pos_emb(self, x):
        B, S, d = x.shape
        t_i = torch.arange(S, device=x.device)
        # (S, D)
        pos = self.time_embed(t_i)
        # (B, S, d) <- (1, S, d) <- (S, d)
        pos = pos.view(1, S, d).repeat(B, 1, 1)
        return pos


class DoNothing(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        return x


class CrossBlock(torch.nn.Module):
    '''Similar to transformer.Block (which is an Encoder layer), but this one has cross attention in
    the middle (as in decoder layers of the original transformer)'''

    def __init__(self, config):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(config.n_embd)
        self.ln2 = torch.nn.LayerNorm(config.n_embd)
        self.ln3 = torch.nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.xattn = CrossAttention(config)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.n_embd, 4 * config.n_embd),
            torch.nn.GELU(),  # nice
            torch.nn.Linear(4 * config.n_embd, config.n_embd),
            torch.nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: tuple) -> tuple:
        x, context = x
        x = x + self.attn(self.ln1(x))
        x = x + self.xattn(self.ln2(x), context)
        x = x + self.mlp(self.ln3(x))
        return x, context


class CrossAttention(SelfAttention):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, x, context):
        B, Tx, C = x.size()
        B, Tc, C = context.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, Tx, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(context).view(B, Tc, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(context).view(B, Tc, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = self.attn_drop(att) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, Tx, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        return y


if __name__ == '__main__':
    v_feats, a_feats = torch.rand(3, 16, 512, 7, 7), torch.rand(3, 512, 9, 27)
    # num_selectors, n_layer, n_hear, dout_p = 64, 6, 8, 0.1

    # (B, tv, d, h, w), (B, d, f, ta) = v_feats.shape, a_feats.shape

    # v_selector = FeatureSelectorTransformer(
    #     num_selectors=64,
    #     context_pos_emb_module={
    #         'target': 'model.modules.transformer.PositionEmbeddingLearnedVisual',
    #         'params': {'block_shape': [tv, h, w], 'n_embd': d}},
    #     context_norm={'target': 'torch.nn.Identity', 'params': {'normalized_shape': d}},
    #     n_layer=6, n_head=8, n_embd=d, embd_pdrop=dout_p, resid_pdrop=dout_p, attn_pdrop=dout_p,
    # )
    # a_selector = FeatureSelectorTransformer(
    #     num_selectors=64,
    #     context_pos_emb_module={
    #         'target': 'model.modules.transformer.PositionEmbeddingLearnedAudio',
    #         'params': {'block_shape': [f, ta], 'n_embd': d}},
    #     context_norm={'target': 'torch.nn.Identity', 'params': {'normalized_shape': d}},
    #     n_layer=6, n_head=8, n_embd=d, embd_pdrop=dout_p, resid_pdrop=dout_p, attn_pdrop=dout_p,
    # )
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('./configs/av_sync.yaml')
    transformer = instantiate_from_config(cfg.model.params.transformer)
    out = transformer(v_feats, a_feats)

    print(out.shape)
