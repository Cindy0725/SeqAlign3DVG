import math
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextSlotter(nn.Module):
    def __init__(self, text_dim: int, n_slots: int = 1, pad_id: int = 0):
        super().__init__()
        self.n_slots = n_slots
        self.pad_id = pad_id

        self.slot_queries = nn.Parameter(torch.randn(n_slots, text_dim) * 0.02)
        self.ln_tokens = nn.LayerNorm(text_dim)
        self.ln_slots = nn.LayerNorm(text_dim)
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2),
            nn.GELU(),
            nn.Linear(text_dim * 2, text_dim),
        )

    def forward(self, token_embeds: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        # token_embeds: (B,T,D), token_ids: (B,T)
        B, T, D = token_embeds.shape
        x = self.ln_tokens(token_embeds)

        q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)  # (B,K,D)
        logits = torch.einsum('bkd,btd->bkt', q, x) / math.sqrt(D)  # (B,K,T)

        pad_mask = (token_ids == self.pad_id)  # (B,T)
        logits = logits.masked_fill(pad_mask.unsqueeze(1), -1e9)

        attn = logits.softmax(dim=-1)
        slots = torch.einsum('bkt,btd->bkd', attn, x)

        slots = self.ln_slots(slots)
        slots = slots + self.mlp(slots)
        return slots


class TextSlotterV2(nn.Module):
    """
    改进点：
    1) mask PAD + (可选) SOT/EOT
    2) slot query 不再只靠 learnable query，而是 learnable + pooled_text 条件化（更稳定）
    3) 可选注入 voxel_ctx，让“选哪些token更重要”依赖场景（创新点）
    4) 返回 attn，方便 debug/可视化/做 token top-k
    """
    def __init__(
        self,
        text_dim: int,
        n_slots: int = 1,
        pad_id: int = 0,
        sot_id: int = 49406,
        eot_id: int = 49407,
        mask_special: bool = True,
        use_voxel_ctx: bool = True,
        voxel_ctx_dim: Optional[int] = None,  # 若 use_voxel_ctx=True，则需要给 voxel_ctx_dim=text_dim
    ):
        super().__init__()
        self.n_slots = n_slots
        self.pad_id = pad_id
        self.sot_id = sot_id
        self.eot_id = eot_id
        self.mask_special = mask_special
        self.use_voxel_ctx = use_voxel_ctx

        # learnable query（保留你的原始思想）
        self.slot_queries = nn.Parameter(torch.randn(n_slots, text_dim) * 0.02)

        # pooled_text -> query（让 query 随句子变化）
        in_q = text_dim
        if use_voxel_ctx:
            assert voxel_ctx_dim == text_dim, "use_voxel_ctx=True 时建议 voxel_ctx_dim=text_dim"
            in_q = text_dim * 2
        self.q_from_cond = nn.Linear(in_q, n_slots * text_dim, bias=True)

        # learnable 与 conditional query 的混合权重（sigmoid后在[0,1]）
        self.query_mix = nn.Parameter(torch.tensor(0.0))  # 初始约 0.5

        self.ln_tokens = nn.LayerNorm(text_dim)
        self.ln_slots = nn.LayerNorm(text_dim)
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2),
            nn.GELU(),
            nn.Linear(text_dim * 2, text_dim),
        )

    def forward(
        self,
        token_embeds: torch.Tensor,        # (B,T,D)
        token_ids: torch.Tensor,           # (B,T)
        pooled_text: torch.Tensor,         # (B,D)
        voxel_ctx: Optional[torch.Tensor] = None,  # (B,D) 可选
        return_attn: bool = False,
    ):
        B, T, D = token_embeds.shape
        x = self.ln_tokens(token_embeds)

        if self.use_voxel_ctx:
            if voxel_ctx is None:
                raise ValueError("use_voxel_ctx=True 但 voxel_ctx=None")
            cond = torch.cat([pooled_text, voxel_ctx], dim=-1)  # (B,2D)
        else:
            cond = pooled_text                                  # (B,D)

        q_cond = self.q_from_cond(cond).view(B, self.n_slots, D)  # (B,K,D)
        q_learn = self.slot_queries.unsqueeze(0).expand(B, -1, -1)

        alpha = torch.sigmoid(self.query_mix)  # scalar
        q = alpha * q_learn + (1.0 - alpha) * q_cond

        logits = torch.einsum("bkd,btd->bkt", q, x) / math.sqrt(D)  # (B,K,T)

        # mask PAD 以及 SOT/EOT（建议）
        pad_mask = (token_ids == self.pad_id)
        if self.mask_special:
            pad_mask = pad_mask | (token_ids == self.sot_id) | (token_ids == self.eot_id)

        logits = logits.masked_fill(pad_mask.unsqueeze(1), -1e9)

        attn = logits.softmax(dim=-1)                 # (B,K,T)
        slots = torch.einsum("bkt,btd->bkd", attn, x) # (B,K,D)

        slots = self.ln_slots(slots)
        slots = slots + self.mlp(slots)

        if return_attn:
            return slots, attn
        return slots

class ProgressiveTextVoxelFusion(nn.Module):
    """
    现在支持：
      - baseline (global pooled + concat + conv) 作为保底通路
      - level1: token slot -> target prior (logits/map) -> residual gating

    fusion_level:
      0: 仅 baseline
      1: baseline + level1 residual (推荐你现在用这个)
    """
    def __init__(
        self,
        voxel_dim: int,
        text_dim: int,
        n_voxels: Tuple[int, int, int] = (40, 40, 16),
        voxel_size: Tuple[float, float, float] = (0.16, 0.16, 0.16),
        fusion_level: int = 1,
        baseline_method: str = "concat",   # 这里只保留 concat（跟你 baseline 一致）
        sim_normalize: bool = True,
        use_baseline: bool = True,         # 新增：level1 时也启用 baseline
    ):
        super().__init__()
        assert fusion_level in (0, 1, 2)
        assert baseline_method in ("concat",)

        self.voxel_dim = voxel_dim
        self.text_dim = text_dim
        self.fusion_level = fusion_level
        self.baseline_method = baseline_method
        self.sim_normalize = sim_normalize
        self.use_baseline = use_baseline

        # CLIP-style temperature
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))  # ~2.659

        # baseline concat+conv（不论 level0/1，只要 use_baseline=True 就会创建）
        # if self.use_baseline:
        self.baseline_fusion = nn.Sequential(
                nn.Conv3d(voxel_dim + text_dim, voxel_dim, kernel_size=1, bias=False),
                nn.BatchNorm3d(voxel_dim),
                nn.ReLU(inplace=True),
            )

        self.prior_fuse = nn.Sequential(
            nn.Conv3d(voxel_dim + 1, voxel_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(voxel_dim),
            nn.ReLU(inplace=True),
        )
        self.prior_fuse_v2 = nn.Sequential(
            nn.Conv3d(voxel_dim * 2 + 1, voxel_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(voxel_dim),
            nn.ReLU(inplace=True),
        )
        # I want to add a mlp to slot_tgt to control the gate more flexibly
        self.gate_mlp = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2),
            nn.GELU(),
            nn.Linear(text_dim * 2, 1),
            nn.Sigmoid(),
        )

        # level1 modules
        if fusion_level >= 1:
            self.ln_text = nn.LayerNorm(text_dim)
            self.slotter = TextSlotter(text_dim=text_dim, n_slots=1, pad_id=0)  # level1 用 1 个 slot 最稳
            self.voxel_to_text = nn.Conv3d(voxel_dim, text_dim, kernel_size=1, bias=False)
            self.gate = nn.Parameter(torch.tensor(0.0))  # 从 0 开始更稳
            
            # 2026/01/17凌晨应用一种新的 slotter，用 pooled_text 和 voxel_ctx 条件化 query
            use_voxel_ctx_for_slot = True
            self.use_voxel_ctx_for_slot = use_voxel_ctx_for_slot
            self.slotter_v2 = TextSlotterV2(
                text_dim=text_dim,
                n_slots=1,
                pad_id=0,
                sot_id=49406,
                eot_id=49407,
                mask_special=True,
                use_voxel_ctx=use_voxel_ctx_for_slot,
                voxel_ctx_dim=text_dim if use_voxel_ctx_for_slot else None,
            )
            self.voxel_ctx_proj = nn.Linear(voxel_dim, text_dim, bias=False)
        if fusion_level >= 2:
            self.topk_vox = 512        # 建议先 512；X*Y*Z=25600，512≈2%
            self.attn_heads = 8

            # voxel(C) -> query(D)
            self.q_proj = nn.Linear(voxel_dim, text_dim, bias=False)
            self.ln_q = nn.LayerNorm(text_dim)

            # 3D positional embedding： (x,y,z) -> D
            self.pos_mlp = nn.Sequential(
                nn.Linear(3, text_dim),
                nn.GELU(),
                nn.Linear(text_dim, text_dim),
            )

            # sparse cross-attention：Q=voxels, K/V=tokens
            self.sparse_attn = nn.MultiheadAttention(
                embed_dim=text_dim,
                num_heads=self.attn_heads,
                batch_first=True,
            )

            # attn out(D) -> voxel(C)
            self.out_proj = nn.Linear(text_dim, voxel_dim, bias=False)

            # 稳定训练：让 level2 初始接近 identity
            self.gate2 = nn.Parameter(torch.tensor(-2.0))  # sigmoid≈0.12

    def _baseline_forward(self, voxel_feat: torch.Tensor, pooled_text: torch.Tensor) -> torch.Tensor:
        B, C, X, Y, Z = voxel_feat.shape
        text_exp = pooled_text.view(B, -1, 1, 1, 1).expand(-1, -1, X, Y, Z)
        return self.baseline_fusion(torch.cat([voxel_feat, text_exp], dim=1))

    def _slot_to_logits(
        self,
        voxel_feat: torch.Tensor,   # (B,C,X,Y,Z)
        slot_vec: torch.Tensor,     # (B,D)
    ) -> torch.Tensor:
        B, _, X, Y, Z = voxel_feat.shape

        vtxt = self.voxel_to_text(voxel_feat)              # (B,D,X,Y,Z)
        vflat = vtxt.flatten(2).transpose(1, 2)            # (B,N,D)

        if self.sim_normalize:
            vflat = F.normalize(vflat, dim=-1)
            slot_vec = F.normalize(slot_vec, dim=-1)

        sim = (vflat * slot_vec.unsqueeze(1)).sum(dim=-1)  # (B,N), cosine in [-1,1]
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = (sim * scale).view(B, 1, X, Y, Z)
        return logits

    def _voxel_global_ctx(self, voxel_feat: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        voxel_feat: (B,C,X,Y,Z)
        valid_mask: (B,1,X,Y,Z) bool
        return: (B,C) -> project到 text_dim 在 slotter 用
        """
        B, C, X, Y, Z = voxel_feat.shape
        if valid_mask is None:
            ctx = voxel_feat.mean(dim=(2,3,4))  # (B,C)
        else:
            m = valid_mask.float()
            denom = m.sum(dim=(2,3,4)).clamp(min=1.0)  # (B,1)
            ctx = (voxel_feat * m).sum(dim=(2,3,4)) / denom  # (B,C)
        return ctx

    def forward(
        self,
        voxel_feat: torch.Tensor,                    # (B,C,X,Y,Z)
        pooled_text: Optional[torch.Tensor] = None,  # (B,D) baseline 需要
        token_embeds: Optional[torch.Tensor] = None, # (B,T,D) level1 需要
        token_ids: Optional[torch.Tensor] = None,    # (B,T)
        valid_mask: Optional[torch.Tensor] = None,   # (B,1,X,Y,Z) bool
        return_debug: bool = False,
    ):
        debug: Dict[str, Any] = {}

        x = voxel_feat

        # baseline 保底
        if self.use_baseline:
            if pooled_text is None:
                raise ValueError("use_baseline=True 时必须提供 pooled_text (B,D)")
            x = self._baseline_forward(x, pooled_text)

        if self.fusion_level == 0:
            if valid_mask is not None:
                x = x * valid_mask.float()
            return (x, debug) if return_debug else x

        # level1 residual prior
        if token_embeds is None or token_ids is None:
            raise ValueError("fusion_level=1 需要 token_embeds/token_ids")

        token_embeds = self.ln_text(token_embeds)
        # slots = self.slotter(token_embeds, token_ids)  # (B,1,D)
        # slot_tgt = slots[:, 0]

        # 2026/01/17凌晨应用一种新的 slotter，用 pooled_text 和 voxel_ctx 条件化 query
        voxel_ctx_c = self._voxel_global_ctx(voxel_feat, valid_mask)   # (B,C)
        voxel_ctx = self.voxel_ctx_proj(voxel_ctx_c)                    # (B,D)
        slots = self.slotter_v2(
            token_embeds=token_embeds,
            token_ids=token_ids,
            pooled_text=pooled_text,
            voxel_ctx=voxel_ctx if self.use_voxel_ctx_for_slot else None,
            return_attn=False,
        )
        slot_tgt = slots[:, 0]     
        
        # 2026/01/16 新增：level1 时用slot_tgt做expansion+concat+conv. 
        # the following line commented for 2026/04/09 experiments
        # x = self._baseline_forward(x, slot_tgt)

        target_logits = self._slot_to_logits(voxel_feat, slot_tgt)  # 注意：用原 voxel_feat 更“纯”
        target_map = torch.sigmoid(target_logits)

        if valid_mask is not None:
            target_map = target_map * valid_mask.float()

        # out = x + self.gate * target_map * x #20260111_192156.log & baseline_prior_multiply_best_Acc@0.25_epoch_9.pth & self.gate init 0.0
        
        # 先简单 concat 融合 20260112_105450.log & self.gate init 0.0, 效果比 baseline_prior_multiply_best_Acc@0.25_epoch_9.pth 略低
        # delta = self.prior_fuse(torch.cat([x, target_map], dim=1))
        # out = x + self.gate * delta

        # 另一种更复杂的融合方式，效果最好。
        inp = torch.cat([x, target_map, target_map * x], dim=1)  # (B, 2C+1, X,Y,Z)
        delta = self.prior_fuse_v2(inp)
        g = torch.sigmoid(self.gate)
        out = x + g * delta
        
        # 直接用 mlp 预测 gate，更灵活一些，效果比不用self.gate_mlp略差。
        # inp = torch.cat([x, target_map, target_map * x], dim=1)  # (B, 2C+1, X,Y,Z)
        # delta = self.prior_fuse_v2(inp)
        # g = self.gate_mlp(slot_tgt).view(-1, 1, 1, 1, 1)
        # out = x + g * delta
        
        # 在out = x + self.gate * target_map * x的基础上，用mlp预测gate 效果并不好。
        # g = torch.sigmoid(self.gate)
        # out = x + g * target_map * x

        if valid_mask is not None:
            out = out * valid_mask.float()
        
        if self.fusion_level >= 2:
            B, C, X, Y, Z = out.shape
            N = X * Y * Z

            # 用 logits 排序比用 sigmoid 后的 map 更不容易饱和
            score = target_logits.squeeze(1)  # (B,X,Y,Z)

            if valid_mask is not None:
                score = score.masked_fill(~valid_mask.squeeze(1), -1e9)

            score_flat = score.flatten(1)     # (B,N)
            K = min(self.topk_vox, N)
            topk_idx = score_flat.topk(K, dim=1).indices   # (B,K)
            out_flat = out.flatten(2).transpose(1, 2)  # (B,N,C)

            q_vox = out_flat.gather(
                1, topk_idx.unsqueeze(-1).expand(-1, -1, C)
            )  # (B,K,C)

            q = self.q_proj(q_vox)  # (B,K,D)
            x_idx = topk_idx // (Y * Z)
            yz = topk_idx % (Y * Z)
            y_idx = yz // Z
            z_idx = yz % Z

            coords = torch.stack([x_idx, y_idx, z_idx], dim=-1).to(dtype=q.dtype)  # (B,K,3)

            # normalize to [-1,1]，避免跟体素数量绑定
            coords[..., 0] = coords[..., 0] / max(X - 1, 1) * 2 - 1
            coords[..., 1] = coords[..., 1] / max(Y - 1, 1) * 2 - 1
            coords[..., 2] = coords[..., 2] / max(Z - 1, 1) * 2 - 1

            q = self.ln_q(q + self.pos_mlp(coords))  # (B,K,D)
            key_padding_mask = (
                (token_ids == 0) |
                (token_ids == 49406) |
                (token_ids == 49407)
            )  # (B,T) True=ignore

            kv = token_embeds  # 你上面已经 ln_text(token_embeds) 过更好

            attn_out, attn_w = self.sparse_attn(
                q, kv, kv,
                key_padding_mask=key_padding_mask,
                need_weights=return_debug
            )  # attn_out: (B,K,D)
            delta_q = self.out_proj(attn_out)  # (B,K,C)
            g2 = torch.sigmoid(self.gate2)

            q_vox_new = q_vox + g2 * delta_q

            out_flat = out_flat.scatter(
                1,
                topk_idx.unsqueeze(-1).expand(-1, -1, C),
                q_vox_new
            )

            out = out_flat.transpose(1, 2).view(B, C, X, Y, Z)

            if valid_mask is not None:
                out = out * valid_mask.float()

        if return_debug:
            debug["target_logits"] = target_logits
            debug["target_map"] = target_map
            debug["gate"] = self.gate.detach().clone()
            debug["logit_scale"] = self.logit_scale.exp().detach().clone()
            return out, debug
        return out