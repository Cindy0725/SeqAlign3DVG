from typing import Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .plvf import TextSlotterV2

def _make_gn(channels: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class TemporalLanguageGuidedVoxelMemory(nn.Module):
    """
    Conservative Query-Guided Multi-View Voxel Memory

    核心设计目标：
    1) memory 只做 query-guided view selection + conservative refinement
    2) 不在 memory 里做太强的 text semantic rewriting
    3) 写入 support 更尖、更依赖 selected-view consensus
    4) recurrent update 按 relevance order，不按 camera index 伪时间顺序
    """

    def __init__(
        self,
        voxel_dim: int,
        text_dim: int,
        hidden_dim: int = 128,
        top_m: int = 4,
        score_temperature: float = 1.0,
        sim_normalize: bool = True,
        prior_gamma: float = 1.5,
        memory_update_order: str = "relevance",
        random_order_seed: int = -1,
        view_scorer_ablation: str = "none",
    ):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim 最好设成偶数"

        self.voxel_dim = voxel_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.top_m = top_m
        self.score_temperature = max(float(score_temperature), 1e-6)
        self.sim_normalize = sim_normalize
        self.prior_gamma = float(prior_gamma)

        self.memory_update_order = self._normalize_memory_update_order(memory_update_order)
        self.random_order_seed = int(random_order_seed)
        self.view_scorer_ablation = self._normalize_view_scorer_ablation(view_scorer_ablation)

        # ---- query-aware target slot ----
        self.slotter = TextSlotterV2(
            text_dim=text_dim,
            n_slots=1,
            pad_id=0,
            sot_id=49406,
            eot_id=49407,
            mask_special=True,
            use_voxel_ctx=True,
            voxel_ctx_dim=text_dim,
        )
        self.voxel_ctx_proj = nn.Linear(voxel_dim, text_dim, bias=False)
        self.ln_text = nn.LayerNorm(text_dim)

        # ---- per-view target prior ----
        self.voxel_to_text = nn.Conv3d(voxel_dim, text_dim, kernel_size=1, bias=False)
        self.text_to_voxel = nn.Linear(text_dim, voxel_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))  # CLIP-style

        # ---- view scoring ----
        self.view_ln = nn.LayerNorm(voxel_dim)
        self.view_proj = nn.Linear(voxel_dim, hidden_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, hidden_dim, bias=False)

        # cue: [view_pos, 1-view_pos, valid_ratio, prior_focus]
        self.cue_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.view_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # ---- lightweight memory update ----
        self.mem_reduce = nn.Conv3d(voxel_dim, hidden_dim // 2, kernel_size=1, bias=False)
        self.cur_reduce = nn.Conv3d(voxel_dim, hidden_dim // 2, kernel_size=1, bias=False)

        # gate input:
        # mem_r(H/2) + cur_r(H/2)
        # + prior(1) + diff(1) + cur_valid(1) + obs_valid(1) + mem_conf(1)
        # + view_weight(1) + new_ratio(1) + overlap_ratio(1) + stage(1)
        gate_in_ch = hidden_dim + 9

        self.gate_net = nn.Sequential(
            nn.Conv3d(gate_in_ch, hidden_dim, kernel_size=1, bias=False),
            _make_gn(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, 1, kernel_size=1, bias=True),
        )

        self.out_norm = _make_gn(voxel_dim)

        # 初始更保守
        self.text_write_scale = nn.Parameter(torch.tensor(-2.0))  # sigmoid≈0.12
        self.blend_alpha = nn.Parameter(torch.tensor(-1.0))       # sigmoid≈0.27

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
        dim: int = 0,
    ) -> torch.Tensor:
        logits = logits / temperature
        logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(logits, dim=dim)
        probs = probs * mask.float()
        probs = probs / probs.sum(dim=dim, keepdim=True).clamp(min=1e-6)
        return probs

    @staticmethod
    def _masked_pool_single(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        x: (1,C,X,Y,Z)
        w: (1,1,X,Y,Z)
        out: (1,C)
        """
        w = w.to(dtype=x.dtype)
        denom = w.sum(dim=(2, 3, 4)).clamp(min=1e-6)
        pooled = (x * w).sum(dim=(2, 3, 4)) / denom
        return pooled

    @staticmethod
    def _global_voxel_ctx(
        view_volumes: torch.Tensor,
        view_valids: torch.Tensor,
    ) -> torch.Tensor:
        """
        view_volumes: (V,C,X,Y,Z)
        view_valids : (V,1,X,Y,Z)
        return: (1,C)
        """
        m = view_valids.float()
        denom = m.sum(dim=(0, 2, 3, 4)).clamp(min=1.0)
        ctx = (view_volumes * m).sum(dim=(0, 2, 3, 4)) / denom
        return ctx.unsqueeze(0)

    @staticmethod
    def _masked_topk_mean_single(x: torch.Tensor, mask: torch.Tensor, k: int = 64) -> torch.Tensor:
        """
        x:    (1,1,X,Y,Z)
        mask: (1,1,X,Y,Z)
        return: scalar
        """
        vals = x[mask.bool()]
        if vals.numel() == 0:
            return x.new_zeros(())
        kk = min(int(k), int(vals.numel()))
        return vals.topk(kk).values.mean()

    @staticmethod
    def _normalize_memory_update_order(order: str) -> str:
        """
        Normalize aliases used in configs / cfg-options.

        relevance:              original ROVM, write Top-M views by descending relevance score.
        chronological:          write the same Top-M views by increasing frame index.
        reverse_chronological:  write the same Top-M views by decreasing frame index.
        random:                 write the same Top-M views in a random order.
        """
        order = str(order).lower().strip()
        aliases = {
            "rel": "relevance",
            "score": "relevance",
            "score_order": "relevance",
            "relevance_order": "relevance",
            "relevance_ordered": "relevance",
            "relevance": "relevance",
            "chronological": "chronological",
            "chrono": "chronological",
            "time": "chronological",
            "temporal": "chronological",
            "reverse": "reverse_chronological",
            "reverse_chrono": "reverse_chronological",
            "reverse_chronological": "reverse_chronological",
            "rev": "reverse_chronological",
            "random": "random",
            "shuffle": "random",
            "shuffled": "random",
        }
        if order not in aliases:
            raise ValueError(
                f"Unsupported memory_update_order={order!r}. "
                "Use one of: relevance, chronological, reverse_chronological, random."
            )
        return aliases[order]

    @staticmethod
    def _normalize_view_scorer_ablation(ablation: str) -> str:
        """
        Normalize cue-ablation names.

        The scorer cue vector is [view_pos, 1-view_pos, valid_ratio, prior_focus].
        - wo_position: zero both positional entries.
        - wo_visibility: zero valid_ratio.
        - wo_focus: zero prior_focus.
        - wo_all_scalar_cues: zero all four scalar cue entries.
        """
        ablation = str(ablation).lower().strip()
        aliases = {
            "": "none",
            "none": "none",
            "full": "none",
            "default": "none",
            "no_ablation": "none",
            "wo_visibility": "wo_visibility",
            "w/o_visibility": "wo_visibility",
            "without_visibility": "wo_visibility",
            "no_visibility": "wo_visibility",
            "wo_visibility_cue": "wo_visibility",
            "wo_focus": "wo_focus",
            "w/o_focus": "wo_focus",
            "without_focus": "wo_focus",
            "no_focus": "wo_focus",
            "wo_focus_cue": "wo_focus",
            "wo_position": "wo_position",
            "w/o_position": "wo_position",
            "without_position": "wo_position",
            "no_position": "wo_position",
            "wo_position_cue": "wo_position",
            "wo_all_scalar_cues": "wo_all_scalar_cues",
            "w/o_all_scalar_cues": "wo_all_scalar_cues",
            "without_all_scalar_cues": "wo_all_scalar_cues",
            "no_all_scalar_cues": "wo_all_scalar_cues",
            "wo_scalar_cues": "wo_all_scalar_cues",
            "no_scalar_cues": "wo_all_scalar_cues",
            "wo_all_cues": "wo_all_scalar_cues",
            "all_scalar_cues": "wo_all_scalar_cues",
        }
        if ablation not in aliases:
            raise ValueError(
                f"Unsupported view_scorer_ablation={ablation!r}. "
                "Use one of: none, wo_visibility, wo_focus, wo_position, wo_all_scalar_cues."
            )
        return aliases[ablation]

    def _apply_view_scorer_ablation(
        self,
        view_pos: torch.Tensor,
        valid_ratio: torch.Tensor,
        prior_focus: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the 4-D scalar cue vector after applying the requested cue ablation.

        This keeps the network architecture and tensor dimensions unchanged, which is
        useful for fair retraining ablations and checkpoint compatibility. Zeroing a
        cue removes view-specific information from that cue; if all scalar cues are
        zeroed, cue_proj can only contribute a view-independent learned bias.
        """
        inv_view_pos = 1.0 - view_pos
        ablation = self.view_scorer_ablation

        if ablation == "wo_position":
            view_pos = torch.zeros_like(view_pos)
            inv_view_pos = torch.zeros_like(inv_view_pos)
        elif ablation == "wo_visibility":
            valid_ratio = torch.zeros_like(valid_ratio)
        elif ablation == "wo_focus":
            prior_focus = torch.zeros_like(prior_focus)
        elif ablation == "wo_all_scalar_cues":
            view_pos = torch.zeros_like(view_pos)
            inv_view_pos = torch.zeros_like(inv_view_pos)
            valid_ratio = torch.zeros_like(valid_ratio)
            prior_focus = torch.zeros_like(prior_focus)
        elif ablation != "none":
            # Should not happen because we validate in __init__, but keep this for safety.
            raise ValueError(f"Unsupported view_scorer_ablation={ablation!r}")

        return torch.stack([view_pos, inv_view_pos, valid_ratio, prior_focus], dim=-1)

    def _order_selected_views(
        self,
        selected_idx_by_relevance: torch.Tensor,
        selected_w_by_relevance: torch.Tensor,
    ):
        """
        Reorder the same Top-M selected views for trained order ablations.

        The set of selected views is always produced by the trained view scorer.
        This function only changes the recurrent memory write order, which isolates
        relevance-order vs chronological/random/reverse update behavior.
        """
        if selected_idx_by_relevance.numel() <= 1:
            return selected_idx_by_relevance, selected_w_by_relevance

        order = self.memory_update_order
        if order == "relevance":
            perm = torch.arange(
                selected_idx_by_relevance.numel(),
                device=selected_idx_by_relevance.device,
                dtype=torch.long,
            )
        elif order == "chronological":
            perm = torch.argsort(selected_idx_by_relevance, descending=False)
        elif order == "reverse_chronological":
            perm = torch.argsort(selected_idx_by_relevance, descending=True)
        elif order == "random":
            if self.random_order_seed >= 0:
                # Deterministic mode, useful for fixed-seed evaluation.
                # The seed is intentionally not advanced here; for training, keep
                # random_order_seed=-1 so the runner/global RNG controls randomness.
                gen = torch.Generator(device="cpu")
                gen.manual_seed(int(self.random_order_seed))
                perm = torch.randperm(
                    selected_idx_by_relevance.numel(),
                    generator=gen,
                    device="cpu",
                ).to(selected_idx_by_relevance.device)
            else:
                # Stochastic mode, recommended for retraining the random-update ablation.
                perm = torch.randperm(
                    selected_idx_by_relevance.numel(),
                    device=selected_idx_by_relevance.device,
                )
        else:
            raise ValueError(f"Unsupported memory_update_order={order!r}")

        return (
            selected_idx_by_relevance.index_select(0, perm),
            selected_w_by_relevance.index_select(0, perm),
        )

    def _compute_target_prior_single(
        self,
        volume: torch.Tensor,   # (1,C,X,Y,Z)
        valid: torch.Tensor,    # (1,1,X,Y,Z)
        slot_tgt: torch.Tensor, # (1,D)
    ):
        _, _, X, Y, Z = volume.shape

        vtxt = self.voxel_to_text(volume)                 # (1,D,X,Y,Z)
        vflat = vtxt.flatten(2).transpose(1, 2)          # (1,N,D)
        slot = slot_tgt                                   # (1,D)

        if self.sim_normalize:
            vflat = F.normalize(vflat, dim=-1, eps=1e-6)
            slot = F.normalize(slot, dim=-1, eps=1e-6)

        sim = (vflat * slot.unsqueeze(1)).sum(dim=-1)     # (1,N)
        scale = self.logit_scale.exp().clamp(max=50.0).to(sim.dtype)
        logits = (sim * scale).view(1, 1, X, Y, Z)

        logits = logits.masked_fill(~valid.bool(), -20.0)
        prior = torch.sigmoid(logits) * valid.float()

        return logits, prior

    def forward(
        self,
        view_volumes: torch.Tensor,   # (V,C,X,Y,Z)
        view_valids: torch.Tensor,    # (V,1,X,Y,Z)
        pooled_text: torch.Tensor,    # (1,D) or (D,)
        token_embeds: torch.Tensor = None,
        token_ids: torch.Tensor = None,
        return_debug: bool = False,
    ):
        if view_volumes.dim() != 5:
            raise ValueError(f"view_volumes must be (V,C,X,Y,Z), got {view_volumes.shape}")
        if view_valids.dim() != 5:
            raise ValueError(f"view_valids must be (V,1,X,Y,Z), got {view_valids.shape}")

        if pooled_text.dim() == 1:
            pooled_text = pooled_text.unsqueeze(0)

        V, C, X, Y, Z = view_volumes.shape
        device = view_volumes.device
        dtype = view_volumes.dtype
        valids = view_valids.float()

        # ------------------------------------------------------------
        # 1) Query-aware target slot
        # ------------------------------------------------------------
        if token_embeds is not None and token_ids is not None:
            voxel_ctx_c = self._global_voxel_ctx(view_volumes, view_valids)   # (1,C)
            voxel_ctx = self.voxel_ctx_proj(voxel_ctx_c)                      # (1,D)
            slots = self.slotter(
                token_embeds=token_embeds,
                token_ids=token_ids,
                pooled_text=pooled_text,
                voxel_ctx=voxel_ctx,
                return_attn=False,
            )  # (1,1,D)
            slot_tgt = slots[:, 0]
        else:
            slot_tgt = pooled_text

        slot_tgt = self.ln_text(slot_tgt)
        text_delta = self.text_to_voxel(slot_tgt).view(1, C, 1, 1, 1)

        # ------------------------------------------------------------
        # 2) Per-view target prior + query-aware view descriptor
        # ------------------------------------------------------------
        has_obs = view_valids.reshape(V, -1).bool().any(dim=1)

        if V == 1:
            view_pos = torch.zeros(1, device=device, dtype=dtype)
        else:
            view_pos = torch.linspace(0, 1, steps=V, device=device, dtype=dtype)

        target_priors = []
        view_descs = []
        prior_focus = []
        valid_ratio = []

        for t in range(V):
            cur = view_volumes[t:t+1]
            cur_valid = valids[t:t+1]

            _, prior_t = self._compute_target_prior_single(cur, cur_valid, slot_tgt)

            # target-aware pooling + small valid fallback
            pool_w = prior_t + 0.05 * cur_valid
            desc_t = self._masked_pool_single(cur, pool_w)

            focus_t = self._masked_topk_mean_single(prior_t, cur_valid > 0.5, k=64)
            valid_ratio_t = cur_valid.mean()

            target_priors.append(prior_t)
            view_descs.append(desc_t)
            prior_focus.append(focus_t)
            valid_ratio.append(valid_ratio_t)

        target_priors = torch.cat(target_priors, dim=0)               # (V,1,X,Y,Z)
        view_desc = torch.cat(view_descs, dim=0)                      # (V,C)
        prior_focus = torch.stack(prior_focus, dim=0).to(view_desc.dtype)   # (V,)
        valid_ratio = torch.stack(valid_ratio, dim=0).to(view_desc.dtype)   # (V,)

        # ------------------------------------------------------------
        # 3) View scoring
        # 用 prior_focus 替代 prior_mass，避免偏向“大而散”的 view
        # ------------------------------------------------------------
        view_desc = self.view_ln(view_desc)
        view_feat = self.view_proj(view_desc)
        text_feat = self.text_proj(slot_tgt).expand(V, -1)

        cues = self._apply_view_scorer_ablation(
            view_pos=view_pos,
            valid_ratio=valid_ratio,
            prior_focus=prior_focus,
        )

        cue_feat = self.cue_proj(cues)
        score_feat = torch.cat([view_feat, text_feat, cue_feat], dim=-1)
        view_logits = self.view_score_head(score_feat).squeeze(-1)
        view_logits = view_logits.masked_fill(~has_obs, -20.0)

        view_scores = torch.sigmoid(view_logits)
        view_weights = self._masked_softmax(
            view_logits,
            has_obs,
            temperature=self.score_temperature,
            dim=0,
        )

        num_valid_views = int(has_obs.sum().item())
        if num_valid_views == 0:
            fused_volume = view_volumes.new_zeros((C, X, Y, Z))
            fused_valid = torch.zeros((1, X, Y, Z), device=device, dtype=torch.bool)

            if return_debug:
                debug: Dict[str, Any] = {
                    "view_logits": view_logits.unsqueeze(0),
                    "view_scores": view_scores.unsqueeze(0).detach(),
                    "view_weights": view_weights.unsqueeze(0).detach(),
                    "mem_conf": torch.zeros((1, 1, X, Y, Z), device=device, dtype=dtype),
                    "selected_idx": torch.empty(0, device=device, dtype=torch.long),
                    "selected_idx_by_relevance": torch.empty(0, device=device, dtype=torch.long),
                    "memory_update_order": self.memory_update_order,
                    "view_scorer_ablation": self.view_scorer_ablation,
                    "coverage": fused_valid.float().mean().detach(),
                }
                return fused_volume, fused_valid, debug
            return fused_volume, fused_valid

        # ------------------------------------------------------------
        # 4) Weighted-average baseline over all views
        # ------------------------------------------------------------
        w_all = view_weights.view(V, 1, 1, 1, 1)
        avg_valid = (valids * w_all).sum(dim=0)
        avg_volume = (view_volumes * valids * w_all).sum(dim=0) / avg_valid.clamp(min=1e-6)
        avg_volume = avg_volume * (avg_valid > 0).float()

        # ------------------------------------------------------------
        # 5) Select top-M views
        # 重要：按 relevance order 写 memory，不再按 camera index 排序
        # ------------------------------------------------------------
        M = min(self.top_m, num_valid_views)

        # Top-k 默认按 score 从高到低，得到相同的 selected view set。
        # memory_update_order 只改变 recurrent memory 的写入顺序；
        # 对 reverse/random ablation，请重新训练对应 variant。
        selected_idx_by_relevance = torch.topk(view_logits, k=M, dim=0).indices

        selected_w_by_relevance = view_weights.index_select(0, selected_idx_by_relevance)
        selected_w_by_relevance = selected_w_by_relevance / selected_w_by_relevance.sum().clamp(min=1e-6)

        selected_idx, selected_w = self._order_selected_views(
            selected_idx_by_relevance=selected_idx_by_relevance,
            selected_w_by_relevance=selected_w_by_relevance,
        )

        # 重新归一化一次，避免极端数值误差。
        selected_w = selected_w / selected_w.sum().clamp(min=1e-6)

        selected_priors = target_priors.index_select(0, selected_idx)
        consensus_prior = (selected_priors * selected_w.view(M, 1, 1, 1, 1)).sum(dim=0, keepdim=True)
        consensus_prior = consensus_prior.clamp(0.0, 1.0)

        # ------------------------------------------------------------
        # 6) Conservative recurrent memory update
        # ------------------------------------------------------------
        mem = view_volumes.new_zeros((1, C, X, Y, Z))
        obs_valid = view_volumes.new_zeros((1, 1, X, Y, Z))   # 观测覆盖
        mem_conf = view_volumes.new_zeros((1, 1, X, Y, Z))    # target support / confidence

        write_scale = torch.sigmoid(self.text_write_scale)

        for rank, idx in enumerate(selected_idx.tolist()):
            cur = view_volumes[idx:idx+1]
            cur_valid = valids[idx:idx+1]

            # raw prior for gate feature
            prior = target_priors[idx:idx+1] * cur_valid

            # sharpened + consensus-gated support for actual write
            prior_write = prior.clamp(0.0, 1.0).pow(self.prior_gamma)
            prior_write = prior_write * (0.5 + 0.5 * consensus_prior)
            prior_write = prior_write * cur_valid

            diff = (cur - mem).abs().mean(dim=1, keepdim=True)

            prior_denom = prior_write.sum().clamp(min=1e-6)
            support_detached = mem_conf.detach()

            new_ratio = (((1.0 - support_detached) * prior_write).sum() / prior_denom).detach()
            overlap_ratio = ((support_detached * prior_write).sum() / prior_denom).detach()

            view_w = selected_w[rank]
            view_w_map = view_w.view(1, 1, 1, 1, 1).expand(1, 1, X, Y, Z)

            # 用 update stage 代替伪 temporal index
            stage = 0.0 if M == 1 else float(rank) / float(M - 1)
            stage_map = view_volumes.new_full((1, 1, X, Y, Z), stage)

            new_map = new_ratio.view(1, 1, 1, 1, 1).expand(1, 1, X, Y, Z)
            overlap_map = overlap_ratio.view(1, 1, 1, 1, 1).expand(1, 1, X, Y, Z)

            mem_r = self.mem_reduce(mem)
            cur_r = self.cur_reduce(cur)

            gate_in = torch.cat([
                mem_r,        # H/2
                cur_r,        # H/2
                prior,        # 1
                diff,         # 1
                cur_valid,    # 1
                obs_valid,    # 1
                mem_conf,     # 1
                view_w_map,   # 1
                new_map,      # 1
                overlap_map,  # 1
                stage_map,    # 1
            ], dim=1)

            gate = torch.sigmoid(self.gate_net(gate_in)) * cur_valid
            gate = gate * (0.10 + 0.90 * prior_write)
            gate = gate * (0.35 + 0.65 * view_w_map)

            # 只对“新的 target evidence”做强写入
            first_write = prior_write * (1.0 - mem_conf) * (0.25 + 0.75 * new_map)

            update = torch.maximum(gate, first_write).clamp(max=1.0)

            # memory 内只保留很弱的 text 引导，避免和后面的 text_voxel_fuser 叠加过强
            cand = cur + 0.25 * write_scale * prior_write * text_delta

            mem = mem + update * (cand - mem)
            obs_valid = torch.maximum(obs_valid, cur_valid)

            # 只在实际 target support 区域提高 confidence
            mem_conf = torch.maximum(mem_conf, prior_write * (0.5 + 0.5 * update))

        mem = self.out_norm(mem)
        mem = mem * obs_valid

        # ------------------------------------------------------------
        # 7) Confidence-guided residual blend
        # ------------------------------------------------------------
        avg = avg_volume.unsqueeze(0)
        blend = torch.sigmoid(self.blend_alpha) * mem_conf
        fused = avg + blend * (mem - avg)

        fused_valid = (avg_valid > 0).unsqueeze(0)
        fused = fused * fused_valid.float()

        fused_volume = fused.squeeze(0)
        fused_valid = fused_valid.squeeze(0).bool()

        if return_debug:
            debug: Dict[str, Any] = {
                "view_logits": view_logits.unsqueeze(0),
                "view_scores": view_scores.unsqueeze(0).detach(),
                "view_weights": view_weights.unsqueeze(0).detach(),
                "mem_conf": mem_conf.detach(),
                "selected_idx": selected_idx.detach(),
                "selected_idx_by_relevance": selected_idx_by_relevance.detach(),
                "selected_weights": selected_w.detach(),
                "cue_values": cues.detach(),
                "memory_update_order": self.memory_update_order,
                "view_scorer_ablation": self.view_scorer_ablation,
                "coverage": fused_valid.float().mean().detach(),
            }
            return fused_volume, fused_valid, debug

        return fused_volume, fused_valid