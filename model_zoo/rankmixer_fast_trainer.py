"""Fast trainer for RankMixer: same training algorithm as BaseModel.fit(), restructured for speed.

Enabled per experiment with `fast_trainer: True` in config/model_config.yaml; run_expid.py then uses
build_fast_trainer() + fit() below instead of DDP + model.fit(). Validation, checkpointing and the
test evaluation still use the original model (`trainer.model`, a model_zoo.RankMixer whose
parameters are the trained ones) and its evaluate().

Origin: the submitted solution (run 14) of an autonomous Claude Code optimization run on the
UniRank training-speed task (agent_task_train/, docs/agent_train_opt.md), checked on full-epoch
training in fulltrain_check/ (RESULTS.md): 2,593 s -> 706 s per epoch on 2 GPUs with the original
data pipeline, same loss curve, test AUC within run-to-run noise. Only the epoch-level fit() and
this docstring were added; the trainer itself is unchanged.

Requirements (asserted where cheap): RankMixer with single-head target attention, all features
categorical with padding_idx 0, PerTokenFeedForward layers, dense AdamW + Adagrad on the embedding
tables with lr_decay = weight_decay = 0 (UniRank hard-codes weight_decay=0 for the sparse
optimizer), accumulation_steps 1, a torch.distributed process group (run_expid.py creates a
1-rank group for single-GPU runs). Not supported: reduce_lr_on_plateau / early stopping across
epochs (the trainer keeps its own optimizers).

How it works (from the solution's notes):
- The model is built exactly as before (same parameter initialization, rank 0's parameters broadcast
  as DDP does); then all 37 embedding tables are merged into one fp32 table (the original
  nn.Embedding weights become views of it, so state_dict() is unchanged): [small tables | user_id,
  item_id | sentinel row]. One gather per step for the forward; the gradient is taken w.r.t. the
  gathered rows.
- small tables (<= 4096 rows): their dense gradient (partitioned scatter-add) is all-reduced with
  the dense gradients and updated with dense Adagrad, exactly as before;
- large tables: per-rank coalescing (sort + segment sum), all-gather, union coalescing (<= 2 terms
  per row: exact and identical on both ranks) and an exact sparse Adagrad over the touched rows
  (rows with zero gradient are not changed by Adagrad, lr_decay = weight_decay = 0);
- the dense part (attention, tokenizer, RankMixer layers, towers) is one torch.compile graph; the
  target attention uses q.(W_k s) = (W_k^T q).s and sum_j a_j W_v s_j = W_v sum_j a_j s_j, so the
  100 history items are not projected; sigmoid + BCE stay eager (bf16-rounded predictions as in the
  original model);
- dense gradients: one flat buffer, one all_reduce, clip at 10, fused AdamW;
- the whole step (input preparation, forward, backward, NCCL collectives, both optimizers) is
  captured in CUDA graphs (two, for double-buffered inputs filled by a copy stream); per step the
  CPU only copies the batch and replays a graph: no host synchronization, nothing that holds the
  GIL for long (the data loader's pin-memory thread lives in this process and is the bottleneck).
"""
import collections
import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

import model_zoo


def build_trainer(ctx):
    return Trainer(ctx)


# --------------------------------------------------------------------------------------------- dense
def _linear(x, lin):
    return F.linear(x, lin.weight, lin.bias)


class DenseNet(nn.Module):
    """The dense part of RankMixer, sharing the parameters of the original model's modules."""

    def __init__(self, model, n_ctx_fields, n_item_fields, emb_dim):
        super().__init__()
        unwrap = lambda m: getattr(m, "_orig_mod", m)
        self.att = unwrap(model.attention_layers)
        self.tok = unwrap(model.unified_tokenizer_layer)
        self.mixer = unwrap(model.unified_interaction_layers)
        self.towers = nn.ModuleList([unwrap(t) for t in model.tower])
        self.n_ctx = n_ctx_fields
        self.n_item = n_item_fields
        self.emb_dim = emb_dim
        self.item_dim = n_item_fields * emb_dim
        self.att_scale = 1.0 / self.att.scale
        self.num_tasks = len(self.towers)

    def attention(self, target, seq, mask):
        # target [B, Di], seq [B, L, Di] (fp32), mask [B, L] bool
        att = self.att
        q = F.linear(target, att.W_q.weight)                         # [B, A] (bf16 under autocast)
        qk = torch.matmul(q, att.W_k.weight).float()                  # [B, Di]  == W_k^T q
        scores = torch.einsum("bd,bld->bl", qk, seq) * self.att_scale  # fp32
        scores = scores.masked_fill(~mask, -1e30)
        w = torch.softmax(scores, dim=-1)
        w = torch.where(mask, w, torch.zeros_like(w))                 # fully masked rows -> 0
        pooled = torch.einsum("bl,bld->bd", w, seq)                   # [B, Di] fp32
        v = F.linear(pooled, att.W_v.weight)
        return F.linear(v, att.W_o.weight)

    def mixer_forward(self, x):
        m = self.mixer
        B, T, D = x.shape
        for i in range(m.num_layers):
            h = m.mixer_norms[i](x)
            mix = m.mixer_layers[i]
            assert mix.pad_dim == 0
            h = h.view(B, T, T, D // T).transpose(1, 2).reshape(B, T, D)
            x = h + x
            ffn = m.pffn_layers[i]
            h = m.pffn_norms[i](x)
            h = torch.einsum("btd,tdk->btk", h, ffn.W1) + ffn.b1
            h = ffn.act(h)
            h = torch.einsum("btk,tkd->btd", h, ffn.W2) + ffn.b2
            x = h + x
        return x

    def forward(self, emb, mask, labels):
        # emb [N, E] fp32: B*n_ctx context rows, then B*(L+1)*n_item item rows
        B = mask.shape[0]
        L1 = mask.shape[1] + 1
        E = self.emb_dim
        n_ctx_rows = B * self.n_ctx
        ctx_emb = emb[:n_ctx_rows].view(B, self.n_ctx * E)
        item_emb = emb[n_ctx_rows:].view(B, L1, self.item_dim)
        mask = mask.bool()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            target = item_emb[:, -1, :]
            seq = item_emb[:, :-1, :]
            pooled = self.attention(target, seq, mask)
            feat = torch.cat([ctx_emb, target, pooled], dim=-1)
            tokens = self.tok(feat)
            tokens = self.mixer_forward(tokens)
            bottom = tokens.mean(dim=1)
            logits = torch.cat([t(bottom) for t in self.towers], dim=1)   # [B, tasks] bf16
        return logits


def bce_loss(logits, labels):
    # eager on purpose: as in the original model, the sigmoid output is rounded to bf16 before the
    # fp32 BCE (Inductor would fuse that rounding away).
    preds = torch.sigmoid(logits).float()
    return F.binary_cross_entropy(preds, labels, reduction="sum") / labels.shape[0]


# --------------------------------------------------------------------------------------------- sparse
SMALL_VOCAB = 4096          # tables with at most this many rows get dense (all-reduced) gradients
N_PART = 128                # partitions of the small-table gradient scatter (avoids atomic contention)


def _gather_rows(table, idx):
    return table[idx]


def _seg(sids):
    """Segment ids of a sorted id vector: 0-based index of the distinct value at each position."""
    flag = torch.ne(sids[1:], sids[:-1]).long()
    return torch.cat([flag.new_zeros(1), torch.cumsum(flag, 0)])


class Trainer:
    def __init__(self, ctx):
        params = dict(ctx.params)
        self.ctx = ctx
        self.device = dev = ctx.device
        self.rank, self.world = ctx.rank, ctx.world_size
        model = model_zoo.RankMixer(ctx.feature_map, **params)
        if ctx.rank == 0:
            model.count_parameters()
        self.model = model
        self.max_norm = float(params.get("max_gradient_norm", 10.0))
        fm = ctx.feature_map
        self.labels = list(fm.labels)

        # DDP broadcasts rank 0's parameters and buffers at construction (the ranks are seeded differently)
        with torch.no_grad():
            for t in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(t.data, src=0)

        # ---- merged embedding table: [small tables | large tables | sentinel row]
        emb_layers = model.embedding_layer.embedding_layer.embedding_layers
        self.ctx_feats, self.item_feats = [], []
        for feat, spec in fm.features.items():
            if feat in fm.labels or spec.get("type") == "meta":
                continue
            assert spec["type"] == "categorical" and feat in emb_layers
            assert emb_layers[feat].padding_idx == 0
            if spec.get("source") in ["item", "action"]:
                self.item_feats.append(feat)
            else:
                self.ctx_feats.append(feat)
        feats = self.ctx_feats + self.item_feats
        self.vocab = {f: emb_layers[f].weight.shape[0] for f in feats}
        small = [f for f in feats if self.vocab[f] <= SMALL_VOCAB]
        large = [f for f in feats if self.vocab[f] > SMALL_VOCAB]
        offsets, off = {}, 0
        for f in small + large:
            offsets[f] = off
            off += self.vocab[f]
        self.R_small = sum(self.vocab[f] for f in small)
        self.num_rows = off
        self.sentinel = off                        # scratch row: padding positions / unused slots
        self.emb_dim = E = emb_layers[feats[0]].weight.shape[1]
        with torch.no_grad():
            table = torch.zeros(off + 1, E, dtype=torch.float32, device=dev)
            for f in feats:
                table[offsets[f]:offsets[f] + self.vocab[f]] = emb_layers[f].weight.data
        for f in feats:
            emb_layers[f].weight.data = table[offsets[f]:offsets[f] + self.vocab[f]]
        self.table = table
        self.state_sum = torch.zeros_like(table)
        self.offsets = offsets
        self.L1 = int(params.get("max_len", 100)) + 1
        self.ctx_off = torch.tensor([offsets[f] for f in self.ctx_feats], dtype=torch.long, device=dev)
        self.item_off = torch.tensor([offsets[f] for f in self.item_feats], dtype=torch.long, device=dev)
        self.c_small = [j for j, f in enumerate(self.ctx_feats) if f in small]
        self.c_large = [j for j, f in enumerate(self.ctx_feats) if f in large]
        self.i_small = [j for j, f in enumerate(self.item_feats) if f in small]
        self.i_large = [j for j, f in enumerate(self.item_feats) if f in large]
        self.pad_rows_small = torch.tensor([offsets[f] for f in small], dtype=torch.long, device=dev)
        self.sparse_lr = float(params.get("sparse_learning_rate", 0.05))
        self.adagrad_eps = 1e-10

        # ---- dense parameters + small-table gradients in one flat buffer (one all_reduce), fused AdamW
        emb_param_ids = {id(emb_layers[f].weight) for f in feats}
        self.dense_params = [p for p in model.parameters() if p.requires_grad and id(p) not in emb_param_ids]
        self.n_dense = n_dense = sum(p.numel() for p in self.dense_params)
        self.flat_grad = torch.zeros(n_dense + self.R_small * E, dtype=torch.float32, device=dev)
        o = 0
        for p in self.dense_params:
            p.grad = self.flat_grad[o:o + p.numel()].view_as(p)
            o += p.numel()
        self.small_grad = self.flat_grad[n_dense:].view(self.R_small, E)
        # drop the original optimizers (the dense Adagrad state alone is 2 GB)
        model.optimizers, model.dense_optimizer, model.sparse_optimizer, model.optimizer = [], None, None, None
        self.adamw = torch.optim.AdamW(self.dense_params, lr=float(params.get("dense_learning_rate", 1e-4)),
                                       weight_decay=float(params.get("dense_weight_decay", 0) or 0),
                                       fused=True, capturable=True)
        torch.cuda.empty_cache()

        self.dense = DenseNet(model, len(self.ctx_feats), len(self.item_feats), E)
        self.dense_c = torch.compile(self.dense)
        self.gather_c = torch.compile(_gather_rows)
        self.small_grad_c = torch.compile(self._small_grads)
        self.split_c = torch.compile(self._large_grads)
        self.adagrad_c = torch.compile(self._adagrad_rows)
        self.use_graph = True
        self.copy_stream = torch.cuda.Stream(dev)

    # --------------------------------------------------------------------------------------- input
    def _alloc_static(self, batch):
        batch_dict, item_dict, mask, _ = batch
        dev = self.device
        B = mask.shape[0]
        self.B = B
        self.cols = self.ctx_feats + self.labels
        # The batch_dict columns are strided 1-D views, each with its own pinned storage. A direct H2D
        # copy of a strided view goes through a synchronous pageable staging copy; DMA-ing their whole
        # pinned spans (96 MB/step) slows the data loader down. Gather them on the CPU into a small ring
        # of pinned buffers instead (torch.stack releases the GIL) and copy that in one piece.
        v0 = batch_dict[self.cols[0]]
        self.col_dtype = v0.dtype
        self.ring = [torch.empty(len(self.cols), B, dtype=v0.dtype).pin_memory() for _ in range(4)]
        self.ring_ev = [None] * len(self.ring)
        self.ring_i = 0
        self.sets = []
        for _ in range(2):                    # double-buffered static inputs (one CUDA graph per set)
            c = torch.empty(len(self.cols), B, dtype=v0.dtype, device=dev)
            items = [torch.empty(item_dict[f].shape, dtype=item_dict[f].dtype, device=dev) for f in self.item_feats]
            m = torch.empty(mask.shape, dtype=mask.dtype, device=dev)
            self.sets.append({"cols": c, "items": items, "mask": m, "dst": [c] + items + [m]})
        L1 = self.L1
        self.part_c = (torch.arange(B, device=dev) % N_PART).view(B, 1)
        self.part_i = (torch.arange(B * L1, device=dev) % N_PART).view(B, L1, 1)

    def _batch_matches(self, batch):
        batch_dict, item_dict, mask, _ = batch
        if mask.shape[0] != self.B or mask.shape[1] + 1 != self.L1:
            return False
        return all(batch_dict[c].dtype == self.col_dtype for c in self.cols)

    def _copy_in(self, batch, st):
        """Issue the H2D copies of `batch` into the static input set `st` on the current stream."""
        batch_dict, item_dict, mask, _ = batch
        k = self.ring_i % len(self.ring)
        self.ring_i += 1
        buf, ev = self.ring[k], self.ring_ev[k]
        if ev is not None:
            ev.synchronize()                 # issued 4 steps ago (blocking-sync event: sleeps, no spin)
        torch.stack([batch_dict[c] for c in self.cols], out=buf)
        src = [buf] + [item_dict[f] for f in self.item_feats] + [mask]
        torch._foreach_copy_(st["dst"], src, non_blocking=True)
        ev = torch.cuda.Event(blocking=True)
        ev.record()
        self.ring_ev[k] = ev

    # --------------------------------------------------------------------------------------- step
    def _small_grads(self, G, ids_c, ids_i):
        """Gradient of the small tables [R_small, E] from the gradient G [N, E] of the gathered rows."""
        B, L1, E, R = self.B, self.L1, self.emb_dim, self.R_small
        nc = B * len(self.ctx_feats)
        Gc = G[:nc].view(B, len(self.ctx_feats), E)[:, self.c_small]
        Gi = G[nc:].view(B, L1, len(self.item_feats), E)[:, :, self.i_small]
        tc = self.part_c * R + ids_c[:, self.c_small]
        ti = self.part_i * R + ids_i[:, :, self.i_small]
        tgt = torch.cat([tc.reshape(-1), ti.reshape(-1)])
        src = torch.cat([Gc.reshape(-1, E), Gi.reshape(-1, E)])
        buf = torch.zeros(N_PART * R, E, dtype=G.dtype, device=G.device)
        buf.index_put_((tgt,), src, accumulate=True)
        return buf.view(N_PART, R, E).sum(0)

    def _large_grads(self, G, ids_c, ids_i, loc_c, loc_i):
        """(ids, grads) of the positions of the large tables; padding positions -> sentinel row."""
        B, L1, E = self.B, self.L1, self.emb_dim
        nc = B * len(self.ctx_feats)
        Gc = G[:nc].view(B, len(self.ctx_feats), E)[:, self.c_large]
        Gi = G[nc:].view(B, L1, len(self.item_feats), E)[:, :, self.i_large]
        ic = torch.where(loc_c[:, self.c_large] == 0, self.sentinel, ids_c[:, self.c_large])
        ii = torch.where(loc_i[:, :, self.i_large] == 0, self.sentinel, ids_i[:, :, self.i_large])
        return torch.cat([ic.reshape(-1), ii.reshape(-1)]), torch.cat([Gc.reshape(-1, E), Gi.reshape(-1, E)])

    def _adagrad_rows(self, table, state, uid, g):
        # exact Adagrad (lr_decay 0, weight_decay 0, initial accumulator 0, eps 1e-10)
        s = state[uid] + g * g
        w = table[uid] - self.sparse_lr * g / (torch.sqrt(s) + self.adagrad_eps)
        return s, w

    def _coalesce(self, ids, g, cap):
        """Sum rows of g with equal ids (static shapes). Returns (uniq [cap], sums [cap, E])."""
        sids, perm = torch.sort(ids)
        seg = _seg(sids)
        uniq = torch.full((cap,), self.sentinel, dtype=ids.dtype, device=ids.device)
        uniq.scatter_(0, seg, sids)
        sums = torch.zeros(cap, g.shape[1], dtype=g.dtype, device=g.device)
        sums.index_add_(0, seg, self.gather_c(g, perm))
        return uniq, sums

    def _step_body(self, st):
        """One full training step on the static input set `st`. Returns the loss (scalar tensor)."""
        B, nc = self.B, len(self.ctx_feats)
        cols = st["cols"]                                                       # [n_cols, B]
        loc_c = cols[:nc].t().long()                                            # [B, Fc]
        loc_i = torch.stack([t.long() for t in st["items"]], dim=-1)            # [B, L+1, Fi]
        ids_c = loc_c + self.ctx_off
        ids_i = loc_i + self.item_off
        ids = torch.cat([ids_c.reshape(-1), ids_i.reshape(-1)])
        labels = cols[nc:].t().float()
        mask = st["mask"]

        emb = self.gather_c(self.table, ids).requires_grad_(True)
        loss = bce_loss(self.dense_c(emb, mask, labels), labels)
        loss.backward()
        G = emb.grad
        with torch.no_grad():
            # small tables: dense gradient into the flat buffer, padding rows get no gradient
            self.small_grad.copy_(self.small_grad_c(G, ids_c, ids_i))
            self.small_grad.index_fill_(0, self.pad_rows_small, 0.0)
            work = dist.all_reduce(self.flat_grad, async_op=True)
            # large tables: coalesce locally, all-gather, coalesce the union (<= 2 terms per row: exact
            # and identical on both ranks), sparse Adagrad
            lid, lg = self.split_c(G, ids_c, ids_i, loc_c, loc_i)
            cap = lid.shape[0]
            uniq, gsum = self._coalesce(lid, lg, cap)
            recv_ids = torch.empty(self.world * cap, dtype=uniq.dtype, device=self.device)
            recv_g = torch.empty(self.world * cap, gsum.shape[1], dtype=gsum.dtype, device=self.device)
            dist.all_gather_into_tensor(recv_ids, uniq)
            dist.all_gather_into_tensor(recv_g, gsum)
            work.wait()
            # dense parameters: average, clip (dense only), AdamW; small tables: dense Adagrad
            self.flat_grad.mul_(1.0 / self.world)
            torch.nn.utils.clip_grad_norm_(self.dense_params, self.max_norm, foreach=True)
            self.adamw.step()
            gs = self.small_grad
            s_small = self.state_sum[:self.R_small]
            s_small.addcmul_(gs, gs)
            self.table[:self.R_small].addcdiv_(gs, s_small.sqrt().add_(self.adagrad_eps), value=-self.sparse_lr)
            self.flat_grad.zero_()
            uid, gu = self._coalesce(recv_ids, recv_g, self.world * cap)
            gu.mul_(1.0 / self.world)
            s, w = self.adagrad_c(self.table, self.state_sum, uid, gu)
            self.state_sum.index_copy_(0, uid, s)
            self.table.index_copy_(0, uid, w)
        return loss.detach()

    def train(self, batches, report_loss):
        n_warm = 3
        graphs = [None, None]
        g_loss = [None, None]
        side = torch.cuda.Stream(self.device)
        main = torch.cuda.current_stream()
        self._inflight = collections.deque()
        done_ev = [None, None]              # last replay that read input set k
        for i, batch in enumerate(batches):
            if i == 0:
                self._alloc_static(batch)
            if not self._batch_matches(batch) or (i >= n_warm and not self.use_graph):
                main.wait_stream(side)
                loss = self._eager_step(batch)
            elif i < n_warm:
                # warm-up (compilation, NCCL, cuBLAS) on a side stream, as required before graph capture
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    self._copy_in(batch, self.sets[0])
                    loss = self._step_body(self.sets[0])
                main.wait_stream(side)
            else:
                k = i % 2
                st = self.sets[k]
                self._throttle()
                cs = self.copy_stream
                if done_ev[k] is not None:
                    cs.wait_event(done_ev[k])
                else:
                    cs.wait_stream(main)
                with torch.cuda.stream(cs):
                    self._copy_in(batch, st)
                main.wait_stream(cs)
                if graphs[k] is None:
                    g = torch.cuda.CUDAGraph()
                    other = graphs[1 - k]
                    with torch.cuda.graph(g, pool=other.pool() if other is not None else None):
                        g_loss[k] = self._step_body(st)
                    graphs[k] = g
                graphs[k].replay()
                ev = torch.cuda.Event(blocking=True)
                ev.record(main)
                done_ev[k] = ev
                self._inflight.append(ev)
                loss = g_loss[k]
            report_loss(loss)
            if i % 200 == 0 and self.rank == 0:
                logging.info("step %d issued", i)

    def _throttle(self, max_ahead=3):
        # Bound how far the CPU runs ahead of the GPU (the faster rank's GPU waits for the other rank in
        # NCCL). The events are blocking-sync events: the wait sleeps (GIL released) instead of the
        # driver's default spin-wait, which would burn a CPU core next to the data-loader threads.
        q = self._inflight
        while len(q) >= max_ahead:
            q.popleft().synchronize()

    def _eager_step(self, batch):
        # fallback for an unexpected batch shape: same step on freshly allocated inputs, no graph
        names = ("B", "cols", "col_dtype", "ring", "ring_ev", "ring_i", "sets", "part_c", "part_i")
        saved = {n: getattr(self, n) for n in names}
        self._alloc_static(batch)
        self._copy_in(batch, self.sets[0])
        loss = self._step_body(self.sets[0])
        for n, v in saved.items():
            setattr(self, n, v)
        return loss

    def state_dict(self):
        # torch.compile wraps sub-modules (keys get "_orig_mod."); the reference model has none.
        return {k.replace("_orig_mod.", ""): v for k, v in self.model.state_dict().items()}


# --------------------------------------------------------------------------------------------- epoch loop
def build_fast_trainer(ctx):
    """ctx: rank, world_size, local_rank, device, feature_map, params, cpu_group."""
    return Trainer(ctx)


def fit(trainer, train_gen, valid_gen, params, cpu_group=None):
    """Epoch loop around Trainer.train(): what BaseModel.fit() does at the epoch level.

    Both ranks train min(len(loader)) steps per epoch (the blocked loader can give the ranks a few
    batches more or less; the original fit() let them run unevenly under DDP Join). Returns True for
    an MR_MAX_STEPS short run (no validation, no checkpoint), as BaseModel.fit() does.
    """
    import itertools
    import os
    import time
    from unirank.pytorch import observables as mr_observables

    model = trainer.model
    main = trainer.rank == 0
    epochs = int(params.get("epochs", 1))
    max_steps = mr_observables.max_steps()
    best = None
    for epoch in range(epochs):
        if hasattr(train_gen, "set_epoch"):
            train_gen.set_epoch(epoch)
        n = len(train_gen)
        if trainer.world > 1:
            t = torch.tensor([n], dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.MIN, group=cpu_group)
            n = int(t.item())
        if max_steps:
            n = min(n, max_steps)
        losses = torch.zeros(n, dtype=torch.float32, device=trainer.device)
        count = [0]

        def report_loss(loss):
            losses[count[0]] = loss.detach().reshape(()).float()      # no host sync
            count[0] += 1

        if main:
            logging.info("************ Epoch=%d start ************", epoch + 1)
            logging.info("Fast trainer: %d steps per rank (local batches: %d)", n, len(train_gen))
        t0 = time.time()
        trainer.train(itertools.islice(train_gen, n), report_loss)
        torch.cuda.synchronize()
        dt = time.time() - t0
        if count[0] != n:
            raise RuntimeError(f"rank {trainer.rank}: trained {count[0]} of {n} steps")
        if main:
            logging.info("Train loss: %.6f (%.1f s, %.1f ms/step)", losses.mean().item(), dt, dt / n * 1000)
        if max_steps:
            if main:
                logging.info("[MR] short run: stopping at step %d", n)
            return True
        if trainer.world > 1:
            dist.barrier(group=cpu_group)
        if main:
            logging.info("Evaluation @epoch %d - batch %d: ", epoch + 1, n)
        val_logs = model.evaluate(valid_gen, metrics=model._monitor.get_metrics())
        if main:
            value = model._monitor.get_value(val_logs)
            better = best is None or (value > best if model._monitor_mode == "max" else value < best)
            if better:
                best = value
                os.makedirs(os.path.dirname(model.checkpoint), exist_ok=True)
                logging.info("Save best model: monitor(%s)=%.6f", model._monitor_mode, value)
                model.save_weights(model.checkpoint)
            else:
                logging.info("Monitor(%s)=%.6f did not improve (no lr decay / early stop in the fast trainer)",
                             model._monitor_mode, value)
        if trainer.world > 1:
            dist.barrier(group=cpu_group)
        model.train()
    if epochs > 1:
        if trainer.world > 1:
            dist.barrier(group=cpu_group)
        model.load_weights(model.checkpoint)        # best epoch, as BaseModel.fit()
    if main:
        logging.info("Training finished.")
    return False
