"""Design A hybrid MP-Transolver: graph builder + local branch (CPU smoke).

Covers the G3-P3 must-fix checks: (a) hybrid_mp=false is byte-identical vanilla
Transolver; (b) hybrid forward/backward is finite and a tiny batch overfits;
(c) THE key check — the MP edge MLP receives nonzero gradient after one backward
(else the gate is a dead no-op); (d) the MP branch changes the one-step
prediction; plus a bounded-edge-count check on a contact-dense cloud and the
per-example partition invariant.
"""

import numpy as np
import pytest
import torch

from structbench.models.transolver import TransolverSimulator
from structbench.models.transolver.graph_ops import (
    _capped_radius_edges_single,
    build_radius_graph,
)
from structbench.models.transolver.network import TransolverNet, hybrid_block_indices

# --- graph builder ---------------------------------------------------------


def test_capped_radius_graph_is_degree_bounded_and_self_free():
    # Contact-dense: many points inside one radius ball. Uncapped this would be
    # O(P^2); the cap must hold E <= P * max_neighbors, with no self-loops.
    torch.manual_seed(0)
    p, k = 200, 8
    pos = torch.rand(p, 2) * 0.01  # all within a tiny ball
    ei = _capped_radius_edges_single(pos, radius=1.0, max_neighbors=k)
    assert ei.shape[0] == 2
    assert ei.shape[1] <= p * k  # bounded
    assert ei.shape[1] > 0  # dense cloud => edges exist
    assert (ei[0] != ei[1]).all()  # no self-loops
    # per-node in-degree never exceeds the cap
    deg = torch.zeros(p, dtype=torch.int64)
    deg.index_add_(0, ei[1], torch.ones_like(ei[1]))
    assert int(deg.max()) <= k


def test_build_radius_graph_partitions_per_example():
    # A 2-example batch must produce block-diagonal edges (no edge crosses the
    # example boundary), the invariant that keeps index_add_ segment-safe.
    torch.manual_seed(0)
    a, b = torch.rand(7, 2), torch.rand(5, 2)
    pos = torch.cat([a, b])
    n_per = torch.tensor([7, 5])
    ei, ef = build_radius_graph(pos, n_per, radius=0.5, max_neighbors=8)
    example_of = torch.repeat_interleave(torch.arange(2), n_per)
    assert (example_of[ei[0]] == example_of[ei[1]]).all()
    # edge_feats = [x_ij, |x_ij|]
    assert ef.shape == (ei.shape[1], 3)
    x_ij = pos[ei[0]] - pos[ei[1]]
    torch.testing.assert_close(ef[:, :2], x_ij)
    torch.testing.assert_close(ef[:, 2], torch.linalg.norm(x_ij, dim=-1))


def test_build_radius_graph_batched_equals_per_example():
    # The batched graph must equal the two per-example graphs concatenated
    # (offset), the graph analogue of test_batched_forward_matches_per_example.
    torch.manual_seed(1)
    a, b = torch.rand(9, 2), torch.rand(6, 2)
    ei_a, _ = build_radius_graph(a, None, 0.4, 8)
    ei_b, _ = build_radius_graph(b, None, 0.4, 8)
    ei_batch, _ = build_radius_graph(
        torch.cat([a, b]), torch.tensor([9, 6]), 0.4, 8
    )
    expected = torch.cat([ei_a, ei_b + 9], dim=1)
    # column order matches (per-example built in the same order)
    assert torch.equal(ei_batch, expected)


def test_hybrid_block_indices_are_middle_only():
    # Never block 0 or the last (decoder) block.
    assert hybrid_block_indices(8, 1) == {4}
    idx = hybrid_block_indices(8, 3)
    assert idx == {3, 4, 5}
    assert all(1 <= i <= 6 for i in idx)
    with pytest.raises(ValueError, match="exceeds"):
        hybrid_block_indices(4, 3)  # only 2 middle blocks available


# --- vanilla byte-identity -------------------------------------------------


def test_hybrid_off_adds_no_params_and_ignores_graph():
    net_plain = TransolverNet(
        node_in=7, out_size=4, hidden_dim=16, n_layers=4, n_heads=2, slice_num=4
    )
    net_off = TransolverNet(
        node_in=7,
        out_size=4,
        hidden_dim=16,
        n_layers=4,
        n_heads=2,
        slice_num=4,
        hybrid_mp=False,
    )
    # hybrid_mp=False creates NO extra params (byte-identical architecture).
    assert sum(p.numel() for p in net_plain.parameters()) == sum(
        p.numel() for p in net_off.parameters()
    )
    assert not any(getattr(b, "hybrid_mp", False) for b in net_off.blocks)
    # forward accepts the 2-arg call unchanged.
    out = net_off(torch.randn(5, 7), None)
    assert out.shape == (5, 4)


def test_hybrid_net_requires_graph():
    net = TransolverNet(
        node_in=7,
        out_size=4,
        hidden_dim=16,
        n_layers=4,
        n_heads=2,
        slice_num=4,
        hybrid_mp=True,
        hybrid_blocks=1,
        hybrid_edge_in=3,
    )
    with pytest.raises(ValueError, match="requires a graph"):
        net(torch.randn(5, 7), None)  # graph omitted


# --- simulator: the P1 smoke checks ---------------------------------------


def _hybrid_sim(**kw):
    return TransolverSimulator(
        dim=2,
        hidden_dim=8,
        n_layers=4,
        n_heads=2,
        slice_num=2,
        hybrid_mp=True,
        hybrid_radius=0.5,
        hybrid_max_neighbors=8,
        hybrid_blocks=1,
        **kw,
    )


def _mp_block(sim):
    for b in sim._net.blocks:
        if getattr(b, "hybrid_mp", False):
            return b
    raise AssertionError("no hybrid block found")


def test_hybrid_requires_positive_radius():
    with pytest.raises(ValueError, match="hybrid_radius"):
        TransolverSimulator(dim=2, hybrid_mp=True, hybrid_radius=0.0)


def test_mp_edge_mlp_receives_gradient():
    # THE key check (G3-P3): after one backward at init, the MP edge MLP must
    # receive nonzero gradient. If it is exactly 0 the gate is a dead no-op and
    # the whole run is a plain-Transolver wearing a costume.
    torch.manual_seed(0)
    sim = _hybrid_sim()
    p = 12
    x = torch.rand(p, 2)
    nxt = x + 0.05
    aux = torch.rand(p)
    types = torch.zeros(p, dtype=torch.int64)
    ref = torch.rand(p, 2)
    pred, target = sim.forward_train(
        x, nxt, aux, types, ref, torch.tensor([p]), accumulate=False
    )
    loss = ((pred - target) ** 2).mean()
    loss.backward()

    block = _mp_block(sim)
    edge_lin = block.mp.edge_mlp[0]  # first Linear of the edge MLP
    assert isinstance(edge_lin, torch.nn.Linear)
    assert edge_lin.weight.grad is not None
    gnorm = float(edge_lin.weight.grad.norm())
    assert gnorm > 0.0, f"MP edge MLP grad is zero ({gnorm}); gate is a dead no-op"
    # the gate itself also receives gradient
    assert block.mp_gate.grad is not None
    assert float(block.mp_gate.grad.norm()) > 0.0


def test_hybrid_forward_backward_finite_and_loss_decreases():
    # Forward/backward finite, and a tiny fixed batch overfits over ~20 steps.
    torch.manual_seed(0)
    sim = _hybrid_sim()
    p = 16
    x = torch.rand(p, 2)
    nxt = x + 0.05 * torch.rand(p, 2)
    aux = torch.rand(p)
    types = torch.zeros(p, dtype=torch.int64)
    ref = torch.rand(p, 2)
    npp = torch.tensor([p])
    opt = torch.optim.Adam(sim.parameters(), lr=1e-2)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        pred, target = sim.forward_train(x, nxt, aux, types, ref, npp, accumulate=False)
        loss = ((pred - target) ** 2).mean()
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_hybrid_mp_changes_offgrid_prediction():
    # BONUS (locality does something): on a bound case, a nonzero MP gate must
    # change the one-step prediction vs a zeroed gate (which removes the branch).
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    T, P = 6, 9
    sim = _hybrid_sim()
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    ref = torch.tensor(rng.random((P, 2)), dtype=torch.float32)
    types = torch.zeros(P, dtype=torch.int64)
    gt = torch.tensor(rng.random((T, P, 2)), dtype=torch.float32).cumsum(0)

    # amplify the gate so the effect is visible above float noise
    block = _mp_block(sim)
    with torch.no_grad():
        block.mp_gate.fill_(1.0)
    sim.eval()
    sim.bind_case(cells, ref, types, gt)
    sim.reset_rollout()
    win = gt[0:2].permute(1, 0, 2).contiguous()
    with torch.no_grad():
        nxt_on, _ = sim.predict_positions(win, torch.tensor([P]), types)

    with torch.no_grad():
        block.mp_gate.zero_()  # remove the MP branch entirely
    sim.reset_rollout()
    with torch.no_grad():
        nxt_off, _ = sim.predict_positions(win, torch.tensor([P]), types)

    assert not torch.allclose(nxt_on, nxt_off), (
        "MP branch had no effect on the one-step prediction"
    )


def test_hybrid_edge_frame_rejects_unknown_value():
    # The frame enum is validated at construction (mirrors the config-load
    # guard), so a typo fails loudly rather than silently defaulting.
    with pytest.raises(ValueError, match="hybrid_edge_frame"):
        _hybrid_sim(hybrid_edge_frame="bogus")


def test_reference_frame_graph_is_static_under_position_perturbation():
    # (iter3 KEY) With hybrid_edge_frame='reference' the MP graph is built from
    # the fixed rest coordinates, so perturbing the CURRENT positions leaves the
    # edge index AND the edge features byte-identical: the graph is truly static
    # across the rollout, which is what kills the predicted-state drift feedback.
    torch.manual_seed(0)
    p = 30
    ref = torch.rand(p, 2)
    xa = torch.rand(p, 2)
    xb = torch.rand(p, 2)  # a genuinely different current-position cloud
    sim = _hybrid_sim(hybrid_edge_frame="reference")
    ei_a, ef_a = sim._build_graph(xa, ref, None)
    ei_b, ef_b = sim._build_graph(xb, ref, None)
    assert ei_a.shape[1] > 0  # non-empty, so the equality is a real check
    assert torch.equal(ei_a, ei_b)  # edge_index invariant to x_t
    assert torch.equal(ef_a, ef_b)  # rest-frame relative-position feats too


def test_current_frame_graph_changes_under_position_perturbation():
    # (unchanged behaviour) With the default 'current' frame the graph tracks the
    # current positions, so a different position cloud gives a different graph.
    torch.manual_seed(0)
    p = 30
    ref = torch.rand(p, 2)
    xa = torch.rand(p, 2)
    xb = torch.rand(p, 2)
    sim = _hybrid_sim()  # default hybrid_edge_frame='current'
    ei_a, _ = sim._build_graph(xa, ref, None)
    ei_b, _ = sim._build_graph(xb, ref, None)
    assert not torch.equal(ei_a, ei_b)  # graph moved with x_t


def test_reference_frame_mp_edge_mlp_receives_gradient():
    # Reference-frame variant of the key grad-flow check: the static-graph MP
    # branch must still receive nonzero gradient (else the gate is a dead no-op
    # even with the rest-frame graph).
    torch.manual_seed(0)
    sim = _hybrid_sim(hybrid_edge_frame="reference")
    p = 12
    x = torch.rand(p, 2)
    nxt = x + 0.05
    aux = torch.rand(p)
    types = torch.zeros(p, dtype=torch.int64)
    ref = torch.rand(p, 2)
    pred, target = sim.forward_train(
        x, nxt, aux, types, ref, torch.tensor([p]), accumulate=False
    )
    loss = ((pred - target) ** 2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    block = _mp_block(sim)
    edge_lin = block.mp.edge_mlp[0]
    assert isinstance(edge_lin, torch.nn.Linear)
    assert edge_lin.weight.grad is not None
    assert float(edge_lin.weight.grad.norm()) > 0.0


def test_hybrid_no_decay_parameters_lists_mp_branch():
    sim = _hybrid_sim()
    no_decay = sim.hybrid_no_decay_parameters()
    assert len(no_decay) > 0
    block = _mp_block(sim)
    ids = {id(p) for p in no_decay}
    assert id(block.mp_gate) in ids
    assert all(id(p) in ids for p in block.mp.parameters())
    # off => empty (vanilla optimizer recipe untouched)
    plain = TransolverSimulator(dim=2, hidden_dim=8, n_layers=4, n_heads=2, slice_num=2)
    assert plain.hybrid_no_decay_parameters() == []
