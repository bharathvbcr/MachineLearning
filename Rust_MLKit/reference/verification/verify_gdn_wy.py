import torch
import time


def gdn_naive(q, k, v, alpha, beta):
    B, H, L, D = k.shape
    S = torch.zeros(B, H, D, D, device=k.device, dtype=k.dtype)
    outputs = []
    for t in range(L):
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]
        q_t = q[:, :, t, :]
        a_t = alpha[:, :, t]
        b_t = beta[:, :, t]

        v_hat = (S @ k_t.unsqueeze(-1)).squeeze(-1)
        e_t = v_t - v_hat
        S = a_t.unsqueeze(-1).unsqueeze(-1) * S + b_t.unsqueeze(-1).unsqueeze(-1) * (
            e_t.unsqueeze(-1) @ k_t.unsqueeze(-2)
        )
        y_t = (S @ q_t.unsqueeze(-1)).squeeze(-1)
        outputs.append(y_t)
    return torch.stack(outputs, dim=2)


def fwd_intra_chunk(k, v, q, alpha, beta, S_in):
    """
    Computes exact outputs and final state for a single chunk.
    Inputs: [B, H, C, D]
    S_in: [B, H, D, D]
    """
    C = k.shape[2]
    S = S_in
    outputs = []

    for t in range(C):
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]
        q_t = q[:, :, t, :]
        a_t = alpha[:, :, t]
        b_t = beta[:, :, t]

        v_hat = (S @ k_t[..., None]).squeeze(-1)
        e_t = v_t - v_hat
        S = a_t[..., None, None] * S + b_t[..., None, None] * (
            e_t[..., None] @ k_t[..., None, :]
        )
        y_t = (S @ q_t[..., None]).squeeze(-1)
        outputs.append(y_t)

    return torch.stack(outputs, dim=2), S


def bwd_intra_chunk(k, v, q, alpha, beta, S_in, grad_y, grad_S_out):
    """
    Computes exact gradients for a single chunk.
    Inputs/Grads: [B, H, C, D]
    S_in, grad_S_out: [B, H, D, D]
    """
    C = k.shape[2]

    # 1. Forward pass to collect intermediate states
    S = S_in
    states = [S]
    errors = []
    for t in range(C):
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]
        a_t = alpha[:, :, t]
        b_t = beta[:, :, t]

        v_hat = (S @ k_t[..., None]).squeeze(-1)
        e_t = v_t - v_hat
        errors.append(e_t)

        S = a_t[..., None, None] * S + b_t[..., None, None] * (
            e_t[..., None] @ k_t[..., None, :]
        )
        states.append(S)

    # 2. Backward pass
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)
    grad_q = torch.zeros_like(q)
    grad_alpha = torch.zeros_like(alpha)
    grad_beta = torch.zeros_like(beta)

    dS = grad_S_out.clone()

    for t in reversed(range(C)):
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]
        q_t = q[:, :, t, :]
        a_t = alpha[:, :, t]
        b_t = beta[:, :, t]
        e_t = errors[t]
        S_prev = states[t]
        S_cur = states[t + 1]

        dy_t = grad_y[:, :, t, :]

        # y_t = q_t^T S_cur  =>  grad_q = S_cur^T dy_t
        grad_q[:, :, t, :] = (S_cur.transpose(-1, -2) @ dy_t[..., None]).squeeze(-1)

        # dS gets contribution from dy_t
        dS += dy_t[..., None] @ q_t[..., None, :]

        # S_cur = a_t S_prev + b_t e_t k_t^T
        # d(a_t) = tr(dS^T S_prev)
        grad_alpha[:, :, t] = (dS * S_prev).sum(dim=(-1, -2))

        # d(b_t) = tr(dS^T e_t k_t^T)
        ekT = e_t[..., None] @ k_t[..., None, :]
        grad_beta[:, :, t] = (dS * ekT).sum(dim=(-1, -2))

        # d(ekT) = b_t * dS
        dekT = b_t[..., None, None] * dS

        # ekT = e_t k_t^T  => de_t = dekT @ k_t
        de_t = (dekT @ k_t[..., None]).squeeze(-1)
        # dk_t_from_ekT = dekT^T @ e_t
        dk_t_1 = (dekT.transpose(-1, -2) @ e_t[..., None]).squeeze(-1)

        # e_t = v_t - S_prev k_t
        grad_v[:, :, t, :] = de_t
        # dk_t_from_e = - S_prev^T @ de_t
        dk_t_2 = -(S_prev.transpose(-1, -2) @ de_t[..., None]).squeeze(-1)
        grad_k[:, :, t, :] = dk_t_1 + dk_t_2

        # dS_prev = a_t * dS - de_t k_t^T
        dS = a_t[..., None, None] * dS - de_t[..., None] @ k_t[..., None, :]

    return grad_k, grad_v, grad_q, grad_alpha, grad_beta, dS


class GDNExactFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, chunk_size):
        B, H, L, D = k.shape
        N = L // chunk_size

        ctx.chunk_size = chunk_size

        q_c = q.view(B, H, N, chunk_size, D)
        k_c = k.view(B, H, N, chunk_size, D)
        v_c = v.view(B, H, N, chunk_size, D)
        a_c = alpha.view(B, H, N, chunk_size)
        b_c = beta.view(B, H, N, chunk_size)

        S = torch.zeros((B, H, D, D), device=k.device, dtype=k.dtype)
        outputs = []
        S_carries = [S]

        for m in range(N):
            y_c, S = fwd_intra_chunk(
                k_c[:, :, m], v_c[:, :, m], q_c[:, :, m], a_c[:, :, m], b_c[:, :, m], S
            )
            outputs.append(y_c)
            S_carries.append(S)

        ctx.save_for_backward(q, k, v, alpha, beta)
        ctx.S_carries = S_carries  # Only N * [B, H, D, D] = very small!

        return torch.cat(outputs, dim=2)

    @staticmethod
    def backward(ctx, grad_y):
        q, k, v, alpha, beta = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        S_carries = ctx.S_carries

        B, H, L, D = k.shape
        N = L // chunk_size

        q_c = q.view(B, H, N, chunk_size, D)
        k_c = k.view(B, H, N, chunk_size, D)
        v_c = v.view(B, H, N, chunk_size, D)
        a_c = alpha.view(B, H, N, chunk_size)
        b_c = beta.view(B, H, N, chunk_size)
        grad_y_c = grad_y.view(B, H, N, chunk_size, D)

        grad_k = torch.zeros_like(k_c)
        grad_v = torch.zeros_like(v_c)
        grad_q = torch.zeros_like(q_c)
        grad_alpha = torch.zeros_like(a_c)
        grad_beta = torch.zeros_like(b_c)

        dS = torch.zeros((B, H, D, D), device=k.device, dtype=k.dtype)

        for m in reversed(range(N)):
            gk, gv, gq, ga, gb, dS = bwd_intra_chunk(
                k_c[:, :, m],
                v_c[:, :, m],
                q_c[:, :, m],
                a_c[:, :, m],
                b_c[:, :, m],
                S_carries[m],
                grad_y_c[:, :, m],
                dS,
            )
            grad_k[:, :, m] = gk
            grad_v[:, :, m] = gv
            grad_q[:, :, m] = gq
            grad_alpha[:, :, m] = ga
            grad_beta[:, :, m] = gb

        return (
            grad_q.view(B, H, L, D),
            grad_k.view(B, H, L, D),
            grad_v.view(B, H, L, D),
            grad_alpha.view(B, H, L),
            grad_beta.view(B, H, L),
            None,
        )


def gdn_chunked(q, k, v, alpha, beta, chunk_size=32):
    return GDNExactFunction.apply(q, k, v, alpha, beta, chunk_size)


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, L, D = 4, 4, 2048, 64
    chunk_size = 32
    print(f"B={B}, H={H}, L={L}, D={D}, Chunk={chunk_size}")

    q = torch.randn(B, H, L, D, device=device, dtype=torch.float64) / (D**0.5)
    k = torch.randn(B, H, L, D, device=device, dtype=torch.float64) / (D**0.5)
    v = torch.randn(B, H, L, D, device=device, dtype=torch.float64) / (D**0.5)
    alpha = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float64))
    beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float64))

    q_n = q.clone().detach().requires_grad_()
    k_n = k.clone().detach().requires_grad_()
    v_n = v.clone().detach().requires_grad_()
    a_n = alpha.clone().detach().requires_grad_()
    b_n = beta.clone().detach().requires_grad_()

    q_c = q.clone().detach().requires_grad_()
    k_c = k.clone().detach().requires_grad_()
    v_c = v.clone().detach().requires_grad_()
    a_c = alpha.clone().detach().requires_grad_()
    b_c = beta.clone().detach().requires_grad_()

    # Naive
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    y_naive = gdn_naive(q_n, k_n, v_n, a_n, b_n)
    loss_naive = y_naive.sum()
    loss_naive.backward()
    t1 = time.time()
    print(
        f"Naive: {t1 - t0:.3f}s, Mem: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB"
    )

    # Chunked
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    y_chunked = gdn_chunked(q_c, k_c, v_c, a_c, b_c, chunk_size)
    loss_chunked = y_chunked.sum()
    loss_chunked.backward()
    t1 = time.time()
    print(
        f"Chunked: {t1 - t0:.3f}s, Mem: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB"
    )

    print(f"Max diff Y: {(y_naive - y_chunked).abs().max().item()}")
    print(f"Max diff dQ: {(q_n.grad - q_c.grad).abs().max().item()}")
    print(f"Max diff dK: {(k_n.grad - k_c.grad).abs().max().item()}")
    print(f"Max diff dV: {(v_n.grad - v_c.grad).abs().max().item()}")
    print(f"Max diff dAlpha: {(a_n.grad - a_c.grad).abs().max().item()}")
    print(f"Max diff dBeta: {(b_n.grad - b_c.grad).abs().max().item()}")


if __name__ == "__main__":
    test()
