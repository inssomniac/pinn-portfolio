import torch


def uniform_random(n_int, n_bnd, device='cuda'):
    """
    Исходный метод генерации точек.
    Внутренние точки равномерно случайные, для граничных точек - равномерная сетка по каждой из 4 сторон.
    Граница фиксирована, не перегенерируется в процессе обучения.
    """
    x_int = torch.rand(n_int, 1)
    y_int = torch.rand(n_int, 1)
    interior = torch.cat([x_int, y_int], dim=1).to(device)
    interior.requires_grad_(True)

    n_side = n_bnd // 4
    t = torch.linspace(0, 1, n_side).unsqueeze(1)

    b0 = torch.cat([t, torch.zeros(n_side, 1)], dim=1)   # y = 0
    b1 = torch.cat([t, torch.ones(n_side, 1)],  dim=1)   # y = 1
    b2 = torch.cat([torch.zeros(n_side, 1), t], dim=1)   # x = 0
    b3 = torch.cat([torch.ones(n_side, 1),  t], dim=1)   # x = 1
    boundary = torch.cat([b0, b1, b2, b3], dim=0).to(device)

    return interior, boundary


def rad(model, M, residual_fn, k=1.0, c=1.0, n_candidates=10000, device='cuda'):
    """
    Residual-based Adaptive Distribution
    Семплируем точки пропорционально невязке модели,
    концентрируя их там, где ошибка сейчас наибольшая.
    """
    x_cand = torch.rand(n_candidates, 1, device=device)
    y_cand = torch.rand(n_candidates, 1, device=device)
    S0 = torch.cat([x_cand, y_cand], dim=1)
    S0.requires_grad_(True)

    eps = residual_fn(model, S0).abs().squeeze()

    eps_k = eps ** k
    p_unnorm = eps_k + c * eps_k.mean()
    p = p_unnorm / p_unnorm.sum()

    indices = torch.multinomial(p, num_samples=M, replacement=False)
    interior = S0[indices]

    return interior.detach().requires_grad_(True)


def _grad_norm(residual_fn, model, S0):
    """
    Вспомогательная функция: считает норму пространственного градиента невязки.
    Используется в RAD-G и RAR-G вместо самой невязки |r(x)|.
    """
    r = residual_fn(model, S0)
    dr = torch.autograd.grad(r, S0, grad_outputs=torch.ones_like(r), create_graph=False)[0]
    return dr.norm(dim=1)


def rad_g(model, M, residual_fn, k=1.0, c=1.0, n_candidates=10000, device='cuda'):
    """
    Идентичен RAD, но вместо невязки |r(x)| использует норму её
    пространственного градиента. Это позволяет концентрировать
    точки не там, где ошибка велика, а там, где она резко меняется.
    """
    x_cand = torch.rand(n_candidates, 1, device=device)
    y_cand = torch.rand(n_candidates, 1, device=device)
    S0 = torch.cat([x_cand, y_cand], dim=1)
    S0.requires_grad_(True)

    g = _grad_norm(residual_fn, model, S0)

    g_k = g ** k
    p_unnorm = g_k + c * g_k.mean()
    p = p_unnorm / p_unnorm.sum()

    indices = torch.multinomial(p, num_samples=M, replacement=False)
    interior = S0[indices]

    return interior.detach().requires_grad_(True)


def rar_g(model, existing_interior, m, residual_fn, n_candidates=10000, device='cuda'):
    """К существующим точкам добавляются m новых. Набор растёт с каждым вызовом."""
    x_cand = torch.rand(n_candidates, 1, device=device)
    y_cand = torch.rand(n_candidates, 1, device=device)
    S0 = torch.cat([x_cand, y_cand], dim=1)
    S0.requires_grad_(True)

    g = _grad_norm(residual_fn, model, S0)

    # Берём топ-m точек с наибольшим градиентом невязки
    top_indices = torch.topk(g, k=m).indices
    new_points = S0[top_indices].detach()

    # Объединяем с существующим набором
    combined = torch.cat([existing_interior.detach(), new_points], dim=0)
    return combined.requires_grad_(True)
