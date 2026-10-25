import copy, math
import numpy as np
import torch
import torch.nn as nn

ce = nn.CrossEntropyLoss()


def _logits(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def eval_loss_acc(model, loader, device):
    model.eval()
    tot_l = tot_c = n = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        o = _logits(model, x)
        tot_l += ce(o, y).item() * len(y)
        tot_c += (o.argmax(1) == y).sum().item()
        n += len(y)
    return tot_l / n, 100.0 * tot_c / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        ps.append(_logits(model, x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    fs, ys = [], []
    for x, y, _ in loader:
        fs.append(model.extract(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(fs).numpy(), torch.cat(ys).numpy()


def train_baseline(model, train_loader, val_loader, device, epochs=40, lr=1e-3,
                   weight_decay=1e-4, patience=8):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_loss, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        tl = tc = n = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            o = model(x)
            loss = ce(o, y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y); tc += (o.argmax(1) == y).sum().item(); n += len(y)
        vl, va = eval_loss_acc(model, val_loader, device)
        sched.step(vl)
        for k, v in zip(hist, [tl / n, 100 * tc / n, vl, va]):
            hist[k].append(v)
        print(f"ep {ep+1:02d} | train {tl/n:.3f}/{100*tc/n:.1f}% | val {vl:.3f}/{va:.1f}%")
        if vl < best_loss:
            best_loss, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print("early stopping"); break
    model.load_state_dict(best_state)       # seleksi model hanya dari VAL SOURCE (bukan target)
    return hist


def train_dann(model, src_loader, val_loader, tgt_loader, device, epochs=40, lr=1e-3,
               weight_decay=1e-4, lambda_adv=1.0, patience=8):
    """Label gestur target TIDAK dipakai; target hanya untuk domain loss."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "dom_loss", "dom_acc"]}
    best_loss, best_state, bad = float("inf"), None, 0
    n_iter, it = epochs * len(src_loader), 0
    tgt_iter = iter(tgt_loader)
    for ep in range(epochs):
        model.train()
        tl = tc = n = dl = da = nb = 0
        for xs, ys, _ in src_loader:
            try:
                xt, _, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader); xt, _, _ = next(tgt_iter)
            alpha = lambda_adv * (2.0 / (1.0 + math.exp(-10.0 * it / n_iter)) - 1.0)
            it += 1
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            cls_out, dom_out = model(torch.cat([xs, xt]), alpha)
            d_lbl = torch.cat([torch.zeros(len(xs)), torch.ones(len(xt))]).long().to(device)
            l_cls = ce(cls_out[:len(xs)], ys)
            l_dom = ce(dom_out, d_lbl)
            opt.zero_grad()
            (l_cls + l_dom).backward()
            opt.step()
            tl += l_cls.item() * len(ys); tc += (cls_out[:len(xs)].argmax(1) == ys).sum().item(); n += len(ys)
            dl += l_dom.item(); da += (dom_out.argmax(1) == d_lbl).float().mean().item(); nb += 1
        vl, va = eval_loss_acc(model, val_loader, device)
        sched.step(vl)
        for k, v in zip(hist, [tl / n, 100 * tc / n, vl, va, dl / nb, 100 * da / nb]):
            hist[k].append(v)
        print(f"ep {ep+1:02d} | cls {tl/n:.3f}/{100*tc/n:.1f}% | val {vl:.3f}/{va:.1f}% | "
              f"dom {dl/nb:.3f}/{100*da/nb:.1f}%")
        if vl < best_loss:
            best_loss, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print("early stopping"); break
    model.load_state_dict(best_state)
    return hist
