"""El molde en TPU: entrena en float (JAX) el mismo transformer de caracteres que train.rs
y guarda checkpoints .llg con el formato exacto del Rust. El motor entero sigue en Rust:
el .llg se funde a .lli en casa (int_check) y ahí no hay ni un float.

Subcomandos:
  corpus   descarga Wikipedia es (todos los shards menos el 10, del que sale valid.bin)
           y la codifica con vocab.txt -> train.bin
  train    entrena (float o QAT con STE) y guarda el mejor .llg según valid
  paridad  pérdida en valid de un .llg (para comparar con eval_llg de Rust)

Configuración por variables de entorno, como train.rs:
  D L H CTX BATCH STEPS LR WARMUP WD SEED EVAL_EVERY EVAL_CHARS OUT INIT QAT
  DATA (carpeta con vocab.txt, valid.bin, train.bin)  TRAIN_CHARS (para corpus)
"""
import json, math, os, re, struct, sys, time, urllib.request

import numpy as np


def env(k, d):
    v = os.environ.get(k)
    return type(d)(v) if v is not None else d


DATA = env("DATA", "data")

# ---------------------------------------------------------------- corpus

HF = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.es/train-{:05d}-of-00013.parquet"
CUT = re.compile(r"\n(Referencias|Enlaces externos|Bibliografía|Véase también|Notas|Fuentes)\s*\n")


def load_vocab():
    with open(os.path.join(DATA, "vocab.txt"), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def corpus():
    import pyarrow.parquet as pq

    vocab = load_vocab()
    idx = {c: i for i, c in enumerate(vocab)}
    unk = len(vocab) - 1
    table = np.full(0x110000, unk, dtype=np.uint8)
    for c, i in idx.items():
        table[ord(c)] = i
    target = env("TRAIN_CHARS", 1_500_000_000)
    out_path = os.path.join(DATA, "train.bin")
    n = 0
    with open(out_path, "wb") as out:
        for shard in [s for s in range(13) if s != 10]:
            if n >= target:
                break
            tmp = f"/tmp/wiki_{shard}.parquet"
            print(f"shard {shard}: descargando", flush=True)
            urllib.request.urlretrieve(HF.format(shard), tmp)
            texts = pq.read_table(tmp, columns=["text"]).column("text").to_pylist()
            os.remove(tmp)
            docs = []
            for t in texts:
                m = CUT.search(t)
                if m:
                    t = t[: m.start()]
                t = re.sub(r"[ \t]+\n", "\n", t)
                t = re.sub(r"\n{3,}", "\n\n", t).strip()
                if len(t) >= 500:
                    docs.append(t)
            s = "\n\n".join(docs) + "\n\n"
            ids = table[np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)]
            ids = ids[: max(0, target - n)]
            out.write(ids.tobytes())
            n += len(ids)
            print(f"shard {shard}: {len(docs)} docs, total {n/1e6:.0f}M caracteres, unk {np.mean(ids == unk):.5f}", flush=True)
    print("train.bin listo:", n)


# ---------------------------------------------------------------- modelo (espejo de model.rs)

import jax
import jax.numpy as jnp


def layout(c):
    """Lista (nombre, forma) en el orden exacto de Layout::new en model.rs."""
    v, T, d = c["vocab"], c["ctx"], c["d"]
    lay = [("wte", (v, d)), ("wpe", (T, d))]
    for l in range(c["layers"]):
        lay += [
            (f"l{l}.ln1_g", (d,)), (f"l{l}.ln1_b", (d,)),
            (f"l{l}.qkv_w", (d, 3 * d)), (f"l{l}.qkv_b", (3 * d,)),
            (f"l{l}.proj_w", (d, d)), (f"l{l}.proj_b", (d,)),
            (f"l{l}.ln2_g", (d,)), (f"l{l}.ln2_b", (d,)),
            (f"l{l}.fc_w", (d, 4 * d)), (f"l{l}.fc_b", (4 * d,)),
            (f"l{l}.fc2_w", (4 * d, d)), (f"l{l}.fc2_b", (d,)),
        ]
    lay += [("lnf_g", (d,)), ("lnf_b", (d,))]
    return lay


MATS = lambda c: ["wte"] + [f"l{l}.{m}" for l in range(c["layers"]) for m in ("qkv_w", "proj_w", "fc_w", "fc2_w")]


def save_llg(path, c, p):
    with open(path + ".tmp", "wb") as f:
        f.write(b"LLG1")
        f.write(struct.pack("<5I", c["vocab"], c["ctx"], c["d"], c["layers"], c["heads"]))
        for name, _ in layout(c):
            f.write(np.asarray(p[name], dtype="<f4").tobytes())
    os.replace(path + ".tmp", path)


def load_llg(path):
    b = open(path, "rb").read()
    assert b[:4] == b"LLG1", "checkpoint inválido"
    v, T, d, L, H = struct.unpack("<5I", b[4:24])
    c = dict(vocab=v, ctx=T, d=d, layers=L, heads=H)
    flat = np.frombuffer(b[24:], dtype="<f4")
    p, o = {}, 0
    for name, shape in layout(c):
        k = int(np.prod(shape))
        p[name] = flat[o:o + k].reshape(shape).copy()
        o += k
    assert o == len(flat), "tamaño no calza con la config"
    return c, p


def init_params(c, seed):
    rng = np.random.default_rng(seed)
    d, L = c["d"], c["layers"]
    ps = 0.02 / math.sqrt(2 * L)
    p = {}
    for name, shape in layout(c):
        base = name.split(".")[-1]
        if base in ("ln1_g", "ln2_g", "lnf_g"):
            p[name] = np.ones(shape, np.float32)
        elif base.endswith("_b"):
            p[name] = np.zeros(shape, np.float32)
        else:
            std = {"wte": 0.02, "wpe": 0.01, "proj_w": ps, "fc2_w": ps}.get(base, 0.02)
            p[name] = (rng.standard_normal(shape) * std).astype(np.float32)
    return p


def fmt_fn(name):
    """Cuantización simulada (misma regla que quant.rs) sobre bloques contiguos del row-major."""
    if not name:
        return None
    m = re.fullmatch(r"ternario-g(\d+)", name)
    if m:
        g = int(m.group(1))

        def q(w):
            b = w.reshape(-1, g)
            a = jnp.abs(b)
            mask = a > 0.7 * a.mean(axis=1, keepdims=True)
            cnt = mask.sum(axis=1, keepdims=True)
            s = jnp.where(cnt > 0, (a * mask).sum(axis=1, keepdims=True) / jnp.maximum(cnt, 1), 0.0)
            return (jnp.sign(b) * s * mask).reshape(w.shape)

        return q
    m = re.fullmatch(r"binario-g(\d+)", name)
    if m:
        g = int(m.group(1))

        def q(w):
            b = w.reshape(-1, g)
            s = jnp.abs(b).mean(axis=1, keepdims=True)
            return jnp.where(b >= 0, s, -s).reshape(w.shape)

        return q
    m = re.fullmatch(r"trit-e(\d+)-g(\d+)", name)
    if m:
        eb, g = int(m.group(1)), int(m.group(2))

        def q(w):
            b = w.reshape(-1, g)
            a = jnp.abs(b)
            mask = a > 0.7 * a.mean(axis=1, keepdims=True)
            cnt = mask.sum(axis=1, keepdims=True)
            s = jnp.where(cnt > 0, (a * mask).sum(axis=1, keepdims=True) / jnp.maximum(cnt, 1), 0.0)
            e = jnp.where(s > 0, jnp.round(jnp.log2(jnp.maximum(s, 1e-30))), -1e9)
            emax = e.max()
            s2 = jnp.where(s > 0, jnp.exp2(jnp.maximum(e, emax - (2 ** eb - 1))), 0.0)
            return (jnp.sign(b) * s2 * (a > 0.5 * s2) * (s2 > 0)).reshape(w.shape)

        return q
    raise ValueError(f"formato desconocido: {name}")


def quantize(c, p, q):
    """Pesos vistos por el forward. Con q: STE (adelante cuantizado, gradiente directo al maestro)."""
    if q is None:
        return p
    p = dict(p)
    for n in MATS(c):
        w = p[n]
        p[n] = w + jax.lax.stop_gradient(q(w) - w)
    return p


def ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + 1e-5) * g + b


def forward(c, p, inp):
    """inp (B,T) int32 -> logits (B,T,V)."""
    B, T = inp.shape
    d, H = c["d"], c["heads"]
    hd = d // H
    x = p["wte"][inp] + p["wpe"][:T]
    mask = jnp.tril(jnp.ones((T, T), bool))
    for l in range(c["layers"]):
        P = lambda k: p[f"l{l}.{k}"]
        h = ln(x, P("ln1_g"), P("ln1_b"))
        qkv = h @ P("qkv_w") + P("qkv_b")
        q, k, v = (qkv[..., i * d:(i + 1) * d].reshape(B, T, H, hd) for i in range(3))
        att = jnp.einsum("bihc,bjhc->bhij", q, k) / math.sqrt(hd)
        att = jax.nn.softmax(jnp.where(mask, att, -1e30), axis=-1)
        ao = jnp.einsum("bhij,bjhc->bihc", att, v).reshape(B, T, d)
        x = x + ao @ P("proj_w") + P("proj_b")
        h = ln(x, P("ln2_g"), P("ln2_b"))
        x = x + jax.nn.gelu(h @ P("fc_w") + P("fc_b"), approximate=True) @ P("fc2_w") + P("fc2_b")
    x = ln(x, p["lnf_g"], p["lnf_b"])
    return x @ p["wte"].T


def loss_fn(c, p, seq, q):
    logits = forward(c, quantize(c, p, q), seq[:, :-1])
    lp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return -jnp.take_along_axis(lp, seq[:, 1:, None], axis=-1).mean()


def evaluate(c, p, valid, chars, q, eval_fn, bs=64):
    """Igual que eval.rs: ventanas contiguas de ctx sin solape, media de la pérdida por ventana."""
    T = c["ctx"]
    n_win = min(chars, len(valid) - 1) // T
    tot = 0.0
    for s in range(0, n_win, bs):
        ks = np.arange(s, min(s + bs, n_win))
        seq = np.stack([valid[k * T:k * T + T + 1] for k in ks]).astype(np.int32)
        tot += float(eval_fn(p, seq)) * len(ks)
    return tot / n_win


def train():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    init = env("INIT", "")
    if init:
        c, p = load_llg(init)
    else:
        c = dict(vocab=len(load_vocab()), ctx=env("CTX", 256), d=env("D", 512), layers=env("L", 8), heads=env("H", 8))
        p = init_params(c, env("SEED", 1))
    batch, steps, lr_max = env("BATCH", 256), env("STEPS", 20000), env("LR", 1e-3)
    warmup, wd, seed = env("WARMUP", 300), env("WD", 0.1), env("SEED", 1)
    eval_every, eval_chars = env("EVAL_EVERY", 1000), env("EVAL_CHARS", 2_000_000)
    out, qname = env("OUT", "/kaggle/working/molde.llg"), env("QAT", "")
    q = fmt_fn(qname)
    train_d = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint8, mode="r")
    valid = np.fromfile(os.path.join(DATA, "valid.bin"), dtype=np.uint8)
    T = c["ctx"]
    npar = sum(int(np.prod(s)) for _, s in layout(c))
    devs = jax.devices()
    print(f"config {c} params {npar} batch {batch} steps {steps} lr {lr_max} init '{init}' qat '{qname}' "
          f"train {len(train_d)/1e6:.0f}M dispositivos {len(devs)}x{devs[0].platform}", flush=True)

    mesh = Mesh(np.array(devs), ("data",))
    rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("data"))
    p = jax.device_put({k: jnp.asarray(v) for k, v in p.items()}, rep)
    mats = set(MATS(c))
    decay = {k: (1.0 if k in mats else 0.0) for k in p}
    st = dict(m=jax.tree.map(jnp.zeros_like, p), v=jax.tree.map(jnp.zeros_like, p))
    b1, b2, eps = 0.9, 0.95, 1e-8

    @jax.jit
    def step_fn(p, st, seq, lr, t):
        loss, g = jax.value_and_grad(lambda pp: loss_fn(c, pp, seq, q))(p)
        gn = jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32) ** 2) for x in jax.tree.leaves(g)))
        clip = jnp.where(gn > 1.0, 1.0 / gn, 1.0)
        m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g * clip, st["m"], g)
        v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * (g * clip) ** 2, st["v"], g)
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        p = {k: p[k] - lr * ((m[k] / bc1) / (jnp.sqrt(v[k] / bc2) + eps) + wd * decay[k] * p[k]) for k in p}
        return p, dict(m=m, v=v), loss, gn

    eval_fn = jax.jit(lambda p, seq: loss_fn(c, p, seq, q))
    rng = np.random.default_rng(seed ^ 0xABCDEF)
    best, t0, ema = float("inf"), time.time(), None
    for step in range(1, steps + 1):
        offs = rng.integers(0, len(train_d) - T - 1, size=batch)
        seq = np.stack([train_d[o:o + T + 1] for o in offs]).astype(np.int32)
        seq = jax.device_put(seq, shard)
        if step <= warmup:
            lr = lr_max * step / warmup
        else:
            pr = (step - warmup) / max(steps - warmup, 1)
            lr = lr_max * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * pr)))
        p, st, loss, gn = step_fn(p, st, seq, jnp.float32(lr), jnp.float32(step))
        if step % 50 == 0 or step == 1:
            loss = float(loss)
            ema = loss if ema is None else 0.98 ** 50 * ema + (1 - 0.98 ** 50) * loss
            el = time.time() - t0
            print(f"step {step:6} loss {loss:.4f} ema {ema:.4f} bpc~{ema/math.log(2):.3f} gnorm {float(gn):.2f} "
                  f"lr {lr:.2e} tok/s {step*batch*T/el:.0f} t {el:.0f}s", flush=True)
        if step % eval_every == 0 or step == steps:
            vl = evaluate(c, p, valid, eval_chars, q, eval_fn)
            mark = ""
            if vl < best:
                best = vl
                save_llg(out, c, jax.device_get(p))
                mark = " *guardado"
            print(f"EVAL step {step} valid_loss {vl:.4f} valid_bpc {vl/math.log(2):.4f}{mark}", flush=True)
    print(f"fin: mejor valid_bpc {best/math.log(2):.4f} en {time.time()-t0:.0f}s")


def paridad():
    c, p = load_llg(env("MODEL", "runs/base.llg"))
    q = fmt_fn(env("FMT", ""))
    valid = np.fromfile(os.path.join(DATA, "valid.bin"), dtype=np.uint8)
    jax.config.update("jax_default_matmul_precision", "highest")
    p = {k: jnp.asarray(v) for k, v in p.items()}
    eval_fn = jax.jit(lambda p, seq: loss_fn(c, p, seq, q))
    vl = evaluate(c, p, valid, env("EVAL_CHARS", 100_000), q, eval_fn)
    print(f"JAX: valid_loss {vl:.6f} valid_bpc {vl/math.log(2):.6f}")


if __name__ == "__main__":
    {"corpus": corpus, "train": train, "paridad": paridad}[sys.argv[1]]()
