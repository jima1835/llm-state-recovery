"""Model backends behind one small interface.

    generate(messages, max_tokens, temperature, seed) -> Generation
    reply_residuals(gen, layers)                      -> np.ndarray [len(layers), d_model], float32
    option_probs(messages, options)                   -> (probs over options, total mass on them)

`reply_residuals` is the "Prior" readout: the residual stream at each layer, averaged over the
tokens of the model's own reply, computed on exactly the token ids the model saw and produced.

MLXBackend runs on Apple Silicon. FakeBackend is deterministic and needs no GPU; it exists so the
pipeline can be tested anywhere. An HF/CUDA backend (Day 5) implements the same three methods.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


@dataclass
class Generation:
    text: str
    prompt_ids: list[int]
    reply_ids: list[int]
    finish_reason: str
    meta: dict = field(default_factory=dict)


class Backend:
    name: str
    n_layers: int
    d_model: int

    def generate(self, messages: list[Message], max_tokens: int, temperature: float, seed: int) -> Generation:
        raise NotImplementedError

    def reply_residuals(self, gen: Generation, layers: list[int]) -> np.ndarray:
        raise NotImplementedError

    def option_probs(self, messages: list[Message], options: list[str]) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- fake (tests, dry runs)

def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class FakeBackend(Backend):
    """Deterministic stand-in. Residuals and answers are functions of the input text only."""

    def __init__(self, model_id: str = "fake", n_layers: int = 6, d_model: int = 16):
        self.name, self.n_layers, self.d_model = model_id, n_layers, d_model
        self.calls = {"generate": 0, "residuals": 0, "option_probs": 0}

    def _key(self, messages):
        return _h(self.name, *[m["role"] + m["content"] for m in messages])

    def generate(self, messages, max_tokens, temperature, seed):
        self.calls["generate"] += 1
        k = _h(self._key(messages), seed, temperature)
        n = 3 + k % 5
        words = [f"w{(k >> i) % 97}" for i in range(n)][:max_tokens]
        return Generation(" ".join(words), [k % 1000] * 4, list(range(len(words))), "stop")

    def reply_residuals(self, gen, layers):
        self.calls["residuals"] += 1
        rng = np.random.default_rng(_h(gen.text, *gen.prompt_ids))
        return rng.standard_normal((len(layers), self.d_model)).astype(np.float32)

    def option_probs(self, messages, options):
        self.calls["option_probs"] += 1
        rng = np.random.default_rng(self._key(messages))
        p = rng.dirichlet(np.ones(len(options)))
        return p, 0.97


# ---------------------------------------------------------------- MLX (Apple Silicon)

class MLXBackend(Backend):
    def __init__(self, model_id: str, chat_template_kwargs: dict | None = None, answer_prefill: str = "Option"):
        import mlx.core as mx
        from mlx_lens import LensModel

        self.mx = mx
        self.name = model_id
        self.lens = LensModel(model_id)
        self.tok = self.lens.tokenizer
        self.n_layers = self.lens.n_layers
        self.d_model = self.lens.d_model
        self.chat_kwargs = dict(chat_template_kwargs or {})
        self.answer_prefill = answer_prefill
        self._option_cache: dict[tuple, tuple[str, list[int]]] = {}

    # -- tokenisation helpers
    def _ids(self, messages, add_generation_prompt: bool, suffix: str = "") -> list[int]:
        text = self.tok.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=False, **self.chat_kwargs
        )
        # Chat templates already insert BOS where the model needs it.
        return list(self.tok.encode(text + suffix, add_special_tokens=False))

    def _option_tokens(self, options: tuple[str, ...]) -> tuple[str, list[int]]:
        """Pick an assistant prefill and one token id per option such that prefill+option adds exactly
        one token. Tries "Option " then "Option" (+ " 1") then no prefill."""
        if options in self._option_cache:
            return self._option_cache[options]
        enc = lambda s: list(self.tok.encode(s, add_special_tokens=False))  # noqa: E731
        base = self.answer_prefill
        for prefill, sep in ((base + " ", ""), (base, " "), ("", "")):
            head = enc("\n" + prefill)  # chat templates end the generation prompt with a newline
            ids = []
            for o in options:
                full = enc("\n" + prefill + sep + o)
                if full[: len(head)] == head and len(full) == len(head) + 1:
                    ids.append(full[-1])
            if len(ids) == len(options) and len(set(ids)) == len(ids):
                # store the prefill including the separator the model would emit itself
                self._option_cache[options] = (prefill, ids)
                return prefill, ids
        raise RuntimeError(f"No single-token encoding found for options {options} with {self.name}")

    # -- interface
    def generate(self, messages, max_tokens, temperature, seed):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        mx = self.mx
        mx.random.seed(seed)
        prompt_ids = self._ids(messages, add_generation_prompt=True)
        reply_ids, finish = [], None
        for r in stream_generate(
            self.lens.model, self.tok, prompt=mx.array(prompt_ids), max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        ):
            if r.finish_reason is None:
                reply_ids.append(int(r.token))
            else:
                finish = r.finish_reason
                # On "length" the last yield carries a new, real token; on "stop" it is the EOS token.
                if finish == "length":
                    reply_ids.append(int(r.token))
        text = self.tok.decode(reply_ids)
        return Generation(text.strip(), prompt_ids, reply_ids, finish or "stop",
                          {"prompt_tokens": len(prompt_ids), "reply_tokens": len(reply_ids)})

    def reply_residuals(self, gen, layers):
        mx = self.mx
        if not gen.reply_ids:
            return np.full((len(layers), self.d_model), np.nan, dtype=np.float32)
        ids = gen.prompt_ids + gen.reply_ids
        start = len(gen.prompt_ids)
        with self.lens.capture(layers=layers) as cache:
            self.lens.forward(ids)
        # cache is filled when the block exits
        out = mx.stack([cache[l][0, start:, :].astype(mx.float32).mean(axis=0) for l in layers])
        mx.eval(out)
        return np.array(out)

    def option_probs(self, messages, options):
        mx = self.mx
        prefill, ids = self._option_tokens(tuple(options))
        prompt = self._ids(messages, add_generation_prompt=True, suffix=prefill)
        logits = self.lens.forward(prompt)[0, -1].astype(mx.float32)
        p = mx.softmax(logits, axis=-1)
        sel = p[mx.array(ids)]
        mx.eval(sel)
        sel = np.array(sel, dtype=np.float64)
        mass = float(sel.sum())
        return (sel / mass if mass > 0 else np.full(len(ids), np.nan)), mass

    def peak_memory_gb(self) -> float:
        return float(self.mx.get_peak_memory()) / 1e9

    def reset_peak_memory(self) -> None:
        self.mx.reset_peak_memory()


def make_backend(kind: str, model_id: str, cfg: dict) -> Backend:
    if kind == "fake":
        return FakeBackend(model_id)
    if kind == "mlx":
        kwargs = {}
        if cfg.get("qwen3_thinking") is False:
            kwargs["enable_thinking"] = False  # ignored by templates that don't use it
        return MLXBackend(model_id, kwargs, cfg.get("answer_prefill", "Option"))
    raise ValueError(f"Unknown backend {kind!r}")
