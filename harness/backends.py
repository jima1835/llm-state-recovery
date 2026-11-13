"""Model backends behind one small interface.

    generate(messages, max_tokens, temperature, seed) -> Generation
    reply_residuals(gen, layers, first_k)             -> {"all": [L, d], "first": [L, d]} float32
                                                         (mean over the whole reply / its first k tokens)
    option_probs(messages, options)                   -> (probs over options, total mass on them)
    option_probs_many(list of messages, options)      -> [option_probs(...) for each], possibly faster
    battery(list of messages, options, gen_messages)  -> (option_probs_many(...), greedy generate(gen_messages))

`reply_residuals` is the "Prior" readout: the residual stream at each layer, averaged over the
tokens of the model's own reply, computed on exactly the token ids the model saw and produced.

MLXBackend runs on Apple Silicon. FakeBackend is deterministic and needs no GPU; it exists so the
pipeline can be tested anywhere. An HF/CUDA backend (Day 5) implements the same three methods.
"""
from __future__ import annotations

import hashlib
import os
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

    def reply_residuals(self, gen: Generation, layers: list[int], first_k: int = 64) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def option_probs(self, messages: list[Message], options: list[str]) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def option_probs_many(self, message_lists: list[list[Message]], options: list[str]) -> list[tuple[np.ndarray, float]]:
        """Same answers as calling option_probs on each; backends may share work across them."""
        return [self.option_probs(m, options) for m in message_lists]

    def battery(self, message_lists, options, gen_messages=None, gen_max_tokens: int = 8, gen_seed: int = 0):
        """All self-report questions at one checkpoint: option probabilities for each questionnaire fork,
        plus one greedy generation (the 0-100 rating). Backends may share work across them."""
        answers = self.option_probs_many(message_lists, options)
        gen = self.generate(gen_messages, gen_max_tokens, 0.0, gen_seed) if gen_messages is not None else None
        return answers, gen

    stai_impl = "per_item"

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

    def reply_residuals(self, gen, layers, first_k=64):
        self.calls["residuals"] += 1
        rng = np.random.default_rng(_h(gen.text, *gen.prompt_ids))
        a = rng.standard_normal((len(layers), self.d_model)).astype(np.float32)
        return {"all": a, "first": a if len(gen.reply_ids) <= first_k else a * 0.5}

    def option_probs(self, messages, options):
        self.calls["option_probs"] += 1
        rng = np.random.default_rng(self._key(messages))
        p = rng.dirichlet(np.ones(len(options)))
        return p, 0.97


def common_prefix_len(prompts: list[list[int]], min_suffix: int = 2) -> int:
    """Length of the longest shared token prefix, leaving at least `min_suffix` tokens in every prompt."""
    limit = min(len(p) for p in prompts) - min_suffix
    first, n = prompts[0], 0
    while n < limit and all(p[n] == first[n] for p in prompts):
        n += 1
    return max(n, 0)


# ---------------------------------------------------------------- MLX (Apple Silicon)

class MLXBackend(Backend):
    def __init__(self, model_id: str, chat_template_kwargs: dict | None = None, answer_prefill: str = "Option",
                 reuse_prefix: bool = True):
        import mlx.core as mx
        from mlx_lens import LensModel

        from .lens_compat import install
        install()  # Gemma 4 layers return a tuple; no change for array-returning layers
        self.mx = mx
        self.name = model_id
        self.lens = LensModel(model_id)
        self.tok = self.lens.tokenizer
        self.n_layers = self.lens.n_layers
        self.d_model = self.lens.d_model
        self.chat_kwargs = dict(chat_template_kwargs or {})
        self.answer_prefill = answer_prefill
        self._option_cache: dict[tuple, tuple[str, list[int]]] = {}
        self.reuse_prefix = reuse_prefix

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
        stream = stream_generate(self.lens.model, self.tok, prompt=mx.array(prompt_ids), max_tokens=max_tokens,
                                 sampler=make_sampler(temp=temperature))
        return self._collect(stream, prompt_ids)

    def _collect(self, stream, prompt_ids) -> Generation:
        reply_ids, finish = [], None
        for r in stream:
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

    def reply_residuals(self, gen, layers, first_k=64):
        mx = self.mx
        if not gen.reply_ids:
            nan = np.full((len(layers), self.d_model), np.nan, dtype=np.float32)
            return {"all": nan, "first": nan.copy()}
        ids = gen.prompt_ids + gen.reply_ids
        start = len(gen.prompt_ids)
        with self.lens.capture(layers=layers) as cache:
            self.lens.forward(ids)
        # cache is filled when the block exits
        reply = [cache[l][0, start:, :].astype(mx.float32) for l in layers]
        out_all = mx.stack([r.mean(axis=0) for r in reply])
        out_first = mx.stack([r[:first_k].mean(axis=0) for r in reply])
        mx.eval(out_all, out_first)
        return {"all": np.array(out_all), "first": np.array(out_first)}

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

    @property
    def stai_impl(self) -> str:
        return "prefix_reuse" if self.reuse_prefix else "per_item"

    def option_probs_many(self, message_lists, options):
        return self.battery(message_lists, options)[0]

    def _battery_per_item(self, message_lists, options, gen_messages, gen_max_tokens, gen_seed):
        """The slow, simple path: one full pass per question. Must not call battery/option_probs_many
        (they route back here when reuse is off)."""
        answers = [self.option_probs(m, options) for m in message_lists]
        gen = self.generate(gen_messages, gen_max_tokens, 0.0, gen_seed) if gen_messages is not None else None
        return answers, gen

    def battery(self, message_lists, options, gen_messages=None, gen_max_tokens: int = 8, gen_seed: int = 0):
        """Read the shared conversation once, then answer each question from a copy of that state.

        All questionnaire forks share the same tokens up to the item text. We run that prefix once into
        the model's key/value cache, then run each item's remaining tokens on a shallow copy of it.
        Safe for the two cache types used by our models (checked in mlx-lm 0.32):
          - KVCache: a suffix only writes past the prefix, and each copy has its own offset, so the
            prefix region is never changed and stale entries beyond the offset are never read.
          - RotatingKVCache (Gemma's sliding-window layers): multi-token updates always build new
            arrays, so the saved prefix is untouched. Suffixes are kept at >= 2 tokens to stay on
            that path.
        Anything else falls back to one full pass per item. harness/check_prefix_reuse.py compares
        the two paths on real conversations.

        Two levels: level 1 is shared by every question including the 0-100 rating (system prompt +
        conversation); level 2 extends it with the questionnaire's own shared start (its preamble).
        The rating is generated LAST: token-by-token generation writes into cache arrays in place,
        which is harmless once nothing else will read them.
        """
        if not self.reuse_prefix or len(message_lists) < 2:
            return self._battery_per_item(message_lists, options, gen_messages, gen_max_tokens, gen_seed)

        import copy

        from mlx_lm import stream_generate
        from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        mx = self.mx
        model = self.lens.model
        prefill, ids = self._option_tokens(tuple(options))
        prompts = [self._ids(m, add_generation_prompt=True, suffix=prefill) for m in message_lists]
        gen_prompt = self._ids(gen_messages, add_generation_prompt=True) if gen_messages is not None else None
        n2 = common_prefix_len(prompts, min_suffix=2)
        n1 = min(n2, common_prefix_len(prompts + [gen_prompt], min_suffix=2)) if gen_prompt else n2
        if n2 - n1 < 2:          # a one-token extension would take the in-place update path; skip level 2
            n2 = n1
        level1 = make_prompt_cache(model)
        if n1 < 16 or not all(type(c) in (KVCache, RotatingKVCache) for c in level1):
            return self._battery_per_item(message_lists, options, gen_messages, gen_max_tokens, gen_seed)

        def ev(cache):
            mx.eval([c.keys for c in cache] + [c.values for c in cache])

        model(mx.array(prompts[0][:n1])[None], cache=level1); ev(level1)
        if n2 > n1:
            level2 = [copy.copy(c) for c in level1]
            model(mx.array(prompts[0][n1:n2])[None], cache=level2); ev(level2)
        else:
            level2 = level1

        sel = mx.array(ids)
        answers = []
        for p in prompts:
            c_i = [copy.copy(c) for c in level2]
            logits = model(mx.array(p[n2:])[None], cache=c_i)[0, -1].astype(mx.float32)
            probs = mx.softmax(logits, axis=-1)[sel]
            mx.eval(probs)
            probs = np.array(probs, dtype=np.float64)
            mass = float(probs.sum())
            answers.append((probs / mass if mass > 0 else np.full(len(ids), np.nan), mass))

        gen = None
        if gen_prompt is not None:
            c_g = [copy.copy(c) for c in level1]
            stream = stream_generate(model, self.tok, prompt=mx.array(gen_prompt[n1:]), max_tokens=gen_max_tokens,
                                     sampler=make_sampler(temp=0.0), prompt_cache=c_g)
            gen = self._collect(stream, gen_prompt)
        return answers, gen

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
        reuse = os.environ.get("LSR_PREFIX_REUSE", "1") != "0"  # implementation switch, not protocol
        return MLXBackend(model_id, kwargs, cfg.get("answer_prefill", "Option"), reuse_prefix=reuse)
    raise ValueError(f"Unknown backend {kind!r}")
