"""Transformer estilo SmolLM2 y denoisers independientes para pruebas de viabilidad."""
from __future__ import annotations

import math
from statistics import NormalDist

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Normalización RMS sin estadísticas del dataset."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normaliza en float32 y conserva el dtype de entrada."""
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def attention_mask(length: int, diffusion: bool, device: torch.device) -> Tensor:
    """True permite atención; denoising ve pasado limpio y solo su propio ruido."""
    causal = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    if not diffusion:
        return causal
    mask = torch.zeros(2 * length, 2 * length, dtype=torch.bool, device=device)
    mask[:length, :length] = causal
    mask[length:, :length] = causal.tril(-1)
    mask[length:, length:] = torch.eye(length, dtype=torch.bool, device=device)
    return mask


class Layer(nn.Module):
    """Atención GQA con RoPE, RMSNorm y MLP SwiGLU."""

    def __init__(self, cfg: dict, conditioned: bool) -> None:
        super().__init__()
        d, f = cfg['hidden_size'], cfg['intermediate_size']
        self.heads, self.kv_heads = cfg['num_attention_heads'], cfg['num_key_value_heads']
        self.head_dim = d // self.heads
        if d % self.heads or self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError('Dimensiones incompatibles para GQA/RoPE')
        self.norm1 = RMSNorm(d, cfg['rms_norm_eps'])
        self.norm2 = RMSNorm(d, cfg['rms_norm_eps'])
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, f, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)
        self.modulation = nn.Linear(cfg['condition_dim'], 6 * d) if conditioned else None
        self.register_buffer('freq', 1 / cfg['rope_theta'] ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim), persistent=False)

    def _rope(self, x: Tensor, positions: Tensor) -> Tensor:
        angles = positions.float()[:, None] * self.freq[None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        first, second = x.chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)

    def forward(self, x: Tensor, mask: Tensor, positions: Tensor, cond: Tensor | None) -> Tensor:
        """Aplica una capa; el condicionamiento usa desplazamiento, escala y gates."""
        if self.modulation is not None:
            if cond is None:
                raise ValueError('Falta condicionamiento de ruido')
            a, b, g, c, d, h = self.modulation(F.silu(cond)).unsqueeze(1).chunk(6, -1)
        else:
            a = b = c = d = 0
            g = h = 1
        normalized = self.norm1(x) * (1 + b) + a
        batch, length, _ = x.shape
        q = self._rope(self.q(normalized).view(batch, length, self.heads, self.head_dim).transpose(1, 2), positions)
        k = self._rope(self.k(normalized).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2), positions)
        v = self.v(normalized).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + g * self.o(out.transpose(1, 2).reshape(batch, length, -1))
        normalized = self.norm2(x) * (1 + d) + c
        return x + h * self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class Block(nn.Module):
    """Grupo de capas y módulos de ruido propios, sin parámetros compartidos entrenables."""

    def __init__(self, cfg: dict, count: int, conditioned: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer(cfg, conditioned) for _ in range(count)])
        self.norm = RMSNorm(cfg['hidden_size'], cfg['rms_norm_eps'])
        self.condition_dim = cfg['condition_dim']
        self.frequency_base = cfg['condition_frequency_base']
        self.condition = nn.Sequential(nn.Linear(self.condition_dim, self.condition_dim), nn.SiLU(), nn.Linear(self.condition_dim, self.condition_dim)) if conditioned else None

    def forward(self, x: Tensor, mask: Tensor, positions: Tensor, sigma: Tensor | None) -> Tensor:
        """Ejecuta exclusivamente las capas de este bloque."""
        cond = None
        if self.condition is not None and sigma is not None:
            frequencies = torch.exp(-math.log(self.frequency_base) * torch.arange(self.condition_dim // 2, device=x.device) / (self.condition_dim // 2))
            angles = sigma.log()[:, None] * frequencies[None, :]
            cond = self.condition(torch.cat((angles.cos(), angles.sin()), dim=-1))
        for layer in self.layers:
            x = layer(x, mask, positions, cond)
        return self.norm(x)


class LanguageModel(nn.Module):
    """Baseline causal o ensemble DiffusionBlocks con codebook congelado."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        noise = cfg['noise']
        if cfg['num_hidden_layers'] < 1 or cfg['num_blocks'] < 1 or cfg['condition_dim'] < 2 or cfg['condition_dim'] % 2:
            raise ValueError('Capas/bloques positivos y condition_dim par >= 2 requeridos')
        if not (0 < noise['min'] < noise['max']) or any(noise[key] <= 0 for key in ('log_std', 'sigma_data', 'logit_temperature', 'steps_per_block')):
            raise ValueError('Distribución y escalas de ruido inválidas')
        self.diffusion = cfg['method'] == 'diffusion'
        if cfg['method'] not in ('baseline', 'diffusion'):
            raise ValueError('method debe ser baseline o diffusion')
        if self.diffusion and not cfg['freeze_codebook']:
            raise ValueError('Este prototipo requiere codebook congelado para independencia')
        blocks = cfg['num_blocks'] if self.diffusion else 1
        if cfg['num_hidden_layers'] % blocks:
            raise ValueError('Las capas deben dividirse exactamente entre bloques')
        self.embedding = nn.Embedding(cfg['vocab_size'], cfg['hidden_size'])
        # Inicializar antes de construir capas/adaptadores garantiza el mismo codebook
        # en ambos métodos cuando parten de la misma seed.
        nn.init.normal_(self.embedding.weight, std=cfg['initializer_range'])
        self.blocks = nn.ModuleList([Block(cfg, cfg['num_hidden_layers'] // blocks, self.diffusion) for _ in range(blocks)])
        self.apply(self._initialize)
        if self.diffusion or cfg['freeze_codebook']:
            self.embedding.weight.requires_grad_(False)
        if self.diffusion:
            for module in self.modules():
                if isinstance(module, Layer) and module.modulation is not None:
                    nn.init.zeros_(module.modulation.weight)
                    nn.init.zeros_(module.modulation.bias)
        noise = cfg['noise']
        normal = NormalDist(noise['log_mean'], noise['log_std'])
        lower, upper = normal.cdf(math.log(noise['min'])), normal.cdf(math.log(noise['max']))
        boundaries = [noise['max']]
        boundaries += [math.exp(normal.inv_cdf(upper - (upper - lower) * i / blocks)) for i in range(1, blocks)]
        boundaries += [noise['min']]
        self.register_buffer('boundaries', torch.tensor(boundaries))

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.cfg['initializer_range'])
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def sample_sigma(self, block: int, batch: int, device: torch.device) -> Tensor:
        """Muestrea log-normal truncada por cuantiles equiprobables, sin reseeding."""
        noise = self.cfg['noise']
        normal = NormalDist(noise['log_mean'], noise['log_std'])
        lo = normal.cdf(math.log(float(self.boundaries[block + 1])))
        hi = normal.cdf(math.log(float(self.boundaries[block])))
        uniform = (lo + (hi - lo) * torch.rand(batch, device=device)).clamp(torch.finfo(torch.float32).eps, 1 - torch.finfo(torch.float32).eps)
        return torch.exp(noise['log_mean'] + noise['log_std'] * math.sqrt(2) * torch.erfinv(2 * uniform - 1))

    def denoise(self, tokens: Tensor, noisy: Tensor, sigma: Tensor, block: int) -> Tensor:
        """Predice embedding limpio con precondicionamiento tipo EDM."""
        clean = self.embedding(tokens)
        length = tokens.size(1)
        sd = self.cfg['noise']['sigma_data']
        scale = sigma[:, None, None]
        x = torch.cat((clean, noisy / (scale.square() + sd ** 2).sqrt()), dim=1)
        positions = torch.arange(length, device=tokens.device).repeat(2)
        output = self.blocks[block](x, attention_mask(length, True, tokens.device), positions, sigma)[:, length:]
        skip = sd ** 2 / (scale.square() + sd ** 2)
        out = scale * sd / (scale.square() + sd ** 2).sqrt()
        return skip * noisy + out * output

    def forward(self, tokens: Tensor, block: int | None = None, sigma: Tensor | None = None, noise: Tensor | None = None) -> Tensor:
        """Devuelve logits [batch, longitud, vocabulario]; diffusion recupera tokens actuales."""
        if not self.diffusion:
            x = self.embedding(tokens)
            x = self.blocks[0](x, attention_mask(tokens.size(1), False, tokens.device), torch.arange(tokens.size(1), device=tokens.device), None)
            return F.linear(x, self.embedding.weight)
        if block is None:
            raise ValueError('Diffusion requiere índice de bloque')
        if sigma is None:
            sigma = self.sample_sigma(block, tokens.size(0), tokens.device)
        clean = self.embedding(tokens)
        if noise is None:
            noise = torch.randn_like(clean)
        result = self.denoise(tokens, clean + sigma[:, None, None] * noise, sigma, block)
        return F.linear(result, self.embedding.weight) / self.cfg['noise']['logit_temperature']

    def _clamp(self, prediction: Tensor) -> Tensor:
        """Clamping de Diffusion-LM: sustituye la predicción por el embedding de su token más probable."""
        if not self.cfg['noise']['clamp']:
            return prediction
        # Sin esto la media posterior queda entre embeddings y la trayectoria nunca se compromete a un token.
        return self.embedding(F.linear(prediction, self.embedding.weight).argmax(-1))

    @torch.no_grad()
    def generate(self, prompt: Tensor, count: int) -> Tensor:
        """Genera autoregresivamente; diffusion integra desde ruido por todos los bloques."""
        self.eval()
        tokens = prompt.clone()
        for _ in range(count):
            if not self.diffusion:
                next_token = self(tokens)[:, -1].argmax(-1)
            else:
                candidate = torch.cat((tokens, torch.zeros_like(tokens[:, :1])), dim=1)
                # La posición objetivo solo puede ver el pasado limpio; el placeholder no es visible.
                noisy = torch.randn(candidate.size(0), candidate.size(1), self.cfg['hidden_size'], device=tokens.device) * self.boundaries[0]
                for block in range(len(self.blocks)):
                    steps = torch.linspace(float(self.boundaries[block]), float(self.boundaries[block + 1]), self.cfg['noise']['steps_per_block'] + 1, device=tokens.device)
                    for current, following in zip(steps[:-1], steps[1:]):
                        sigma = current.expand(candidate.size(0))
                        prediction = self._clamp(self.denoise(candidate, noisy, sigma, block))
                        noisy = noisy + (following - current) * (noisy - prediction) / current
                # Paso final hasta sigma=0 (EDM): se decodifica la predicción, no el ruido residual de sigma_min.
                final = self.denoise(candidate, noisy, self.boundaries[-1].expand(candidate.size(0)), len(self.blocks) - 1)
                logits = F.linear(final[:, -1], self.embedding.weight) / self.cfg['noise']['logit_temperature']
                next_token = logits.argmax(-1)
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
        return tokens
