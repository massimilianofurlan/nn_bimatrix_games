import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Tuple


class AttentionExposingEncoderLayer(nn.Module):
    """Transformer encoder layer with an explicit attention-readout method."""

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, need_weights=False)[0]
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self._ff_block(x))
        return x

    @torch.jit.export
    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        assert attn_weights is not None
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self._ff_block(x))
        return x, attn_weights


class RawPayoffTransformer_Bimatrix(nn.Module):
    """Transformer over raw scalar bimatrix-payoff tokens.

    This model intentionally does not center payoffs, does not compute
    payoff-comparison features, and does not apply an attention mask. For a
    2x2 game it creates eight payoff tokens: four current-player payoff tokens
    and four opponent-payoff tokens. All tokens, including CLS, attend fully.
    """

    def __init__(self, n_actions: int, n_layers: int, hidden_dim: int):
        super().__init__()
        self.n_actions = n_actions
        self.n_profiles = n_actions * n_actions
        self.n_tokens = 2 * self.n_profiles

        n_heads = 4 if hidden_dim % 4 == 0 else 2
        self.input_projection = nn.Linear(1, hidden_dim)
        self.payoff_owner_embedding = nn.Embedding(2, hidden_dim)
        self.row_embedding = nn.Embedding(n_actions, hidden_dim)
        self.col_embedding = nn.Embedding(n_actions, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.encoder_layers = nn.ModuleList(
            [AttentionExposingEncoderLayer(hidden_dim, n_heads) for _ in range(n_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, n_actions)
        self.softmax = nn.Softmax(dim=1)

        profile_rows = torch.arange(n_actions).repeat_interleave(n_actions)
        profile_cols = torch.arange(n_actions).repeat(n_actions)
        self.register_buffer("token_rows", torch.cat((profile_rows, profile_rows)))
        self.register_buffer("token_cols", torch.cat((profile_cols, profile_cols)))
        self.register_buffer(
            "token_payoff_owner",
            torch.cat(
                (
                    torch.zeros(self.n_profiles, dtype=torch.long),
                    torch.ones(self.n_profiles, dtype=torch.long),
                )
            ),
        )

        self.apply(self._initialize_weights)
        init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)

    def _raw_payoff_tokens(self, G: torch.Tensor) -> torch.Tensor:
        batch_size = G.shape[0]
        raw_payoffs = torch.cat(
            (
                G[:, 0].reshape(batch_size, self.n_profiles),
                G[:, 1].reshape(batch_size, self.n_profiles),
            ),
            dim=1,
        )
        tokens = self.input_projection(raw_payoffs.reshape(batch_size, self.n_tokens, 1))
        tokens = tokens + self.payoff_owner_embedding(self.token_payoff_owner).unsqueeze(0)
        tokens = tokens + self.row_embedding(self.token_rows).unsqueeze(0)
        tokens = tokens + self.col_embedding(self.token_cols).unsqueeze(0)
        return tokens

    def _logits_from_encoded(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.output_norm(encoded[:, 0]))

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        batch_size = G.shape[0]
        tokens = self._raw_payoff_tokens(G)
        cls = self.cls_token.expand(batch_size, -1, -1)
        encoded = torch.cat((cls, tokens), dim=1)
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        return self.softmax(self._logits_from_encoded(encoded))

    @torch.jit.export
    def forward_with_attention(self, G: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size = G.shape[0]
        attention_weights = torch.jit.annotate(List[torch.Tensor], [])
        tokens = self._raw_payoff_tokens(G)
        cls = self.cls_token.expand(batch_size, -1, -1)
        encoded = torch.cat((cls, tokens), dim=1)
        for layer in self.encoder_layers:
            encoded, weights = layer.forward_with_attention(encoded)
            attention_weights.append(weights)
        return self.softmax(self._logits_from_encoded(encoded)), attention_weights

    @torch.jit.ignore
    def token_labels(self) -> List[str]:
        labels = ["CLS"]
        for owner in ["own", "opponent"]:
            for row in range(self.n_actions):
                for col in range(self.n_actions):
                    labels.append(f"{owner}_payoff[{row},{col}]")
        return labels


PayoffTokenTransformer_Bimatrix = RawPayoffTransformer_Bimatrix


class PayoffOrderTransformer_Bimatrix(nn.Module):
    """Transformer over action-profile tokens with cardinal and rank payoff features.

    Each token is one action profile.  The two raw payoff coordinates retain the
    cardinal information needed for mixed strategies, while the rank coordinates
    expose the payoff order structure directly to the attention layers.
    """

    def __init__(self, n_actions: int, n_layers: int, hidden_dim: int):
        super().__init__()
        self.n_actions = n_actions
        self.n_profiles = n_actions * n_actions

        n_heads = 4 if hidden_dim % 4 == 0 else 2
        self.input_projection = nn.Linear(4, hidden_dim)
        self.row_embedding = nn.Embedding(n_actions, hidden_dim)
        self.col_embedding = nn.Embedding(n_actions, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, n_actions)
        self.softmax = nn.Softmax(dim=1)

        row_idx = torch.arange(n_actions).repeat_interleave(n_actions)
        col_idx = torch.arange(n_actions).repeat(n_actions)
        self.register_buffer("row_indices", row_idx)
        self.register_buffer("col_indices", col_idx)

        self.apply(self._initialize_weights)
        init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)

    def _rank_features(self, payoffs: torch.Tensor) -> torch.Tensor:
        lower_count = (payoffs.unsqueeze(2) > payoffs.unsqueeze(1)).to(payoffs.dtype).sum(dim=2)
        return lower_count / float(max(self.n_profiles - 1, 1))

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        batch_size = G.shape[0]
        payoffs = G.permute(0, 2, 3, 1).reshape(batch_size, self.n_profiles, 2)
        own_rank = self._rank_features(payoffs[:, :, 0]).unsqueeze(2)
        opponent_rank = self._rank_features(payoffs[:, :, 1]).unsqueeze(2)
        features = torch.cat((payoffs, own_rank, opponent_rank), dim=2)

        tokens = self.input_projection(features)
        tokens = tokens + self.row_embedding(self.row_indices).unsqueeze(0)
        tokens = tokens + self.col_embedding(self.col_indices).unsqueeze(0)

        cls = self.cls_token.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1))
        logits = self.output_layer(self.output_norm(encoded[:, 0]))
        return self.softmax(logits)


class MaskedPayoffOrderTransformer_Bimatrix(nn.Module):
    """Transformer over payoff-comparison tokens with non-strategic components masked.

    For the current player, own-payoff tokens are centered across own actions
    for each opponent action. Opponent-payoff tokens are centered across the
    opponent's actions for each current action. This removes the payoff levels
    that do not affect best-response comparisons before the attention stack.
    """

    def __init__(self, n_actions: int, n_layers: int, hidden_dim: int):
        super().__init__()
        self.n_actions = n_actions
        self.n_profiles = n_actions * n_actions
        self.n_tokens = 2 * self.n_profiles

        n_heads = 4 if hidden_dim % 4 == 0 else 2
        self.input_projection = nn.Linear(4, hidden_dim)
        self.payoff_owner_embedding = nn.Embedding(2, hidden_dim)
        self.row_embedding = nn.Embedding(n_actions, hidden_dim)
        self.col_embedding = nn.Embedding(n_actions, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, n_actions)
        self.softmax = nn.Softmax(dim=1)

        profile_rows = torch.arange(n_actions).repeat_interleave(n_actions)
        profile_cols = torch.arange(n_actions).repeat(n_actions)
        self.register_buffer("token_rows", torch.cat((profile_rows, profile_rows)))
        self.register_buffer("token_cols", torch.cat((profile_cols, profile_cols)))
        self.register_buffer(
            "token_payoff_owner",
            torch.cat(
                (
                    torch.zeros(self.n_profiles, dtype=torch.long),
                    torch.ones(self.n_profiles, dtype=torch.long),
                )
            ),
        )
        self.register_buffer("attention_mask", self._build_attention_mask(n_actions))

        self.apply(self._initialize_weights)
        init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_attention_mask(self, n_actions: int) -> torch.Tensor:
        seq_len = 1 + 2 * n_actions * n_actions
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        mask[0, :] = False
        mask[:, 0] = False
        for col in range(n_actions):
            own_pair = [1 + row * n_actions + col for row in range(n_actions)]
            for i in own_pair:
                for j in own_pair:
                    mask[i, j] = False
        offset = 1 + n_actions * n_actions
        for row in range(n_actions):
            opponent_pair = [offset + row * n_actions + col for col in range(n_actions)]
            for i in opponent_pair:
                for j in opponent_pair:
                    mask[i, j] = False
        return mask

    def _comparison_features(self, centered: torch.Tensor) -> torch.Tensor:
        centered = centered.reshape(centered.shape[0], self.n_tokens, 1)
        sign = torch.sign(centered)
        abs_value = centered.abs()
        pair_rank = (centered > 0).to(centered.dtype) + 0.5 * (centered == 0).to(centered.dtype)
        return torch.cat((centered, sign, abs_value, pair_rank), dim=2)

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        batch_size = G.shape[0]
        own_payoffs = G[:, 0]
        opponent_payoffs = G[:, 1]

        own_centered = own_payoffs - own_payoffs.mean(dim=1, keepdim=True)
        opponent_centered = opponent_payoffs - opponent_payoffs.mean(dim=2, keepdim=True)
        centered = torch.cat(
            (
                own_centered.reshape(batch_size, self.n_profiles),
                opponent_centered.reshape(batch_size, self.n_profiles),
            ),
            dim=1,
        )
        tokens = self.input_projection(self._comparison_features(centered))
        tokens = tokens + self.payoff_owner_embedding(self.token_payoff_owner).unsqueeze(0)
        tokens = tokens + self.row_embedding(self.token_rows).unsqueeze(0)
        tokens = tokens + self.col_embedding(self.token_cols).unsqueeze(0)

        cls = self.cls_token.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1), mask=self.attention_mask)
        logits = self.output_layer(self.output_norm(encoded[:, 0]))
        return self.softmax(logits)
