"""
Transformer-based models for ShengJi++.

Drop-in replacement for networks/Models.py — same class names, same forward() signatures.
Copy this file into a model folder via --model-architecture transformer to use it.

x_batch layout (no oracle, dynamic encoding):
  MainModel (Q-function), 873-dim:
    0:108   hand (108)
    108:112 dealer_pos (4)
    112:132 trump (20)
    132:136 declarer_pos (4)
    136:140 chaodi_times (4)
    140:220 points (80)
    220:328 unplayed_cards (108)
    328:332 trick_leader_pos (4)
    332:440 right_current_trick (108)
    440:548 opp_current_trick (108)
    548:656 left_current_trick (108)
    656:764 kitty (108)
    764:765 dominates_all (1)
    765:873 action (108)

  ValueModel (V-function), 764-dim: same minus dominates_all and action.

history_batch layout: (B, 15, 436)
  0:4     trick leader one-hot position
  4:112   self cards (108)
  112:220 right cards (108)
  220:328 opposite cards (108)
  328:436 left cards (108)
"""

import math
import torch
import torch.nn as nn


# ── Unchanged stage models (declaration, kitty, chaodi) ──────────────────────

class DeclarationModel(nn.Module):
    "The declaration model's observation includes: player's cards, player's position relative to the dealer, current declaration (rank and suit), position of current declaration, known trump cards in each player's hands."
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(172 + 7, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class KittyModel(nn.Module):
    "The kitty model's observation includes: the player's cards, the player's position relative to the dealer, current declaration, position of current declaration, known trump cards in each player's hand."
    def __init__(self, dynamic_kitty=False) -> None:
        super().__init__()
        self.dynamic_kitty = dynamic_kitty
        self.single_card_embedding = nn.Embedding(54, 54)
        self.fc1 = nn.Linear(172 + 54, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, card: torch.Tensor):
        if self.dynamic_kitty:
            card_embeddings = torch.zeros((x.shape[0], 54), device=x.device)
            card_embeddings[torch.arange(x.shape[0], dtype=torch.long), card.long()] = 1
        else:
            card_embeddings = self.single_card_embedding(card)
        x = torch.relu(self.fc1(torch.hstack([x, card_embeddings])))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class KittyRNNModel(nn.Module):
    def __init__(self, rnn_type='lstm') -> None:
        super().__init__()
        self.rnn_type = rnn_type
        self.card_embedding = nn.Embedding(54, 64)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.in1 = nn.Linear(64, 72)
        self.in2 = nn.Linear(72, 128)
        self.out1 = nn.Linear(128, 128)
        self.out2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, hand: torch.Tensor, card: torch.Tensor):
        sequence = self.card_embedding(torch.cat([hand, card], dim=-1))
        initial_state = torch.relu(self.in1(x))
        initial_state = self.in2(initial_state)
        if self.rnn_type == 'gru':
            _, h_n = self.rnn.forward(sequence, initial_state.unsqueeze(0))
        else:
            _, (h_n, _) = self.rnn.forward(sequence, (initial_state.unsqueeze(0), initial_state.unsqueeze(0)))
        x = torch.relu(self.out1(h_n.squeeze(0)))
        return self.out2(x)


class KittyArgmaxModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.single_card_embedding = nn.Embedding(54, 10)
        self.fc1 = nn.Linear(172 + 33 * 10, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 33)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, cards: torch.Tensor):
        card_embeddings = self.single_card_embedding(cards).view((x.shape[0], -1))
        x = torch.relu(self.fc1(torch.hstack([x, card_embeddings])))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.softmax(self.fc4(x))


class ChaodiModel(nn.Module):
    "The chaodi model's observation includes: the player's cards, the player's position relative to the dealer, current declaration, position of current declaration, known trump cards in each player's hand."
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(178, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# ── Transformer backbone ──────────────────────────────────────────────────────

D_MODEL = 128


def _make_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
    """Returns sinusoidal positional encoding of shape (1, max_len, d_model)."""
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, max_len, d_model)


class _TransformerBase(nn.Module):
    """
    Shared transformer backbone for MainModel (Q-function) and ValueModel (V-function).

    Token sequence:
      MainModel  (122 tokens): CLS + hand×54 + scalar + action + unplayed + right + opp + left + kitty + hist×60
      ValueModel (121 tokens): CLS + hand×54 + scalar +          unplayed + right + opp + left + kitty + hist×60

    Hand tokens at positions [1:55] may be masked when a card slot is empty.
    All other tokens are always attended to.
    """

    def __init__(self, has_action: bool, d_model: int = D_MODEL, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.has_action = has_action

        # Shared card embedding: 54 slots × d_model
        self.card_embedding = nn.Embedding(54, d_model)

        # Projection after weighted-sum card-set encoding
        self.cardset_proj = nn.Linear(d_model, d_model)

        # Per-token projection for hand card tokens
        self.hand_proj = nn.Linear(d_model, d_model)

        # Scalar context: MainModel includes dominates_all(1), ValueModel does not
        scalar_dim = 117 if has_action else 116
        self.scalar_proj = nn.Linear(scalar_dim, d_model)

        # Learned CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Sinusoidal positional encoding for 60 history slots (15 tricks × 4 players)
        self.register_buffer('hist_pos_enc', _make_sinusoidal(60, d_model))

        # Transformer encoder (Pre-LN for training stability)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _enc_cardset(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes a (B, 108) dynamic card tensor to (B, d_model).

        Dynamic encoding: tensor[i]=1 iff count≥1, tensor[54+i]=1 iff count≥2.
        count[i] = x[i] + x[54+i] ∈ {0, 1, 2}.
        We take the count-weighted sum of card embeddings.
        """
        count = x[:, :54] + x[:, 54:]              # (B, 54)
        emb = count @ self.card_embedding.weight     # (B, d_model)
        return self.cardset_proj(emb)

    def _enc_hand(self, x: torch.Tensor):
        """
        Encodes a (B, 108) hand tensor to 54 per-card tokens.

        Returns:
            tokens:   (B, 54, d_model) — zero for absent cards
            pad_mask: (B, 54) bool     — True means the token should be ignored
        """
        count = x[:, :54] + x[:, 54:]                          # (B, 54)
        # (B, 54, 1) * (1, 54, d_model) → (B, 54, d_model)
        tokens = count.unsqueeze(-1) * self.card_embedding.weight.unsqueeze(0)
        tokens = self.hand_proj(tokens)
        pad_mask = (count == 0)                                 # True = absent
        return tokens, pad_mask

    def _enc_history(self, history: torch.Tensor) -> torch.Tensor:
        """
        Encodes (B, 15, 436) history into (B, 60, d_model) trick tokens.

        Each trick row: [leader_pos(4), self(108), right(108), opp(108), left(108)].
        4 player tokens per trick × 15 tricks = 60 tokens total.
        Sinusoidal positional encoding is added to preserve trick order.
        """
        B = history.shape[0]
        # Extract card tensors for all 4 players across all 15 tricks
        cards = history[:, :, 4:].reshape(B * 15, 4, 108)       # (B*15, 4, 108)
        per_player = torch.stack(
            [self._enc_cardset(cards[:, i, :]) for i in range(4)],
            dim=1,
        )                                                         # (B*15, 4, d_model)
        tokens = per_player.reshape(B, 60, self.d_model)         # (B, 60, d_model)
        tokens = tokens + self.hist_pos_enc                       # add positional encoding
        return tokens

    def _forward_impl(self, x: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        B, device = x.shape[0], x.device

        # ── Parse x_batch ──────────────────────────────────────────────────────
        hand         = x[:, 0:108]
        dealer_pos   = x[:, 108:112]
        trump        = x[:, 112:132]
        declarer_pos = x[:, 132:136]
        chaodi_times = x[:, 136:140]
        points       = x[:, 140:220]
        unplayed     = x[:, 220:328]
        leader_pos   = x[:, 328:332]
        right_trick  = x[:, 332:440]
        opp_trick    = x[:, 440:548]
        left_trick   = x[:, 548:656]
        kitty        = x[:, 656:764]

        if self.has_action:
            dominates = x[:, 764:765]
            action    = x[:, 765:873]
            scalars = torch.cat(
                [dealer_pos, trump, declarer_pos, chaodi_times, points, leader_pos, dominates],
                dim=-1,
            )  # (B, 117)
        else:
            scalars = torch.cat(
                [dealer_pos, trump, declarer_pos, chaodi_times, points, leader_pos],
                dim=-1,
            )  # (B, 116)

        # ── Build tokens ───────────────────────────────────────────────────────
        hand_tok, hand_mask = self._enc_hand(hand)                 # (B, 54, d), (B, 54)
        scalar_tok  = self.scalar_proj(scalars).unsqueeze(1)       # (B, 1, d)
        unplay_tok  = self._enc_cardset(unplayed).unsqueeze(1)     # (B, 1, d)
        right_tok   = self._enc_cardset(right_trick).unsqueeze(1)  # (B, 1, d)
        opp_tok     = self._enc_cardset(opp_trick).unsqueeze(1)    # (B, 1, d)
        left_tok    = self._enc_cardset(left_trick).unsqueeze(1)   # (B, 1, d)
        kitty_tok   = self._enc_cardset(kitty).unsqueeze(1)        # (B, 1, d)
        hist_tok    = self._enc_history(history)                   # (B, 60, d)
        cls         = self.cls_token.expand(B, -1, -1)             # (B, 1, d)

        if self.has_action:
            action_tok = self._enc_cardset(action).unsqueeze(1)    # (B, 1, d)
            tokens = torch.cat(
                [cls, hand_tok, scalar_tok, action_tok, unplay_tok,
                 right_tok, opp_tok, left_tok, kitty_tok, hist_tok],
                dim=1,
            )  # (B, 122, d)
        else:
            tokens = torch.cat(
                [cls, hand_tok, scalar_tok, unplay_tok,
                 right_tok, opp_tok, left_tok, kitty_tok, hist_tok],
                dim=1,
            )  # (B, 121, d)

        # ── Padding mask: hand tokens [1:55] masked where card is absent ───────
        total = tokens.shape[1]
        key_padding_mask = torch.zeros(B, total, dtype=torch.bool, device=device)
        key_padding_mask[:, 1:55] = hand_mask  # hand always occupies positions 1..54

        # ── Transformer + output head ──────────────────────────────────────────
        out = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        return self.head(out[:, 0, :])  # CLS token → (B, 1)


# ── Public model classes ──────────────────────────────────────────────────────

class MainModel(_TransformerBase):
    """
    Transformer Q-function for the trick-taking phase.
    Same forward(x, history) signature as networks.Models.MainModel.
    use_oracle is accepted for API compatibility but ignored (oracle hurts generalization).
    """

    def __init__(self, use_oracle: bool = False, d_model: int = D_MODEL, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__(
            has_action=True,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x, history)


class ValueModel(_TransformerBase):
    """
    Transformer V-function for SAC (state-only, no action).
    Same forward(x, history) signature as networks.Models.ValueModel.
    Receives 764-dim x (state without dominates_all and action).
    """

    def __init__(self, d_model: int = D_MODEL, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__(
            has_action=False,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x, history)
