import math
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

try:
    from .tAPE_PositionalEncoding import eRPE_Transformer
except ImportError:
    eRPE_Transformer = None

try:
    from .CotarFormer import CotarFormer
except ImportError:
    CotarFormer = None

feature_len_dict = {
    "SleePyCo": [
        [5, 24, 120],
        [10, 48, 240],
        [15, 72, 360],
        [20, 96, 480],
        [24, 120, 600],
        [29, 144, 720],
        [34, 168, 840],
        [39, 192, 960],
        [44, 216, 1080],
        [48, 240, 1200],
    ],
    "XSleepNet": [
        [6, 12, 24],
        [12, 24, 47],
        [18, 36, 71],
        [24, 47, 94],
        [30, 59, 118],
        [36, 71, 141],
        [42, 83, 165],
        [47, 94, 188],
        [53, 106, 211],
        [59, 118, 236],
    ],
    "UTime": [
        [7, 15, 62],
        [15, 31, 125],
        [23, 45, 187],
        [31, 62, 250],
        [39, 78, 312],
        [46, 93, 375],
        [54, 109, 437],
        [62, 125, 500],
        [70, 140, 562],
        [78, 156, 625],
    ],
}


class PlainRNN(nn.Module):
    def __init__(self, config):
        super(PlainRNN, self).__init__()
        self.cfg = config["classifier"]
        self.num_classes = self.cfg["num_classes"]
        self.input_dim = self.cfg["input_dim"]
        self.hidden_dim = self.cfg["hidden_dim"]
        self.num_layers = self.cfg["num_rnn_layers"]
        self.bidirectional = self.cfg["bidirectional"]

        self.rnn = nn.RNN(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.fc = nn.Linear(
            self.hidden_dim * 2 if self.bidirectional else self.hidden_dim,
            self.num_classes,
        )

    def init_hidden(self, x):
        h0 = torch.zeros(
            (
                self.num_layers * (2 if self.bidirectional else 1),
                x.size(0),
                self.hidden_dim,
            )
        ).cuda()

        return h0

    def forward(self, x):
        hidden = self.init_hidden(x)
        rnn_output, hidden = self.rnn(x, hidden)

        if self.bidirectional:
            output_f = rnn_output[:, -1, : self.hidden_dim]
            output_b = rnn_output[:, 0, self.hidden_dim :]
            output = torch.cat((output_f, output_b), dim=1)
        else:
            output = rnn_output[:, -1, :]

        output = self.fc(output)
        return output


class PlainGRU(PlainRNN):
    def __init__(self, config):
        super(PlainGRU, self).__init__(config)
        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )


class PlainLSTM(PlainRNN):
    def __init__(self, config):
        super(PlainLSTM, self).__init__(config)
        self.rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    def init_hidden(self, x):
        h0 = torch.zeros(
            (
                self.num_layers * (2 if self.bidirectional else 1),
                x.size(0),
                self.hidden_dim,
            )
        ).cuda()
        c0 = torch.zeros(
            (
                self.num_layers * (2 if self.bidirectional else 1),
                x.size(0),
                self.hidden_dim,
            )
        ).cuda()

        return h0, c0


class AttRNN(PlainRNN):
    def __init__(self, config):
        super(AttRNN, self).__init__(config)
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        self.w_ha = nn.Linear(
            self.hidden_dim * 2 if self.bidirectional else self.hidden_dim,
            self.hidden_dim,
            bias=True,
        )
        self.w_att = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, x):
        hidden = self.init_hidden(x)
        rnn_output, hidden = self.rnn(x, hidden)
        a_states = self.w_ha(rnn_output)
        alpha = torch.softmax(self.w_att(a_states), dim=1).view(x.size(0), 1, x.size(1))
        weighted_sum = torch.bmm(alpha, a_states)

        output = weighted_sum.view(x.size(0), -1)
        output = self.fc(output)
        return output


class AttGRU(AttRNN):
    def __init__(self, config):
        super(AttGRU, self).__init__(config)
        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )


class AttLSTM(AttRNN):
    def __init__(self, config):
        super(AttLSTM, self).__init__(config)
        self.rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    def init_hidden(self, x):
        h0 = torch.zeros(
            (
                self.num_layers * (2 if self.bidirectional else 1),
                x.size(0),
                self.hidden_dim,
            )
        ).cuda()
        c0 = torch.zeros(
            (
                self.num_layers * (2 if self.bidirectional else 1),
                x.size(0),
                self.hidden_dim,
            )
        ).cuda()

        return h0, c0


class GRUAttn(AttRNN):
    def __init__(self, config):
        super(GRUAttn, self).__init__(config)
        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.rnn_out_dim = (
            self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        )
        self.layer_norm = nn.LayerNorm(self.rnn_out_dim)
        fusion_dim = self.hidden_dim + self.rnn_out_dim
        self.fc = nn.Linear(fusion_dim, self.num_classes)

    def forward(self, x):
        hidden = self.init_hidden(x)
        rnn_output, hn = self.rnn(x, hidden)
        rnn_output = self.layer_norm(rnn_output)
        a_states = torch.tanh(self.w_ha(rnn_output))
        alpha = torch.softmax(self.w_att(a_states), dim=1).transpose(1, 2)

        weighted_sum = torch.bmm(alpha, a_states).squeeze(1)
        if self.bidirectional:
            last_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            last_hidden = hn[-1, :, :]

        combined = torch.cat((weighted_sum, last_hidden), dim=1)
        output = self.fc(combined)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, config, in_features, out_features, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.cfg = config["classifier"]["pos_enc"]
        self.num_scales = config["feature_pyramid"]["num_scales"]

        if self.cfg["dropout"]:
            self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.act_fn = nn.PReLU()

        if self.num_scales > 1:
            self.max_len = feature_len_dict[config["backbone"]["name"]][
                config["dataset"]["seq_len"] - 1
            ][config["feature_pyramid"]["num_scales"] - 1]
        else:
            self.max_len = 5000

        print("[INFO] Maximum length of pos_enc: {}".format(self.max_len))

        pe = torch.zeros(self.max_len, out_features)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, out_features, 2).float()
            * (-math.log(10000.0) / out_features)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = self.act_fn(self.fc(x))

        if self.num_scales > 1:
            hop = self.max_len // x.size(0)
            pe = self.pe[hop // 2 :: hop, :]
        else:
            pe = self.pe

        if pe.shape[0] != x.size(0):
            pe = pe[: x.size(0), :]

        x = x + pe

        if self.cfg["dropout"]:
            x = self.dropout(x)

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = True

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class Transformer(nn.Module):
    def __init__(self, config, nheads, num_encoder_layers, pool="mean"):
        super(Transformer, self).__init__()
        self.cfg = config["classifier"]
        self.model_dim = self.cfg["model_dim"]
        self.feedforward_dim = self.cfg["feedforward_dim"]
        self.in_features = config["feature_pyramid"]["dim"]
        self.out_features = self.cfg["model_dim"]

        self.pos_encoding = PositionalEncoding(
            config, self.in_features, self.out_features
        )

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nheads,
            dim_feedforward=self.feedforward_dim,
            dropout=0.1 if self.cfg["dropout"] else 0.0,
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_encoder_layers
        )

        self.pool = pool

        if self.cfg["dropout"]:
            self.dropout = nn.Dropout(p=0.5)

        if pool == "attn":
            self.w_ha = nn.Linear(self.model_dim, self.model_dim, bias=True)
            self.w_at = nn.Linear(self.model_dim, 1, bias=False)

        self.fc = nn.Linear(self.model_dim, self.cfg["num_classes"])

    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.transpose(0, 1)

        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "last":
            x = x[:, -1]
        elif self.pool == "attn":
            a_states = torch.tanh(self.w_ha(x))
            alpha = torch.softmax(self.w_at(a_states), dim=1).view(
                x.size(0), 1, x.size(1)
            )
            x = torch.bmm(alpha, a_states).view(x.size(0), -1)
        elif self.pool is None:
            x = x
        else:
            raise NotImplementedError

        if self.cfg["dropout"]:
            x = self.dropout(x)

        out = self.fc(x)
        return out


class CotarFormerClassifier(nn.Module):
    def __init__(self, config):
        super(CotarFormerClassifier, self).__init__()
        self.cfg = config["classifier"]
        self.model_dim = self.cfg["model_dim"]
        self.positional_encoding = PositionalEmbedding(self.model_dim)
        self.transformer = CotarFormer(config)
        self.pool = self.cfg["pool"]
        self.fc = nn.Linear(self.model_dim, self.cfg["num_classes"])

        if self.pool == "attn":
            self.w_ha = nn.Linear(self.model_dim, self.model_dim, bias=True)
            self.w_at = nn.Linear(self.model_dim, 1, bias=False)

    def forward(self, x):
        x = x + self.positional_encoding(x)
        x = self.transformer(x)

        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "last":
            x = x[:, -1]
        elif self.pool == "attn":
            a_states = torch.tanh(self.w_ha(x))
            alpha = torch.softmax(self.w_at(a_states), dim=1).view(
                x.size(0), 1, x.size(1)
            )
            x = torch.bmm(alpha, a_states).view(x.size(0), -1)
        elif self.pool is None:
            x = x
        else:
            raise NotImplementedError

        if self.cfg["dropout"]:
            x = self.dropout(x)

        out = self.fc(x)
        return out


class MambaClassifier(nn.Module):
    def __init__(self, config, num_layers=3, pool="mean"):
        super(MambaClassifier, self).__init__()

        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required for MambaClassifier. Install it or use a different classifier."
            )

        self.cfg = config["classifier"]
        self.model_dim = self.cfg["model_dim"]
        self.in_features = config["feature_pyramid"]["dim"]
        self.out_features = self.cfg["model_dim"]
        self.pool = pool

        self.proj = nn.Linear(self.in_features, self.model_dim)
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=self.model_dim,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                )
                for _ in range(num_layers)
            ]
        )

        if self.cfg["dropout"]:
            self.dropout = nn.Dropout(p=0.5)

        if pool == "attn":
            self.w_ha = nn.Linear(self.model_dim, self.model_dim, bias=True)
            self.w_at = nn.Linear(self.model_dim, 1, bias=False)

        self.fc = nn.Linear(self.model_dim, self.cfg["num_classes"])

    def forward(self, x):
        x = self.proj(x)

        for layer in self.mamba_layers:
            x = layer(x)

        if self.pool == "mean":
            out = x.mean(dim=1)
        elif self.pool == "max":
            out, _ = x.max(dim=1)
        elif self.pool == "attn":
            attn_score = self.w_at(torch.tanh(self.w_ha(x)))
            attn_weight = torch.softmax(attn_score, dim=1)
            out = torch.sum(attn_weight * x, dim=1)
        else:
            raise NotImplementedError

        if self.cfg["dropout"]:
            out = self.dropout(out)

        logits = self.fc(out)
        return logits


class FC_Classifier(nn.Module):
    def __init__(self, config):
        super(FC_Classifier, self).__init__()
        self.cfg = config["classifier"]
        self.num_classes = self.cfg["num_classes"]
        self.input_dim = self.cfg.get("input_dim", config["model"].get("embed_dim"))
        self.dropout_rate = self.cfg.get("dropout", 0.0)

        self.fc = nn.Linear(self.input_dim, self.num_classes)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.mean(dim=1)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        logits = self.fc(x)
        return logits


def get_classifier(config):
    classifier_name = config["classifier"].get("name", "FC")

    if classifier_name == "PlainRNN":
        classifier = PlainRNN(config)
    elif classifier_name == "AttentionRNN":
        classifier = AttRNN(config)
    elif classifier_name == "PlainLSTM":
        classifier = PlainLSTM(config)
    elif classifier_name == "AttentionLSTM":
        classifier = AttLSTM(config)
    elif classifier_name == "PlainGRU":
        classifier = PlainGRU(config)
    elif classifier_name == "AttentionGRU":
        classifier = AttGRU(config)
    elif classifier_name == "GRUAttn":
        classifier = GRUAttn(config)
    elif classifier_name == "Transformer":
        if eRPE_Transformer is None:
            raise ImportError(
                "eRPE_Transformer dependency is missing. Install the required module or use a different classifier."
            )
        classifier = Transformer(
            config, nheads=8, num_encoder_layers=6, pool=config["classifier"]["pool"]
        )
    elif classifier_name == "eRPETransformer":
        if eRPE_Transformer is None:
            raise ImportError(
                "eRPE_Transformer dependency is missing. Install the required module or use a different classifier."
            )
        classifier = eRPE_Transformer(
            config, nheads=8, num_encoder_layers=6, pool=config["classifier"]["pool"]
        )
    elif classifier_name == "Mamba":
        classifier = MambaClassifier(
            config, num_layers=1, pool=config["classifier"]["pool"]
        )
    elif classifier_name == "CotarFormer" or classifier_name == "CotarFormer6":
        if CotarFormer is None:
            raise ImportError(
                "CotarFormer dependency is missing. Copy CotarFormer.py into the local models package or use a different classifier."
            )
        classifier = CotarFormerClassifier(config)
    elif classifier_name == "FC":
        classifier = FC_Classifier(config)
    else:
        raise ValueError(f"Unknown classifier name: {classifier_name}")

    return classifier
