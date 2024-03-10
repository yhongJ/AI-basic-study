import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model , n_heads):
    super().__init__()

    self.d_model = d_model
    self.n_heads  = n_heads
    self.d_k = d_model / n_heads # = d_head

    self.function_q = nn.Linear(d_model, d_model) #d_model -> d_k로 가는 과정은 Linear이후 차원을 늘리면서 시행
    self.function_k = nn.Linear(d_model, d_model)
    self.function_v = nn.Linear(d_model, d_model)
    self.function_o = nn.Linear(d_model, d_model)
    self.dropOut = nn.Dropout(0.1)

  def forward(self, query, key, value, mask):
    # query, key, value = [batch_size, seq_len, d_model]
    batch_size = query.size()[0]

    Q = self.function_q(query)
    K = self.function_k(key)
    V = self.function_v(value)

    Q = Q.view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3) #-1은 함수가 알아서 계산. [batch_size, self.n_heads, seq_len, d_k]
    K = K.view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
    V = V.view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)

    alpha = Q @ K.permute(0,1, 3, 2 )  / torch.tensor([math.sqrt(self.d_k)]) #d_k끼리 곱해질 수 있도록 transpose(permute) -> K = [batch_size, self.n_heads, d_k, seq_len]
    #alpha = [batch_size, self.n_heads, seq_len, seq_len]

    if mask is not None:
      alpha = alpha.masked_fill(mask == 0, -1e10)

    attention = torch.softmax(alpha, dim = -1)
    attention = self.dropOut(attention)

    z = attention @ V
    z = z.permute(0, 1, 3, 2).contiguous()
    z = z.view([batch_size, -1, self.d_model])
    z = self.function_o(z)

    return z


class FFN(nn.Module):
  def __init__(self, d_model, d_ffn, dropOut_ratio):
    super().__init__()

    self.function1 = nn.Linear(d_model, d_ffn)
    self.function2 = nn.Linear(d_ffn, d_model)

    self.dropOut = nn.Dropout(dropOut_ratio)

  def forward(self, x):
    x = self.function1(x)
    x = torch.relu(x)
    x = self.dropOut(x)
    x = self.function2(x)

    return x


class EncoderLayer(nn.Module):
  def __init__(self, d_model, n_heads, d_ffn, dropOut_ratio):
    super().__init__()

    self.self_attention = MultiHeadAttention(d_model, n_heads)
    self.layerNorm1 = nn.LayerNorm(d_model)
    self.ffn = FFN(d_model, d_ffn, dropOut_ratio)
    self.layerNorm2 = nn.LayerNorm(d_model)

    self.dropOut = nn.Dropout(dropOut_ratio)

  def forward(self, x, mask):
    #x = [batch_size, seq_len, d_model]

    residual = x #residual learning
    x = self.self_attention(query = x, key = x, value = x, mask = mask)

    x = self.dropOut(x) + residual # H(x) = F(x) + x
    x = self.layerNorm1(x)

    residual = x

    x = self.ffn(x)

    x = self.dropOut(x) + residual
    x = self.layerNorm2(x)

    return x



class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model): #max_len : 최대 단어 수
        super().__init__()


        self.PE = torch.zeros(max_len, d_model) #[pos, i]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)  #(max_len) -> (max_len, 1)
        _2i = torch.arange(0, d_model,  2, dtype=torch.float)

        #index slicing -> 모든 행에 대해. 0번째부터 2씩 증가
        self.PE[:, 0: :2] = torch.sin(pos / 10000 ** (_2i / d_model)) #i짝 -> sin
        #index slicing -> 모든 행에 대해. 1번째부터 2씩 증가
        self.PE[:, 1: :2] = torch.cos(pos / 10000 ** (_2i / d_model)) #i홀 ->cos

    def forward(self,x):
#x = [batch, seq_len, d_model] -> x.size(1), 즉 seq_len까지만 PE적용
        return x + self.PE[ :, x.size(1) ].detach()

class Encoder(nn.Module):
  def __init__(self, N, vocab_size , d_model, n_heads, d_ffn, max_len, dropOut_ratio):
    super().__init__()
    self.input_emb = nn.Embedding(vocab_size, d_model)
    self.pos_enc = PositionalEncoding(max_len, d_model)
    self.dropOut = nn.Dropout(dropOut_ratio)

    self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ffn, dropOut_ratio) for i in range(N)])

  def forward(self, x):
    input_emb = self.input_emb(x)
    pos_enc = self.pos_enc(x)

    x = self.dropOut(input_emb + pos_enc)

    for encoder_layer in self.encoder_layers:
      x = encoder_layer(x, None)

    return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, dropOut_ratio):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.encoder_attention = MultiHeadAttention(d_model, n_heads) #encoder output의 key, value들과 decoder의 query들과 multihead attention
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ffn, dropOut_ratio)
        self.layerNorm3 = nn.LayerNorm(d_model)
        self.dropOut = nn.Dropout(dropOut_ratio)

    def forward(self, x, encoder_output, self_mask):
        # x = [batch_size, seq_len, d_model]

        residual = x
        x = self.self_attention(query=x, key=x, value=x, mask=self_mask)
        x = self.dropOut(x) + residual
        x = self.layerNorm1(x)

        # Encoder-Decoder Attention
        residual = x
        x = self.encoder_attention(query=x, key=encoder_output, value=encoder_output, mask = True)
        x = self.dropOut(x) + residual
        x = self.layerNorm2(x)

        # Feedforward
        residual = x

        x = self.ffn(x)
        x = self.dropOut(x) + residual
        x = self.layerNorm3(x)

        return x


class Decoder(nn.Module):
    def __init__(self, N, vocab_size, d_model, n_heads, d_ffn, max_len, dropOut_ratio):
        super().__init__()
        self.input_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(max_len, d_model)
        self.dropOut = nn.Dropout(dropOut_ratio)

        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ffn, dropOut_ratio) for _ in range(N)])

    def forward(self, x, encoder_output, self_mask, encoder_mask):
        input_emb = self.input_emb(x)
        pos_enc = self.pos_enc(x)

        x = self.dropOut(input_emb + pos_enc)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, self_mask, encoder_mask)

        return x

class Transformer(nn.Module):
    def __init__(self, N, vocab_size, d_model, n_heads, d_ffn, max_len, dropOut_ratio):
        super().__init__()
        self.encoder = Encoder(N, vocab_size, d_model, n_heads, d_ffn, max_len, dropOut_ratio)
        self.decoder = Decoder(N, vocab_size, d_model, n_heads, d_ffn, max_len, dropOut_ratio)

    def forward(self, encoder_input, decoder_input):
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output, self_mask=None, encoder_mask=None)
        return decoder_output

