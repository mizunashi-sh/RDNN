import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def compared_version(ver1, ver2):
	list1 = str(ver1).split(".")
	list2 = str(ver2).split(".")	

	for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
		if int(list1[i]) == int(list2[i]):
			pass
		elif int(list1[i]) < int(list2[i]):
			return -1
		else:
			return 1

	if len(list1) == len(list2):
		return True
	elif len(list1) < len(list2):
		return False
	else:
		return True


class PositionalEmbedding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super().__init__()
		pe = torch.zeros(max_len, d_model).float()
		pe.requires_grad = False

		position = torch.arange(0, max_len).float().unsqueeze(1)
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		pe = pe.unsqueeze(0)
		self.register_buffer("pe", pe)

	def forward(self, x):
		return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
	def __init__(self, c_in, d_model):
		super().__init__()
		padding = 1 if compared_version(torch.__version__, "1.5.0") else 2
		self.tokenConv = nn.Conv1d(
			in_channels=c_in,
			out_channels=d_model,
			kernel_size=3,
			padding=padding,
			padding_mode="circular",
			bias=False,
		)
		for module in self.modules():
			if isinstance(module, nn.Conv1d):
				nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")

	def forward(self, x):
		return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
	def __init__(self, c_in, d_model):
		super().__init__()

		weights = torch.zeros(c_in, d_model).float()
		weights.requires_grad = False

		position = torch.arange(0, c_in).float().unsqueeze(1)
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

		weights[:, 0::2] = torch.sin(position * div_term)
		weights[:, 1::2] = torch.cos(position * div_term)

		self.emb = nn.Embedding(c_in, d_model)
		self.emb.weight = nn.Parameter(weights, requires_grad=False)

	def forward(self, x):
		return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
	def __init__(self, d_model, embed_type="fixed", freq="h"):
		super().__init__()

		minute_size = 4
		hour_size = 24
		weekday_size = 7
		day_size = 32
		month_size = 13

		embed_cls = FixedEmbedding if embed_type == "fixed" else nn.Embedding
		if freq == "t":
			self.minute_embed = embed_cls(minute_size, d_model)
		self.hour_embed = embed_cls(hour_size, d_model)
		self.weekday_embed = embed_cls(weekday_size, d_model)
		self.day_embed = embed_cls(day_size, d_model)
		self.month_embed = embed_cls(month_size, d_model)

	def forward(self, x):
		x = x.long()

		minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
		hour_x = self.hour_embed(x[:, :, 3])
		weekday_x = self.weekday_embed(x[:, :, 2])
		day_x = self.day_embed(x[:, :, 1])
		month_x = self.month_embed(x[:, :, 0])

		return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
	def __init__(self, d_model, embed_type="timeF", freq="h"):
		super().__init__()

		freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
		d_inp = freq_map[freq]
		self.embed = nn.Linear(d_inp, d_model, bias=False)

	def forward(self, x):
		return self.embed(x)


class DataEmbedding(nn.Module):
	def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
		super().__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		self.temporal_embedding = (
			TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
			if embed_type != "timeF"
			else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
		)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, x_mark=None):
		if x_mark is None:
			x = self.value_embedding(x) + self.position_embedding(x)
		else:
			x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
		return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
	def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
		super().__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
		self.temporal_embedding = (
			TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
			if embed_type != "timeF"
			else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
		)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, x_mark=None):
		x = self.value_embedding(x)
		if x_mark is not None:
			x = x + self.temporal_embedding(x_mark)
		return self.dropout(x)


class moving_avg(nn.Module):
	def __init__(self, kernel_size, stride):
		super().__init__()
		self.kernel_size = kernel_size
		self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

	def forward(self, x):
		front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
		end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
		x = torch.cat([front, x, end], dim=1)
		x = self.avg(x.permute(0, 2, 1))
		return x.permute(0, 2, 1)


class series_decomp(nn.Module):
	def __init__(self, kernel_size):
		super().__init__()
		self.moving_avg = moving_avg(kernel_size, stride=1)

	def forward(self, x):
		moving_mean = self.moving_avg(x)
		res = x - moving_mean
		return res, moving_mean


class my_Layernorm(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.layernorm = nn.LayerNorm(channels)

	def forward(self, x):
		x_hat = self.layernorm(x)
		bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
		return x_hat - bias


class AutoCorrelation(nn.Module):
	def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
		super().__init__()
		self.factor = factor
		self.scale = scale
		self.mask_flag = mask_flag
		self.output_attention = output_attention
		self.dropout = nn.Dropout(attention_dropout)

	def time_delay_agg_training(self, values, corr):
		head = values.shape[1]
		channel = values.shape[2]
		length = values.shape[3]
		top_k = max(1, int(self.factor * math.log(length)))
		mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
		index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
		weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
		tmp_corr = torch.softmax(weights, dim=-1)
		tmp_values = values
		delays_agg = torch.zeros_like(values).float()
		for i in range(top_k):
			pattern = torch.roll(tmp_values, -int(index[i]), -1)
			delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
				1, head, channel, length
			)
		return delays_agg

	def time_delay_agg_inference(self, values, corr):
		batch = values.shape[0]
		head = values.shape[1]
		channel = values.shape[2]
		length = values.shape[3]
		init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
		top_k = max(1, int(self.factor * math.log(length)))
		mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
		weights, delay = torch.topk(mean_value, top_k, dim=-1)
		tmp_corr = torch.softmax(weights, dim=-1)
		tmp_values = values.repeat(1, 1, 1, 2)
		delays_agg = torch.zeros_like(values).float()
		for i in range(top_k):
			tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
			pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
			delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
				1, head, channel, length
			)
		return delays_agg

	def time_delay_agg_full(self, values, corr):
		batch = values.shape[0]
		head = values.shape[1]
		channel = values.shape[2]
		length = values.shape[3]
		init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
		top_k = max(1, int(self.factor * math.log(length)))
		weights, delay = torch.topk(corr, top_k, dim=-1)
		tmp_corr = torch.softmax(weights, dim=-1)
		tmp_values = values.repeat(1, 1, 1, 2)
		delays_agg = torch.zeros_like(values).float()
		for i in range(top_k):
			tmp_delay = init_index + delay[..., i].unsqueeze(-1)
			pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
			delays_agg = delays_agg + pattern * tmp_corr[..., i].unsqueeze(-1)
		return delays_agg

	def forward(self, queries, keys, values, attn_mask):
		batch_size, length, heads, depth = queries.shape
		_, source_length, _, _ = values.shape

		if length > source_length:
			zeros = torch.zeros_like(queries[:, :(length - source_length), :]).float()
			values = torch.cat([values, zeros], dim=1)
			keys = torch.cat([keys, zeros], dim=1)
		else:
			values = values[:, :length, :, :]
			keys = keys[:, :length, :, :]

		q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
		k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
		res = q_fft * torch.conj(k_fft)
		corr = torch.fft.irfft(res, n=length, dim=-1)

		if self.training:
			out = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
		else:
			out = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

		if self.output_attention:
			return out.contiguous(), corr.permute(0, 3, 1, 2)
		return out.contiguous(), None


class AutoCorrelationLayer(nn.Module):
	def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
		super().__init__()

		d_keys = d_keys or (d_model // n_heads)
		d_values = d_values or (d_model // n_heads)

		self.inner_correlation = correlation
		self.query_projection = nn.Linear(d_model, d_keys * n_heads)
		self.key_projection = nn.Linear(d_model, d_keys * n_heads)
		self.value_projection = nn.Linear(d_model, d_values * n_heads)
		self.out_projection = nn.Linear(d_values * n_heads, d_model)
		self.n_heads = n_heads

	def forward(self, queries, keys, values, attn_mask):
		batch_size, query_len, _ = queries.shape
		_, key_len, _ = keys.shape
		heads = self.n_heads

		queries = self.query_projection(queries).view(batch_size, query_len, heads, -1)
		keys = self.key_projection(keys).view(batch_size, key_len, heads, -1)
		values = self.value_projection(values).view(batch_size, key_len, heads, -1)

		out, attn = self.inner_correlation(queries, keys, values, attn_mask)
		out = out.view(batch_size, query_len, -1)
		return self.out_projection(out), attn


class EncoderLayer(nn.Module):
	def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
		super().__init__()
		d_ff = d_ff or 4 * d_model
		self.attention = attention
		self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
		self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
		self.decomp1 = series_decomp(moving_avg)
		self.decomp2 = series_decomp(moving_avg)
		self.dropout = nn.Dropout(dropout)
		self.activation = F.relu if activation == "relu" else F.gelu

	def forward(self, x, attn_mask=None):
		new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
		x = x + self.dropout(new_x)
		x, _ = self.decomp1(x)
		y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
		y = self.dropout(self.conv2(y).transpose(-1, 1))
		res, _ = self.decomp2(x + y)
		return res, attn


class Encoder(nn.Module):
	def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
		super().__init__()
		self.attn_layers = nn.ModuleList(attn_layers)
		self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
		self.norm = norm_layer

	def forward(self, x, attn_mask=None):
		attns = []
		if self.conv_layers is not None:
			for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
				x, attn = attn_layer(x, attn_mask=attn_mask)
				x = conv_layer(x)
				attns.append(attn)
			x, attn = self.attn_layers[-1](x)
			attns.append(attn)
		else:
			for attn_layer in self.attn_layers:
				x, attn = attn_layer(x, attn_mask=attn_mask)
				attns.append(attn)

		if self.norm is not None:
			x = self.norm(x)

		return x, attns


class DecoderLayer(nn.Module):
	def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
		super().__init__()
		d_ff = d_ff or 4 * d_model
		self.self_attention = self_attention
		self.cross_attention = cross_attention
		self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
		self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
		self.decomp1 = series_decomp(moving_avg)
		self.decomp2 = series_decomp(moving_avg)
		self.decomp3 = series_decomp(moving_avg)
		self.dropout = nn.Dropout(dropout)
		self.projection = nn.Conv1d(
			in_channels=d_model,
			out_channels=c_out,
			kernel_size=3,
			stride=1,
			padding=1,
			padding_mode="circular",
			bias=False,
		)
		self.activation = F.relu if activation == "relu" else F.gelu

	def forward(self, x, cross, x_mask=None, cross_mask=None):
		x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
		x, trend1 = self.decomp1(x)
		x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
		x, trend2 = self.decomp2(x)
		y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
		y = self.dropout(self.conv2(y).transpose(-1, 1))
		x, trend3 = self.decomp3(x + y)

		residual_trend = trend1 + trend2 + trend3
		residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
		return x, residual_trend


class Decoder(nn.Module):
	def __init__(self, layers, norm_layer=None, projection=None):
		super().__init__()
		self.layers = nn.ModuleList(layers)
		self.norm = norm_layer
		self.projection = projection

	def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
		for layer in self.layers:
			x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
			trend = trend + residual_trend

		if self.norm is not None:
			x = self.norm(x)

		if self.projection is not None:
			x = self.projection(x)
		return x, trend


class AutoformerCore(nn.Module):
	def __init__(
		self,
		enc_in,
		dec_in,
		c_out,
		seq_len,
		pred_len,
		label_len=0,
		output_attention=False,
		moving_avg=25,
		factor=1,
		d_model=128,
		n_heads=4,
		e_layers=2,
		d_layers=1,
		d_ff=None,
		dropout=0.1,
		activation="gelu",
		embed="fixed",
		freq="h",
		init_size=None,
	):
		super().__init__()
		self.seq_len = seq_len
		self.label_len = label_len
		self.pred_len = pred_len
		self.output_attention = output_attention
		self.c_out = c_out
		self.dec_in = dec_in

		self.decomp = series_decomp(moving_avg)
		self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
		self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

		self.encoder = Encoder(
			[
				EncoderLayer(
					AutoCorrelationLayer(
						AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=output_attention),
						d_model,
						n_heads,
					),
					d_model,
					d_ff,
					moving_avg=moving_avg,
					dropout=dropout,
					activation=activation,
				)
				for _ in range(e_layers)
			],
			norm_layer=my_Layernorm(d_model),
		)
		self.decoder = Decoder(
			[
				DecoderLayer(
					AutoCorrelationLayer(
						AutoCorrelation(True, factor, attention_dropout=dropout, output_attention=False),
						d_model,
						n_heads,
					),
					AutoCorrelationLayer(
						AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False),
						d_model,
						n_heads,
					),
					d_model,
					c_out,
					d_ff,
					moving_avg=moving_avg,
					dropout=dropout,
					activation=activation,
				)
				for _ in range(d_layers)
			],
			norm_layer=my_Layernorm(d_model),
			projection=nn.Linear(d_model, c_out, bias=True),
		)

		self.trend_projection = nn.Linear(enc_in, c_out) if enc_in != c_out else None
		self.init_enc_projection = nn.Linear(init_size, d_model) if init_size is not None else None
		self.init_trend_projection = nn.Linear(init_size, c_out) if init_size is not None else None

	def forward(
		self,
		x_enc,
		x_dec=None,
		x_mark_enc=None,
		x_mark_dec=None,
		enc_self_mask=None,
		dec_self_mask=None,
		dec_enc_mask=None,
		init_state=None,
	):
		batch_size = x_enc.shape[0]
		device = x_enc.device
		dtype = x_enc.dtype

		if x_dec is None:
			x_dec = torch.zeros(batch_size, self.pred_len, self.dec_in, device=device, dtype=dtype)

		mean = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.pred_len, 1)
		zeros = torch.zeros(batch_size, self.pred_len, x_dec.shape[2], device=device, dtype=dtype)
		_, trend_init = self.decomp(x_enc)
		if self.trend_projection is not None:
			trend_init = self.trend_projection(mean)
		else:
			trend_init = mean

		if init_state is not None:
			if self.init_enc_projection is not None:
				init_context = self.init_enc_projection(init_state).unsqueeze(1)
				init_trend = self.init_trend_projection(init_state).unsqueeze(1)
				trend_init = trend_init + init_trend
			else:
				init_context = None
		else:
			init_context = None

		seasonal_init = zeros

		enc_out = self.enc_embedding(x_enc, x_mark_enc)
		if init_context is not None:
			enc_out = enc_out + init_context
		enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

		dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
		seasonal_part, trend_part = self.decoder(
			dec_out,
			enc_out,
			x_mask=dec_self_mask,
			cross_mask=dec_enc_mask,
			trend=trend_init,
		)
		dec_out = trend_part + seasonal_part

		pred = dec_out[:, -self.pred_len :, :]
		hidden = enc_out

		if self.output_attention:
			return pred, attns
		return pred, hidden


class AngularVelocityAutoformer(nn.Module):
	def __init__(
		self,
		hidden_size: int = 128,
		seq_len: int = 256,
		n_heads: int = 4,
		e_layers: int = 2,
		d_layers: int = 1,
		d_ff: int | None = None,
		moving_avg: int = 25,
		dropout: float = 0.1,
		activation: str = "gelu",
		factor: float = 1.0,
		embed: str = "fixed",
		freq: str = "h",
	):
		super().__init__()
		self.core = AutoformerCore(
			enc_in=1,
			dec_in=2,
			c_out=2,
			seq_len=seq_len,
			pred_len=seq_len,
			label_len=0,
			output_attention=False,
			moving_avg=moving_avg,
			factor=factor,
			d_model=hidden_size,
			n_heads=n_heads,
			e_layers=e_layers,
			d_layers=d_layers,
			d_ff=d_ff,
			dropout=dropout,
			activation=activation,
			embed=embed,
			freq=freq,
			init_size=2,
		)

	def forward(self, velocity: torch.Tensor, init_pos: torch.Tensor | None = None):
		return self.core(velocity, init_state=init_pos)


class MemoryGuidedSaccadeAutoformer(nn.Module):
	def __init__(
		self,
		hidden_size: int = 128,
		seq_len: int = 512,
		n_heads: int = 4,
		e_layers: int = 2,
		d_layers: int = 1,
		d_ff: int | None = None,
		moving_avg: int = 25,
		dropout: float = 0.1,
		activation: str = "gelu",
		factor: float = 1.0,
		embed: str = "fixed",
		freq: str = "h",
	):
		super().__init__()
		self.core = AutoformerCore(
			enc_in=3,
			dec_in=2,
			c_out=2,
			seq_len=seq_len,
			pred_len=seq_len,
			label_len=0,
			output_attention=False,
			moving_avg=moving_avg,
			factor=factor,
			d_model=hidden_size,
			n_heads=n_heads,
			e_layers=e_layers,
			d_layers=d_layers,
			d_ff=d_ff,
			dropout=dropout,
			activation=activation,
			embed=embed,
			freq=freq,
			init_size=None,
		)

	def forward(self, inputs: torch.Tensor):
		return self.core(inputs)


class DoubleAngularVelocityAutoformer(nn.Module):
	def __init__(
		self,
		hidden_size: int = 128,
		seq_len: int = 256,
		n_heads: int = 4,
		e_layers: int = 2,
		d_layers: int = 1,
		d_ff: int | None = None,
		moving_avg: int = 25,
		dropout: float = 0.1,
		activation: str = "gelu",
		factor: float = 1.0,
		embed: str = "fixed",
		freq: str = "h",
	):
		super().__init__()
		self.core = AutoformerCore(
			enc_in=2,
			dec_in=4,
			c_out=4,
			seq_len=seq_len,
			pred_len=seq_len,
			label_len=0,
			output_attention=False,
			moving_avg=moving_avg,
			factor=factor,
			d_model=hidden_size,
			n_heads=n_heads,
			e_layers=e_layers,
			d_layers=d_layers,
			d_ff=d_ff,
			dropout=dropout,
			activation=activation,
			embed=embed,
			freq=freq,
			init_size=4,
		)

	def forward(self, velocity: torch.Tensor, init_pos: torch.Tensor | None = None):
		return self.core(velocity, init_state=init_pos)


#AngularVelocityIntegrationAutoformer = AngularVelocityAutoformer
#MemoryGuidedSaccadeIntegrationAutoformer = MemoryGuidedSaccadeAutoformer
#DoubleAngularVelocityIntegrationAutoformer = DoubleAngularVelocityAutoformer


#__all__ = [
#	"AngularVelocityAutoformer",
#	"AngularVelocityIntegrationAutoformer",
#	"MemoryGuidedSaccadeAutoformer",
#	"MemoryGuidedSaccadeIntegrationAutoformer",
#	"DoubleAngularVelocityAutoformer",
#	"DoubleAngularVelocityIntegrationAutoformer",
#]
