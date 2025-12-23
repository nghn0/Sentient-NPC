# transformer_chatbot.py
# --- FIX: Changed all 'keras' imports to 'tensorflow.keras' ---
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout, LayerNormalization, Input, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
# --- END FIX ---
import tensorflow as tf

# -------------------------
# Serializable Mask Expansion Layer
# -------------------------
@register_keras_serializable()
class ExpandMask(Layer):
    """
    Converts mask (batch, seq_len) -> (batch, 1, seq_len) for MultiHeadAttention
    """
    def call(self, mask):
        if mask is None:
            return None
        return tf.cast(mask[:, tf.newaxis, :], tf.bool)

# -------------------------
# Positional Encoding Layer
# -------------------------
@register_keras_serializable()
class PositionalEncoding(Layer):
    def __init__(self, max_len=64, dim=128, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_len = max_len
        self.pos_embedding = Embedding(input_dim=max_len, output_dim=dim)

    def build(self, input_shape):
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            tf.print(
                "Warning: PositionalEncoding dim does not match input embedding dim:",
                "pos_dim=", self.dim, "inp_dim=", input_shape[-1]
            )
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(seq_len)[tf.newaxis, :]
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb

    def compute_mask(self, inputs, mask=None):
        return mask

# -------------------------
# Encoder Block
# -------------------------
def EncoderBlock(x, mask=None, dim=128, heads=4, dim_ff=256, dropout=0.1):
    x_norm = LayerNormalization()(x)
    attn_mask = ExpandMask()(mask) if mask is not None else None
    attn = MultiHeadAttention(num_heads=heads, key_dim=dim // heads, dropout=dropout)(
        x_norm, x_norm, attention_mask=attn_mask
    )
    x = x + attn
    x_norm = LayerNormalization()(x)
    ffn = Dense(dim_ff, activation='gelu')(x_norm)
    ffn = Dropout(dropout)(ffn)
    ffn = Dense(dim)(ffn)
    return x + ffn

# -------------------------
# Decoder Block
# -------------------------
@register_keras_serializable()
class DecoderBlock(Layer):
    def __init__(self, dim=128, heads=4, dim_ff=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.self_attn = MultiHeadAttention(num_heads=heads, key_dim=dim // heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(num_heads=heads, key_dim=dim // heads, dropout=dropout)
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.ln3 = LayerNormalization()
        self.ffn1 = Dense(dim_ff, activation='gelu')
        self.ffn2 = Dense(dim)
        self.dropout = Dropout(dropout)

    def call(self, x, enc, training=False, mask=None, enc_mask=None):
        batch_size = tf.shape(x)[0]
        tgt_len = tf.shape(x)[1]

        # Self-attention (causal + padding)
        if mask is not None:
            causal = tf.linalg.band_part(tf.ones((tgt_len, tgt_len), dtype=tf.bool), -1, 0)
            causal = tf.expand_dims(causal, 0)
            causal = tf.tile(causal, [batch_size, 1, 1])
            pad = tf.cast(mask[:, tf.newaxis, :], tf.bool)
            self_attn_mask = tf.logical_and(causal, pad)
        else:
            self_attn_mask = None

        x_norm = self.ln1(x)
        attn1 = self.self_attn(x_norm, x_norm, x_norm, attention_mask=self_attn_mask)
        x = x + attn1

        # Cross-attention
        if enc_mask is not None:
            enc_attn_mask = tf.cast(enc_mask[:, tf.newaxis, :], tf.bool)
        else:
            enc_attn_mask = None

        x_norm = self.ln2(x)
        attn2 = self.cross_attn(x_norm, enc, enc, attention_mask=enc_attn_mask)
        x = x + attn2

        # Feed-forward
        x_norm = self.ln3(x)
        ffn_out = self.ffn2(self.dropout(self.ffn1(x_norm), training=training))
        return x + ffn_out

    def compute_mask(self, inputs, mask=None):
        return mask

# -------------------------
# Transformer Builder
# -------------------------
def build_transformer(Q_vocab_size,
                      A_vocab_size,
                      dim=128,
                      heads=4,
                      blocks=4,
                      dim_ff=256,
                      dropout=0.1,
                      max_positional_len=64):
    vocab_size = max(Q_vocab_size, A_vocab_size)
    shared_emb = Embedding(input_dim=vocab_size, output_dim=dim, mask_zero=True, name="shared_embedding")

    # Encoder
    enc_in = Input(shape=(None,), dtype=tf.int32, name="encoder_input")
    enc_mask = shared_emb.compute_mask(enc_in)
    enc_x = shared_emb(enc_in)
    enc_x = PositionalEncoding(max_len=max_positional_len, dim=dim, name="pos_enc")(enc_x)
    enc_x = Dropout(dropout)(enc_x)
    for i in range(blocks):
        enc_x = EncoderBlock(enc_x, mask=enc_mask, dim=dim, heads=heads, dim_ff=dim_ff, dropout=dropout)

    # Decoder
    dec_in = Input(shape=(None,), dtype=tf.int32, name="decoder_input")
    dec_mask = shared_emb.compute_mask(dec_in)
    dec_x = shared_emb(dec_in)
    dec_x = PositionalEncoding(max_len=max_positional_len, dim=dim, name="pos_dec")(dec_x)
    dec_x = Dropout(dropout)(dec_x)
    for i in range(blocks):
        dec_x = DecoderBlock(dim=dim, heads=heads, dim_ff=dim_ff, dropout=dropout, name=f"decoder_block_{i}")(
            dec_x, enc_x, mask=dec_mask, enc_mask=enc_mask
        )

    # Output logits
    logits = Dense(A_vocab_size, name="logits_dense")(dec_x)

    model = Model([enc_in, dec_in], logits, name="chat_transformer")
    return model

# -------------------------
# Greedy decode helper
# -------------------------
def greedy_decode(transformer_model, tokenizer, encoder_input_ids, start_token_id, end_token_id, max_len=64):
    if isinstance(encoder_input_ids, (list, tuple)):
        encoder_input_ids = tf.constant([encoder_input_ids], dtype=tf.int32)
    elif len(tf.shape(encoder_input_ids)) == 1:
        encoder_input_ids = tf.expand_dims(encoder_input_ids, 0)

    batch_size = tf.shape(encoder_input_ids)[0]
    decoded = tf.constant([[start_token_id]] * batch_size, dtype=tf.int32)

    for _ in range(max_len - 1):
        logits = transformer_model([encoder_input_ids, decoded])
        next_logits = logits[:, -1, :]
        next_id = tf.argmax(next_logits, axis=-1, output_type=tf.int32)
        next_id = tf.expand_dims(next_id, axis=1)
        decoded = tf.concat([decoded, next_id], axis=1)
        if tf.reduce_all(tf.equal(next_id, end_token_id)):
            break

    return decoded.numpy().tolist()