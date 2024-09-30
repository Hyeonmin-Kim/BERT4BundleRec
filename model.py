import copy
import json
from typing import Optional
import six

import tensorflow as tf
import keras



class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
        vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
        hidden_size: Size of the encoder layers and the pooler layer.
        num_hidden_layers: Number of hidden layers in the Transformer encoder.
        num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
        intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
        hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
        max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
        type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
        initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    

class BertEmbedding(keras.layers.Layer):
    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 initializer_range=0.02,
                 use_one_hot_embeddings=False,
                 use_token_type=False,
                 token_type_vocab_size=16,
                 use_position_embeddings=True,
                 max_position_embeddings=512,
                 word_embedding_name="word_embeddings",
                 token_type_embedding_name="token_type_embeddings",
                 position_embedding_name="position_embeddings",
                 dropout_prob=0.1):
        super().__init__(name=word_embedding_name)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_rage = initializer_range
        self.use_one_hot_embeddings = use_one_hot_embeddings

        self.word_embedding_layer = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range), # type: ignore
            name=word_embedding_name
        )

        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size

        if self.use_token_type:
            self.token_type_embedding_layer = keras.layers.Embedding(
                input_dim=self.token_type_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range), # type: ignore
                name=token_type_embedding_name
            )

        self.use_position_embeddings = use_position_embeddings
        self.max_position_embeddings = max_position_embeddings
        
        if self.use_position_embeddings:
            self.position_embedding_layer = keras.layers.Embedding(
                input_dim=self.max_position_embeddings,
                output_dim=self.embedding_size,
                embeddings_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range), # type: ignore
                name=position_embedding_name
            )

        self.dropout_prob = dropout_prob
        self.dropout = keras.layers.Dropout(rate=self.dropout_prob)

        self.layernorm = keras.layers.LayerNormalization()

    def build(self, input_shapes: tuple):
        seq_length = input_shapes[1]
        if self.use_one_hot_embeddings:
            self.word_embedding_layer.build((None, seq_length))

    def call(self, input_ids: tf.Tensor, token_type_ids: Optional[tf.Tensor] = None):
        # The shape of input_ids are standardized to [batch, seqlen]
        input_shape = input_ids.shape
        batch_size, seq_length = input_shape[0], input_shape[1]

        if self.use_one_hot_embeddings:
            flat_input_ids = tf.reshape(input_ids, [-1])
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=self.vocab_size)
            embedding_table = self.word_embedding_layer.get_weights()[0]
            output = tf.matmul(one_hot_input_ids, embedding_table)
        else:
            output = self.word_embedding_layer(input_ids)
        output = tf.reshape(output, [batch_size, seq_length, self.embedding_size])

        if self.use_token_type:
            assert token_type_ids is not None, "token_type_ids` must be specified if `use_token_type` is True."
            token_type_embedding = self.token_type_embedding_layer(token_type_ids)
            output += token_type_embedding
        
        if self.use_position_embeddings:
            assert seq_length <= self.max_position_embeddings  # type: ignore
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
            positional_embedding = self.position_embedding_layer(position_ids)
            output += positional_embedding

        output = self.layernorm(output)
        output = self.dropout(output)

        return output
    
    def get_embedding_tables(self):
        return {
            'word_embeddings': self.word_embedding_layer.embeddings,
            'token_type_embeddings': self.token_type_embedding_layer.embeddings if self.use_token_type else None,
            'position_embeddings': self.position_embedding_layer.embeddings if self.use_position_embeddings else None,
        }


class BertSelfAttention(keras.layers.Layer):
    def __init__(self,
                 num_attention_heads=1,
                 size_per_heads=512,
                 attention_probs_dropout_prob=0.0, 
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1):
        super().__init__()
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=size_per_heads,
            dropout=attention_probs_dropout_prob,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range), # type: ignore
        )
        self.dropout = keras.layers.Dropout(hidden_dropout_prob)
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, attention_mask: tf.Tensor):
        attention_output = self.attention(query=query, key=key, value=value, attention_mask=attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm(query + attention_output)
        return attention_output
    

class BertFeedForward(keras.layers.Layer):
    def __init__(self,
                 intermediate_size=3072,
                 hidden_size=768,
                 intermediate_act_fn=keras.activations.gelu,
                 hidden_drop_prob=0.1,
                 initializer_range=0.02):
        super().__init__()

        self.dense1 = keras.layers.Dense(
            intermediate_size, 
            activation=intermediate_act_fn,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range) # type: ignore
        )
        self.dense2 = keras.layers.Dense(
            hidden_size,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range) # type: ignore
        )
        self.dropout = keras.layers.Dropout(hidden_drop_prob)
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, input_tensor: tf.Tensor):
        intermediate_output = self.dense1(input_tensor)
        layer_output = self.dense2(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layer_norm(input_tensor + layer_output)
        return layer_output


class BertTransformer(keras.layers.Layer):
    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 intermediate_act_fn=keras.activations.gelu,
                 hidden_drop_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02):
        super().__init__()
        
        assert hidden_size % num_attention_heads == 0, \
            f"The hidden size {hidden_size} is not a multiple of the number of attention heads {num_attention_heads}."
        attention_head_size = int(hidden_size / num_attention_heads)

        self.attention = BertSelfAttention(
            num_attention_heads=num_attention_heads,
            size_per_heads=attention_head_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_drop_prob
        )

        self.feed_forward = BertFeedForward(
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            intermediate_act_fn=intermediate_act_fn,
            hidden_drop_prob=hidden_drop_prob,
            initializer_range=initializer_range
        )
    
    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor):
        attention_output = self.attention(input_tensor, input_tensor, input_tensor, attention_mask=attention_mask)
        transformer_output = self.feed_forward(attention_output)
        return transformer_output


class BertModel(keras.Model):
    def __init__(self, 
                 config: BertConfig,
                 use_one_hot_embeddings: bool = True):
        super().__init__()
        config = copy.deepcopy(config)

        self.embeddings = BertEmbedding(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            use_one_hot_embeddings=use_one_hot_embeddings,
            use_token_type=True,
            token_type_vocab_size=config.type_vocab_size,
            use_position_embeddings=True,
            max_position_embeddings=config.max_position_embeddings,
            word_embedding_name="word_embeddings",
            token_type_embedding_name="token_type_embeddings",
            position_embedding_name="position_embeddings",
            dropout_prob=config.hidden_dropout_prob)
        
        self.transformer = BertTransformer(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range
        )

    def call(self, inputs: dict):
        input_ids = inputs['input_ids']
        input_mask = inputs.get('input_mask', tf.ones_like(input_ids, dtype=tf.int32))
        token_type_ids = inputs.get('token_type_ids', tf.zeros_like(input_ids, dtype=tf.int32))

        embedding = self.embeddings(input_ids, token_type_ids=token_type_ids)

        attention_mask = self._create_attention_mask_from_input_mask(input_ids, input_mask)
        transformer_output = self.transformer(embedding, attention_mask)

        return transformer_output
    
    def _create_attention_mask_from_input_mask(self, from_tensor: tf.Tensor, to_mask: tf.Tensor):
        """Create 3D attention mask from a 2D tensor mask.

        Args:
            from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
            to_mask: int32 Tensor of shape [batch_size, to_seq_length].

        Returns:
            float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
        from_shape = tf.shape(from_tensor)
        batch_size, from_seq_length = from_shape[0], from_shape[1] # type: ignore
        to_shape = tf.shape(to_mask)
        to_seq_length = to_shape[1] # type: ignore

        to_mask = tf.cast(
            tf.reshape(to_mask, [batch_size, 1, to_seq_length]), dtype=tf.float32
        ) # type: ignore

        broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
        mask = broadcast_ones * to_mask

        return mask