from typing import Dict, Any

from tensorflow import keras

from kashgari.layers import L
from kashgari.tasks.classification.abc_model import ABCClassificationModel


class FC_Model(ABCClassificationModel):
    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_output': {

            },
        }

    def build_model_arc(self) -> None:
        output_dim = self.label_processor.vocab_size
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # Single FC
        layer_stack = [
            L.Dense(output_dim, **config['layer_output']),
            self._activation_layer()
        ]

        x = embed_model.output
        x = keras.layers.Lambda(lambda x: x[:, 0])(x)   # 第一维为 CLS 直接拿出来
        for layer in layer_stack:
            x = layer(x)

        self.tf_model: keras.Model = keras.Model(embed_model.inputs, x)