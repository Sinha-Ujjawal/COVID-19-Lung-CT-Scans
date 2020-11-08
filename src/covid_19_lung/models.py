from tensorflow import keras
import tensorflow_hub as hub


class Covid19Lung_BiTModel(keras.Model):
    def __init__(self, bit_module: hub.KerasLayer) -> None:
        super().__init__()
        self._bit_module = bit_module


class Covid19Lung_BiTModel_s_r50x1(Covid19Lung_BiTModel):
    BIT_URL = "https://tfhub.dev/google/bit/s-r50x1/1"

    def __init__(self) -> None:
        super().__init__(bit_module=hub.KerasLayer(self.BIT_URL))
