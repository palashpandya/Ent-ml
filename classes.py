from dataclasses import dataclass
from typing import Callable
from typing import Any, List

import tensorflow as tf
from ame4d import AME_loss, generate_TT_data, early_stop


@dataclass
class Parameters:
    numLayers : int = 1
    inputLayerWidth : int = 2
    outputLayerWidth : int = 2
    layerWidth: int = inputLayerWidth + outputLayerWidth
    optimizer: str = "adam"
    activation: str = "selu"
    lossFunction : Callable[[Any],Any] = AME_loss
    generateData : Callable[[Any],Any] = generate_TT_data
    callbackFunctions : List[Callable[[Any],Any]] | None = None

class NN:
    parameters : Parameters = Parameters()
    batchsize : int
    numEpochs : int
    stepsPerEpoch : int

    model : tf.keras.models.Sequential

    def __init__(self, data : Parameters):
        self.parameters = data
        self.model= tf.keras.models.Sequential([tf.keras.Input(
            shape=(self.parameters.inputLayerWidth,))])

    def make_model(self):
        mod = self.model
        try:
            for i in range(self.parameters.numLayers):
                mod.add(tf.keras.layers.Dense(self.parameters.layerWidth,
                                                     activation=self.parameters.activation))
            mod.add(tf.keras.layers.Dense(self.parameters.outputLayerWidth,
                                                 activation=self.parameters.activation))
        except Exception as e:
            print("Failed to make the Sequential model: ", e)

    def compile_model(self):
        mod = self.model
        try:
            mod.compile(optimizer=self.parameters.optimizer,
                           loss=self.parameters.lossFunction,
                           metrics=[self.parameters.lossFunction])
        except Exception as e:
            print("Model compilation failed with the Exception: ", e)

    def fit_model(self):
        try:
            mod = self.model
            mod.fit(
                self.parameters.generateData(self.parameters.inputLayerWidth),
                batch_size = self.batchsize,
                epochs = self.numEpochs,
                steps_per_epoch = self.stepsPerEpoch,
                validation_data = self.parameters.generateData(self.parameters.inputLayerWidth),
                callbacks = self.parameters.callbackFunctions)
        except Exception as e:
            print("Failed to fit the model: ", e)