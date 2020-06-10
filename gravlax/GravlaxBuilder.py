from gravlax.Gravlax import Gravlax
import re

class GravlaxBuilder():

    def __init__(self, input_length, num_train, num_validate, cnn_filters=256, lstm_units=200):
        self.input_length = input_length
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.rnn_padding = 0
        self.maxpool_layer = 0
        self.batch_normalization = False
        self.dropout = False
        self.none_input = False
        self.model_name = f"gravlax{num_train}{num_validate}-{cnn_filters}CNN-{lstm_units}LSTM"

    def with_batch_normalization(self):
        self.batch_normalization = True
        self.model_name += "-bn"
        return self

    def with_rnn_padding(self, rnn_padding):
        self.rnn_padding = rnn_padding
        self.model_name += f"-pad{rnn_padding}"
        return self

    def with_maxpool(self, maxpool_layer):
        self.maxpool_layer = maxpool_layer
        self.model_name += f"-maxpool{maxpool_layer}"
        return self

    def with_dropout(self):
        self.dropout = True
        self.model_name += f"-dropout"
        return self
    
    '''
    Sets the input to the first CNN layer to None
    This ignores `input_length`
    
    Used for prediction where we can thus predict inputs of any length
    '''
    def with_None_input(self):
        self.none_input = True
        return self
    
    def build(self):
        return Gravlax(
            input_length=self.input_length,
            cnn_filters=self.cnn_filters,
            lstm_units=self.lstm_units,
            rnn_padding = self.rnn_padding,
            batch_normalization = self.batch_normalization,
            maxpool_layer=self.maxpool_layer,
            model_name=self.model_name,
            dropout=self.dropout,
            use_None_input=self.none_input)


'''
makes a gravlax for the model file
loads the weights
returns name of model and predict func
'''
def gravlax_for_file(input_length, file, num_train, num_validate, with_None_input=False, use_our_predict=False):
    description = file.split("/")[1]
    if "CNN" in description:
        cnn = int(re.findall(r"\d+CNN", description)[0][:-3])
        lstm = int(re.findall(r"\d+LSTM", description)[0][:-4])
    else:
        cnn = 256
        lstm = 200

    gb = GravlaxBuilder(input_length, num_train, num_validate, cnn_filters=cnn, lstm_units=lstm)
    if "bn" in description:
        gb = gb.with_batch_normalization()
    if "pad5" in description:
        gb = gb.with_rnn_padding(5)
    if "maxpool3" in description:
        gb = gb.with_maxpool(3)
    if with_None_input:
        gb = gb.with_None_input()
    gravlax = gb.build()
    gravlax.load_weights(file)
    if use_our_predict:
        return (gravlax.name, gravlax.predict)
    return (gravlax.name, gravlax.predict_beam_search) # using get_model_name instead of description for safety