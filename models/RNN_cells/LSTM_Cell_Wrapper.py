from tensorflow.keras.layers import Layer, LSTMCell

class LSTM_Cell_Wrapper(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)	      
        self.units = units	       
        self.state_size = (units,units)	       
        self.lstm_cell = LSTMCell(self.units)
    
    def call(self, inputs, states):	    
        outputs, new_states = self.lstm_cell(inputs, states)
        return outputs, new_states 