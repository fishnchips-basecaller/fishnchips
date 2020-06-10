#%% 
from gravlax.GravlaxBuilder import GravlaxBuilder
from gravlax.Callback import SaveCB

from utils.preprocessing.DataGenerator import DataGenerator
from utils.Other import set_gpu_growth

set_gpu_growth()

input_length = 300 
rnn_padding = 5
use_maxpool = True

filename = "/ssd/mapped_therest.hdf5"
train_bacteria = ["Bacillus", "Staphylococcus", "Lactobacillus", "Pseudomonas", "Listeria", "Enterococcus", "Salmonella"]
generator = DataGenerator(filename, bacteria=train_bacteria, batch_size=1000, input_length=input_length, stride=30, reads_count=5, rnn_pad_size=rnn_padding, use_maxpool=use_maxpool)

filename = "/ssd/mapped_umi11to5.hdf5"
test_bacteria = ["Escherichia"]
val_generator = DataGenerator(filename, bacteria=test_bacteria, batch_size=500, input_length=input_length, stride=150, reads_count=5, rnn_pad_size=rnn_padding, use_maxpool=use_maxpool)

#%%

gb = GravlaxBuilder(input_length, len(train_bacteria), len(test_bacteria), cnn_filters=256, lstm_units=250)\
        .with_rnn_padding(rnn_padding)\
        .with_batch_normalization()
gb = gb.with_maxpool(3) if use_maxpool else gb
gravlax=gb.build()

save_cb = SaveCB(gravlax, val_generator)\
    .withCheckpoints()\
    .withImageOutput()\
    .withPatience(patience=300)
save_cb = save_cb.withMaxPool() if use_maxpool else save_cb

#%%

for epoch in range(2000):
    if gravlax.stop_training:
        break
    try:
        X,y = next(generator.get_batch())
        gravlax.fit(X, y, initial_epoch=epoch, epochs=epoch+1, callbacks=[save_cb])
    except Exception as e:
        print(e)