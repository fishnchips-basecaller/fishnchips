import sys
import json
import os
import datetime

from models.FishNChips import FishNChips
from controllers.TrainingController import TrainingController
from controllers.TestingController import TestingController

def create_path(path):
    if os.path.exists(path):
        answer = input(f"*** path {path} already exist, are you sure you want to overwrite it? [y/N]:")
        if answer not in "Yy":
            sys.exit()
    else:
        os.makedirs(path)

def load_config(filepath):
    print(f"*** loading config {filepath}...")
    with open(filepath, "r") as f:
        return json.load(f)

def get_model_save_filepath(model_config, train_config, validation_config, run_name):
    try:
        path = f"./trained_models/{run_name}"
        create_path(path)

        model_filepath = f"{path}/fishnchips{len(train_config['bacteria'])}{len(validation_config['bacteria'])}_{model_config['d_model']}_{model_config['cnn_blocks']}CNN_{model_config['num_heads']}H_{model_config['attention_blocks']}B"
        if model_config['maxpool_kernel'] != 2:
            model_filepath = f"{model_filepath}_{model_config['maxpool_kernel']}MPK"  

        print(f"model name:{model_filepath}")    
        return model_filepath
    except OSError:
        print (f"*** failed to create directory {os.getcwd()}/trained_models/{run_name}")

def make_fish_from_config(model_config):
    print("*** making fish...")
    return FishNChips(
        num_cnn_blocks=model_config['cnn_blocks'], 
        max_pool_layer_idx=model_config['maxpool_idx'], 
        max_pool_kernel_size=model_config['maxpool_kernel'],
        num_layers=model_config['attention_blocks'], 
        d_model=model_config['d_model'], 
        output_dim=1 + 4 + 1 + 1, # PAD + ATCG + START + STOP
        num_heads=model_config['num_heads'],
        dff=model_config['dff'], 
        pe_encoder_max_length=model_config['encoder_max_length'], 
        pe_decoder_max_length=model_config['decoder_max_length'], 
        rate=model_config['dropout_rate'])

def print_run_parameters(config):
    model_config, training_config, validation_config, test_config = split_config(config)

    print("MODEL parameters")
    print(json.dumps(model_config, indent=4, sort_keys=True))

    print("TRAINING parameters")
    print(json.dumps(training_config, indent=4, sort_keys=True))

    print("VALIDATION parameters")
    print(json.dumps(validation_config, indent=4, sort_keys=True))

    print("TEST parameters")
    print(json.dumps(test_config, indent=4, sort_keys=True))
    input("*** verify parameters")

def split_config(config):
    return config['model'], config['train'], config['validate'], config['test']

def run(config_file, run_name):
    config = load_config(config_file)
    print_run_parameters(config)
    model_config, training_config, validation_config, test_config = split_config(config)

    model_filepath = get_model_save_filepath(model_config, training_config, validation_config, run_name)
    model = make_fish_from_config(model_config)

    training_controller = TrainingController(model_config, training_config, validation_config, model, model_filepath)
    test_controller = TestingController(model_config, test_config, model_filepath)

    trained_model = training_controller.train()
    test_controller.test(trained_model)

if __name__ == "__main__":
    config_file = sys.argv[1]
    if len(sys.argv) > 1:
        run_name = sys.argv[2]
    else:
        run_name = datetime.datetime.now().strftime('%m%d%Y-%H%M%S')
    run(config_file, run_name)