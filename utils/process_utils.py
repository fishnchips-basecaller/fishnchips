from utils.preprocessing.AttentionDataGenerator import AttentionDataGenerator

def get_generator(model_config, process_config, kind=""):
    data = process_config['data']
    bacteria = process_config['bacteria']
    stride = process_config['stride']
    print(f"*** constructin {kind} generator...")
    return AttentionDataGenerator(
        data, 
        bacteria, 
        process_config['batch_size'], 
        stride, 
        model_config['encoder_max_length'], 
        model_config['decoder_max_length'])