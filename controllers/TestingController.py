import mappy as mp
import json
import time
import datetime
import os 

from utils.process_utils import get_generator
from utils.attention_evaluation_utils import evaluate_batch
from utils.assembler import assemble
from utils.Other import analyse_cigar

class TestingController():
    def __init__(self, model_config, test_config, model_filepath):
        self._generator = get_generator(model_config, test_config, kind="testing")
        self._reads = test_config['reads']
        self._batch_size = test_config['batch_size']
        self._aligner = mp.Aligner("../useful_files/zymo-ref-uniq_2019-03-15.fa")
        self._with_assembler = model_config['encoder_max_length'] == test_config['stride']
        self._model_file_path = model_filepath
        self._result_dic = self._get_result_dic(self._model_file_path)

    def _get_result_dic(self, model_filepath):
        if os.path.isfile(f"{model_filepath}_evaluation.json"):
            answer = input("*** This models evaluation already exists, do you want to append to it [Y/n]?:")
            if answer not in "Nn" or answer == "":
                with open(f"{model_filepath}_evaluation.json", "r") as f:
                    return json.load(f)
        return []

    def _get_cig_result(self, assembly, read_id):
        try:
            besthit = next(self._aligner.map(assembly))
            return {
                    'read_id':read_id,
                    'ctg': besthit.ctg,
                    'r_st': besthit.r_st,
                    'r_en': besthit.r_en,
                    'NM': besthit.NM,
                    'blen': besthit.blen,
                    'cig': analyse_cigar(besthit.cigar_str),
                    'cigacc': 1-(besthit.NM/besthit.blen)
                }    
        except:
            return {
                'read_id':read_id,
                'ctg': 0,
                'r_st': 0,
                'r_en': 0,
                'NM': 0,
                'blen': 0,
                'cig': 0,
                'cigacc': 0
            }

    def _get_assembly(self, y_pred):
        if self._with_assembler:
            return "".join(y_pred)
        return assemble(y_pred)

    def _pretty_print_progress(self, current_begin, current_end, total):
        progstr = "["
        for i in range(0, total, total//50):
            if i>=current_begin and i<current_end:
                progstr += "x"
            else:
                progstr += "-"
        progstr += "]"
        return progstr

    def test(self, model):
        print("*** testing...")

        for read in range(len(self._result_dic), self._reads):
            try:
                x_windows, y_windows, _, _, read_id = next(self._generator.get_window_batch(label_as_bases=True))
                nr_windows = len(x_windows)

                assert nr_windows == len(y_windows)
                start_time = time.time()

                y_pred = []
                for b in range(0,nr_windows, self._batch_size):
                    x_batch = x_windows[b:b+self._batch_size]
                    print(f"{read:02d}/{self._reads:02d} Predicting windows {self._pretty_print_progress(b, b+len(x_batch), nr_windows)} {b:04d}-{b+len(x_batch):04d}/{nr_windows:04d}", end="\r")

                    # y_batch_true = y_windows[b:b+config['BATCH_SIZE']]
                    y_batch_pred, _ = evaluate_batch(x_batch, model, len(x_batch), as_bases=True)
                    y_pred.extend(y_batch_pred)

                assembly = self._get_assembly(y_pred)
                result = self._get_cig_result(assembly, read_id)
                result['time'] = time.time() - start_time
                self._result_dic.append(result)

                print(f"{read:02d}/{self._reads} Done read... cigacc {result['cigacc']}"+" "*50) # 50 blanks to overwrite the previous print
                with open(f"{self._model_file_path}_evaluation.json", 'w') as jsonfile:
                    json.dump(self._result_dic, jsonfile, indent=4)
                with open(f"{self._model_file_path}_evaluation.fa", 'a') as f:
                    f.write(f"@{read_id};{round(time.time())};{datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n")
                    f.write(f"{assembly}\n")
            except Exception as e:
                print(e)
