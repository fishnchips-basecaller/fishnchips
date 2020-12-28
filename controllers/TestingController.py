import mappy as mp
import json
import time
import datetime
import os 

from utils.process_utils import get_generator
from utils.attention_evaluation_utils import evaluate_batch
from utils.assembler import assemble, assemble_and_output
from utils.Other import analyse_cigar
from utils.preprocessing.chiron_files.chiron_data_utils import process_label_str

class TestingController():
    def __init__(self, model_config, test_config, model_filepath):
        self._reference_path = test_config['data']
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

    def _get_cig_result(self, assembly, aligner, read_id):
        try:
            besthit = next(aligner.map(assembly))
            return {
                    'read_id':read_id,
                    'ctg': besthit.ctg,
                    'r_st': besthit.r_st,
                    'r_en': besthit.r_en,
                    'NM': besthit.NM,
                    'blen': besthit.blen,
                    'cig': analyse_cigar(besthit.cigar_str),
                    'cigacc_old': 1-(besthit.NM/besthit.blen),
                    'cigacc': besthit.mlen / besthit.blen
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
                'cigacc_old':0,
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

    def _set_aligner(self, reference_str):
        pass

    def _compute_reference(self, read_id):
        try:
            path = f'{self._reference_path}/{read_id}.label'
            with open(path, 'r') as f:
                data = f.read()
            reference_queue, _ = process_label_str(data, as_bases=True)
            return ''.join(list(reference_queue))
            
        except Exception as e:
            print(f'*** unable to get reference for read id {read_id}')
            return ''

    def test(self, model):
        print("*** testing...")

        for read in range(len(self._result_dic), self._reads):
            try:
                x_windows, y_windows, _, _, read_id = next(self._generator.get_window_batch(label_as_bases=True))
                
                reference = self._compute_reference(read_id)
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

                path = f'temps/{read_id}.txt'
                with open(path, 'w') as f:
                    f.write(f'>{read_id}\n')
                    f.write(reference)             

                aligner = mp.Aligner(path)
                result = self._get_cig_result(assembly, aligner, read_id)
                os.remove(path)

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
