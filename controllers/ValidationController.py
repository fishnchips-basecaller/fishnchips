import mappy as mp

from utils.process_utils import get_generator
from utils.attention_evaluation_utils import build, evaluate_batch
from utils.assembler import assemble
import editdistance

class ValidationController():
    def __init__(self, model_config, validation_config):
        self._generator = get_generator(model_config, validation_config, kind="validation")
        self._reads = validation_config['reads']
        self._batch_size = validation_config['batch_size']
        self._aligner = mp.Aligner("../useful_files/zymo-ref-uniq_2019-03-15.fa")
        assert validation_config['stride'] <= model_config['encoder_max_length']
        self._with_assembler = model_config['encoder_max_length'] == validation_config['stride']

        assert validation_config['algorithm'] in ["assembly","editdistance"]
        self._val_algo = validation_config['algorithm']

    def get_cig_loss(self, assembly):
        try:
            besthit = next(self._aligner.map(assembly))
            return besthit.NM/besthit.blen
        except:
            return 0

    def get_assembly(self, y_pred):
        if self._with_assembler:
            return "".join(y_pred)
        return assemble(y_pred)

    def validate(self, model):
        print("*** validating...")

        if self._val_algo == "editdistance":
            return self.validate_editdistance(model)
        elif self._val_algo == "assembly":
            return self.validate_assembly(model)
        else:
            raise Exception("Algorithm mistyped")

    def validate_assembly(self, model):

        val_loss = 0
        performed = 0

        for r in range(self._reads):
            print(f"{r+1}/{self._reads}", end="\r")
            try:
                x_windows, y_windows, _, _, _ = next(self._generator.get_window_batch(label_as_bases=True))
                nr_windows = len(x_windows)

                assert nr_windows == len(y_windows)

                y_pred = []
                for b in range(0,nr_windows,self._batch_size):
                    x_batch = x_windows[b:b+self._batch_size]          
                    y_batch_pred, _ = evaluate_batch(x_batch, model, len(x_batch), as_bases=True)
                    y_pred.extend(y_batch_pred)

                assembly = self.get_assembly(y_pred)
                acc = self.get_cig_loss(assembly)
                val_loss += acc
                performed += 1
            except Exception as e:
                print(e)

        if performed == 0:
            return 0
        return val_loss/performed # not using READS_TO_VALIDATE in case of error caught by catch

    def validate_editdistance(self, model):

            val_loss = 0
            performed = 0

            for r in range(self._reads):
                print(f"*** validating {r+1}/{self._reads}", end="\r")
                try:
                    x_windows, y_windows, _, _, _ = next(self._generator.get_window_batch(label_as_bases=True))
                    nr_windows = len(x_windows)

                    assert nr_windows == len(y_windows)

                    y_pred = []
                    for b in range(0,nr_windows,self._batch_size):
                        x_batch = x_windows[b:b+self._batch_size] 
                        y_batch_pred, _ = evaluate_batch(x_batch, model, len(x_batch), as_bases=True)
                        y_pred.extend(y_batch_pred)

                    tot_ed = 0
                    for pred, gt in zip(y_pred, y_windows):
                        tot_ed += editdistance.eval(pred, gt)
                    aed = tot_ed / len(y_pred)

                    val_loss += aed
                    performed += 1
                except Exception as e:
                    print(e)

            if performed == 0:
                return 0
            return val_loss/performed # not using READS_TO_VALIDATE in case of error caught by catch
