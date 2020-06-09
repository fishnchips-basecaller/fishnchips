import mappy as mp
from collections import deque
import re

reference_file = "./temps/ref-uniq.fa"

def get_comparison(dna_pred, use_color=True):
    dna_pred, dna_true, dna_cigar = _align(dna_pred)
    dna_pred, dna_true = _compare(dna_pred, dna_true, dna_cigar, use_color)
    cig_analysis = _analyse_cigar(dna_cigar)
    return dna_pred, dna_true, cig_analysis

def get_miss_matches(dna_pred):
    dna_pred, dna_true, dna_cigar = _align(dna_pred)
    return _calc_miss_matches(dna_pred, dna_true, dna_cigar)

def output_comparison(dna_pred, dna_true, filename):
    with open(filename, 'a') as f:
        f.write(f"PRED: {dna_pred} \n")
        f.write(f"TRUE: {dna_true} \n")
        f.write('\n')

def print_comparison(dna_pred, dna_true):
    incr = 200
    for i in range(0, len(dna_pred), incr):
        print(f"Segments {i}:")
        print(f"PRED:{dna_pred[i:i+incr]}")
        print(f"TRUE:{dna_true[i:i+incr]}")

def print_mismatches(dna_pred, dna_true, amount_per_read=200):

    assert len(dna_pred) == len(dna_true)
    
    class style():
        RED = lambda x: f"\033[31m{x}\033[0m"
        GREEN = lambda x: f"\033[32m{x}\033[0m"

    str_len = 100
    for i in range(0, amount_per_read, str_len):
        for j in range(i, i+str_len):
            if(j >= len(dna_pred)):
                continue
            if dna_true[j] == dna_pred[j]:
                print(dna_pred[j], end="")
            else:
                print(style.RED(dna_pred[j]), end="")
        print()
        for j in range(i, i+str_len):
            if(j >= len(dna_pred)):
                continue
            if dna_true[j] == dna_pred[j]:
                print(dna_pred[j], end="")
            else:
                print(style.GREEN(dna_pred[j]), end="")
        print("\n----")

def get_mapping(pred):
    try:
        aligner = mp.Aligner(reference_file)
        return next(aligner.map(pred))
    except Exception as e:
        print(e)
        return None

def get_reference(key):
    if not key:
        raise Exception(f"Attempting to get reference with a non-key:{key}")
    with open(reference_file, "r") as f:
        ref_file_str = f.read()
        idx = ref_file_str.find(key)
        if idx == -1:
            raise Exception("Didnt match reference file.")
        start_idx = idx + len(key)
        end_idx = ref_file_str.find(">", start_idx)
        if end_idx == -1:
            end_idx = len(ref_file_str)-1
        return ref_file_str[start_idx:end_idx].replace("\n","")

def _calc_miss_matches(dna_pred, dna_true, dna_cigar):
    
    dna_cigar_operations = re.findall(r'[\d]+[SMDI]', dna_cigar)

    result_true = ""
    result_pred = ""
    
    refdeque = deque(dna_true)
    assdeque = deque(dna_pred)
    cigdeque = deque(dna_cigar_operations)
    while len(cigdeque) > 0:
        cc = cigdeque.popleft()
        count = int(cc[:-1])
        action = cc[-1]

        for _ in range(count):
            if action == "M":
                result_true += refdeque.popleft()
                result_pred += assdeque.popleft()
            elif action == "I":
                assdeque.popleft()
            elif action == "D":
                refdeque.popleft()
    num_mismatches = sum([a!=b for a,b in zip(result_true, result_pred)])
    return result_pred, result_true, num_mismatches

def _align(dna_pred):
    mapped = get_mapping(dna_pred)
    if mapped == None:
        raise Exception("Unable to map prediction.")
    
    dna_cigar = mapped.cigar_str
    dna_true = get_reference(mapped.ctg)
    dna_true = dna_true[mapped.r_st:mapped.r_en]
    dna_pred = dna_pred[mapped.q_st:mapped.q_en] 

    if mapped.strand == -1:
        dna_pred = mp.revcomp(dna_pred)
    
    return dna_pred, dna_true, dna_cigar   

def _analyse_cigar(cigar_string):
    res = re.findall(r'[\d]+[SMDI]', cigar_string)
    d = {"S":0,"M":0,"D":0,"I":0}
    for r in res:
        d[r[-1]] += int(r[:-1])
    return d

def _with_color(color):
    def inner_decorator(func):

        def wrapper(o_amount, dna_result, dna_true, dna_pred, true_idx, pred_idx, use_color):
            if use_color:
                dna_result += _get_colors()[color]
                dna_result, dna_true, dna_pred, true_idx, pred_idx = func(o_amount, dna_result, dna_true, dna_pred, true_idx, pred_idx, use_color)
                dna_result += _get_colors()['reset']
                return dna_result, dna_true, dna_pred, true_idx, pred_idx
            return func(o_amount, dna_result, dna_true, dna_pred, true_idx, pred_idx)
        return wrapper
    return inner_decorator

def _compare(dna_pred, dna_true, dna_cigar, use_color):
    
    dna_cigar_operations = re.findall(r'[\d]+[SMDI]', dna_cigar)
    dna_result = ""
    pred_idx = 0
    true_idx = 0

    cigar_func_dic = {
        "M": _match_op,
        "D": _delete_op,
        "I": _insert_op,
        "S": _substitute_op
    }

    for o in dna_cigar_operations:
        o_type = o[-1]
        o_amount = int(o[:-1])
        o_func = cigar_func_dic[o_type]
        dna_result, dna_true, dna_pred, true_idx, pred_idx = o_func(o_amount, dna_result, dna_true, dna_pred, true_idx, pred_idx, use_color)

    return dna_result, dna_true

@_with_color(color="green")
def _match_op(amount, res, true, pred, true_idx, pred_idx, use_color):
    res += pred[pred_idx:pred_idx+amount]
    pred_idx += amount
    true_idx += amount
    return res, true, pred, true_idx, pred_idx

@_with_color(color="red")
def _delete_op(amount, res, true, pred, true_idx, pred_idx, use_color):
    #res += amount*"*"
    res += true[true_idx:true_idx+amount].lower()
    true_idx += amount
    return res, true, pred, true_idx, pred_idx

@_with_color(color="blue")
def _insert_op(amount, res, true, pred, true_idx, pred_idx, use_color):
    res += pred[pred_idx:pred_idx+amount].lower()
    true = true[:true_idx] + amount*"*" + true[true_idx:]
    true_idx += amount
    pred_idx += amount
    return res, true, pred, true_idx, pred_idx

@_with_color(color="cyan")
def _substitute_op(amount, res, true, pred, true_idx, pred_idx, use_color):
    res += "["
    true = true[:true_idx] + "[" + true[true_idx:]
    true_idx += 1
    for i in range(amount):
        res += pred[pred_idx + i]
    res += "]"
    true = true[:true_idx + amount] + "]" + true[true_idx + amount:]
    true_idx += amount + 1
    pred_idx += amount
    return res, true, pred, true_idx, pred_idx

def _get_colors():
    return {
    'red': "\033[1;31m",  
    'blue': "\033[1;34m",
    'green': "\033[0;32m",
    'cyan':"\033[1;36m",
    'reset': "\033[0;0m"
    }