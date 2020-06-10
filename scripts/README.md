# Scripts and utilities

## create_umi_to_bacteria_dict.py

Creates a dictionary mapping read uids and the respective bacteria

Used to enforce the correct bacteria during evaluation

### Running

Edit the paths to the reference csv which maps uids to the bacteria, and to the folder containing the f5 files. It creates the uids.json file needed to correctly perform evaluation.


## fasta_benchmarker.py

The testing pipelines output a json file with the evaluation, but the evaluation doesn't enforce bacteria as per uids.json

This script evaluates the co-created fasta file and enforces the correct bacteria, even if a better match was found by accident 

Creates another json file containing the new evaluation in `trained_models`

### Running

The script can run multiple fasta files at once by editing the list of tuples inside the script.

It can also be run from the command line:
`python fasta_benchmarker.py path_to_fasta_file experiment_name`