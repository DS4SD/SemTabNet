"""Script for evaluating UD2SD predictions."""

import argparse
import re
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from tqdm import tqdm
from src.io.read import read_jsonl
from src.io.write import save_dict_in_jsonl
from src.parsers.t5parser import T5Parser
from src.utils.scores import Scorer
from src.utils.utils import (get_numbers_in_text, 
                             count_statements_predicates, 
                             test_markdown_structure, 
                             collect_values_from_statement)
from src.utils.tree_utils import StatementTree
from src.utils.ted_utils import TSS

from functools import partial
from multiprocessing import Manager, Pool, Queue
import sys,os

class HiddenPrints:
    """Disables prints."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def evaluate_statement(item: dict, parser: T5Parser):
    """Evaluates a statement."""
    original_output = item['output']
    model_output = item['model_output']
    parsed_output = parser.remove_special_tokens(model_output)
    norm_model_output = parser.normalize_text(parsed_output)
    norm_original_output = parser.normalize_text(original_output)

    # ========================        
    # did model contain eos and 
    # output stop token?
    # ========================
    item['output_has_eos_token'] = parser.check_eos_tokens(model_output=model_output)

    # ========================        
    # exact match before norm
    # ========================
    if parsed_output == original_output:
        item["exact_match_before_norm"] = True
    else:
        item["exact_match_before_norm"] = False

    # ========================
    # exact match after norm
    # ========================
    if norm_original_output == norm_model_output:
        item["exact_match_after_norm"] = True
    else:
        item["exact_match_after_norm"] = False

    # ========================
    # Scores before norm
    # ========================
    before_norm = Scorer(original_text=original_output,
                        prediction_text=parsed_output).compare_strings()
    for k,v in before_norm.items():
        if k != 'rouge_score':
            item[f'before_norm_{k}'] = v
        if k == 'rouge_score':
            for kk, vv in v.items():
                item[f"before_norm_{kk}_precision"] = vv.precision
                item[f"before_norm_{kk}_recall"] = vv.recall
                item[f"before_norm_{kk}_fmeasure"] = vv.fmeasure

    # ========================
    # scores after norm
    # ========================
    after_norm = Scorer(original_text=norm_original_output,
                        prediction_text=norm_model_output).compare_strings()
    for k,v in after_norm.items():
        if k != 'rouge_score':
            item[f'after_norm_{k}'] = v
        if k == 'rouge_score':
            for kk, vv in v.items():
                item[f"after_norm_{kk}_precision"] = vv.precision
                item[f"after_norm_{kk}_recall"] = vv.recall
                item[f"after_norm_{kk}_fmeasure"] = vv.fmeasure

    # ========================        
    # number of predicates/statements
    # ========================
    cs, cp = count_statements_predicates(original_output, parser=parser)
    item['original_statements_count'] = cs
    item['original_predicates_count'] = cp

    # ========================
    # outputs w/ correct structure
    # ========================
    cs, cp = count_statements_predicates(model_output, parser=parser)
    item['prediction_statements_count'] = cs
    item['prediction_predicates_count'] = cp
    ccs, ccp, ws,wp = test_markdown_structure(model_output,parser=parser)
    item['prediction_statements_good_format'] = ccs
    item['prediction_predicates_good_format'] = ccp
    item['prediction_statements_bad_format'] = ws
    item['prediction_predicates_bad_format'] = wp

    # ========================
    # Score on retrieval of statement attributes
    # ========================
    keys = ['subject','subject_value','property','property_value','unit']
    # keys = ['subject','subject_value','property','property_value']
    for key in keys:
        values_original = collect_values_from_statement(output=original_output,
                                                        parser=parser, 
                                                        key=key)
        values_prediction = collect_values_from_statement(output=model_output,
                                                        parser=parser, 
                                                        key=key)
        quant_scores = calculate_metrics(values_original, values_prediction)
        for k,v in quant_scores.items():
            item[f"{key}_{k}"] = v

    # ========================
    # Calculate TREE EDIT Distance
    # ========================
    ted_types = ["ted_with_subject", "ted_without_subject"]
    cases = ["unnormalized", "normalized"]
    
    for ted_type in ted_types:
        if 'without' in ted_type:
            include_subjects = False
        else:
            include_subjects = True
        for case in cases:
            # get statements
            if case == "normalized":
                model_statements = parser.parse(model_output.lower())
                original_statements = parser.parse(original_output.lower())
            else:
                model_statements = parser.parse(model_output)
                original_statements = parser.parse(original_output)                    
                
            current_metric = f"{case}_{ted_type}_"
            try:
                df_original_statements = [parser.convert_markdown_to_dataframe(item) for item in original_statements]             
                tree_original_statement = StatementTree(df_original_statements)
                root_node_original = tree_original_statement.get_root_node()

                df_model_statements = [parser.convert_markdown_to_dataframe(item) for item in model_statements]
                tree_model_statements = StatementTree(df_model_statements)
                root_node_model = tree_model_statements.get_root_node()

                ted = TED(root_node_model, root_node_original,include_subjects=include_subjects)
                distance, edits = ted.get_tree_edit_distance()
                item[current_metric+"distance"] = distance
                item[current_metric+"edits"] = edits
                item[current_metric+"distance_normalized"] = ted.get_normalized_distance()
                item[current_metric+"similarity"] = ted.get_tree_similarity()

            except Exception as e:
                item[current_metric+"distance"] = None
                item[current_metric+"edits"] = None
                item[current_metric+"distance_normalized"] = None
                item[current_metric+"similarity"] = None
                print(f"Exception occurred! Details: {e}")
    return item

def evaluate_statement_worker(item: dict, message_queue: Queue, parser: T5Parser):
    """Evaluate a statement and send the output to the message queue."""
    with HiddenPrints():
        item=evaluate_statement(item, parser)
        message_queue.put(item)

def file_writer(rfile: Path, message_queue: Queue):
    """Save every received item from the message queue into the rfile."""
    while True:
        save_dict_in_jsonl(message_queue.get(), rfile)

def main():
    args = parse_arguments()
    ifile = Path(args.input_file)
    data = read_jsonl(ifile.parent, ifile.name)
    parser = T5Parser(task='ud2sd')
    rfilename = str(ifile.name).removesuffix('.jsonl')+"_v00_evaluations.jsonl"
    rfile = ifile.parent.joinpath(rfilename)
    rfile = get_unique_name(rfile)
    print(f"Your results are saved at: {rfile}")   

    with Manager() as manager:
        pool = Pool()  # By default pool will size depending on cores available
        message_queue = manager.Queue()  # Queue for sending messages to file writer listener
        pool.apply_async(file_writer, (rfile, message_queue, ))  # Start file listener ahead of doing the work
        print('mapping...')
        progress=tqdm(pool.imap(partial(evaluate_statement_worker, message_queue=message_queue,parser=parser), data), total=len(data)) 
        # Partial function allows us to use map to divide workload
        print('running...')
        tuple(progress)
        print('done')

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a predictions with ground truth data."
    )

    parser.add_argument(
        "-i",
        "--input-file",
        help="Path to predictions jsonl file.",
        required=True
    )
    return parser.parse_args()

def get_unique_name(rpath:Path):
    while(rpath.exists()):
        print("input path: ", rpath)
        print("Does input exist: ", rpath.exists())
        # fname = rpath.name.removesuffix('_evaluations.jsonl')
        # v = int(fname[-1:])+1

        # Use regular expression to find digits at the end of the string
        match = re.search(r'\d+$', rpath.name)
        
        if match:
            print('yes')
            # Convert the matched digits to an integer and increment
            v = int(match.group()) + 1
        else:
            # If no digits found, set v to 1
            v = 1
            
        fname = rpath.name.removesuffix('_evaluations.jsonl')
        fname = fname + f"{v:02d}" + "_evaluations.jsonl"
        rpath = rpath.parent.joinpath(fname)
        print("new path: ", rpath)
    print("returning path: ", rpath)
    return rpath

def calculate_metrics(values_original:list, values_prediction:list):
    _ = Scorer.count_tp_tn_fp_fn(ground_truth= values_original, 
                                 model_prediction=values_prediction)
    tp, tn, fp, fn, count_common, count_total = _
    scores = Scorer.calculate_metrics(tp, tn, fp, fn, count_common, count_total)
    return scores
    

# def main():

    # args = parse_arguments()
    # ifile = Path(args.input_file)
    # data = read_jsonl(ifile.parent, ifile.name)
    # parser = T5Parser(task='ud2sd')
    # rfilename = str(ifile.name).removesuffix('.jsonl')+"_v00_evaluations.jsonl"
    # rfile = ifile.parent.joinpath(rfilename)
    # rfile = get_unique_name(rfile)
    # print(f"Your results are saved at: {rfile}")    

    # counts_exact_matches_before_norm = 0
    # counts_exact_matches_after_norm = 0
    # total_input_data = 0
    
    # for item in tqdm(data, total=len(data)):
    #     total_input_data+=1
    #     original_output = item['output']
    #     model_output = item['model_output']
    #     parsed_output = parser.remove_special_tokens(model_output)
    #     norm_model_output = parser.normalize_text(parsed_output)
    #     norm_original_output = parser.normalize_text(original_output)

    #     # ========================        
    #     # did model contain eos and 
    #     # output stop token?
    #     # ========================
    #     item['output_has_eos_token'] = parser.check_eos_tokens(model_output=model_output)

    #     # ========================        
    #     # exact match before norm
    #     # ========================
    #     if parsed_output == original_output:
    #         counts_exact_matches_before_norm +=1
    #         item["exact_match_before_norm"] = True
    #     else:
    #         item["exact_match_before_norm"] = False
        
    #     # ========================
    #     # exact match after norm
    #     # ========================
    #     if norm_original_output == norm_model_output:
    #         counts_exact_matches_after_norm +=1
    #         item["exact_match_after_norm"] = True
    #     else:
    #         item["exact_match_after_norm"] = False

    #     # ========================
    #     # Scores before norm
    #     # ========================
    #     before_norm = Scorer(original_text=original_output,
    #                         prediction_text=parsed_output).compare_strings()
    #     for k,v in before_norm.items():
    #         if k != 'rouge_score':
    #             item[f'before_norm_{k}'] = v
    #         if k == 'rouge_score':
    #             for kk, vv in v.items():
    #                 item[f"before_norm_{kk}_precision"] = vv.precision
    #                 item[f"before_norm_{kk}_recall"] = vv.recall
    #                 item[f"before_norm_{kk}_fmeasure"] = vv.fmeasure

    #     # ========================
    #     # scores after norm
    #     # ========================
    #     after_norm = Scorer(original_text=norm_original_output,
    #                         prediction_text=norm_model_output).compare_strings()
    #     for k,v in after_norm.items():
    #         if k != 'rouge_score':
    #             item[f'after_norm_{k}'] = v
    #         if k == 'rouge_score':
    #             for kk, vv in v.items():
    #                 item[f"after_norm_{kk}_precision"] = vv.precision
    #                 item[f"after_norm_{kk}_recall"] = vv.recall
    #                 item[f"after_norm_{kk}_fmeasure"] = vv.fmeasure

    #     # ========================        
    #     # number of predicates/statements
    #     # ========================
    #     cs, cp = count_statements_predicates(original_output, parser=parser)
    #     item['original_statements_count'] = cs
    #     item['original_predicates_count'] = cp

    #     # ========================
    #     # outputs w/ correct structure
    #     # ========================
    #     cs, cp = count_statements_predicates(model_output, parser=parser)
    #     item['prediction_statements_count'] = cs
    #     item['prediction_predicates_count'] = cp
    #     ccs, ccp, ws,wp = test_markdown_structure(model_output,parser=parser)
    #     item['prediction_statements_good_format'] = ccs
    #     item['prediction_predicates_good_format'] = ccp
    #     item['prediction_statements_bad_format'] = ws
    #     item['prediction_predicates_bad_format'] = wp

    #     # ========================
    #     # Score on retrieval of statement attributes
    #     # ========================
    #     keys = ['subject','subject_value','property','property_value','unit']
    #     # keys = ['subject','subject_value','property','property_value']
    #     for key in keys:
    #         values_original = collect_values_from_statement(output=original_output,
    #                                                         parser=parser, 
    #                                                         key=key)
    #         values_prediction = collect_values_from_statement(output=model_output,
    #                                                         parser=parser, 
    #                                                         key=key)
    #         quant_scores = calculate_metrics(values_original, values_prediction)
    #         for k,v in quant_scores.items():
    #             item[f"{key}_{k}"] = v

    #     # ========================
    #     # Calculate TREE Similarity Score
    #     # ========================
    #     ted_types = ["ted_with_subject", "ted_without_subject"]
    #     cases = ["unnormalized", "normalized"]
        
    #     for ted_type in ted_types:
    #         if 'without' in ted_type:
    #             include_subjects = False
    #         else:
    #             include_subjects = True
    #         for case in cases:
    #             # get statements
    #             if case == "normalized":
    #                 model_statements = parser.parse(model_output.lower())
    #                 original_statements = parser.parse(original_output.lower())
    #             else:
    #                 model_statements = parser.parse(model_output)
    #                 original_statements = parser.parse(original_output)                    
                    
    #             current_metric = f"{case}_{ted_type}_"
    #             try:
    #                 df_original_statements = [parser.convert_markdown_to_dataframe(item) for item in original_statements]             
    #                 tree_original_statement = StatementTree(df_original_statements)
    #                 root_node_original = tree_original_statement.get_root_node()

    #                 df_model_statements = [parser.convert_markdown_to_dataframe(item) for item in model_statements]
    #                 tree_model_statements = StatementTree(df_model_statements)
    #                 root_node_model = tree_model_statements.get_root_node()

    #                 ted = TED(root_node_model, root_node_original,include_subjects=include_subjects)
    #                 distance, edits = ted.get_tree_edit_distance()
    #                 item[current_metric+"distance"] = distance
    #                 item[current_metric+"edits"] = edits
    #                 item[current_metric+"distance_normalized"] = ted.get_normalized_distance()
    #                 item[current_metric+"similarity"] = ted.get_tree_similarity()

    #             except Exception as e:
    #                 item[current_metric+"distance"] = None
    #                 item[current_metric+"edits"] = None
    #                 item[current_metric+"distance_normalized"] = None
    #                 item[current_metric+"similarity"] = None
    #                 print(f"Exception occurred! Details: {e}")
            
    #     # ========================
    #     #save data:
    #     # ========================
    #     save_dict_in_jsonl(item, rfile)
    #     total_input_data +=1

    # print(f"=== Total items: {total_input_data}")
    # print(f"=== Exact matches (before norm): {counts_exact_matches_before_norm} ({100*counts_exact_matches_before_norm/total_input_data:.2f})%")
    # print(f"=== Exact matches (after norm): {counts_exact_matches_after_norm} ({100*counts_exact_matches_after_norm/total_input_data:.2f})%")
    # print(f"Your results are saved at: {rfile}")

if __name__ == '__main__':
    main()