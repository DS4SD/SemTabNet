import re

import pandas as pd
from src.parsers.t5parser import T5Parser
from src.input_data.table import Table

def get_numbers_in_text(text):
    """Count the number of numbers in a dataframe."""
    sep = "[\.\,]"
    number_pattern = rf"(?P<decimal>\d*{sep}?\d*{sep}?\d*)"
    matches = re.findall(pattern=number_pattern, string=text)
    matches = [item for item in matches if item != ""]
    return matches

def count_statements_predicates(output:str, parser:T5Parser):
    """Count number of statements and predicates.

    Args:
        output (str): model output containing markdown format statement table
        parser (T5Parser): parser

    Returns:
        cs (int) : count statements
        ws (int) : count statements
    """
    statements = parser.parse(output)
    list_of_predicates = [y.split('\n') for y in statements]
    cp = sum([len(item) for item in list_of_predicates])
    cs = len(statements)
    cp = cp - 2*cs # because each statement markdown contains a row of headers and ---|
    return cs, cp

def test_markdown_structure(output:str, parser:T5Parser):
    """Test markdown structured of statments.
    - test 1 : markdown pattern exists
    - test 2 : number of columns is same in a statement
    - test 3 : special keys are present and only once (subject property .. )
    - test 4 : all tables should have the new line characters!

    Args:
        output (str): structutred data or model output
        parser (T5Parser): parser to convert structured data to markdown

    Returns:
        cs (int) : count correct statements
        ws (int) : count wrong statements
        cp (int) : count correct predicates
        wp (int) : count wrong predicates

    """
    row_pattern = r'\|.*|'
    statements = parser.parse(output)
    
    ws = 0
    wp = 0
    cs = 0
    cp = 0

    keys_to_check = {
                    'subject':2,
                    'subject_value':1, 
                    'property':2, 
                    'property_value':1,
                    'unit':1
                    }
    # subject occurs twice in 'subject' and 'subject_value'

    for idx, statement in enumerate(statements):
        
        # Split the table into rows
        predicates = statement.split('\n')
        num_columns_in_first_row = len(re.findall(r'\|', predicates[0]))
        
        # set flags
        test_markdown_predicate = True
        test_ncol_predicate = True
        test_col_headers = True
        test_newline_char = True

        
        for predicate in predicates:

            # Check if each row matches the pattern
            if not re.match(pattern=row_pattern, string=predicate):
                wp +=1
                test_markdown_predicate = False
                continue # without doing the next test

            # Check if all rows have the same number of columns
            if len(re.findall(r'\|', predicate)) != num_columns_in_first_row:
                wp +=1
                test_ncol_predicate = False
                continue # without doing the next test

            # check if predicate has newline character
            if '\n' not in predicate:
                wp +=1
                test_newline_char = False

        # Test for column headers
        for k, v in keys_to_check.items():
            count = statement.count(k)
            if count != v:
                test_col_headers=False

        if (test_markdown_predicate 
            and test_ncol_predicate 
            and test_col_headers
            and test_newline_char):
            cs +=1
            cp += len(predicates)-2 # remove count for header and --- row
        else:
            ws +=1

    return cs,cp,ws,wp

def collect_values_from_statement(output, parser, key):
    """Collect values from statements corresponding to key."""
    statements = parser.parse(output)
    values = []
    for statement in statements:
        try:
            df = parser.convert_markdown_to_dataframe(statement)
            if isinstance(df, pd.DataFrame):
                try:
                    values.extend(df[key].to_list())
                except KeyError:
                    print("KeyError occurred! Trying to fing key={key}.")
                    #get column name corresponding to key 'property_value' (it may have space padding around)
                    val_columns = [item for item in df.columns if key in item]
                    print(val_columns)
                    if len(val_columns) == 1:
                        values.extend(df[val_columns[0]].to_list())
                    elif len(val_columns) > 1:
                        print(f"More than one key='{key}' found! Using first from: {val_columns}")
                        values.extend(df[val_columns[0]].to_list())
                    else:
                        print(f"Key='{key}' not found!!!")
            else:
                pass # do nothing
        except Exception:
            pass
    # remove whitespace
    values = [item.strip() if isinstance(item,str) else item for item in values ]
    return values

def extract_labels_table_from_inference(model_output:str, parser:T5Parser)->pd.DataFrame:
    """Extract dataframe of labels table from model output.

    Args:
        model_output (str): raw model output

    Returns:
        pd.DataFrame: labels table
    """
    md = parser.parse(model_output)
    try:
        df = parser.convert_markdown_to_dataframe(md[0], set_headers=True)
        # df = pd.DataFrame(parser.parse_cell_annotations(df.values.tolist()))
        return df
    except:
        pass

def convert_labels_table_to_statement(labels_table:pd.DataFrame, input_data:dict):
    """Generate statements from input table and labels table.

    Args:
        labels_table (pd.DataFrame): dataframe of labels table
        input_data (dict): dict of table (as output by ds)

    Returns:
        list: list of statements
    """
    statements = None
    t = Table(table=input_data, cell_annotations=labels_table)
    # nrows_labels = labels_table.shape[0]
    # nrows_table = t.table.shape[0]
    # if nrows_labels-nrows_table == 1:
    #     labels_table = labels_table[1:]
    #     labels_table.index = range(labels_table.shape[0])    
    # return t
    try:
        t = Table(table=input_data, cell_annotations=labels_table)
        t.convert_to_structured_data()
        statements = t.annotations
    except:
        pass
    return statements   