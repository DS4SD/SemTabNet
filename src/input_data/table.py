"""Definition of Table class."""
import re
from typing import List, Union

import numpy as np
import pandas as pd

from src.input_data.utils import create_hash, extract_data_from_table


class Table:
    """A class for table object with annotations and verification."""

    def __init__(
        self,
        table: Union[dict, pd.DataFrame],
        collection=None,
        annotations: List[dict] = [],
        verifications: List[dict] = [],
        cell_annotations: pd.DataFrame = None,
        classification: str = "",
        data_hash: str = "",
        document_hash: str = "",
    ) -> None:
        """Create table object.

        Args:
            table (dict, DataFrame): table either as extracted from Deep Search (dict)
                                    or in dataframe format.
            collection (str) : name of collection
            annotations (List[dict]) : list of annotations
            verifications (List[dict]) : list of verifications
            cell_annotations (DataFrame) : cell annotations table
            classification (str) : table classification
            data_hash (str) : hash of the table
            document_hash (str) : hash of the source document
        """
        if isinstance(table, pd.DataFrame):
            self.ds_table = None
            self.table = table
        elif isinstance(table, dict):
            self.ds_table = table  # deep search table
            self.table = pd.DataFrame(
                extract_data_from_table(self.ds_table)
            )  # dataframe table
        else:
            raise TypeError
        self.annotations = annotations
        self.verifications = verifications

        if cell_annotations is not None:
            if isinstance(cell_annotations, pd.DataFrame):
                self.add_cell_annotations(cell_annotations)
            else:
                raise TypeError
        self.classification = classification

        if collection is None:
            self.collection = ""
        else:
            self.collection = collection
        self.data_hash = data_hash
        self.document_hash = document_hash
        return

    def calculate_data_hash(self):
        """Calculate the data_hash of the table."""
        if self.data_hash == "":
            self.data_hash = create_hash(self.table.values.tolist())
        return

    def convert_to_structured_data(self):
        """Extracts the annotations in a rule based way.

        Search for property_value cells and upon finding one,
        constructs an annotation related to it.
        A couple of notes:
        - upon finding a property, it will append all the headers on top
          (or to the left) to better specify the meaning of the property
        - if a property_value cell is found without a property, it will not
          be added.
        """
        assert not self.cell_annotations.empty, "Cell annotations are missing."

        self.annotations = []

        for i, row in self.cell_annotations.iterrows():
            for j, cell in enumerate(row.to_numpy()):
                # searches for the property values cells
                # and upon finding one, annotates
                if cell == "property_value":
                    annotation, key_value_pairs = self.extract_annotation(i, j)
                    # we save only if we have a corresponding property
                    if annotation["property"] != "":
                        predicate = self.construct_predicate(
                            annotation, key_value_pairs
                        )
                        self.annotations.append(predicate)

    def construct_predicate(self, annotation, key_value_pairs):
        """Construct a predicate with multiple key key_value pairs."""
        predicate = {}
        # concatenate the property with the keys from the key value pairs
        predicate["property"] = [annotation["property"]] + key_value_pairs["property"]
        # concatenate the property_value with the key_values
        # from the key value pairs
        predicate["property_value"] = [annotation["property_value"]] + key_value_pairs[
            "property_value"
        ]
        predicate["unit"] = [annotation["unit"]] + [
            "" for _ in range(len(predicate["property"]) - 1)
        ]

        for key in [
            "subject",
            "subject_value",
            # "data_hash",
            # "document_hash",
            # "predicate_hash",
        ]:
            predicate[key] = [
                annotation[key] for _ in range(len(predicate["property"]))
            ]

        return predicate

    def extract_annotation(self, i, j):
        """Method to extract an annotation from a property_value_cell.

        Conduct a search the on the same row (or column) to find
        the associated property.
        Next, search for associated unit.
        Next, search for associated subject_value, time_value or
        key_value cells.
        - For key_value, search for related key
        - for each k,kv create new rows with the same id (to create statement)
        Args:
            i: row index of the cell
            j: column index of the cell

        Returns:
            annotation: main annotation dictionary
            annotation_key_values: key value pairs related to the annotation
        """
        # main annotation
        annotation = {
            "subject": "",
            "subject_value": "",
            "property": "",
            "property_value": "",
            "unit": "",
        }
        annotation["property_value"] = self.table.iat[i, j]
        # searches for the property related to the property value
        annotation["property"] = self.search_property(i, j)
        # search for pairs supposed to be unique
        _, annotation["unit"] = self.search_unique_pair(i, j, "unit", "unit_value")
        annotation["subject"], annotation["subject_value"] = self.search_unique_pair(
            i, j, "subject", "subject_value"
        )
        _, time_value = self.search_unique_pair(i, j, "time", "time_value")

        # search for every additional key_value pair
        annotation_key_values = self.search_multiple_pairs(i, j, "key", "key_value")
        # append time as a key_value pair
        if time_value != "":
            annotation_key_values["property"].append("time")
            annotation_key_values["property_value"].append(time_value)
        return annotation, annotation_key_values

    def search_property(self, i, j):
        """Conduct a search on the same row and column for a property cell.

        The search includes also subproperties.
        Upon finding a property, search and append all related headers on
        top of it.
        """
        # search on the left looking for the property.
        for row_index, cell in enumerate(
            self.cell_annotations.iloc[i, :j].to_numpy()[::-1]
        ):
            if cell == "property":
                property = self.table.iat[i, j - row_index - 1]
                # search for headers to help better describe the property.
                # after appending them, return the property.
                return self.append_headers(i, j - row_index - 1, property, 4)
            if cell == "subproperty":
                subproperty = self.table.iat[i, j - row_index - 1]
                # upon finding a subproperty, search for the relative property in the
                # cells above it and uppend property and relativve headers.
                for column_index, search_cell in enumerate(
                    self.cell_annotations.iloc[:i, j - row_index - 1].to_numpy()[::-1]
                ):
                    if search_cell == "property":
                        property = self.table.iat[
                            i - column_index - 1, j - row_index - 1
                        ]
                        # append headers
                        property = self.append_headers(
                            i - column_index - 1, j - row_index - 1, property, 4
                        )
                        return f"{property}: {subproperty}"
                # if no property is found, we treat the subproperty as a property
                return self.append_headers(i, j - row_index - 1, subproperty, 4)

        # search on top of the cell looking for the property.
        for column_index, cell in enumerate(
            self.cell_annotations.iloc[:i, j].to_numpy()[::-1]
        ):
            if cell == "property":
                property = self.table.iat[i - column_index - 1, j]
                # search for headers to help better describe the property.
                # after appending them, return the property.
                return self.append_headers(i - column_index - 1, j, property, 4)

        # if nothing is found, return an empty string
        return ""

    def append_headers(self, i, j, property, header_index):
        """Search for headers to help better describe the property."""
        # look for headers among the cells that are above
        # and left to the property cell.
        subtable = self.cell_annotations.iloc[: i + 1, : j + 1].to_numpy()
        for i, row in enumerate(subtable[::-1]):
            for j, cell in enumerate(row[::-1]):
                if cell.startswith("header") and int(cell[-1]) < header_index:
                    property = (
                        f"{self.table.iat[len(subtable)-1-i,len(row)-1-j]}: {property}"
                    )
                    if int(cell[-1]) == 1:
                        return property
                    else:
                        return self.append_headers(
                            len(subtable) - 1 - i,
                            len(row) - 1 - j,
                            property,
                            int(cell[-1]),
                        )
        return property

    def search_unique_pair(self, i, j, key, key_value):
        """Conduct a search for a supposedly unique key,key_value pair.

        Upon finding a key_value, search for its key and return them.
        If the key is not found (or both), empty strings will be returned.
        The search is conducted in a precise order, looking first to the left,
        then among the cells above and lastly to the right of the given cell.
        """
        # search orderly, to the left, on top and to the right
        # for an eventual pair
        functions = [self.left_search, self.top_search, self.right_search]
        for func in functions:
            str_keys, str_key_values = func(i, j, key, key_value)
            if len(str_keys) > 0:
                # returns the first found, because closest to the cell
                return str_keys[0], str_key_values[0]
        # nothing is found, therefore return two empty string
        return "", ""

    def left_search(self, i, j, key, key_value):
        """Searches all possible pairs to the left of the cell at i,j."""
        # search on the left looking for the key_value cell.
        output_keys, output_key_values = [], []
        for row_index, value_cell in enumerate(
            self.cell_annotations.iloc[i, :j].to_numpy()[::-1]
        ):
            if value_cell == key_value:
                output_key_values.append(self.table.iat[i, j - row_index - 1])
                # if it finds the key_value on the left,
                # search for the key of it in the cells above
                for column_index, key_cell in enumerate(
                    self.cell_annotations.iloc[:i, j - row_index - 1].to_numpy()[::-1]
                ):
                    if key_cell == key:
                        output_keys.append(
                            self.table.iat[i - column_index - 1, j - row_index - 1]
                        )
                # if it doesn't find the related key, appends an empty string
                if len(output_key_values) > len(output_keys):
                    output_keys.append("")
        return output_keys, output_key_values

    def top_search(self, i, j, key, key_value):
        """Searches all possible pairs above the cell at i,j."""
        # search on top looking for the key_value cell.
        output_keys, output_key_values = [], []
        for column_index, value_cell in enumerate(
            self.cell_annotations.iloc[:i, j].to_numpy()[::-1]
        ):
            if value_cell == key_value:
                output_key_values.append(self.table.iat[i - column_index - 1, j])
                # if it finds the key_value on the top of the cell,
                # search for the key of it on the left of the key_value cell
                for row_index, key_cell in enumerate(
                    self.cell_annotations.iloc[i - column_index - 1, :j].to_numpy()[
                        ::-1
                    ]
                ):
                    if key_cell == key:
                        output_keys.append(
                            self.table.iat[i - column_index - 1, j - row_index - 1]
                        )
                # if it doesn't find the related key, appends an empty string
                if len(output_key_values) > len(output_keys):
                    output_keys.append("")
        return output_keys, output_key_values

    def right_search(self, i, j, key, key_value):
        """Searches all possible pairs to the right of the cell at i,j."""
        # search on the right looking for the key_value cell.
        output_keys, output_key_values = [], []
        for row_index, value_cell in enumerate(
            self.cell_annotations.iloc[i, j:].to_numpy()
        ):
            if value_cell == key_value:
                output_key_values.append(self.table.iat[i, j + row_index])
                # if it finds the key_value on the right,
                # search for the key of it in the cells above
                for column_index, key_cell in enumerate(
                    self.cell_annotations.iloc[:i, j + row_index].to_numpy()[::-1]
                ):
                    if key_cell == key:
                        output_keys.append(
                            self.table.iat[i - column_index - 1, j + row_index]
                        )
                # if it doesn't find the related key, appends an empty string
                if len(output_key_values) > len(output_keys):
                    output_keys.append("")
        return output_keys, output_key_values

    def search_multiple_pairs(self, i, j, key, key_value):
        """Conduct a search for multiple key,key_value pairs.

        Upon finding a key_value, search for its key and return them.
        If the key is not found (or both), empty strings will be returned.
        The search is conducted looking for every possible key_value
        on the same row or column of the given cell (not below the cell).
        The keys returned will be unique.
        """
        # the key key_value pairs will be returned in this form
        key_key_value_pairs = {"property": [], "property_value": []}

        functions = [self.left_search, self.top_search, self.right_search]
        for func in functions:
            str_keys, str_key_values = func(i, j, key, key_value)
            for str_key, str_key_value in zip(str_keys, str_key_values):
                # we name give a key to the keyless key_values
                if str_key == "":
                    str_key = "category"
                # make sure to not add an already existing key
                if str_key not in key_key_value_pairs["property"]:
                    key_key_value_pairs["property"].append(str_key)
                    key_key_value_pairs["property_value"].append(str_key_value)

        return key_key_value_pairs

    def search_additional_information(
        self, index, table, cell_annotations, annotation, annotation_key_values
    ):
        """Conduct a search on the same row for additional information.

        First look for property, then key, key value pairs.
        """
        special_keys = {"time_value": "time", "unit_value": "unit", "key_value": "key"}
        # search row_wise at the index row for additional infos
        for k, cell in enumerate(cell_annotations.iloc[index].to_numpy()):
            if cell == "property":
                annotation["property"] = table.iat[index, k]
                # searches for keys to help better describe the property
                # it looks for them on the cells above the property
                for n, cell_key in enumerate(
                    cell_annotations.iloc[:index, k].to_numpy()
                ):
                    if cell_key == "key":
                        annotation[
                            "property"
                        ] = f"{table.iat[n,k]}: {annotation['property']}"

            # upon finding a key value, it searches its key name
            elif cell in special_keys:
                annotation_key_values["property_value"].append(table.iat[index, k])
                for n, cell_key in enumerate(cell_annotations.iloc[:, k].to_numpy()):
                    if cell_key == special_keys[cell]:
                        annotation_key_values["property"].append(table.iat[n, k])
                        break
                # if a key value have no key, fills the key with the default type
                if len(annotation_key_values["property_value"]) > len(
                    annotation_key_values["property"]
                ):
                    annotation_key_values["property"].append(special_keys[cell])

    def classify_simple(
        self,
        threshold_only_number_cell_ratio: float = 0.05,
        threshold_only_text_cell_ratio: float = 0.75,
    ) -> bool:
        """Returns whether a table is simple or not.

        Args:
            threshold_only_number_cell_ratio (float) :
            minimum ratio of only number cells to all cells in table

            threshold_only_text_cell_ratio (float) :
            maximum ratio of only text cells to all cells in table

        Returns:
            bool : True if table is simple, False if not

        Notes:
        Current heuristic for simple table:
        - At least "only_number_cell_ratio" cells in the entire table as only number;
        - Less that "only_text_cell_ratio" cells in the entire table as only text;
        - No middle row spans.
        """
        _, cell_ratio = self.calculate_cell_labels()
        middle_row_spans = self.check_middle_row_spans()
        if (
            cell_ratio["1"] >= threshold_only_number_cell_ratio
            and cell_ratio["3"] <= threshold_only_text_cell_ratio
            and not middle_row_spans
        ):
            return True
        return False

    def check_middle_row_spans(self) -> bool:
        """Function to calculate the middle row spans.

        Returns:
            bool : True if the table contains a middle row span, False if not

        Notes:
        There are 3 conditions for a span to be considered as such:
        - It needs to be a row span;
        - It needs to span over the entire row;
        - It can't be located in the first or last row of the table.
        """
        # row_wise_equality = self.table.iloc[:, [0]].eq(self.table).all(axis=1)
        a = self.table.values
        # basically compares all the values to the first one and sees if they are equal
        row_wise_equality = (a == a[:, [0]]).all(axis=1)
        try:
            # check whether in any middle row, a value is repeated through all the row
            for val in row_wise_equality[1:-1]:
                if val:
                    return True
            return False
        except IndexError:
            return False

    def calculate_cell_labels(self):
        """Function to calculate the cell labels of a table dictionary.

        Returns:
            list: a table of the same structure with cell labels as values
            dict: a dictionary containing the percentages per label contained.

        Notes:
        The cells are labeled:
        0 if they contain only a number, but on the format of years from 1900 to 2099;
        1 if they contain only a number;
        2 if they contain a number in a text;
        3 if they contain only text;
        4 empty cells
        5 for all other formats.
        """
        nclasses = 6
        only_number = re.compile(r"^([+-]?((\d+([\.\,\'\s]\d+)*)|([\.\,]\d+))%?)$")
        probable_year = re.compile(r"^(19|20)\d{2}$")
        number_in_string = re.compile(
            r"(^|\W)([+-]?((\d\d+([\.\,]\d*)?)|([\.\,]\d+))%?)(\W|$)"
        )
        text_in_string = re.compile(r"[a-zA-Z]+")

        table = self.table.to_numpy()

        increasing_rate = 1 / table.size
        cell_types = np.empty_like(table)
        cells_ratio = {str(i): 0 for i in range(nclasses)}
        for index, text in np.ndenumerate(table):
            text = text.strip()
            if probable_year.search(text):
                cell_types[index] = 0
                cells_ratio["0"] += increasing_rate
            elif only_number.search(text):
                cell_types[index] = 1
                cells_ratio["1"] += increasing_rate
            elif number_in_string.search(text):
                cell_types[index] = 2
                cells_ratio["2"] += increasing_rate
            elif text_in_string.search(text):
                cell_types[index] = 3
                cells_ratio["3"] += increasing_rate
            elif text == "":
                cell_types[index] = 4
                cells_ratio["4"] += increasing_rate
            else:
                cell_types[index] = 5
                cells_ratio["5"] += increasing_rate

        return cell_types.tolist(), cells_ratio

    def to_markdown(self):
        """Returns the table data in a markdown format."""
        return self.table.to_markdown()

    def to_dict(self):
        """Convert a Table object to a dictionary object."""
        dict_table = {}
        dict_table["table"] = self.table.values.tolist()
        dict_table["collection"] = self.collection
        dict_table["annotations"] = self.annotations
        dict_table["verifications"] = self.verifications
        dict_table["classification"] = self.classification
        dict_table["cell_annotations"] = self.cell_annotations
        dict_table["ds_table"] = self.ds_table

        return dict_table

    def to_dataframe(self):
        """Return the table in dataframe format."""
        return self.table

    def add_annotation_from_dict(self, annotated_data: dict) -> None:
        """Add annotations from a dictionary.

        Args:
            annotated_data (dict): dict containing:
                    'subject',
                    'property',
                    'property_value',
                    'unit',
                    'relation',
                    'date_annotation',
                    'annotator'

        Raises:
            ValueError: if a key is missing, a value error is raised.
        """
        list_keys = [
            "subject",
            "count_statements",
            "property",
            "property_value",
            "unit",
            "relation",
            "date_annotation",
            "annotator",
        ]
        _dict = {}
        for k in list_keys:
            if k not in annotated_data.keys():
                raise ValueError(f"Key {k} not found in input. Please check.")
            else:
                _dict[k] = annotated_data[k]
        self.annotations.append(_dict)
        return

    def add_cell_annotations(self, cell_annotations: pd.DataFrame,) -> None:
        """Add cell annotations to object.

        Args:
            cell_annotations (DataFrame): DataFrame containing annotations for each cell
        """
        # x = "Mismatch in shape of cell annotations and table."
        # assert cell_annotations.shape == self.table.shape, x
        self.cell_annotations = cell_annotations
        return


def convert_structured_data_to_string(
    self, structured_data: dict, col_sep: str = "|"
) -> str:
    """String representation (markdown table) of structured data for training.

    Args:
        structured_data (dict): structured data with keys like: subject, property.
        col_sep (str, optional): token to separate columns, Defaults to "|".

    Returns:
        str: _description_
    """
    # TODO : Check if lengths of all atributes is same!
    str_sd = ""  # string representation of structured data

    # get count of key-value pairs
    if "subject" in structured_data.keys():
        counts = len(structured_data["subject"])
    elif "Subject" in structured_data.keys():
        counts = len(structured_data["subject"])
    else:
        raise ValueError("Subject not in structured data.")

    # add header
    keys_to_skip = ["data_hash", "organization"]
    col_keys = [key for key in structured_data.keys() if key not in keys_to_skip]
    str_sd += col_sep.join(col_keys) + col_sep + "\n"
    str_sd += col_sep.join(["--" for item in col_keys]) + col_sep + "\n"

    # create string representation
    for idx in range(counts):
        for key in structured_data.keys():
            if key in keys_to_skip:
                continue
            str_sd += f"{structured_data[key][idx]}{col_sep}"
        str_sd += "\n"
    return str_sd
