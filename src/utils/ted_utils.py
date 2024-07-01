"""This script defines routines to compute Tree Similarity Score."""

from apted import Config, APTED
from src.utils.tree_utils import StatementTree
from anytree import Node
import pandas as pd

import distance
from typing import List

class StatementTreeDiff(Config):

    def __init__(self, include_subjects:bool=True):
        self.include_subjects = include_subjects
        return
    
    def maximum(self, *sequences):
        """Get maximum of length of sequences.
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)
    
    def delete(self,node):
        """Calculates the cost of deleting a node"""
        return 1.

    def insert(self,node):
        """Calculates the cost of inserting a node"""
        return 1.

    def rename(self, node1, node2):
        """Calculates the cost of renaming the label of the source node
        to the label of the destination node
        - test if nodes are of correct and same types => norm levestein distance
        - in all other cases: => max norm levy distance = 1
        - cases: nodes are not of same type, or have none value     
        
        """
        structure_types = ['root','statement','predicate']
        # root is node of statementS vs statement is node of single statement
        subject_types = ['subject', 'subject_value']
        value_types = ['property','property_value','unit']
        
        if node1.type == node2.type:
        # case where nodes have same types

            test_subject_types = node1.type in subject_types
            test_value_types = node1.type in value_types
            test_structure_types = node1.type in structure_types
            
            if self.include_subjects:
                test_node_types = test_subject_types or test_value_types
            else:
                test_node_types = test_value_types

            # print("================================")
            # print("=== test_node_types:", test_node_types)
            # print("=== self.include_subjects:", self.include_subjects)
            # print("=== node1.type",node1.type)
            # print("=== node2.type",node2.type)
            # print("=== node1.value",node1.value)
            # print("=== node2.value",node2.value)
            # print("=== Are they equal?:", node1.value == node2.value)
            # print("=== Node1.value is nan?:", pd.isna(node1.value))
            # print("=== Node2.value is nan?:", pd.isna(node2.value))
            # print("================================")

            if test_node_types or test_structure_types:
                if isinstance(node1.value,str) and isinstance(node2.value, str):
                    # case where both nodes have values
                    return self.normalized_distance(node1.value, node2.value)
                if node1.value == node2.value:
                    # case where both nodes have same value, irrespective of type
                    return 0.
                if pd.isna(node1.value) and pd.isna(node2.value):
                    # case where both node values are either None or NaN
                    # pandas fills empty values with nan (not a number)
                    # we interpret None and nan as empty values and equate them
                    return 0.
            elif test_subject_types:
                # case where node is subject type, but we don't care
                return 0.
            else:
                # case where node is of unknown type
                x = f"Found Node of unknown type: {node1.type}"
                x += " with value: {node1.value}. Change tree or rethink TED calculation."
                x += " Allowed node types are: {structure_types} and {statement_types}."
                raise ValueError(x)
   
        # in all other cases, maximum distance!        
        return 1.0

class TSS():
    """Class for computing tree similarity score for statements.
    """
    def __init__(self, 
                 node_t1:StatementTree, 
                 node_t2:StatementTree,
                 include_subjects:bool
                 ):
        """
        Initialize TSS computation.
        
        Args:
            node_t1 (StatementTree): starting node of tree 1.
            node_t2 (StatementTree): starting node of tree 2.
            include_subjects (bool, optional): include subject/subject_value in TED?. Defaults to True.
        """        
        self.node_t1 = node_t1
        self.node_t2 = node_t2    
        self.include_subjects = include_subjects
        self.diff_tree = APTED(node_t1, node_t2, StatementTreeDiff(include_subjects=include_subjects))
        self.distance = self.diff_tree.compute_edit_distance()
        self.step_count_dict = None
        self.normalized_distance = None

    def get_tree_edit_distance(self, verbose=False,):
        """Get minimum tree edit distance between two trees. 
        
        Args:
            explain (bool, optional): Explains steps to convert one tree to another. Defaults to False.

        Returns:
            distance (float): minimum edit distance between two trees
            step_count_dict (dict): counts of deletions/insertions/renames
        """
        _ = self.explain_tree_edit_distance(verbose=verbose)
        return self.distance, self.step_count_dict

    def explain_tree_edit_distance(self, verbose:bool=False, vv=False):
        """Explains the tree edit distance.

        Args:
            verbose (bool, optional): describes the steps which count. Defaults to False.
            vv (bool, optional): describes all steps. Defaults to False.

        Returns:
            _type_: _description_
        """

        count_del = 0
        count_ins = 0
        count_ren = 0
        print_missing_step = False
        if vv:
            verbose = False

        for idx, step in enumerate(self.diff_tree.compute_edit_mapping()):
            if step[0] is None:
                if verbose:
                    score = StatementTreeDiff(include_subjects=self.include_subjects).insert(step[1])
                    print(f"Step: {idx+1}: === Insertion. Contribution to distance: {score}")
                    print(step)
                count_ins +=1
            elif step[1] is None:
                if verbose:
                    score = StatementTreeDiff(include_subjects=self.include_subjects).delete(step[0])
                    print(f"Step: {idx+1}: === Deletion. Contribution to distance: {score}")
                    print(step)
                count_del +=1
            elif step[0] == step[1]:
                if verbose:
                    print(f"Step: {idx+1}: === SAME NODES! Contribution to distance: 0.0")
                    print(step)
            elif step[0] != step[1]:
                if not pd.isna(step[0].value) or not pd.isna(step[1].value):
                    # renames are counted only when the values weren't none. 
                    score  = StatementTreeDiff(include_subjects=self.include_subjects).rename(step[0],step[1])
                    if score != 0:
                        count_ren +=1
                        if verbose:
                            print(f"Step: {idx+1}: === Rename. Contribution to distance: {score}")
                            print(step)
                    else:
                        print_missing_step = True
                else:
                    print_missing_step = True
            else:
                if verbose:
                    print("This should never happen, or?")
                    print(step)
            if vv:
                print(step)
            
        if print_missing_step and verbose:
            f = " ===== \n"
            f += "Missing steps may be due to:"
            f += " (1) nodes with none values"
            f += " or (2) rename of node with same type and value but different location in tree."
            f += " To see all steps anyway, use 'explain_tree_edit_distance' and set vv to True."
            f += "\n  ===== \n"
            print(f)
        self.step_count_dict = {
            "delete": count_del,
            "insert" : count_ins,
            "rename" : count_ren
        }
        self.total_edits = count_del + count_ins + count_ren
        return self.step_count_dict

    def get_normalized_distance(self):
        """Normalized tree edit distance is distance /total number of edits.
        """
        if self.step_count_dict is None:
            self.get_tree_edit_distance()
        if self.total_edits != 0 :
            self.normalized_distance = self.distance/self.total_edits
        else:
            self.normalized_distance = 0
        return self.normalized_distance

    def get_tree_similarity(self):
        """Calculate similarity between two trees.

        - Similarity is 0 nodes were deleted or inserted!
        """
        if self.normalized_distance is None:
            self.get_normalized_distance()
        self.similarity = 1 - self.normalized_distance
        return self.similarity

        