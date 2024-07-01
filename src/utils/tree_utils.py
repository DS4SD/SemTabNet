"""This module defines the Tree class and its methods."""

from anytree import AnyNode, Node, RenderTree
from anytree.exporter import UniqueDotExporter, DotExporter, DictExporter
import pandas as pd
from typing import List

class StatementTree():

    def __init__(self, 
                 statements:List[pd.DataFrame], 
                 root_name:str="root",
                 print_tree:bool = False):
        """Instantiate a statements tree.

        Args:
            statements (List[pd.DataFrame]): list of dataframes where each dataframe 
            represents a single statement. Each statment is composed of k-kv pairs (predicates).

            root (str) : Name of root node (default = 'root')
        """
        self.root = Node(name=root_name, type="root", value=None)
        for idx, statement in enumerate(statements):
            s = Node(name=f"s{idx}", type = 'statement', value=None, parent=self.root)
            self.convert_statement_dict_to_tree(df=statement, statement_node=s)
        
        if print_tree:
            print_tree()
        return

    def get_root_node(self):
        return self.root

    def print_tree(self, start_node:Node=None):
        if start_node is None:
            start_node = self.root
        print(RenderTree(start_node))

    def convert_statement_dict_to_tree(self, df:pd.DataFrame,statement_node:Node):
        """dictionary of a single statment

        Args:
            df (pd.DataFrame): Dataframe of single statement
            statement_index (int): index of statement from list of statements
        """
        data = df.to_dict(orient='index')
        for k, v in data.items():
            # # create unique id as S3P2 : second predicate in third statement
            # unique_id = f"S{statement_index}P{k}"
            pred_node = Node(name=f"p{k}", parent=statement_node, type="predicate", value=None)
            # graph.extend([pred_node])
            for kk, vv in v.items():
                _ = Node(name=f"{kk}", parent=pred_node, type=kk, value=vv)
                # graph.append([_])
        return

    def to_picture(self, 
                   root_node:Node=None, 
                   filename:str=None):
        
        if filename is None:
            filename = f"graph_{str(root_node.name)}.png"

        if root_node is None:
            root_node = self.root
        
        def nodenamefunc(node):
            if node.parent is not None:
                parent = node.parent.name
            else:
                parent = ""
            if node.value is None:
                v = ''
                name =  f"{parent}:{node.name}"
            else:
                v = node.value
                name =  f"{parent}:{node.name}={v}"
            if name[0] == ':':
                name = name[1:]
            return name
        
        def nodeattrfunc(node):
            return "width=1, height=1, shape=diamond"
            # return "fixedsize=true, width=1, height=1, shape=diamond"
        
        def edgeattrfunc(node, child):
            return 'label="%s:%s"' % (node.name, child.name)
        
        DotExporter(
            node = root_node,
            nodenamefunc=nodenamefunc,
            # nodeattrfunc=nodeattrfunc,
            # edgeattrfunc=edgeattrfunc
        ).to_picture(filename)
        return
        