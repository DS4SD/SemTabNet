# Introduction
This repository is for the ACL 2024 workshop paper [Statements: Universal Information Extraction from Tables with Large Language Models for ESG KPIs](https://arxiv.org/abs/2406.19102).

In this paper, we propose **STATEMENTS** as a new knowledge model for storing quantiative information in a domain agnotic, uniform structure. The task of converting a raw input (table or text) to Statements is called Statement Extraction (SE). The statement extraction task falls under the category of universal information extraction.

![Concept Horizontal](/images/concept_horizontal.png)

Links:

- [Data](https://huggingface.co/datasets/ds4sd/SemTabNet)
- [Code](https://github.com/DS4SD/SemTabNet)
- [Arxiv Paper](https://arxiv.org/abs/2406.19102)

Following notebooks are provided:

- [Data preprocessing](/notebooks/view_data_preprocessing.ipynb)
- [Statements Preview + Convert labels table to statement](/notebooks/statements_preview.ipynb)
- [Tree Similarity Score](/notebooks/tree_similarity_score.ipynb)

Following scripts are provided:
- [Model training](/scripts/train.py)
- [Model inferencing](/scripts/t5inference.py)
- [Statement extraction (if task was indirect statement extraction, see below)](/scripts/tca_statement_extraction.py)
- [Evaluation](/scripts/t5eval_ud2sd.py)


# SemTabNet 

![Model Input Outpu](/images/model_input_output.png)

## Data

The SemTabNet data was originally prepared from annotating 84,890 table cells in 1107 tables from multiple ESG Reports. Each cell in our dataset is classified into one of the following 16 categories:

-  Property, Property Value
-  Sub-property
-  Subject, Subject Value 
-  Unit, Unit Value
-  Time, Time Value
-  Key, Key Value
-  Header 1, Header 2, Header 3
-  Empty, Rubbish

Thus, each original table gives rise to another table (same shape and size) consisting only of the annotation labels. We call this the labels table. Using a set of rules, it is now possible to [write a program](/src/input_data/table.py#L72) which takes in the original table and the labels table to produce the statements. 

In addition, we augment the original tables to give rise to at-most 130 more tables. This leads to the final SemTabNet which contains over 120k tables. 


## Experiments

In our paper, we consider three experiments, for statement extraction from tables, which are described below:

| Alias | Task | Input | Output|
|---|----- |-----|---------|
|`ud2sd_table`| SE Direct | Table in markdown format | Statement in markdown format|
| `tca1d` |SE Indirect 1D | Individual table cell content | Classification label|
| `tca2d` |SE Indirect 2D | Table in markdown format | Labels table in markdown format| 


## DATA Samples

You can see [data samples in this file](/data_sample.md). These are generated using [this notebook](/notebooks/view_data_preprocessing.ipynb).

# Citation

```
@misc{mishra2024statementsuniversalinformationextraction,
      title={Statements: Universal Information Extraction from Tables with Large Language Models for ESG KPIs}, 
      author={Lokesh Mishra and Sohayl Dhibi and Yusik Kim and Cesar Berrospi Ramis and Shubham Gupta and Michele Dolfi and Peter Staar},
      year={2024},
      eprint={2406.19102},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.19102}, 
}
```