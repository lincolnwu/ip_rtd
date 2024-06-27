from .data_processing import ImmunoPhenoData
import numpy as np
import requests
import urllib.parse
import random
import json
import logging
import warnings
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import networkx as nx
from networkx.exception import NetworkXError
from nxontology.imports import from_file
from networkx.algorithms.dag import dag_longest_path

import matplotlib.pyplot as plt
from netgraph import Graph

from sklearn.impute import KNNImputer
# from .stvea_controller import Controller
import math
import scipy
import copy
from importlib.resources import files
from scipy.stats import entropy

class ImmunoPhenoDB_Connect:
    """A class to interact with the ImmunoPheno database

    Performs queries to a database containing curated single cell data from
    different experiments, tissues, antibodies, and cell populations. These queries
    can be used to find antibodies for gating specific cell populations,
    perform automatic annotation of cytometry data, and design optimal antibody panels
    for cytometry experiments.

    Args:
        url (str): URL link to the ImmunoPheno database server.

    Attributes:
        url (str): URL link to the ImmunoPheno database server.
        imputed_reference (pd.DataFrame): reference antigenic dataset returned from
            the database after calling run_stvea. Format: Row (cells) x column (antibodies).
        transfer_matrix (pd.DataFrame): transfer matrix returned after calling run_stvea.
            Format: Row (query cells) x column (reference cells).
    """
    def __init__(self, url: str):
        self.url: str = url
        self.imputed_reference: pd.DataFrame = None
        self.transfer_matrix: pd.DataFrame = None

        self._OWL_graph = None
        self._subgraph = None
        self._db_idCLs = None
        self._db_idCL_names = None
        self._last_stvea_params = None
        self._downsample_pairwise_graph = None
        self._nn_dist = None

        if self.url is None:
            raise Exception("Error. Server URL must be provided")

        if self.url is not None and self.url.endswith("/"):
            # Find the last forward slash
            last_slash_index = self.url.rfind("/")
            
            # Remove everything after the last forward slash
            result_url = self.url[:last_slash_index]
            self.url = result_url
        
        if "://" not in self.url:
            self.url = "http://" + self.url
        
        if self._OWL_graph is None:
            print("Loading necessary files...")
            try:
                owl_link = _update_cl_owl()
                G_nxo = from_file(owl_link)
            except:
                owl_link = str(files('immunopheno.data').joinpath('cl_2024_05_15.owl'))
                G_nxo = from_file(owl_link)
            G = G_nxo.graph
            self._OWL_graph = G
        
        if self._subgraph is None:
            # Make an API call to get our unique idCLs
            print("Connecting to database...") 
            try:
                idCL_response = requests.get(f"{self.url}/api/idcls")
                idCL_JSON = idCL_response.json()
                idCLs = idCL_JSON['idCLs']

                self._db_idCLs = idCLs
                self._subgraph = _subgraph(self._OWL_graph, self._db_idCLs)

                convert_idCL = {
                    "idCL": list(self._subgraph.nodes)
                }

                convert_idCL_res = requests.post(f"{self.url}/api/convertcelltype", json=convert_idCL)
                idCL_names = convert_idCL_res.json()["results"]                
                self._db_idCL_names = idCL_names
                
                print("Connected to database.")
            except:
                raise Exception("Error. Unable to connect to database")
    
    def _find_descendants(self, id_CLs: list) -> dict:
        node_fam_dict = {}
    
        # For each idCL, find all of their unique descendants using the database graph
        for idCL in id_CLs:
            node_family = []
            descendants = nx.descendants(self._subgraph, idCL)
            node_family.extend(list(set(descendants)))
            node_fam_dict[idCL] = node_family

        return node_fam_dict
    
    def plot_db_graph(self, 
                      root: str = None) -> go.Figure:
        """Plots a graph of all cell type ontologies in the database

        The graph will start with the root node of "CL:0000000" representing "cell".
        This root node can be modified to hone in on a particular cell type and 
        their descendant cell types. Nodes colored in red indicate cell ontologies for
        which there are cells in the database containing those ontologies. Nodes in blue
        indicate intermediary cell ontologies that have been dervied from 
        those that are already in the database.

        Args:
            root (str): Root node in the graph. Modifying the root node
                will return a subgraph containing descendants for only
                that modified root node. 

        Returns:
            go.Figure: Graph containing cell ontologies as nodes. This plotly figure 
            can be further updated or styled.
        """
        if root is None:
            # We already calculated the database's subgraph in self._subgraph
            # Find hover names
            hover_names = []
            for node in list(self._subgraph.nodes):
                hover_names.append(self._db_idCL_names[node])
            plotly_graph = _plotly_subgraph(self._subgraph, self._db_idCLs, hover_names)
        else:
            leaf_nodes = _find_leaf_nodes(self._subgraph, root)
            
            if len(leaf_nodes) == 0:
                # If there are no leaf nodes of root, then we were already provided a leaf node
                # We can plot this singular node directly
                nodes_to_plot = [root]
                # Check if this node was in our database
                node_in_db = list(set(nodes_to_plot) & set(self._db_idCLs))
                # Take subgraph using default function
                default_subgraph = nx.subgraph(self._subgraph, nodes_to_plot)
                # Find hover names
                hover_names = []
                for node in list(default_subgraph.nodes):
                    hover_names.append(self._db_idCL_names[node])
                plotly_graph = _plotly_subgraph(default_subgraph, node_in_db, hover_names)
                
            elif len(leaf_nodes) == 1:
                # If there was only one leaf node, we can directly plot the descendants to that node
                nodes_to_plot = list(nx.descendants(self._subgraph, root))
                # Include the original node
                nodes_to_plot.insert(0, root)
                # Check if these were in the database
                node_in_db = list(set(nodes_to_plot) & set(self._db_idCLs))
                # Take subgraph using default function
                default_subgraph = nx.subgraph(self._subgraph, nodes_to_plot)
                # Find hover names
                hover_names = []
                for node in list(default_subgraph.nodes):
                    hover_names.append(self._db_idCL_names[node])
                plotly_graph = _plotly_subgraph(default_subgraph, node_in_db, hover_names)
                
            else:
                # Multiple leaf nodes require finding the lowest common ancestor
                # Use custom subgraph function
                custom_subgraph = _find_subgraph_from_root(self._subgraph, root, leaf_nodes)
                # Include the original node
                nodes_to_plot = list(custom_subgraph.nodes)
                # Check if these were in the database
                node_in_db = list(set(nodes_to_plot) & set(self._db_idCLs))
                # Find hover names
                hover_names = []
                for node in list(custom_subgraph.nodes):
                    hover_names.append(self._db_idCL_names[node])
                plotly_graph = _plotly_subgraph(custom_subgraph, node_in_db, hover_names)
                
        return plotly_graph

    def find_antibodies(self, 
                        id_CLs: list,
                        background_id_CLs: list = None,
                        idBTO: list = None, 
                        idExperiment: list = None)-> tuple:
        """Queries the database to find antibodies that mark a provided list of cell populations.

        This function contains two parameters to accept a list of cell populations in the form of cell ontology IDs.
        If providing a list for "id_CLs", the function will return a table of antibodies that are
        expressed in those cell populations. If a list is also provided for "background_id_CLs", the function
        returns a table of antibodies that can distinguish cells in populations defined in "id_CLs" from those
        defined in "background_id_CLs". Additional filters based on tissue and experiment IDs can be applied
        to restrict the data in the query.

        Args:
            id_CLs (list): list of cell populations in the form of cell ontology IDs. 
            background_id_CLs (list, optional): list of cell populations used for comparison.
            idBTO (list): list of tissues in the form of BRENDA tissue ontology IDs.
            idExperiment (list): list of experiment IDs from the database.

        Returns:
            tuple: Returns a tuple (pd.DataFrame, dictionary). 
            
            The dataframe contains rows for all possible antibodies found in each cell population.
            The columns contain statistics regarding the upregulation or downregulation of 
            an antibody for a cell population. The level of detection of an antibody for a
            cell population is also included. 

            The dictionary contains boxplots of antibodies for each provided 
            cell population in both "id_CLs" and "background_id_CLs". Each plot contains the distribution
            of normalized expression levels for each antibody in a specific cell population.
            Each set of boxplots is accessible by providing the cell population ID as the key. 
        """
        
        # First find all descendants of the provided id_CLs. These will be included 
        # when running the LMM
        try: 
            node_fam_dict = self._find_descendants(id_CLs)
            if background_id_CLs is not None:
                background_fam_dict = self._find_descendants(background_id_CLs)
            else:
                background_fam_dict = None
        except NetworkXError as err:
            err_msg = str(err).split(' ')[2] # Get idCL error
            raise Exception(f"Error. {err_msg} not found in the database")
        
        # Call API endpoint here to get_Abs to return dataframe        
        abs_body = {
            "idCL": node_fam_dict,
            "background": background_fam_dict,
            "idBTO": idBTO,
            "idExperiment": idExperiment
        }
                            
        abs_response = requests.post(f"{self.url}/api/findabs", json=abs_body)

        # Check response from server
        if 'text/html' in abs_response.headers.get('content-type'):
            error_msg = abs_response.text
            raise ValueError("No antibodies found. Please retry with fewer cell types or different parameters.")
        elif 'application/json' in abs_response.headers.get('content-type'):
            res_JSON = abs_response.json()
            res_df = pd.DataFrame.from_dict(res_JSON)

        # Store all summary plots for each idCL here
        idCL_plots = {}
                            
        # Antibodies to send to plot_antibodies endpoint
        antibodies = list(res_df.index) 

        # Store all figures in a dict (key: idCL, value: figure)
        all_figures = {}

        for idCL_parent, descendants in node_fam_dict.items():
            desc_copy = descendants.copy()
            desc_copy.insert(0, idCL_parent) # include parent when sending all idCLs to endpoint

            plot_abs_body = {
                "abs": antibodies,
                "idcls": desc_copy
            }

            abs_plot_response = requests.post(f"{self.url}/api/plotabs", json=plot_abs_body) # send antibodies and idCLs
            abs_plot_JSON = abs_plot_response.json() # returns a dictionary
            abs_plot_df = pd.DataFrame.from_dict(abs_plot_JSON) # Convert into a dataframe
            
            # Add to dictionary
            idCL_plots[idCL_parent] = abs_plot_df

        # Plot using plotly
        for idCL, summary_stats_df in idCL_plots.items():
            fig = _plot_antibodies_graph(idCL, res_df, summary_stats_df)
            all_figures[idCL] = fig
    
        return (res_df, all_figures)
    
    def find_celltypes(self,
                       ab_ids: list,
                       idBTO: list = None, 
                       idExperiment: list = None) -> tuple:
        
        """Queries the database to find cell populations that are marked by a provided list of antibodies.

        This function contains one parameter "ab_ids" to accept a list of antibody IDs. 
        This finds all cell populations that are marked by each antibody provided. Additional filters 
        based on tissue and experiment IDs can be applied to restrict the data in the query.

        Args:
            ab_ids (list): list of antibody IDs.
            idBTO (list, optional): list of tissues in the form of BRENDA tissue ontology IDs.
            idExperiment (list, optional): list of experiment IDs from the database.

        Returns:
            tuple: Returns a tuple (dictionary, dictionary).

            The first dictionary in the tuple will be the query results from the database in the 
            form of a dataframe. The dataframe contains rows for all possible cell populations
            that the antibody marks. The columns contain statistics regarding the regulation of the
            antibody in comparison to all other cell populations. Each dataframe
            is accessibly by providing the antibody ID as the key.

            The second dictionary in the tuple will be the boxplots for each provided antibody
            in "ab_ids". Each plot contains the distribution of normalized expression levels
            for cell populations marked by each antibody. Each set of boxplots is accessible
            by providing the antibody ID as the key.
        """

        # Dict to hold results (dataframe) for each antibody
        ab_df_dict = {}
        
        celltypes_body = {
            "ab": ab_ids,
            "idBTO": idBTO,
            "idExp": idExperiment
        }
        
        # Call API endpoint here to get dataframe
        celltypes_response = requests.post(f"{self.url}/api/findcelltypes", json=celltypes_body)
                           
        if 'text/html' in celltypes_response.headers.get('content-type'):
            res_dict_strjson = celltypes_response.text
            raise ValueError("No cell types found. Please try with different parameters.")
        elif 'application/json' in celltypes_response.headers.get('content-type'):
            res_dict_strjson = celltypes_response.json() # returns a dict of string-jsons
        
        # Convert all of these string-jsons into dataframes
        for key, value in res_dict_strjson.items():
            # Convert string-json into dict
            temp_dict = json.loads(value)
            
            # Convert dict into dataframe
            temp_df = pd.DataFrame.from_dict(temp_dict)
            
            # Add to final dict of dfs
            ab_df_dict[key] = temp_df
            
        # Check for skipped antibodies If so, then the server found no cells in the database
        res_abs = list(res_dict_strjson.keys())
        missing_abs = list(set(ab_ids) - set(res_abs))
        for missing in missing_abs:
            logging.warning(f"No cells found in the database for {missing}. Skipping {missing}")
            
        plotting_dfs = {} # key: ab, value: df
        
        for ab, celltype_res_df in ab_df_dict.items():
            # Extract index from each dataframe
            df_idCLs = list(celltype_res_df.index)
            
            # Send antibody and idCLs to get dataframe
            plot_celltypes_body = {
                "ab": ab,
                "idcls": df_idCLs
            }
            
            celltypes_plot_response = requests.post(f"{self.url}/api/plotcelltypes", json=plot_celltypes_body)
            celltypes_plot_JSON = celltypes_plot_response.json() # returns a dict
            celltypes_plot_df = pd.DataFrame.from_dict(celltypes_plot_JSON) # Convert into a dataframe

            # Store all of these in a dict
            plotting_dfs[ab] = celltypes_plot_df

        # Plot here using plotly
        all_figures = {}
        for ab, celltypes_plot_df in plotting_dfs.items():
            fig = _plot_celltypes_graph(ab, ab_df_dict[ab], celltypes_plot_df)
            all_figures[ab] = fig
            
        return ab_df_dict, all_figures
    
    def find_experiments(self,
                         ab: list = None,
                         idCL: list = None,
                         idBTO: list = None) -> pd.DataFrame:
        """Queries the database to find experiments under specific requirements

        The function must accept at least one parameter when looking up experiments. 
        One parameter is to search by antibodies used in an experiment, which accepts a list of antibody IDs.
        Another parameter is to search by the presence of cell populations, which accepts
        a list of cell ontology IDs. The last parameter is to search by the usage of specific tissues,
        which accepts a list of BRENDA tissue ontology IDs. A combination of these parameters
        can be used to further narrow the number of experiments.

        Args:
            ab (list): list of antibody IDs.
            idCL (list): list of cell populations in the form of cell ontology IDs.
            idBTO (list): list of tissues in the form of BRENDA tissue ontology IDs.

        Returns:
            pd.DataFrame: Returns a dataframe containing information about each experiment.
            This includes its database ID, name, type, PMID, DOI, tissue ID, and tissue name.
        """

        if ab is None and idCL is None and idBTO is None:
            raise Exception("Error. At least one parameter must not be empty.")

        exp_body = {
            "ab": ab,
            "idCL": idCL,
            "idBTO": idBTO
        }
        
        exp_response = requests.post(f"{self.url}/api/findexperiments", json=exp_body)
        if 'text/html' in exp_response.headers.get('content-type'):
            raise Exception(exp_response.text)
        elif 'application/json' in exp_response.headers.get('content-type'):
            res_JSON = exp_response.json()
            res_df = pd.DataFrame.from_dict(res_JSON)
            return res_df
    
    def which_antibodies(self,
                         search_query: str) -> pd.DataFrame:
        """Queries the database to find antibodies based on a search phrase

        Args:
            search_query (str): a plain text input that contains words that will be used
                to look up antibodies in the database. 

        Returns:
            pd.DataFrame: Returns a dataframe containing information about each antibody.
            This includes: antibody ID, name, target, clonality, citation, clone ID, host organism,
            vendor, catalog number, and all experiment IDs in which the antibody is used.
        """

        wa_body = {
            "search_query": search_query
        }
        
        wa_response = requests.post(f"{self.url}/api/whichantibodies", json=wa_body)
        if 'text/html' in wa_response.headers.get('content-type'):
            raise Exception(wa_response.text)
        elif 'application/json' in wa_response.headers.get('content-type'):
            res_JSON = wa_response.json()
            res_df = pd.DataFrame.from_dict(res_JSON)
            return res_df
    
    def which_celltypes(self,
                        search_query: str) -> pd.DataFrame:
        """Queries the database to find cell types based on a search phrase

        Args:
            search_query (str): a plain text input that contains words that will be used
                to look up cell types in the database.

        Returns:
            pd.DataFrame: Returns a dataframe containing information about each cell type.
            This includes: cell ontology ID, cell type name, and all experiment IDs
            in which the cell type is found in. 
        """
        wc_body = {
            "search_query": search_query
        }
        
        wc_response = requests.post(f"{self.url}/api/whichcelltypes", json=wc_body)
        if 'text/html' in wc_response.headers.get('content-type'):
            raise Exception(wc_response.text)
        elif 'application/json' in wc_response.headers.get('content-type'):
            res_JSON = wc_response.json()
            res_df = pd.DataFrame.from_dict(res_JSON)
            return res_df

    def which_experiments(self,
                          search_query: str) -> pd.DataFrame:
        """Queries the database to find all experiments based on a search phrase

        Args:
            search_query (str): a plain text input that contains words that will be used
                to look up experiments in the database.

        Returns:
            pd.DataFrame: Returns a dataframe containing information about each experiment.
            This includes its database ID, name, type, PMID, DOI, tissue ID, and tissue name.
        """

        we_body = {
            "search_query": search_query
        }

        we_response = requests.post(f"{self.url}/api/whichexperiments", json=we_body)
        if 'text/html' in we_response.headers.get('content-type'):
            raise Exception(we_response.text)
        elif 'application/json' in we_response.headers.get('content-type'):
            res_JSON = we_response.json()
            res_df = pd.DataFrame.from_dict(res_JSON)
            return res_df

    def run_stvea(self,
                  IPD: ImmunoPhenoData,
                  idBTO: list = None, 
                  idExperiment: list = None, 
                  parse_option: int = 1,
                  rho: float = 0.5,
                  population_size: int = 50,
                  # STvEA parameters
                  k_find_nn: int = 40,
                  k_find_anchor: int = 20,
                  k_filter_anchor: int = 40,
                  k_score_anchor: int = 30,
                  k_find_weights: int = 40,
                  k_transfer_matrix: int = 40,
                  c_transfer_matrix: float = 0.5,
                  mask_threshold: float = 0.75,
                  mask: bool = True,
                  num_chunks: int = 1,
                  num_cores: int = 1):
        """Automatically transfers single cell annotations to cytometry data

        Uses reference data stored in the ImmunoPhenoDB database to annotate cells
        in a cytometry dataset. This function uses an algorithm called STvEA, which 
        uses a kNN approach in a consolidated protein expression space to map
        annotations from the reference to query data. This function will find the appropriate
        reference dataset using the spreadsheet provided in the ImmunoPhenoData object. This
        spreadsheet must contain all antibodies and antibody IDs that were used in that experiment.
        For any antibodies in the spreadsheet that are not found in the database, a matching algorithm
        is used to find the next best antibody instead. This level of matching can be
        adjusted in the "parse_option" parameter. This function can be parallelized by specifying
        the number of chunks to split the dataset and the number of cores. 

        Args:
            IPD (ImmunoPhenoData): ImmunoPhenoData object that must already contain the
                normalized protein counts and a spreadsheet with all antibody IDs for each
                antibody used in the experiment. 
            idBTO (list, optional): list of tissue IDs used to restrict the
                reference dataset. This is optional, but specifying a tissue will greatly
                improve the accuracy of the annotations.
            idExperiment (list, optional): list of experiment IDs to restrict
                the reference dataset. This is optional, but specifying certain experiments
                can greatly improve the accuracy of the annotations.
            parse_option (int): level of strictness when searching
                antibodies in the database. Levels are as follows:
                    1: parse by clone ID and alias (default)
                    2: parse by alias and antibody ID (most relaxed)
                    3: parse by antibody ID (strictest)
            rho (float): weight parameter to adjust the number of
                cells or antibodies in the reference dataset. A small value of rho
                will provide more cells and less antibodies. A large value of rho
                will provide more antibodies and less cells. Defaults to 0.5.
            population_size (int): the minimum number of cells needed to define 
                a cell type population. This is used to downsample a large reference
                dataset. Defaults to 50 cells as the minimum number to define a population.
            k_find_nn (int): the number of nearest neighbors. Defaults to 40.
            k_find_anchor (int): the number of neibhbors to find anchors. Defaults to 20.
            k_filter_anchor (int): the number of nearest neighbors to find in the original data space.
                Defaults to 40.
            k_score_anchor (int): The number of neighbors to find anchors.
                Fewer k_anchor should mean higher quality of anchors. Defaults to 30.
            k_find_weights (int): the number of nearest anchors to use in correction. Defaults to 40.
            k_transfer_matrix (int): the number of nearest anchors to use in correction. Defaults to 40.
            c_transfer_matrix (float): a constant that controls the width of the Gaussian kernel. Defaults to 0.5.
            mask_threshold (float): specifies threshold to discard query cells. Defaults to 0.75.
            mask (bool): a boolean value to specify whether to discard 
                query cells that don't have nearby reference cells. Defaults to True.
            num_chunks (int): number of chunks to split the protein dataset for parallelization. Defaults to 1.
            num_cores (int): number of cores used to run in parallel. Defaults to 1.

        Returns:
            ImmunoPhenoData: Returns ImmunoPhenoData object containing transferred annotations
            accessible in the "labels" property of the new object.
        """
        
        IPD_new = copy.deepcopy(IPD)

        # Check if reference query parameters have changed OR if the reference table is empty
        if (IPD_new, IPD_new._stvea_correction_value, idBTO, idExperiment, 
            parse_option, rho, population_size) != self._last_stvea_params or self.imputed_reference is None:

            antibody_pairs = [[key, value] for key, value in IPD_new._ab_ids_dict.items()]
        
            stvea_body = {
                "antibody_pairs": antibody_pairs,
                "idBTO": idBTO,
                "idExperiment": idExperiment,
                "parse_option": parse_option,
                "population_size": population_size
            }

            print("Retrieving reference dataset...")
            timeout = (600, 600) # connection, read timeout
            stvea_response = requests.post(f"{self.url}/api/stveareference", json=stvea_body, timeout=timeout)
            if 'text/html' in stvea_response.headers.get('content-type'):
                raise Exception(stvea_response.text)
            elif 'application/json' in stvea_response.headers.get('content-type'):
                res_JSON = stvea_response.json()
                reference_dataset = pd.DataFrame.from_dict(res_JSON)

            # Output statistics on the number of antibodies matched
            columns_to_exclude = ["idCL", "idExperiment"]
            num_antibodies_matched = (~reference_dataset.columns.isin(columns_to_exclude)).sum()
            if parse_option == 1:
                print(f"Number of antibodies matched from database using clone ID: {num_antibodies_matched}")
            elif parse_option == 2:
                print(f"Number of antibodies matched from database using antibody target: {num_antibodies_matched}")
            elif parse_option == 3:
                print(f"Number of antibodies matched from database using antibody ID: {num_antibodies_matched}")

            # Impute any missing values in reference dataset
            print("Imputing missing values...")
            imputed_reference = _impute_dataset_by_type(reference_dataset, rho=rho) 

            # Apply stvea_correction value
            self.imputed_reference = imputed_reference.copy(deep=True).applymap(
                        lambda x: x - IPD_new._stvea_correction_value if (x != 0 and type(x) is not str) else x)

            # Store these parameters to check for subsequent function calls
            self._last_stvea_params = (IPD_new, IPD_new._stvea_correction_value, idBTO, idExperiment, 
                                       parse_option, rho, population_size)

        # Separate out the antibody counts from the cell IDs 
        imputed_antibodies = self.imputed_reference.loc[:, self.imputed_reference.columns != 'idCL']
        imputed_idCLs = self.imputed_reference['idCL'].to_frame()
    
        # Convert antibody names from CODEX normalized counts to their IDs
        codex_normalized_with_ids = IPD_new._normalized_counts_df.rename(columns=IPD_new._ab_ids_dict, inplace=False)
    
        # Perform check for rows/columns with all 0s or NAs
        codex_normalized_with_ids = _remove_all_zeros_or_na(codex_normalized_with_ids)
                        
        # At this stage, we have all the information we need to run STvEA
        print("Running STvEA...")
        cn = Controller()
        cn.interface(codex_protein=codex_normalized_with_ids, 
                     cite_protein=imputed_antibodies,
                     cite_cluster=imputed_idCLs,
                     k_find_nn=k_find_nn,
                     k_find_anchor=k_find_anchor,
                     k_filter_anchor=k_filter_anchor,
                     k_score_anchor=k_score_anchor,
                     k_find_weights=k_find_weights,
                     # transfer_matrix
                     k_transfer_matrix=k_transfer_matrix,
                     c_transfer_matrix=c_transfer_matrix,
                     mask_threshold=mask_threshold,
                     mask=mask,
                     num_chunks=num_chunks,
                     num_cores=num_cores)
        
        # Store original nearest neighbor distances in class
        nn_dist = cn.stvea.nn_dist_matrix
        self._nn_dist = nn_dist

        # Store transfer_matrix in class
        transfer_matrix = cn.stvea.transfer_matrix
        self.transfer_matrix = transfer_matrix
        
        transferred_labels = cn.stvea.codex_cluster_names_transferred
        
        # Add the labels to the IPD object
        labels_df = transferred_labels.to_frame(name="labels")
        convert_idCL = {
            "idCL": list(set(labels_df['labels']))
        }
        convert_idCL_res = requests.post(f"{self.url}/api/convertcelltype", json=convert_idCL)
        idCL_names = convert_idCL_res.json()["results"]     
        labels_df['celltype'] = labels_df['labels'].map(idCL_names)

        # labels_df now contains two columns: 'labels' and 'celltype'
        # However, some rows may have been filtered out during the run_cca step
        # Fill in missing rows from the original dataset with "Not Assigned"
        index_diff = codex_normalized_with_ids.index.difference(labels_df.index)
        filtered_cells = {'labels': ['Not Assigned'] * len(index_diff), 'celltype': ['Not Assigned'] * len(index_diff)}
        new_df = pd.DataFrame(filtered_cells, index=index_diff)
        complete_labels_df = labels_df.append(new_df)

        # Ensure that all NaN values are replaced with "Not Assigned" in the "celltype" column
        complete_labels_df['celltype'] = complete_labels_df['celltype'].fillna('Not Assigned')
        
        # Before setting norm_cell_types, check if it matches the previous. If not, reset norm_umap field
        if not (complete_labels_df.equals(IPD_new._cell_labels_filt_df)):
            IPD_new._norm_umap = None
        IPD_new._cell_labels_filt_df = complete_labels_df
        
        # Add labels to raw cell labels as well. The filtered rows will be marked as "filtered_by_stvea"
        original_cells_index = IPD_new.protein.index
        merged_df = IPD_new._cell_labels_filt_df.reindex(original_cells_index)
        merged_df = merged_df.fillna("filtered_by_stvea")

        # Check if the raw and norm labels have changed. If so, reset the UMAP field in IPD
        if not (merged_df.equals(IPD_new._cell_labels)):
            IPD_new._raw_umap = None
        IPD_new._cell_labels = merged_df

        # Make sure the indexes match
        IPD_new._normalized_counts_df = IPD_new._normalized_counts_df.loc[IPD_new._cell_labels_filt_df.index]
        print("Annotation transfer complete.")
        return IPD_new

    def _part1_localization(self, IPD, p_threshold=0.05):
        """
        Remove: the option to either remove cells completely from the object after filtering
                OR set it to "Not Assigned" instead in the cell annotations
        """
        # We will be working with the normalized_counts and labels in the IPD object
        # First, we will need to ignore all initial cells labeled as "Not Assigned"
        filtered_index = IPD.labels[IPD.labels['labels'] != 'Not Assigned'].index
        norm_counts = pd.DataFrame(IPD.normalized_counts.loc[filtered_index])
        norm_labels = pd.DataFrame(IPD.labels.loc[filtered_index])
    
        # Temporarily rename the "labels" column to "idCL"
        norm_counts_labels = norm_labels.rename(columns={"labels":"idCL"})
        
        # Combine the normalized protein with the labels field containing "idCL"
        norm_combine = pd.concat([norm_counts, norm_counts_labels.loc[norm_labels.index]["idCL"]], axis = 1)
    
        ## GRAPH CREATION ## 
        # Create dictionary of cell barcodes to their cell types
        if self._downsample_pairwise_graph is None:
            ct_lookup = norm_counts_labels["idCL"].to_dict()
        
            # Downsample to 5000 cells and get pairwise distance matrix
            print("Downsampling dataset to 5000 cells...")
            ds_norm = _downsample(norm_combine, downsample_size=5000)
        
            # Calculate the pairwise correlation distances between all rows (cells)
            print("Calculating pairwise distances between cells...")
            df_without_idCL = ds_norm.drop(columns=["idCL"])
            pairwise_correlation = (1 - df_without_idCL.T.corr())
        
            # Create adjacency matrix
            adj_mat = _pearson_correlation_adjaceny_matrix(pairwise_correlation)
        
            # Create graph
            print("Generating cell graph...")
            G = _fast_cell_label_graph(adj_mat)
        
            # Set cell type for each node in graph
            nx.set_node_attributes(G, ct_lookup, "celltype")
        
            ###################################### Store this graph object in the class
            self._downsample_pairwise_graph = G
        else:
            G = self._downsample_pairwise_graph
    
        # Get all unique cell types to filter over
        all_celltypes = set(norm_counts_labels["idCL"])
    
        # Store labels to remove
        labels_to_remove = []

        print("Conducting fisher exact test for each cell type...")
        # Perform fisher for every cell type
        for celltype in all_celltypes:
            p_val = _run_fisher_exact_test(G, celltype)
            if p_val > p_threshold:
                labels_to_remove.append(celltype)
    
        ## RENAME step
        print(f"Number of cell types to rename:", len(labels_to_remove))
        
        affected_cells = norm_counts_labels[norm_counts_labels['idCL'].isin(labels_to_remove)]
        affected_cells_index = list(affected_cells.index)

        # Find which cells contain these labels
        grouped = affected_cells['idCL'].to_frame().groupby('idCL')
        for cell_type, group in grouped:
            if cell_type == "Not Assigned":
                continue
            print(f"{cell_type}: {len(group.index)}")
    
        # Make a copy of the object to return
        temp_copy = copy.deepcopy(IPD)
    
        # This function needs to reset the UMAPs present. So remove _raw_umap and _norm_umap and set to None
        temp_copy._raw_umap = None
        temp_copy._norm_umap = None
        temp_copy._umap_kwargs = None
        
        print(f"Renaming {len(affected_cells_index)} cells with 'Not Assigned' label...")
        temp_copy.labels.loc[affected_cells.index, ['labels', 'celltype']] = "Not Assigned" # Modify the normalized cell labels
        temp_copy._cell_labels.loc[affected_cells.index, ['labels', 'celltype']] = "Not Assigned" # Modify the unnormalized cell labels
        print(f"Renamed {len(affected_cells_index)} cells.")
        return temp_copy

    def _part2_merging(self, IPD, p_threshold=0.05, epsilon=4):
        # Create a deepcopy of the OWL graph. This one will be constantly updated
        owl_graph_deepcopy = copy.deepcopy(self._OWL_graph)

        # If the downsampled pairwise distance graph doesn't exist, create it now
        if self._downsample_pairwise_graph is None:
            # We will be working with the normalized_counts and labels in the IPD object
            # First, we will need to ignore all initial cells labeled as "Not Assigned"
            filtered_index = IPD.labels[IPD.labels['labels'] != 'Not Assigned'].index
            norm_counts = pd.DataFrame(IPD.normalized_counts.loc[filtered_index])
            norm_labels = pd.DataFrame(IPD.labels.loc[filtered_index])
        
            # Temporarily rename the "labels" column to "idCL"
            norm_counts_labels = norm_labels.rename(columns={"labels":"idCL"})
            
            # Combine the normalized protein with the labels field containing "idCL"
            norm_combine = pd.concat([norm_counts, norm_counts_labels.loc[norm_labels.index]["idCL"]], axis = 1)
        
            ## GRAPH CREATION ## 
            # Create dictionary of cell barcodes to their cell types
            ct_lookup = norm_counts_labels["idCL"].to_dict()
        
            # Downsample to 5000 cells and get pairwise distance matrix
            print("Downsampling dataset to 5000 cells...")
            ds_norm = _downsample(norm_combine, downsample_size=5000)
        
            # Calculate the pairwise correlation distances between all rows (cells)
            print("Calculating pairwise distances between cells...")
            df_without_idCL = ds_norm.drop(columns=["idCL"])
            pairwise_correlation = (1 - df_without_idCL.T.corr())
        
            # Create adjacency matrix
            adj_mat = _pearson_correlation_adjaceny_matrix(pairwise_correlation)
        
            # Create graph
            print("Generating cell graph...")
            G = _fast_cell_label_graph(adj_mat)
        
            # Set cell type for each node in graph
            nx.set_node_attributes(G, ct_lookup, "celltype")
        
            ###################################### Store this graph object in the class
            self._downsample_pairwise_graph = G
        
        # Create a deepcopy of the downsampled pairwise distance graph. It will be updated as well
        downsample_graph_deepcopy = copy.deepcopy(self._downsample_pairwise_graph)
    
        # Remove all nodes in OWL graph that are not "CL:" cell ontology
        nodes_to_remove = [node for node in owl_graph_deepcopy.nodes() if "CL:" not in node]
        owl_graph_deepcopy.remove_nodes_from(nodes_to_remove)
    
        # Get all cell types from the IPD object, ignoring "Not Assigned"
        idCLs = list(set(IPD.labels['labels']))
        if "Not Assigned" in idCLs:
            idCLs.remove("Not Assigned")
        cell_labels_df = IPD.labels.copy(deep=True)
    
        # Call filter function until no more labels are filtered out. This modifies cell_labels_df in-place
        celltypes_remaining = _keep_calling_part2(owl_graph_deepcopy.to_undirected(), 
                                                 downsample_graph_deepcopy, 
                                                 idCLs, 
                                                 cell_labels_df, 
                                                 p_threshold, 
                                                 epsilon)
        
        number_merged = sum(1 for element in celltypes_remaining if "_" in element)
        print(f"Number of cell types merged: {number_merged}")
    
        # Create a new object to return
        temp_copy = copy.deepcopy(IPD)
    
        # This function needs to reset the UMAPs present. So remove _raw_umap and _norm_umap and set to None
        temp_copy._raw_umap = None
        temp_copy._norm_umap = None
        temp_copy._umap_kwargs = None
    
        # Set the new cell_labels_df in the temp object to return
        temp_copy.labels = cell_labels_df
    
        return temp_copy
    
    def _part3_reassign_by_distance_ratio(self, IPD, distance_ratio_threshold=10):
        if self._nn_dist is None:
            raise Exception("Missing distance matrix. run_stvea() must be called to filter by distance ratios.")

        cell_labels_df = IPD.labels
        
        # D1 values
        d1_df = _calculate_D1(self._nn_dist)

        # D2 values
        d2_df = _calculate_D2_fast(self._nn_dist)

        # D1/D2 ratio
        ratio_df = _calculate_D1_D2_ratio(d1_df, d2_df)        

        # Find all rows/cells with a ratio greater than the threshold
        cells_to_unlabel = ratio_df[ratio_df["ratio"] > distance_ratio_threshold]

        # Re-assign all affected cells with "Not Assigned"
        temp_labels = cell_labels_df.copy(deep=True)
        
        # Find the number of cells for each cell type getting re-assigned
        cells_group_df = (temp_labels.loc[cells_to_unlabel.index])
        # Disregard cells that were already named "Not assigned"
        num_cells_already_not_assigned = sum(cells_group_df["labels"] == "Not Assigned")
        print("Number of cells renamed to 'Not Assigned':", len(cells_group_df) - num_cells_already_not_assigned )
        
        # Group all rows by cell type
        grouped = cells_group_df.groupby('labels')
        for cell_type, group in grouped:
            if cell_type == "Not Assigned":
                continue
            print(f"{cell_type}: {len(group.index)}")

        # Renaming step
        temp_labels.loc[cells_to_unlabel.index, ['labels', 'celltype']] = "Not Assigned"

        # Set this temp_labels table in a new IPD object
        temp_copy = copy.deepcopy(IPD)
        # This function needs to reset the UMAPs present. So remove _raw_umap and _norm_umap and set to None
        temp_copy._raw_umap = None
        temp_copy._norm_umap = None
        temp_copy._umap_kwargs = None

        # Set the new cell_labels_df in the temp object to return
        temp_copy.labels = temp_labels
        
        return temp_copy
    
    def _part4_reassign_by_entropy(self, IPD, entropy_threshold=10):
        if self.transfer_matrix is None or self.imputed_reference is None:
            raise Exception("Missing transfer_matrix/imputed_reference. run_stvea() must be called to filter by entropies.")
        
        cell_labels_df = IPD.labels
        
        # Find cell indices for each cell type
        cell_indices_by_type = _group_cells_by_type(self.imputed_reference)

        # Calculate entropies for each cell type for each query cell in transfer matrix
        entropies_df = _calculate_entropies_fast(self.transfer_matrix, cell_indices_by_type)
        
        # Find all rows/cells with a ratio greater than the threshold
        cells_to_unlabel = entropies_df[entropies_df["entropy"] > entropy_threshold]
        
        # Re-assign all affected cells with "Not Assigned"
        temp_labels = cell_labels_df.copy(deep=True)
        
        # Find the number of cells for each cell type getting re-assigned
        cells_group_df = (temp_labels.loc[cells_to_unlabel.index])
        # Disregard cells that were already named "Not assigned"
        num_cells_already_not_assigned = sum(cells_group_df["labels"] == "Not Assigned")
        print("Number of cells renamed to 'Not Assigned':", len(cells_group_df) - num_cells_already_not_assigned )
        
        # Group all rows by cell type
        grouped = cells_group_df.groupby('labels')
        for cell_type, group in grouped:
            if cell_type == "Not Assigned":
                continue
            print(f"{cell_type}: {len(group.index)}")

        # Renaming step
        temp_labels.loc[cells_to_unlabel.index, ['labels', 'celltype']] = "Not Assigned"

        # Set this temp_labels table in a new IPD object
        temp_copy = copy.deepcopy(IPD)
        # This function needs to reset the UMAPs present. So remove _raw_umap and _norm_umap and set to None
        temp_copy._raw_umap = None
        temp_copy._norm_umap = None
        temp_copy._umap_kwargs = None

        # Set the new cell_labels_df in the temp object to return
        temp_copy.labels = temp_labels
        
        return temp_copy
    
    def filter_labels(self, 
                      IPD: ImmunoPhenoData,
                      localization: bool = False,               # Filtering step
                      merging: bool = False,                    # Filtering step
                      distance_ratio: bool = False,             # Filtering step
                      entropy: bool = False,                    # Filtering step
                      p_threshold_localization: float = 0.05,    # P value threshold for fisher exact test during localization 
                      p_threshold_merging: float = 0.05,         # P value threshold during merging
                      epsilon_merging: int = 4,                # Epsilon value for deciding to merge two cell types based on proportion ratio
                      distance_ratio_threshold: int = 2,       # Ratio threshold when filtering cells by NN distance ratios (D1/D2)
                      entropy_threshold: int = 2,              # Entropy threshold when filtering cells by total entropy for cell types
                      remove_labels: bool = False):             # Remove cells as "Not Assigned" at the end of filtering:             
        """Filters out poor-quality annotations using the protein expression space

        Args:
            IPD (ImmunoPhenoData): ImmunoPhenoData object that contains cell labels in the
                "labels" property of the object. 
            localization (bool): option to filter annotations by the localization
                of annotations in the protein expression space.
            merging (bool): option to merge two annotations that cannot be separate in the
                protein expression space. 
            distance_ratio (bool): option to filter annotations by a mapping distance ratio
                calculated from the nearest neighbors in the reference data. 
            entropy (bool): option to filter annotations by entropy caluclated from 
                cell type probabilities for each cell.
            p_threshold_localization (float): p value threshold for fisher exact test
                during localization filtering. 
            p_threshold_merging (float): p value threshold for fisher exact test when
                merging two annotations. 
            epsilon_merging (int): epsilon threshold for deciding to merge two cell types
                based on the proportion of each cell type.
            distance_ratio_threshold (int): ratio threshold when filtering cells
                by nearest neighbor distance ratios (D1/D2).
            entropy_threshold (int): entropy threhold when filtering cells by entropies
                calculated from cell type probabilities.
            remove_labels (bool): option to remove rows/cells from the object that are
                labeled as "Not Assigned" after filtering.

        Returns:
            ImmunoPhenoData: Returns ImmunoPhenoData object that contain modified cell labels.
            These could be "Not Assigned" or two labels merged together. This object could also 
            have rows/cells filtered out.
        """
        # Ensure at least one of the filtering steps is enabled
        if not (localization or merging or distance_ratio or entropy):
            raise ValueError("At least one of 'localization', 'merging', 'distance_ratio', or 'entropy' must be set to True.")
        
        # Issue warnings for unused parameters
        if not localization:
            logging.warning("Parameters for localization will be ignored because 'localization' is set to False.")
        
        if not merging:
            logging.warning("Parameters for merging will be ignored because 'merging' is set to False.")
        
        if not distance_ratio:
            logging.warning("Parameters for distance_ratio will be ignored because 'distance_ratio' is set to False.")
        
        if not entropy:
            logging.warning("Parameters for entropy will be ignored because 'entropy' is set to False.")
        
        # Start with the initial IPD
        IPD_new = copy.deepcopy(IPD)
        
        # Call the localization function if needed
        if localization:
            print("Performing localization...")
            IPD_new = self._part1_localization(IPD=IPD_new, p_threshold=p_threshold_localization)
            print("Localization complete.\n")
        # Call the merging function if needed
        if merging:
            print("Performing merging...")
            IPD_new = self._part2_merging(IPD=IPD_new, p_threshold=p_threshold_merging, epsilon=epsilon_merging)
            print("Merging complete.\n")
        # Call the distance_ratio function if needed
        if distance_ratio:
            print("Performing nearest neighbor distance ratio filtering...")
            IPD_new = self._part3_reassign_by_distance_ratio(IPD=IPD_new, distance_ratio_threshold=distance_ratio_threshold)
            print("Distance_ratio filtering complete.\n")
        # Call the entropy function if needed
        if entropy:
            print("Performing cell type entropy filtering...")
            IPD_new = self._part4_reassign_by_entropy(IPD=IPD_new, entropy_threshold=entropy_threshold)
            print("Entropy filtering complete.\n")
        
        # Make sure all "Not Assigned" rows are consistent in both "labels" and "celltypes" column of IPD.labels
        IPD_new.labels.loc[IPD_new.labels["labels"] == "Not Assigned", "celltype"] = "Not Assigned"
        affected_cells_index = IPD_new.labels[IPD_new.labels["labels"] == "Not Assigned"].index
        
        # Remove all rows/cells that had "Not Assigned"
        if remove_labels:
            print(f"Removing {len(affected_cells_index)} cells with 'Not Assigned' from object...")
            # Filter out attributes that start with an underscore and are not properties
            attrs_to_iterate = [attr_name for attr_name in dir(IPD_new) if attr_name.startswith('_') and not isinstance(getattr(IPD_new.__class__, attr_name, None), property)]
            
            for attr_name in attrs_to_iterate:
                attr_value = getattr(IPD_new, attr_name)
                if isinstance(attr_value, pd.DataFrame):
                    # If the attribute is a DataFrame, drop rows from it using the provided index
                    setattr(IPD_new, attr_name, attr_value.drop(index=affected_cells_index, errors='ignore'))
            
            print(f"Removed {len(affected_cells_index)} cells from object.")
        
        return IPD_new
    
    def db_stats(self):
        """Prints summary statistics about the experiments and data stored in the database

        Returns:
            None. Prints statistics about the information stored in the database.
            This includes: number of experiments, tissues, cells, antibodies, antibody targets,
            antibody clones, and the average number of experiments used by each antibody.
        
        """
        stats_res = requests.get(f"{self.url}/api/databasestatistics")  
        if 'text/html' in stats_res.headers.get('content-type'):
            raise Exception(stats_res.text)
        elif 'application/json' in stats_res.headers.get('content-type'):
            stats_JSON = stats_res.json()

        print("Database Statistics\n===================")
        print("Number of experiments:", stats_JSON["num_exp"])
        print("Number of tissues:", stats_JSON["num_tissue"])
        print("Number of cells:", stats_JSON["num_cells"])
        print("Number of antibodies:", stats_JSON["num_ab"])
        print("Number of antibody targets:", stats_JSON["num_targets"])
        print("Number of antibody clones:", stats_JSON["num_clones"])
        print("Average number of experiments per antibody:", "{:.2f}".format(stats_JSON["avg_exp"]))
