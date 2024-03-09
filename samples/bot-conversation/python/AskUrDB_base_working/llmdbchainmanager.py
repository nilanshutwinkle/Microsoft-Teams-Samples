import os
import warnings

import pandas as pd

from sqlalchemy import create_engine

import boto3
from botocore.config import Config

from langchain_community.llms import Ollama
from langchain.utilities import SQLDatabase
from langchain.llms.bedrock import Bedrock
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt

from pandasai import SmartDataframe
from pandasai.llm import HuggingFaceTextGen

warnings.filterwarnings("ignore")

server = 'nwu-capstone-2024.database.windows.net'
database = 'capstone'
username = 'team4'
password = '{capstone#123}'
driver= '{ODBC Driver 17 for SQL Server}'
    
class LLMDBChainManager(object):
    _instance = None
    _llm_connection = None
    _db_connection = None
    _llm_db_chain = None
    _fsp_chain = None
    _pandasai = None

    def __init__(self):
        if not LLMDBChainManager._db_connection:
            self._create_db_connection()
            self._create_llm_connection()

    @classmethod
    def get_connection(cls):
        if not cls._instance:
            cls._instance = cls({})  
        return cls._db_connection, cls._llm_connection

    def _create_db_connection(self):
        driver = '{ODBC Driver 17 for SQL Server}'
        odbc_str = 'mssql+pyodbc:///?odbc_connect=' \
                        'Driver='+driver+ \
                        ';Server=tcp:' + server + ';PORT=1433' + \
                        ';DATABASE=' + database + \
                        ';Uid=' + username+ \
                        ';Pwd=' + password + \
                        ';Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'

        db_engine = create_engine(odbc_str)
        db = SQLDatabase(db_engine)
        LLMDBChainManager._db_connection = db

    def _create_llm_connection_(self, model_id = "meta.llama2-13b-chat-v1"):
        access_key = "AKIAYS2NRH5JQ5B6VGKO"
        secret_access_key = "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"

        retry_config = Config(
                region_name = 'us-east-1',
                retries = {
                    'max_attempts': 5,
                    'mode': 'standard'
                }
        )

        boto3_bedrock_runtime = boto3.client("bedrock-runtime", 
                                            aws_access_key_id = access_key, 
                                            aws_secret_access_key = secret_access_key, 
                                            config=retry_config)

        model_parameter = {"temperature": 0.3, "top_p": .9, "max_gen_len": 200} #parameters define
        LLMDBChainManager._llm_connection = Bedrock(model_id=model_id, model_kwargs=model_parameter,client=boto3_bedrock_runtime)

        LLMDBChainManager._llm_db_chain = SQLDatabaseChain.from_llm(LLMDBChainManager._llm_connection, LLMDBChainManager._db_connection, verbose=False, return_sql=True, use_query_checker=True)

    def _create_llm_connection(self, model_id = "gemma:2b"):

        LLMDBChainManager._llm_connection = Ollama(model=model_id)

        LLMDBChainManager._llm_db_chain = SQLDatabaseChain.from_llm(LLMDBChainManager._llm_connection, LLMDBChainManager._db_connection, verbose=False, return_sql=True, use_query_checker=True)


    def run_query(self, query):
        return LLMDBChainManager._llm_db_chain.run(query)

    def create_examples_with_db_response(self, dbconn, examples):
        for ex in examples:
            df = pd.read_sql_query(ex['sql_cmd'], dbconn)
            ex['answer'] = str(df.head().values.tolist()) #.to_html(justify='center', max_rows= 5)
        # print(examples[:3])
        return examples
    
    def create_embeddings_for_fsp(self, dbconn, examples, prompt_format):
        embeddings = HuggingFaceEmbeddings()

        to_vectorize = [" ".join(str(example.values())) for example in self.create_examples_with_db_response(dbconn, examples)]

        vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vectorstore,
            k=1,
        )

        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=prompt_format,
            prefix=_mysql_prompt,
            suffix=PROMPT_SUFFIX, 
            input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
        )

        LLMDBChainManager._fsp_chain = SQLDatabaseChain.from_llm(LLMDBChainManager._llm_connection, LLMDBChainManager._db_connection, prompt=few_shot_prompt, use_query_checker=True, 
                                        verbose=True, return_sql=True,)
    
    def run_query_with_fsp(self, query):
        if LLMDBChainManager._llm_db_chain is None:
            raise ValueError("No examples created for Few Shots Prompting. Use create_embeddings_for_fsp method for creating embedding")
        
        return LLMDBChainManager._llm_db_chain.run(query)
    
    def get_graph_from_df(self, df):
        smart_df = SmartDataframe(df, config={'llm': LLMDBChainManager._llm_connection})
        return smart_df.chat('''Plot the dataframe with approprite chart using plotly and 
                    and provide its title and labels for x-axis and y-axis and legends if needed''')

