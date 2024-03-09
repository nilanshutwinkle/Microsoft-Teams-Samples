# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import json

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount
from sqldatabasemanager import SQLDatabaseManager
from llmdbchainmanager import LLMDBChainManager
from langchain.prompts.prompt import PromptTemplate
import re
import pandas as pd
import pandasai
from pandasai.llm import HuggingFaceTextGen
import plotly

examples = [
            {
                "input": "I want to perform product order analysis, for that get me the total quantity ordered for different products.",
                "sql_cmd": "SELECT productcode, SUM(quantityordered) AS Total_Quantities_Ordered FROM sales_order GROUP BY productcode ORDER BY Total_Quantities_Ordered DESC;"
            },
            {
                "input": "I want perform product order analysis by querying top 5 products ordered",
                "sql_cmd": "SELECT Top 5 productcode, SUM(quantityordered) AS Total_Quantities_Ordered FROM sales_order GROUP BY productcode ORDER BY Total_Quantities_Ordered DESC;"
            },
            {
                "input": "I want to get three highest products in terms of sales",
                "sql_cmd": "SELECT Top 3 productcode, SUM(sales) AS Total_Sales FROM sales_order GROUP BY productcode ORDER BY Total_Sales DESC;"
            },
            {
                "input": "I need customer names and the city of ten highest who has ordered the most quantities overall to perform customer order analysis",
                "sql_cmd": "SELECT Top 10 customername, city, SUM(quantityordered) AS Total_Quantities_Ordered FROM sales_order GROUP BY customername, city ORDER BY Total_Quantities_Ordered DESC;"
            },
            {
                "input": "I want analyse top five highest ordered products to perform product performance analysis.",
                "sql_cmd": "SELECT Top 5 productcode, SUM(quantityordered) AS Total_Quantities_Ordered FROM sales_order GROUP BY productcode ORDER BY Total_Quantities_Ordered DESC;"
            },
            {
                "input": "Provide me the total sales amount of orders with different status.",
                "sql_cmd": "SELECT status, SUM(sales) AS Total_Sales FROM sales_order GROUP BY status ORDER BY Total_Sales DESC;"
            },
            {
                "input": "For analysing customer purchases I want to query customer wise average sales",
                "sql_cmd": "SELECT customername, AVG(sales) AS Average_Sales FROM sales_order GROUP BY customername ORDER BY Average_Sales DESC;"
            },
            {
                "input": "I want to analyse average quantity ordered to perform customer orders analysis.",
                "sql_cmd": "SELECT customername, AVG(quantityordered) AS Average_Quantity_Ordered FROM sales_order GROUP BY customername ORDER BY Average_Quantity_Ordered DESC;"
            },
            {
                "input": "I want to list products with sufficient stock quantities, where quantity ordered is less than 50.",
                "sql_cmd": "SELECT productcode, SUM(quantityordered) AS Quantity_Ordered FROM sales_order GROUP BY productcode HAVING SUM(quantityordered) <= 50 ORDER BY Quantity_Ordered DESC;"
            },
            {
                "input": "Give me the list of customer name of all the customers.",
                "sql_cmd": "SELECT DISTINCT customername AS Customer_Name FROM sales_order;"
            },
            {
                "input": "List down all cities from where customer has ordered.",
                "sql_cmd": "SELECT DISTINCT city AS Customer_Name FROM sales_order;"
            },
            {
                "input": "Give me the total ordered placed per city.",
                "sql_cmd": "SELECT city, SUM(quantityordered) AS Quantity_Ordered FROM sales_order GROUP BY city ORDER BY Quantity_Ordered DESC;"
            },
            {
                "input": "Give me the total shipped orders for the customers.",
                "sql_cmd": "SELECT customername, COUNT(status) AS Total_Orders_Delivered FROM sales_order GROUP BY customername, status HAVING status = 'Shipped';"
            },
            {
                "input": "Please list down last five state with lowest quantities ordered.",
                "sql_cmd": "SELECT Top 5 state, SUM(quantityordered) AS Quantity_Ordered FROM sales_order GROUP BY state ORDER BY Quantity_Ordered ASC;"
            },
            {
                "input": "Please list the top 5 products with highest quantity ordered.",
                "sql_cmd": "SELECT Top 5 productcode, SUM(quantityordered) AS Quantity_Ordered FROM sales_order GROUP BY productcode ORDER BY Quantity_Ordered DESC;"
            },
            {
                "input": "I want to perform quaterly order analysis. So give me the total orders placed in each quarter.",
                "sql_cmd": "SELECT qtr_id, SUM(quantityordered) AS Quantity_Ordered FROM sales_order GROUP BY qtr_id ORDER BY qtr_id ASC;"
            },
            {
                "input": "Give me the quater with highest quantities ordered.",
                "sql_cmd": "SELECT Top 1 qtr_id, SUM(quantityordered) AS Quantity_Ordered FROM sales_order GROUP BY qtr_id;"
            },
            {
                "input": "Give me the state lowest quantity ordered",
                "sql_cmd": "SELECT TOP 1 state, SUM(quantityordered) AS Quantity_Ordered FROM sales_order GROUP BY state ORDER BY Quantity_Ordered ASC;"
            },
            {
                "input": "Give me the territory with highest qunatity ordered",
                "sql_cmd": "SELECT TOP 1 territory, SUM(quantityordered) AS quantityordered FROM sales_order GROUP BY territory ORDER BY quantityordered DESC;"
            },
            {
                "input": "I want analyze sales based on dealsize.",
                "sql_cmd": "SELECT dealsize, SUM(sales) AS Sales FROM sales_order GROUP BY dealsize;"
            },
            {
                "input": "I want to query the latest month when each product of the order was placed.",
                "sql_cmd": "SELECT productcode, MAX(month_id) AS Latest_Month FROM sales_order GROUP BY productcode;"
            },
            {
                "input": "Give me the total product ordered per quarter.",
                "sql_cmd": "SELECT qtr_id AS Quarter, SUM(quantityordered) AS Quantity_Ordered FROM sales_order GROUP BY qtr_id ORDER BY qtr_id;"
            }
        ]


prompt_format = PromptTemplate(
    input_variables=["input", "sql_cmd", "answer"],
    template="\nQuestion: {input}\nSQLQuery: {sql_cmd}\nResult: {answer}",
)

class MyBot(ActivityHandler):
    _CURRENT_CONTEXT = None
    _DEFAULT_CONTEXT = 0
    _SQL_CONTEXT = 1
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.

    def convert_to_pandas_df(self, lines):
        # lines = sql_results_raw.split('\n')

        # Separate header and data
        header = [h.strip() for h in lines[0].split(',')]
        data_rows = [r.split(',') for r in lines[2:]]
        print(header, data_rows)
        # Create DataFrame 
        return pd.DataFrame(data_rows, columns=header)
    
    def generate_pandas_ai(self, df):
        return pandasai(df, 
                    prompt = '''Plot the dataframe with approprite chart using plotly and 
                    and provide its title and labels for x-axis and y-axis and legends if needed''')

    async def on_message_activity(self, turn_context: TurnContext):
        if 'hi' in turn_context.activity.text.lower() or 'hello' in turn_context.activity.text.lower() or 'exit' in turn_context.activity.text.lower():
            MyBot._CURRENT_CONTEXT = MyBot._DEFAULT_CONTEXT
        if 'query' in turn_context.activity.text.lower() or 'analyse' in turn_context.activity.text.lower()  or 'give' in turn_context.activity.text.lower():
            MyBot._CURRENT_CONTEXT = MyBot._SQL_CONTEXT

        if MyBot._CURRENT_CONTEXT == MyBot._DEFAULT_CONTEXT:
            await turn_context.send_activity(f"You said '{ turn_context.activity.text }'")
        elif MyBot._CURRENT_CONTEXT == MyBot._SQL_CONTEXT:
            formatted_db_chain_results = self.llm_chain._execute_llm_db_chain(turn_context.activity.text)
            if formatted_db_chain_results is None or formatted_db_chain_results == '':
                await turn_context.send_activity('Sorry, its little over my head, can you please rephrase the query?')
            else:
                print('-'*50)
                print(formatted_db_chain_results)
                eda_results = self.llm_chain._execute_eda_chain(formatted_db_chain_results)
                print('-'*50)
                print(eda_results)
                final_results = self.llm_chain._execute_python_code(formatted_db_chain_results, eda_results)
                print(type(final_results))
                # response = self.llm_chain._execute_python_code(turn_context.activity.text)

                await turn_context.send_activity(f"Got it! Please find the response below.\n\n {pd.DataFrame(json.loads(formatted_db_chain_results)).to_markdown()} \n\nBelow is the visual representation of the above data. Hope its useful \n\n {final_results}")

    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                print(member_added.id, turn_context.activity.recipient.id)
                await turn_context.send_activity("""Hello! Welcome to the AskUrDB Bot! Here you can ask your query and I will answer it from the data in database """)

                self.conn = SQLDatabaseManager()
                self.conn = self.conn.__enter__()
                self.llm_chain = LLMDBChainManager()
                self.llm_chain.create_embeddings_for_fsp(self.conn, examples, prompt_format)
                self.llm_chain._create_llm_connection()
                self.llm_chain.initialize_chains()
