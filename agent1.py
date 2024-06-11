import pandas as pd
import os,json,re
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from flasgger import LazyString, LazyJSONEncoder
from werkzeug.exceptions import BadRequest
from training_agent import train,get_data
from flask_cors import CORS
import chromadb,requests
from sqlalchemy import create_engine 

# methods are imported here from the other files
from csv_communication import description_data,generate_summary,generate_sql_query,generate_answer,csv_gererate_graph_type
from mongo_conn import mongo_validation,user_id_validating,source_id_validation,filtered_file_path,validate_mongo_connection_string
from db_conn import db_connect,get_db_schema,db_schema_datatype,get_db_tables_columns,validate_connection_string
from dimensions_measures import table_relationships,group_tables,dimensions_measure_categories
from vanna_connect import vn
from db_communication import db_response_data,generate_system_understanding,gererate_graph_type
from csv_pg import create_table_from_dataframe
from csv_cleaning import cleaning_data
from pdf_bot import all_functions, get_answer_pdf
from market_maven_agents import newsapi1,newsapi2,nlp,new_one,analytics

from document_text_extractor import extract_text_from_pdf, extract_text_from_docx
# from document_text_extractor import extract_text_from_doc
import tempfile
from langchain import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import HarmCategory, HarmBlockThreshold
from langchain.memory import ConversationBufferMemory
import getpass

# To get the secrete keys from the .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = 'sk-proj-MiZXmGVEeQiDND6H1C0VT3BlbkFJI5jnF6GjReWQL3cDIxEV'  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
print('OPENAI_API_KEY:',OPENAI_API_KEY) 

connection_string = os.getenv("mongo_conn_string")
print('mongo_conn_string =:',connection_string)

# create the folder
upload_folder_path = 'uploads'
if not os.path.exists(upload_folder_path):
    os.makedirs(upload_folder_path)


# Initialize Flask app
app = Flask(__name__)
CORS(app)
CORS(app, origins='http://128.199.21.237:3000')
CORS(app, origins='http://localhost:3000')
CORS(app, origins='https://chat.ahexsolutions.com')


# Initialize Swagger
# swagger = Swagger(app)
app.config["SWAGGER"] = {"title": "Swagger-UI", "uiversion": 2}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/swagger/",
}

template = dict(
    swaggerUiPrefix=LazyString(lambda: request.environ.get("HTTP_X_SCRIPT_NAME", ""))
)

app.json_encoder = LazyJSONEncoder
# swagger = Swagger(app, config=swagger_config, template=template)
swagger = Swagger(app,config=swagger_config,)

# chromadb client
chroma_client = chromadb.PersistentClient(path="local_chroma")

@app.route('/sqlagent/v2/api', methods=['POST'])
@swag_from('swagger/agent.yml')
def ask_question():
    """Endpoint to handle incoming questions.

    This function retrieves the question from the request, connects to MongoDB to fetch the required data, 
    establishes a connection to the database, generates SQL query, runs the query, 
    and returns a JSON response containing the SQL query, JSON data, summary, system understanding, status, 
    and chart presentation.

    Returns:
        Response: JSON response containing information about the SQL query, JSON data, summary, 
        system understanding, status, and chart presentation.
    """
    request_data = request.get_json()
    if not request_data:
        error = 'Invalid request. Missing requested data.'
        return jsonify({'error': error}), 400
    data = request_data
    print('data:', data)

    question = data.get('user_input')
    if not question:
        return jsonify({'error':'user_input is missing'}), 400
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({'error':'user_id is missing'}), 400
    elif user_id:
        # Validating whether the user_id is existing or not
        valid_user =user_id_validating(connection_string, user_id)
        if valid_user == False:
            return jsonify({'error':'user_id  is not exist !!!'}), 400
  
    source_id = data.get('source_id')
    if not source_id:
        return jsonify({'error':'source_id is missing'}), 400
    
    elif source_id:
        #  check for mongo db source-id exist or not
        valid_source_ids = source_id_validation(connection_string, user_id, source_id)
        if valid_source_ids == False:
            return jsonify({'error':'source_ids  is not exist or length of source is less or greater than 24 characters !!!'}), 400
    if 'gpt_model' in data:
        gpt_model = data['gpt_model']
    else:
        gpt_model =''
        
    # Retrieving the matched sources information such as filetype, filepath.
    try:
        result = mongo_validation(connection_string,user_id, source_id)
        print('result:',result)
        
    except BadRequest as e:
        return jsonify({'error in connecting mongoDB': str(e)}), 400
    
    if result[0]["fileType"] == 'dbConnection':
        conn_path = result[0]["filePath"]
        # DB connection
        conn = db_connect(conn_path,source_id)
        if not conn:
            error='error in database  connection issue !!!'
            return jsonify({'error': error}), 400

        print('conn:',conn)
        try:
            # first need to initialize the database to the vanna
            def generated_dataframes(sql: str) -> pd.DataFrame:
                """From the trained vanna model, it generate the  sql query"""
                print('sql and conn:',sql,"conn:",conn)
                df = pd.read_sql(sql, conn)
                print('vanna df:',df)
                return df

            vn.run_sql = generated_dataframes
            vn.run_sql_is_set = True

            # Get all the values here
            #try:
            generated_sql_query, json_data, summary_data, data_frame = db_response_data(question)
            print('db dataframe: ',data_frame)
            result_df = pd.DataFrame(data_frame)
            print('result_df:',result_df)
        except:
            return jsonify({'user_input':question,'source_id':source_id,'response':'Please enter valid question !!!'})

        seperate_cols = {}
        # Extract values for each column
        for column in result_df.columns:
            seperate_cols[column] = list(zip(data_frame[column]))

        print('seperate_cols:',seperate_cols)
        # x and y axix keys
        list_dim=[]
        list_mea=[]
        dimension = {}
        measures = {}
        for key,value in seperate_cols.items():
            for data in seperate_cols[key]:
                print('data[0]:',data[0])
                print('=')
                # Check the data type of the value
                if isinstance(data[0], str):
                    # If the value is a string, store it in the dimension dictionary
                    if key not in dimension:
                        dimension[key] = []
                    dimension[key].append(data[0])
                    #print('di:',dimension)
                else:
                    # If the value is not a string, store it in the measures dictionary
                    if key not in measures:
                        measures[key] = []
                    measures[key].append(data[0])
        list_dim.append(dimension)
        list_mea.append(measures)

        # Print the dimension and measures dictionaries
        print("Dimension:")
        # print(dimension)
        print("=========")
        print(list_dim)
        print("\nMeasures:")
        print(list_mea)
        # Get generated system_understanding
        system_understanding = generate_system_understanding(question,gpt_model)

        # Get generated Graph type
        graph_type = gererate_graph_type(data_frame,gpt_model)
        json_data = data_frame.to_dict(orient='records')


        response = {
                    'user_input': question,
                    'source_id' : source_id,
                    'sql_query' : generated_sql_query,
                    'json_data' : json_data,
                    'x_axis': list_dim,
                    'y_axis':list_mea,
                    'response' : summary_data,
                    'system_understanding' : system_understanding,
                    'chart_presentation' : graph_type,
                    'gpt_model':gpt_model}
        return jsonify(response)
    
    elif result[0]['fileType'] == 'csv': 
        table_schema={}
        for ids in result:
            print('ids:',ids)
            csvpath = ids['filePath']
            key = str(ids['_id'])
            print('key:',key)
            df =  pd.read_csv(csvpath) 
            # df = pd.read_excel('604063413-Car-Sales-Kaggle-DV0130EN-Lab3-Start.xlsx') 
            table_name=ids['sourceName']
            conn= create_table_from_dataframe(df, table_name,key)
            print('inside conn:',conn.connect())
            print('sourceName:',ids['sourceName'])
            column_names = df.columns.tolist()
            print('column_names:',column_names)
            # table_schema[table_name]=column_names

            table_schema['table name']=table_name
            table_schema['columns']=column_names

        print('all table_schema:',table_schema)
        summary_data = generate_summary(  table_schema,gpt_model)
        print("Summary : ", summary_data)

        sql_query = generate_sql_query(question, table_schema,gpt_model)
        print("SQL Query : ", sql_query)
        conn_pd = conn.connect() 
        print('conn_pd:',conn_pd) 
        try:
            sql_result = conn_pd.execute(sql_query).fetchall()
        except:
            return jsonify({'user_input':question,'source_id':source_id,'response':'Something went wrong !!!'})
        # df = pd.read_sql_query(sql_query, conn_pd)
        # print("df:",df)
        print("sql_result:",sql_result)
        result_df = pd.DataFrame(sql_result)
        print('result_df:',result_df)
        #if not result_df:
        #   return  jsonify({'response':'enter valid  question ...','user_input':question,'source_id':source_id})
        # Convert the list of tuples to a list of dictionaries
        # dict_data = [dict(zip((column_names), row)) for row in sql_result]


        answer_data = generate_answer(question, result_df,gpt_model)
        print("Answer : ", answer_data)


        chart_presentation = csv_gererate_graph_type(result_df,gpt_model)
        print('sql_result:',sql_result)
        seperate_cols = {}

        # Extract values for each column
        for column in result_df.columns:
            seperate_cols[column] = list(zip(result_df[column]))

        print('seperate_cols:',seperate_cols)
        # x and y axix keys

        list_dim=[]
        list_mea=[]
        dimensions ={}
        measures ={}
        for data in sql_result:
            for key, value in data.items():
                if isinstance(value,str):
                    if key not in dimensions:
                        dimensions[key]=[]
                    dimensions[key].append(value)
                else:
                    if key not in measures:
                        measures[key]=[]
                    measures[key].append(value)
        list_dim.append(dimensions)
        list_mea.append(measures)
        json_data =result_df.to_dict(orient='records')
        print('dimensions:', list_dim)
        print('measures:',list_mea)
        return jsonify({
            'user_input': question,
            'source_id' : source_id,
            'x_axis':list_dim,
            'y_axis':list_mea,
            'json_data' : json_data,
            'chart_presentation' : chart_presentation,
            'sql_query' : sql_query,
            'response' : answer_data,
            'summary' : summary_data,
            'gpt_model':gpt_model
            })
    elif result[0]['fileType'] == 'xlsx':
        table_schema={}
        for ids in result:
            print('ids:',ids)
            excel_path = ids['filePath']
            key = str(ids['_id'])
            print('key:',key)
            df =  pd.read_excel(excel_path)
            # workbook = openpyxl.load_workbook("710386854-Directors-Mobile-Email-Database.xlsx")
            # sheet = workbook.active
            # data = sheet.values
            # columns = next(data)[0:]
            # df = pd.DataFrame(data, columns=columns)
            table_name=ids['sourceName']
            conn= create_table_from_dataframe(df, table_name,key)
            print('inside conn:',conn.connect())
            print('sourceName:',ids['sourceName'])
            column_names = df.columns.tolist()
            print('column_names:',column_names)
            table_schema[table_name]=column_names

            # table_schema['table name']=table_name
            # table_schema['columns']=column_names 

        print('all table_schema:',table_schema)
        summary_data = generate_summary(  table_schema,gpt_model)
        print("Summary : ", summary_data)

        sql_query = generate_sql_query(question, table_schema,gpt_model)
        print("SQL Query : ", sql_query)
        conn_pd = conn.connect()
        print('conn_pd:',conn_pd) 
        try:
            sql_result = conn_pd.execute(sql_query).fetchall()
        except:
            return jsonify({'user_input':question,'source_id':source_id,'response':'Something went wrong !!!'})
        # df = pd.read_sql_query(sql_query, conn_pd)
        # print("df:",df)   
        print("sql_result:",sql_result)
        result_df = pd.DataFrame(sql_result)
        print('result_df:',result_df)
        answer_data = generate_answer(question, result_df,gpt_model)
        print("Answer : ", answer_data)
        chart_presentation = csv_gererate_graph_type(result_df,gpt_model)
        print('sql_result:',sql_result)
        seperate_cols = {}

        # Extract values for each column
        for column in result_df.columns:
            seperate_cols[column] = list(zip(result_df[column]))

        print('seperate_cols:',seperate_cols)
        # x and y axix keys
        list_dim=[]
        list_mea=[]
        dimensions ={}
        measures ={}
        for data in sql_result:
            for key, value in data.items():
                if isinstance(value,str):
                    if key not in dimensions:
                        dimensions[key]=[]
                    dimensions[key].append(value)
                else:
                    if key not in measures:
                        measures[key]=[]
                    measures[key].append(value)
        list_dim.append(dimensions)
        list_mea.append(measures)
        json_data =result_df.to_dict(orient='records')
        print('dimensions:', list_dim)
        print('measures:',list_mea)
        return jsonify({
            'user_input': question,
            'source_id' : source_id,
            'x_axis':list_dim,
            'y_axis':list_mea,
            'json_data' : json_data,
            'chart_presentation' : chart_presentation,
            'sql_query' : sql_query,
            'response' : answer_data,
            'summary' : summary_data,
            'gpt_model':gpt_model
            })
    elif result[0]["fileType"] == 'pdf':
        global pdf_list
        if question is None:
            return jsonify({"message":"No query found"})
        elif user_id is None:
            return jsonify({"message":"No user id found"})
        else:
            collections=chroma_client.list_collections()
            collection_names = [collection.name for collection in collections] # Extract names from Collection objects
            try:
                files_list=[]
                for pdf in result:
                    file_url=pdf['filePath']
                    pdf_name=pdf['fileName']
                    files_list.append(pdf_name)
                if str(user_id) not in collection_names:
                    collection = chroma_client.create_collection(name=user_id)
                    pdf_list=[]
                    for pdf in result:
                        file_url=pdf['filePath']
                        source_name=pdf['fileName']
                        try:
                            response = requests.get(file_url)
                            local_file_path = os.path.join(upload_folder_path, os.path.basename(source_name))
                            if response.status_code == 200:
                                with open(local_file_path, 'wb') as file:
                                    file.write(response.content)
                            else:
                                return jsonify(f"Failed to download file. Status code: {response.status_code}")
                        except Exception as e:
                            return jsonify(f"Error downloading file: {e}")
                        all_functions(collection,str(local_file_path),source_name)
                        os.remove(local_file_path)
                else:
                    collection = chroma_client.get_collection(user_id)
                    files_chroma=collection.get(include=["metadatas"],where={"source": {"$ne": 'None'}})
                    source_names=[]
                    for item in files_chroma['metadatas']:
                        if item['source'] not in source_names:
                            source_names.append(item['source'])
                    for pdf in result:
                        file_url=pdf['filePath']
                        source_name=pdf['fileName']
                        if str(source_name) not in source_names:
                            try:
                                response = requests.get(file_url)
                                local_file_path = os.path.join(upload_folder_path, os.path.basename(source_name))
                                if response.status_code == 200:
                                    with open(local_file_path, 'wb') as file:
                                        file.write(response.content)
                                    print(f"File downloaded successfully to: {local_file_path}")
                                else:
                                    return jsonify(f"Failed to download file. Status code: {response.status_code}")
                            except Exception as e:
                                print(f"Error downloading file: {e}")
                            all_functions(collection,str(local_file_path),source_name)
                            os.remove(local_file_path)
                response = get_answer_pdf(question,collection,files_list)
                response['user_input']=question
                response['source_id']=source_id
                return jsonify(response)
            except Exception as e:
                print(e)
                return jsonify({'error': 'Something went wrong !!!'}), 400


    else:
        return jsonify({'error': 'filepath not found !!!'}), 400
    
@app.route('/train_agent', methods=['POST'])
@swag_from('swagger/training.yml')
def trianging():
    # Get the question parameter from the request
    # request_data = request.form.to_dict()
    request_data = request.get_json()
    if not request_data:
        error = 'Invalid request. Missing requested data.'
        return jsonify({'error': error}), 400
    data = request_data
    print('data:', data)
    question = data.get('question')
    sql = data.get('sql_query')
    source_id = data.get('source_id')
    if not question:
        return jsonify({'error':'question is missing'}), 400
    if not sql:
        return jsonify({'error':'sql_query is missing'}), 400
    if not source_id:
        return jsonify({'error':'source_id is missing'}), 400

    res_trained = train(source_id,question,sql)
    # res_get_data = get_data(source_id)
    print("MODEL is trained")
    response = {'source_id': source_id,

                'status' :res_trained
                }
    return jsonify(response)

@app.route('/get_train_data/', methods=['GET'])
def get_trained_data():
    source_id = request.args.get('source_id')
    print(' source_id data:', source_id)

    if not source_id:
        return jsonify({'error':'source_id is missing'}), 400
    res_get_data = get_data(source_id)

    response = {'source_id': source_id,
                'history' : res_get_data,
                }
    return jsonify(response)

    
    
@app.route('/dimension_measures',methods=['POST'])
def dimension_measures():
    request_data = request.get_json()
    if not request_data:
        error = 'Invalid request. Missing question data.'
        return jsonify({'error': error}), 400
    data = request_data
    print('data:', data)
    
    
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'error':'user_id is missing'}), 400
    elif user_id:
        # Validating whether the user_id is existing or not
        valid_user =user_id_validating(connection_string, user_id)
        if valid_user == False:
            return jsonify({'error':'user_id  is not exist !!!'}), 400
  
    source_id = data.get('source_id')
    if not source_id:
        return jsonify({'error':'source_id is missing'}), 400
    elif source_id:
        #  check for mongo db source-id exist or not
        valid_source_ids = source_id_validation(connection_string, user_id, source_id)
        if valid_source_ids == False:
            return jsonify({'error':'source_ids  is not exist or length of source is less or greater than 24 characters !!!'}), 400

    # Retrieving the matched sources information such as filetype, filepath.
    try:
        result = mongo_validation(connection_string,user_id, source_id)
        print('result:',result)
        
    except BadRequest as e:
        return jsonify({'error in connecting mongoDB': str(e)}), 400
    
    if result[0]["fileType"] == 'dbConnection':
        conn_path = result[0]["filePath"]
        # DB connection
        conn = db_connect(conn_path,source_id) 
        if not conn:
            error='error in database  connection issue !!!'
            return jsonify({'error': error}), 400
        
        # Retrieving  database schema
        db_schema ={}
        schema = get_db_schema(conn,db_schema)
        if not schema:
            error='error in database  while geting the db schemas  table along with column and datatype'
            return jsonify({'error': error}), 400
        
        # Retrieving all the tables relationships together
        relationships= table_relationships(schema)
        if not relationships:
            error= "error in database tables relationships"
            return jsonify({'error':error}), 400
        
        # Retrieving only the unique grouped tables
        group_unique_table = group_tables(relationships)
        if not group_unique_table:
            error= "error in database tables grouping together"
            return jsonify({'error':error}), 400
      

        # Retrieving database schema of the table name along with column with column datatype
        datatype_db_schema = db_schema_datatype(conn,db_schema)
        if not datatype_db_schema:
            error= "error in database  while geting the db schemas  table along with column and datatype"
            return jsonify({'error':error}), 400
        
        # Retrieving the dimensions & measures based on the grouped tables
        result_data = dimensions_measure_categories(group_unique_table,datatype_db_schema)
        if not result_data:
            error= "error in the retrieving dimensions and measures for the group"
            return jsonify({'error':error}), 400
        
        print('result data:',result_data)
        print('result data type:',type(result_data))


        # Convert to JSON format
        # json_data = json.dumps(result_data, indent=1)
        # json_data=json.loads(result_data)
        return result_data
    elif result[0]["fileType"] == 'csv':
        table_schema={}
        dimensions=[]
        measures=[]
        for ids in result:
            print('ids:',ids)
            csvpath = ids['filePath']           
            key = str(ids['_id'])
            print('key:',key)          
            df =  pd.read_csv(csvpath)
            table_name=ids['sourceName']
            conn= create_table_from_dataframe(df, table_name,key)
            print('inside conn:',conn.connect())
            print('sourceName:',ids['sourceName'])
            
            column_names = df.columns.tolist()
            print('column_names:',column_names)
            table_schema[table_name]=column_names

            for column in column_names:
                print(f"{column}: {df[column].dtype}")
                if df[column].dtype=='object':
                    dimensions.append(column)
                else:
                    measures.append(column)
            print()
        print('table_schema:',table_schema)
        print('dimensions:',dimensions)
        print('measures:',measures)
        response = {
                    'source_id' : source_id,
                    'dimension':dimensions,
                    'measure':measures
                    }
        return jsonify(response)
    else:
        return jsonify({'error':"Database filetype should be 'dbConnection' !!!"}), 400

@app.route('/get_data',methods =['POST'])
def get_data_api():
    request_data = request.get_json()
    if not request_data:
        error = 'Invalid request. Missing question data.'
        return jsonify({'error': error}), 400
    data = request_data
    print('data:', data)

    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'error':'user_id is missing'}), 400
    elif user_id:
        # Validating whether the user_id is existing or not
        valid_user =user_id_validating(connection_string, user_id)
        if valid_user == False:
            return jsonify({'error':'user_id  is not exist !!!'}), 400
    source_id = data.get('source_id')
    if not source_id:
        return jsonify({'error':'source_id is missing'}), 400
    elif source_id:
        #  check for mongo db source-id exist or not
        valid_source_ids = source_id_validation(connection_string, user_id, source_id)
        if valid_source_ids == False:
            return jsonify({'error':'source_ids  is not exist or length of source is less or greater than 24 characters !!!'}), 400

        
    original_dimensions = data.get('dimensions')
    print('*'*10)
    print('dimensions:',original_dimensions)
    print('*'*10)
    if not original_dimensions:
        return jsonify({'error':'dimensions is missing'}), 400
    
  
    measures = data.get('measures')
    if not measures:
        return jsonify({'error':'measures is missing'}), 400
    
    # filters = data.get('filters')    
    aggregations = data.get('aggregations')
    if not aggregations:
        return jsonify({'error':'aggregations is missing'}), 400
    
    tables = data.get('table_name')
    print('table:',tables)
    print('table type:',type(tables))
    # if not tables:
    #     return jsonify({'error':'tables is missing'}), 400
    
    # join_tables = ', '.join(tables)
    # print('join table type:',type(join_tables))
    # mea_names = [value.split('.')[1] for value in measures]
    # measures = ', '.join(measures)
    
    filters = data.get('filters')
    if not filters:
        return jsonify({'error':'filters is missing'}), 400
    original_values = data.get('values')
    print('---:values',original_values)
    print('---:values typ;e',type(original_values))
    if not original_values:
        return jsonify({'error':'values is missing'}), 400
    # if not ('original_dimensions' and 'measures' and 'aggregations' and 'table_name' and 'filters' and 'values' and 'user_id' and 'source_id' in data):
    #     return jsonify({'error':' missed the fields'}), 400
    # database connection
    values = ', '.join( original_values)
    
    try:
        result = mongo_validation(connection_string,user_id, source_id)
        print('result:',result)
        
    except BadRequest as e:
        return jsonify({'error in connecting mongoDB': str(e)}), 400
    
    if result[0]["fileType"] == 'dbConnection':
        conn_path = result[0]["filePath"]
        # DB connection
        conn = db_connect(conn_path,source_id) 
        # Retrieving  database schema
        db_schema ={}
        schema = get_db_schema(conn,db_schema)
        if not schema:
            error='error in database  while geting the db schemas  table along with column and datatype'
            return jsonify({'error': error}), 400
        
        # Retrieving all the tables relationships together
        relationships= table_relationships(schema)
        if not relationships:
            error= "error in database tables relationships"
            return jsonify({'error':error}), 400
        
        # Retrieving only the unique grouped tables
        group_unique_table = group_tables(relationships)
        if not conn:
            error='error in database  connection issue !!!'
            return jsonify({'error': error}), 400
    
    # retrieve tables from the database and validating the tables
    table_columns={}
    list_table,table_columns = get_db_tables_columns(conn,table_columns)
    print('list of table:',list_table)
    print('list of table_columns:',table_columns)
    print('list of dimensions:',original_dimensions)
    final_dimensions = ', '.join(original_dimensions)
    # Check if tables exist
    # final_table_col = []
    # for user_table in tables:
    #     if user_table in table_columns:
    #         print('table exist ...')
    #         # validate the table with matched columns
    #         for db_table, column in table_columns.items():
    #             if user_table == db_table:
    #                 for user_dimension in original_dimensions:
    #                     if user_dimension in column:
    #                         final_table_col.append((user_table+'.'+user_dimension))                         
                    
    #     else:
    #         return jsonify({'error':f'Table does not exist:{user_table}'}), 400

    # print('final_table_col:',final_table_col)
    # join_final_table_col = ', '.join(final_table_col)

    final_table_names_dim = [value.split('.')[0] for value in original_dimensions]
    final_table_names_mea = [value.split('.')[0] for value in measures]
    print('final_table_names===>',final_table_names_dim)
    print('final_table_names===>',final_table_names_mea)
    table_concatinate = []
    for table in  final_table_names_dim:
        table_concatinate.append(table)
    for table in  final_table_names_mea:
        table_concatinate.append(table)
    print('==final_table===',table_concatinate)
    join_tables = ', '.join(set(table_concatinate))
    flag=0
    print('group tables  :',group_unique_table)
    for group in group_unique_table.values():
        if all(table in group for table in list(set(table_concatinate))):
            flag = 1

    if flag == 0:
        return f'There is no table having relationship {join_tables}'
    print('join table :',join_tables)
    print('join table type:',type(join_tables))
    # Constructing the WHERE clause based on the filter type
    if filters[0] == "IN":
        # where_clause = f"WHERE {join_final_table_col} IN ('{values}')"
        where_clause = 'WHERE '
        column_list = table_concatinate.split()
        print('column_list:',column_list)
        for i, column in enumerate(column_list):
            print('i:',i,'=>column:',column)
            tp_values = tuple(values.split())
            if i > 0:
                where_clause += ' OR '
            where_clause += f' {column} IN {tp_values} '
        cleaned_where_clause = where_clause.replace(',', ' ')
        print('where_clause:',cleaned_where_clause)
        where_clause=cleaned_where_clause
    elif filters[0] == "LIKE":
        where_clause = 'WHERE '
        # column_list = final_table_names.split()
        column_list = original_dimensions
        print('column_list:',column_list)
        for i, column in enumerate(column_list):
            print('i:',i,'=>column:',column)
            if i > 0:
                where_clause += ' AND '
            where_clause += f' {column} LIKE "{values}" '
        cleaned_where_clause = where_clause.replace(',', ' ')
        print('where_clause:',cleaned_where_clause)
        where_clause=cleaned_where_clause
                

    elif filters[0] == "equals":
        where_clause = f"WHERE {table_concatinate} = '{values}'"
    else:
        # raise ValueError("Unsupported filter type")
        where_clause=''

    aggregations_list = aggregations[0]
    join_measures = ', '.join(measures)
    print('join_measures:',join_measures)
    print('join_measures type:',type(join_measures))
    final_measure =  join_measures.replace('.', '_')
    
    # Constructing the SQL query
    sql_query = f"""
    SELECT {final_dimensions}, {aggregations_list}({join_measures}) AS {aggregations_list}_{final_measure}
    FROM {join_tables}
    {where_clause}
    GROUP BY {final_dimensions}, {join_measures};
    """
    # execute the sql query and get the response
    print('sql :',sql_query)
    print('conn_pd:',conn)
    try:
        sql_result = conn.execute(sql_query).fetchall()
        if not sql_result:
            return jsonify({'response':'Empty data found for this selection !!!'})
        print('sql result:',sql_result)
        individual_dimension = final_dimensions.split(',')
        print('individual_dimension:',individual_dimension)
        print('join_measures:',measures)
        columns=individual_dimension+measures
        print('columns:',columns)
        # Zip the column names with each row to create a list of dictionaries
        result_with_columns = [dict(zip(columns, row)) for row in sql_result]
        print('result_with_columns:',result_with_columns)
    
        result_df = pd.DataFrame(sql_result)
        list_dim=[]
        list_mea=[]
        dimension = {}
        measures = {}
        for data in result_with_columns:
            for key, value in data.items():
                # Check the data type of the value
                if isinstance(value, str):
                    # If the value is a string, store it in the dimension dictionary
                    if key not in dimension:
                        dimension[key] = []
                    dimension[key].append(value)
                    
                else:
                    # If the value is not a string, store it in the measures dictionary
                    if key not in measures:
                        measures[key] = []
                    measures[key].append(value)
            list_dim.append(dimension)
            list_mea.append(measures)

        # Print the dimension and measures dictionaries
        print("Dimension:")
        print(dimension)
        print("=========")
        print(list_dim)
        print("\nMeasures:")
        print(measures)
        # print("result_with_columns:",result_with_columns)
        # print('res df :====',result_df)

    except:
        return jsonify({'result':'dimensions or measures were not  found in db for this selection !!!'})
    response = {
                    'source_id' : source_id,
                    'json_data' : result_with_columns,
                    'dimension':list_dim[0],
                    'measure':list_mea[0]
                    }
    return jsonify(response)



@app.route('/filter_csv_path', methods=['POST'])
def clean_csv_api():
    
    request_data = request.get_json()
    if not request_data:
        error = 'Invalid request. Missing question data.'
        return jsonify({'error': error}), 400
    data = request_data
    print('data:', data)
    
    
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'error':'user_id is missing'}), 400
    elif user_id:
        # Validating whether the user_id is existing or not
        valid_user =user_id_validating(connection_string, user_id)
        if valid_user == False:
            return jsonify({'error':'user_id  is not exist !!!'}), 400
  
    source_id = data.get('source_id')
    if not source_id:
        return jsonify({'error':'source_id is missing'}), 400
    elif source_id:
        #  check for mongo db source-id exist or not
        valid_source_ids = source_id_validation(connection_string, user_id, source_id)
        if valid_source_ids == False:
            return jsonify({'error':'source_ids  is not exist or length of source is less or greater than 24 characters !!!'}), 400
    # Retrieving the matched sources information such as filetype, filepath.
    try:
        result = mongo_validation(connection_string,user_id, source_id)
        print('result:',result)
        
    except BadRequest as e:
        return jsonify({'error in connecting mongoDB': str(e)}), 400


    if result[0]["fileType"] == 'csv':
        csv_path = result[0]["filePath"]
        print("FILE : ", csv_path)
        
        column_updated = cleaning_data(csv_path)
        print('column_updated:',column_updated)
        # update_data = {"FilteredFilePath": csv_path}
        # filtered_file_path(connection_string, user_id, source_id, update_data)

        # df = pd.read_csv(csv_path)
    #     print(df.head(10))

    # return jsonify({'Result': result[0]["fileType"]})
    return 'csv file cleaned !!!'




# Prompt user for Google API key if not set in the environment
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")


safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Initialize the language model
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                             safety_settings=safety_settings)


initial_prompt = """You are an Assistant, who will assist the user for building their project.\n
Provide all the requirements that are need for the project.\n
All the technologies that are needed to build the project.\n
Ask questions one after another if you need to suggest all the requirements.\n
Ask as many questions as are required to make the requirements and recommend the technologies.\n
Questions should be asked 1 at a time.\n 
Give the detailed information.\n
Finally provide detailed SRS."""


# Initialize the conversation buffer memory with the initial prompt
memory = ConversationBufferMemory(initial_context=initial_prompt)

# Initialize the conversation chain with the language model and memory
conversation = ConversationChain(llm=llm, memory=memory)

@app.route('/scope_scout', methods=['POST'])
def requirement_bot():
    if 'file' not in request.files and 'user_input' not in request.form:
        return jsonify({"error": "No file or text part in the request"}), 400

    extracted_data = {}

    if 'file' in request.files:
        file = request.files['file']
        user_input = request.form['user_input']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and file.filename.endswith('.pdf'):
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                file_path = tmp_file.name
                file.save(file_path)

            try:
                # Extract data from the PDF
                extracted_data = extract_text_from_pdf(file_path)
                extracted_data_str = str(extracted_data[1])
                # final_output = requirement_res(input = "This is the data for the project, check if you require something \n",extracted_data)
                # print(final_output)
                # Invoke the conversation chain with the user input
                final_output = conversation.predict(input = user_input + extracted_data_str)
                # Print the response
                print(final_output)
            finally:
                # Remove the temporary file
                os.remove(file_path)
            return jsonify({'final_output': final_output,
                        'user_input' : user_input})
        
        # if file and file.filename.endswith('.doc'):
        #     # Create a temporary file
        #     with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp_file:
        #         file_path = tmp_file.name
        #         file.save(file_path)

        #     try:
        #         # Extract data from the PDF
        #         extracted_data = extract_text_from_doc(file_path)
        #         extracted_data_str = str(extracted_data[1])
        #         # final_output = requirement_res(input = "This is the data for the project, check if you require something \n",extracted_data)
        #         # print(final_output)
        #         # Invoke the conversation chain with the user input
        #         final_output = conversation.predict(input = user_input + extracted_data_str)
        #         # Print the response
        #         print(final_output)
        #     finally:
        #         # Remove the temporary file
        #         os.remove(file_path)
        #     return jsonify({'final_output': final_output,
        #                 'user_input' : user_input})
        
        if file and file.filename.endswith('.docx'):
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                file_path = tmp_file.name
                file.save(file_path)

            try:
                # Extract data from the PDF
                extracted_data = extract_text_from_docx(file_path)
                extracted_data_str = str(extracted_data[1])
                # final_output = requirement_res(input = "This is the data for the project, check if you require something \n",extracted_data)
                # print(final_output)
                # Invoke the conversation chain with the user input
                final_output = conversation.predict(input = user_input + extracted_data_str)
                # Print the response
                print(final_output)
            finally:
                # Remove the temporary file
                os.remove(file_path)
            return jsonify({'final_output': final_output,
                        'user_input' : user_input})
        
        else:
            return jsonify({"error": "Invalid file format. Only PDF, DOC, DOCX files are allowed."}), 400

    elif 'user_input' in request.form:
        user_input = request.form['user_input']
        # Here, you can process the provided text as needed
        # extracted_data['text_input'] = text
        final_output = conversation.predict(input = user_input)
        print(final_output)

        return jsonify({'final_output': final_output,
                        'user_input' : user_input})

@app.route('/market_maven', methods=['POST'])
def market_maken():
    data = request.json
    query = data.get('user_input')
    # model= data.get('model')
    print("query:",query)
    model='gpt-3.5-turbo'
    main_articles=''
    main_response=''
    if query=='':   
        return jsonify({'message': 'Please enter a valid query'})
    else:
        topics=nlp(query)
        print(topics)  
        print(type(topics))
        if type(topics) ==  list:
            for topic in topics: 
                if topic=='NA':
                    return jsonify({'message': 'Please enter a valid query'})
                else:
                    # main_articles=newsapi2(query,model)
                    # main_articles=newsapi1(query,model)
                    main_articles+=new_one(topic,model) 
        print(len(main_articles))
        length=len(main_articles)
        for i in range(0,length,15000):
            main_articles=main_articles[i:i+15000]
            #  print(analytics(topics,main_articles))
            main_response+=analytics(query,main_articles)
            print(len(main_articles))
        main_response=analytics(query,main_articles)
        if len(main_response)==0:
            return jsonify({'message': 'No articles found for the query'})
        else:
            if model=='gpt-3.5-turbo':
                response={"user_input":query,"result":main_response}
                return response
            elif model=="gemini-pro":
                response={"user_input":query,"result":main_response}
                return response
            else:
                return jsonify("Invalid model")

@app.route('/db_string_validator', methods=['POST'])
def validate_dbstring():
    data = request.json 
    db_string = data.get('user_input') 
    if db_string.startswith('postgresql+psycopg2'):
        if validate_connection_string(db_string):
            return jsonify({"message": 'Connection successful'})
        else:
            return jsonify({"message": 'Connection failure'})
    elif db_string.startswith('mysql+pymysql'):
        if validate_connection_string(db_string):
            return jsonify({"message": 'Connection successful'})
        else:
            return jsonify({"message": 'Connection failure'})
    elif db_string.startswith('mongodb+srv'):
        print(connection_string)
        is_valid,db_list = validate_mongo_connection_string(db_string)
        print(db_list)
        if is_valid:
            return jsonify({"message": 'Connection successful'})
        else:
            return jsonify({"message": 'Connection failure'})
    else:
        return jsonify({"message": 'Connection failure'})
if __name__ == '__main__':
    app.run(host='0.0.0.0',port = '2020',debug=True)




