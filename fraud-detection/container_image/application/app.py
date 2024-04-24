import json
import logging
import os
import time
from datetime import datetime
from typing import Dict

import dash
import dash_bootstrap_components as dbc
import dill
import numpy as np
import pandas as pd
import requests
from dash import Input, Output, State, dash_table, dcc, html
from flask import Flask, jsonify, request
from jproperties import Properties
from sklearn_pandas import DataFrameMapper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ------- Model Params -------

MODEL_NAME = "fraud-detection-fd6e7"
NAMESPACE = "user-example-com"
HOST = f"{MODEL_NAME}-predictor-default.{NAMESPACE}"
HEADERS = {"Host": HOST}
MODEL_ENDPOINT = f"http://{MODEL_NAME}-predictor-default/v2/models/model"
PREDICT_ENDPOINT = MODEL_ENDPOINT + "/infer"

# instantiate config
configs = Properties()
# load properties into configs
with open("app-config.properties", "rb") as config_file:
    configs.load(config_file)
# read into dictionary
configs_dict = {}
items_view = configs.items()
for item in items_view:
    configs_dict[item[0]] = item[1].data

merchants = {
    0: {"hash": 0, "name": "Sandwiches & Books Outlet"},
    1: {"hash": 0, "name": "Treasures & Gadgets Shop"},
    2: {"hash": 0, "name": "Lucky Treasures Mart"},
    3: {"hash": 0, "name": "Gas Station"},
    4: {"hash": 0, "name": "The Cool Store"}
}

def load_transaction_data(data):
    with open(data, 'r') as f:
        raw_data = json.load(f)
    
    indexed_data = {}
    for record in raw_data:
        index = record['index']
        merchants[index]['hash'] = record['merchant name']
        
        del record['index']
        indexed_data[index] = record
    
    return indexed_data

def transform_transaction_data(raw_data: Dict):
    transformed = {}
    
    for key, record in data.items():
        logging.info(key, record)
        
        transformed[key] = {
            'ID': key,
            'Amount': record['amount'],
            'Place': merchants[key]['name'],
            'Date': f"{int(record['year'])}-{int(record['month'])}-{int(record['day'])}",
            'Time': record['time'],
            'Fraud Status': "Unchecked"
            
        }

    return transformed


# Sample transactions data
data = load_transaction_data('transactions.json')
# Convert to DataFrame
transactions_df = pd.DataFrame.from_dict(data, orient='index')
# Assuming transactions_df is already defined
transactions_df['Fraud Status'] = 'Unchecked'  # Initialize all transactions as 'Unchecked'
# Assuming transactions_df is already defined
if 'Tested' not in transactions_df.columns:
    transactions_df['Tested'] = False  # Initialize all rows as not tested


transformed_data = transform_transaction_data(data)

server = Flask(__name__)
app = dash.Dash(
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css?family=IBM+Plex+Sans:400,600&display=swap",
    ],
    suppress_callback_exceptions=True,
    server=server
)
app.title = configs_dict["tabtitle"]

navbar_main = dbc.Navbar(
    [
        dbc.Col(
            configs_dict["navbartitle"],
            style={"fontSize": "0.875rem", "fontWeight": "600"},
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem(
                    "View logs",
                    id="view-logs-btn",
                    n_clicks=0,
                    class_name="dmi-class",
                ),
            ],
            toggle_class_name="nav-dropdown-btn",
            caret=False,
            nav=True,
            in_navbar=True,
            label=html.Img(
                src="/assets/settings.svg",
                height="16px",
                width="16px",
                style={"filter": "invert(1)"},
            ),
            align_end=True,
        ),
    ],
    style={
        "paddingLeft": "1rem",
        "height": "3rem",
        "borderBottom": "1px solid #393939",
        "color": "#fff",
    },
    class_name="bg-dark",
)

payload_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("My Payloads")),
        dbc.ModalBody([dbc.Tabs(id="payload-modal-tb", active_tab="payload-tab-0")]),
    ],
    id="payload-modal",
    size="xl",
    scrollable=True,
    is_open=False,
)


transaction_data = dash_table.DataTable(
    id="transactions-table",
    columns=[
        {"name": "ID", "id": "ID"},
        {"name": "Amount", "id": "Amount"},
        {"name": "Place", "id": "Place"},
        {"name": "Date", "id": "Date"},
        {"name": "Time", "id": "Time"}
        # Potentially hide 'Fraud Status' from the view or include it based on your preference
    ],
    data=pd.DataFrame.from_dict(transformed_data, orient='index').drop(columns=["Fraud Status"]).to_dict("records"),
    row_selectable="single",
    selected_rows=[],
    style_data_conditional=[
        {
            "if": {"row_index": "odd"},
            "backgroundColor": "rgb(220, 220, 220)",
        },
        {
            "if": {
                "state": "selected"
            },  # Applies to selected rows; adjust as needed for highlighting
            "backgroundColor": "rgba(0, 116, 217, 0.3)",
            "border": "1px solid blue",
        },
        {
            "if": {
                "filter_query": '{Fraud Status} = "Detected"',
                "column_id": "ID",
            },
            "backgroundColor": "rgba(255, 0, 0, 0.7)",
            "color": "white",
        },
    ],
    page_size=10,
)

generate_button = dbc.Button(
    configs_dict["generate_btn_text"],
    id="generate-button",
    outline=True,
    color="primary",
    n_clicks=0,
    className="carbon-btn",
)

clear_button = dbc.Button(
    "Clear Transactions", 
    id="clear-transactions-btn", 
    outline=True,
    n_clicks=0,
    className="carbon-btn",
    color="warning"
)

export_button = dbc.Button(
    "Export Transactions", 
    id="export-transactions-btn",
    outline=True,
    n_clicks=0,
    className="carbon-btn",
    color="success"
)

buttonsPanel = (
    dbc.Row(
        [
            # dbc.Col(upload_button),
            dbc.Col(generate_button)
        ]
    )
    if configs_dict["show_upload"] in ["true", "True"]
    else dbc.Row(
        [
            dbc.Col(generate_button, className="text-center"),
        ]
    )
)

footer = html.Footer(
    dbc.Row([dbc.Col(configs_dict["footer_text"], className="p-3")]),
    style={
        "paddingLeft": "1rem",
        "paddingRight": "5rem",
        "color": "#c6c6c6",
        "lineHeight": "22px",
    },
    className="bg-dark position-fixed bottom-0",
)


# Construct the button panel
output_buttons_panel = dbc.Row(
    [
        dbc.Col(clear_button, width={"size": 6, "offset": 0}, className="text-center"),
        dbc.Col(export_button, width={"size": 6, "offset": 0}, className="text-center"),
    ],
    justify="around",  # This will space out the buttons evenly
)


vertical_layout = dbc.Row(
    [
        dbc.Col(className="col-2"),
        dbc.Col(
            children=[
                html.H5(configs_dict["Input_title"]),
                html.Div(transaction_data),
                html.Br(),
                buttonsPanel,
                html.Br(),
                html.Hr(),
                html.Div(
                    [
                        html.H5(configs_dict['output_title']),
                        html.Div(
                        [
                            dbc.Label("Model ID"),
                            dcc.Dropdown(
                                id="model_id",
                                options=[
                                    "fraud-detection-fd6e7"
                                ],
                                value="payment method",
                            ),
                        ]
                    ),
                        # Insert the output_buttons_panel here to have it at the top of the generate-output area
                        html.Div(id="generate-output"),  # This div holds the generated transaction output
                    ],
                    style={"padding": "1rem 1rem"},
                ),
            ],
            className="col-8",
        ),
        dbc.Col(className="col-2"),
    ],
    className="px-3 pb-5",
)
download_component = dcc.Download(id='download-transaction-data')

vertical_layout.children[1].children.append(download_component)  # Assuming vertical_layout is your main layout structure


horizontal_layout = dbc.Row(
    [
        dbc.Col(className="col-1"),  # Margin on the left
        dbc.Col(
            children=[
                html.H5("🗃️Transactions"),
                html.Div(
                    transaction_data,
                    style={
                        "overflowY": "auto",
                        "height": "400px",
                    },  # Adjust the height as needed
                ),
                html.Br(),
                buttonsPanel,  # Input buttons panel
                html.Br(),
            ],
            className="col-5 border-end",
            style={"padding": "1rem"},
        ),
        dbc.Col(
            children=[
                html.Div(
                    [
                        html.H5("📋Fraud Detection Reports"),  # Output title
                        html.Div(
                            [
                                dbc.Label("Model ID"),
                                dcc.Dropdown(
                                    id="model_id",
                                    options=[
                                        {
                                            "label": "fraud-detection-fd6e7",
                                            "value": "fraud-detection-fd6e7",
                                        }
                                    ],
                                    value="fraud-detection-fd6e7",
                                ),
                            ],
                            style={
                                "padding": "0 0 1rem 0"
                            },  # Added padding for spacing between elements
                        ),
                        # Assuming output_buttons_panel is a component to be included here
                        output_buttons_panel,  # Output buttons panel
                        dcc.Loading(
                            id="loading-1",
                            type="default",  # Spinner type
                            children=html.Div(id="generate-output"),  # This div holds the generated transaction output
                            color="#119DFF",  # Optional: Spinner color
                        ),
                    ],
                    style={"padding": "1rem 3rem"},
                ),
            ],
            className="col-5",
        ),
        dbc.Col(className="col-1"),  # Margin on the right
    ],
    className="px-3 pb-5",
)


app.layout = html.Div(
    children=[
        navbar_main,
        html.Div(payload_modal),
        html.Br(),
        html.Br(),
        horizontal_layout,
        html.Div(id='fraud-report-status', style={'display': 'none'}),
        html.Br(),
        html.Br(),
        download_component,
        footer,
    ],
    className="bg-white",
    style={"fontFamily": "'IBM Plex Sans', sans-serif"},
)


# ------------------------------ end UI Code ------------------------------

class FraudDatasetTransformer:
    def __init__(self): ...

    def transform(self, dataset: pd.DataFrame, mapper: DataFrameMapper):
        """
        
        dropped columns:
            - mcc
            - zip
            - merchant state

        Args:
            dataset (pd.DataFrame): _description_
            mapper (DataFrameMapper): _description_

        Returns:
            _type_: _description_
        """
        tdf = dataset.copy()
        tdf["merchant name"] = tdf["merchant name"].astype(str)
        tdf.drop(["mcc", "zip", "merchant state"], axis=1, inplace=True)
        tdf.sort_values(by=["user", "card"], inplace=True)
        tdf.reset_index(inplace=True, drop=True)

        tdf = mapper.transform(tdf)
        return tdf


def get_df_mapper():
    with open(os.path.join("encoders", "data", "mapper.pkl"), "rb") as f:
        t_mapper = dill.load(f)
        return t_mapper


def predict(vdf: pd.DataFrame) -> pd.DataFrame:
    
    res_svc = requests.get(MODEL_ENDPOINT, headers=HEADERS)
    response_svc = json.loads(res_svc.text)

    # Data preparation
    x = vdf.drop(vdf.columns.values[0], axis=1).to_numpy()
    y = np.array([vdf[vdf.columns.values[0]].iloc[0]])

    # Adjust the shape of x to match model expectations
    # We need to expand or repeat our single data point to match the sequence length and feature count
    # Assuming your single row is a flat array of features, reshape and repeat it
    sequence_length = response_svc['inputs'][0]['shape'][1]  # Model's expected sequence length
    num_features = response_svc['inputs'][0]['shape'][2]   # Model's expected number of features per sequence element

    # Check if the original features match the required total features
    original_features = x.shape[1]
    logging.info(original_features)
    if original_features < num_features:
        logging.info("pad maybe?")
        # If fewer, we may need to pad or adjust the data; this is situational and may not be exactly correct without more context
        # For now, let's assume padding with zeros is acceptable
        x_padded = np.pad(
            x,
            ((0, 0), (0, num_features - original_features)),
            mode="constant",
            constant_values=0,
        )
    else:
        logging.info("reshape accordingly")
        # If it matches or exceeds, truncate or reshape accordingly (though unusual for a single data point)
        x_padded = x[:, :num_features]

    # Reshape to [1, sequence_length, num_features], replicating the single data point across the new sequence length
    x_reshaped = np.tile(x_padded, (sequence_length, 1)).reshape(
        1, sequence_length, num_features
    )

    # Preparing the payload
    payload = {
        "inputs": [
            {
                "name": "input_1",
                "shape": [1, sequence_length, num_features],
                "datatype": "FP32",
                "data": x_reshaped.tolist(),
            }
        ]
    }

    # Sending the request
    res = requests.post(PREDICT_ENDPOINT, headers=HEADERS, data=json.dumps(payload))
    response = json.loads(res.text)

    # Handle response
    if "error" in response:
        logging.info(f"Error: {response['error']}")
    else:
        logging.info(response["outputs"])
        pred = response["outputs"][0]["data"][0]
        logging.info(f"Actual ({y[0]}) vs. Prediction ({round(pred, 3)} => {int(round(pred, 0))})")
    
    return response

def do_predict(test_data: Dict):
    start = time.time()
    dataset_transfomer = FraudDatasetTransformer()
    test = pd.DataFrame([test_data])
    vdf = dataset_transfomer.transform(test, get_df_mapper())
    result = predict(vdf)
    end = time.time()
    total_time = (end - start) * 1000 # ms
    result['time'] = round(total_time, 3)
    return result

@server.route('/api/model/infer', methods=['POST'])
def predict_endpoint():
    try:
        input_data = request.get_json()
        if not input_data:
            raise ValueError('No input data provided')
        result = do_predict(input_data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@server.route('/api/model/info', methods=['GET'])
def get_model():
    # ------- Model Params -------

    # MODEL_NAME = "fraud-detection-fd6e7"
    # NAMESPACE = "user-example-com"
    # HOST = f"{MODEL_NAME}-predictor-default.{NAMESPACE}"
    # HEADERS = {"Host": HOST}
    # MODEL_ENDPOINT = f"http://{MODEL_NAME}-predictor-default/v2/models/model"
    # PREDICT_ENDPOINT = MODEL_ENDPOINT + "/infer"


    res_svc = requests.get(MODEL_ENDPOINT, headers=HEADERS)
    response_svc = json.loads(res_svc.text)
    logging.info(response_svc)
    return jsonify(response_svc)
    



test_input = {
    "user": 0,
    "card": 0,
    "merchant name": "Stop n Shop",
    "amount": 0,
    "year": 2015,
    "month": 1,
    "day": 1,
    "transaction type": "payment method",
    "merchant city": "Bucyrus",
    "merchant state": "OH",
    "zip": 0,
    "errors": "None",
    "mcc": 0,
    'is fraud?': 'No'
}

prediction_results = {}

@app.callback(
    Output('transactions-table', 'style_data_conditional'),
    Input('fraud-report-status', 'children'),
    State('transactions-table', 'selected_rows'),
    State('transactions-table', 'style_data_conditional'),
    prevent_initial_call=True
)
def update_table_style(fraud_report_status, selected_rows, style):
    if selected_rows:
        selected_row_index = selected_rows[0]
        
        if not transactions_df.iloc[selected_row_index]['Tested']:
                        # Mark the row as tested
            transactions_df.at[selected_row_index, 'Tested'] = True

            # Generating a random fraud confidence interval for demonstration
            logging.info(f"current prediction results: {prediction_results}")
            fraud_confidence = prediction_results.get(selected_row_index, -1)

            # Determine the color based on fraud_confidence
            if fraud_confidence < .20:
                background_color = 'rgba(0, 255, 0, 0.7)'  # Green
            elif .20 <= fraud_confidence <= .50:
                background_color = 'rgba(255, 200, 0, 0.7)'  # Yellow
            else:
                background_color = 'rgba(255, 0, 0, 0.7)'  # Red
                transactions_df.at[selected_row_index, 'Fraud Status'] = 'Detected'

            # Update the style to include the new background color for the selected row
            for condition in style:
                if condition.get('if', {}).get('row_index') == selected_row_index:
                    condition['backgroundColor'] = background_color
                    return style

            style.append({
                'if': {
                    'row_index': selected_row_index,
                },
                'backgroundColor': background_color,
                'color': 'black'
            })
    return style

@app.callback(
    Output('generate-output', 'children'),
    Output('fraud-report-status', 'children'),
    [Input('generate-button', 'n_clicks')],
    [State('generate-output', 'children')] +  # Existing output
    [State('transactions-table', 'selected_rows')],
    prevent_initial_call=True
)
def update_output(n_clicks, existing_output, selected_rows):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Parse existing output
    if not existing_output:
        existing_output = []
    elif isinstance(existing_output, str):
        # In case the existing_output is just a string representation (unlikely, but just in case)
        existing_output = json.loads(existing_output)
        
    selected_row_index = selected_rows[0]
    selected_transaction = data[selected_row_index]
    if transactions_df.iloc[selected_row_index]['Tested']:
        raise dash.exceptions.PreventUpdate 
    
    logging.info(f"row index: {selected_row_index}")
    logging.info(f"selected transaction: {selected_transaction}")
    
    predict_data = -1
    try:
        predict_result = do_predict(selected_transaction)
        logging.info(f"predict results: {predict_result}")
        predict_data = predict_result['outputs'][0]['data'][0]
        logging.info(f"predict score: {predict_data}")
        prediction_results[selected_row_index] = predict_data

    except Exception as e:
        dash.exceptions.PreventUpdate(f"Error occured while running inference: {e}")
        
    
    fraud_icon = "✅" if predict_data < 0.2 else ("⚠️" if .20 <= predict_data <= .50 else "❌")
    
    # Create the new transaction detail as a collapsible element
    new_transaction_detail = html.Details([
        html.Summary(f"📝Additional Transaction Info - ID: {selected_row_index} {fraud_icon}, time⏱️(ms): {predict_result['time']}", style={'cursor': 'pointer'}),
        dash_table.DataTable(
            data=[
                {'Attribute': 'Merchant Name', 'Value': merchants[selected_row_index]['name']},
                {'Attribute': 'Amount', 'Value': selected_transaction['amount']},
                {'Attribute': 'User', 'Value': selected_transaction['user']},
                {'Attribute': 'Card', 'Value': selected_transaction['card']},
                {'Attribute': 'Date', 'Value': f"{int(selected_transaction['year'])}-{int(selected_transaction['month'])}-{int(selected_transaction['day'])}"},
                {'Attribute': 'Transaction Type', 'Value': selected_transaction['use chip']},
                {'Attribute': 'Merchant City', 'Value': selected_transaction['merchant city']},
                {'Attribute': 'Merchant State', 'Value': selected_transaction['merchant state']},
                {'Attribute': 'ZIP', 'Value': selected_transaction['zip']},
                {'Attribute': 'Errors', 'Value': selected_transaction['errors?']},
                {'Attribute': 'Fraud Likelihood %', 'Value': str(round(predict_data * 100, 3))}, # predict_result[0]['data']
                {'Attribute': 'Inference Time (ms)', 'Value': predict_result['time']}
            ],
            columns=[{'name': i, 'id': i} for i in ['Attribute', 'Value']],
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                'whiteSpace': 'normal'
            },
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'left'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Attribute'},
                    'textAlign': 'left'
                },
                {
                    'if': {'column_id': 'Value'},
                    'textAlign': 'right'
                }
            ]
        )
    ], style={'marginTop': '20px'})

    # Append the new transaction to the existing output
    updated_output = existing_output + [new_transaction_detail]
    
        # Set a flag or a timestamp to indicate a new report was generated
    fraud_report_generated_flag = str(datetime.now())
    
    return updated_output, fraud_report_generated_flag

@app.callback(
    Output('generate-output', 'children', allow_duplicate=True),
    [Input('clear-transactions-btn', 'n_clicks')],
    prevent_initial_call=True
)
def clear_transactions(n_clicks):
    # Return an empty list to clear the transactions
    transactions_df["Tested"] = False
        
    return []


import json

from dash.exceptions import PreventUpdate


@app.callback(
    Output('download-transaction-data', 'data'),
    [Input('export-transactions-btn', 'n_clicks')],
    [State('generate-output', 'children')],
    prevent_initial_call=True
)
def export_transactions(n_clicks, content):
    if not content:
        raise PreventUpdate  # If there's no content, do nothing
    
    all_transactions = []  # Initialize a list to hold all the transaction data

    # Loop through each 'Details' component in the content
    for details in content:
        if ('props' in details and 
            'children' in details['props'] and 
            isinstance(details['props']['children'], list)):
            for child in details['props']['children']:
                # Check if this child is a DataTable
                if (child['type'] == 'DataTable' and
                    'props' in child and
                    'data' in child['props']):
                    # Extract the data from this DataTable and append to all_transactions
                    all_transactions.extend(child['props']['data'])

    # Convert all transaction data to JSON format
    transactions_json = json.dumps(all_transactions, indent=4)

    # Return the JSON string for download
    return dcc.send_string(transactions_json, filename="transactions.json")


if __name__ == "__main__":
    SERVICE_PORT = os.getenv("SERVICE_PORT", default="8050")
    DEBUG_MODE = eval(os.getenv("DEBUG_MODE", default="True"))
    app.run(
        host="0.0.0.0", port=SERVICE_PORT, debug=DEBUG_MODE, dev_tools_hot_reload=False
    )
