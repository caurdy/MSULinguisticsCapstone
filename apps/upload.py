import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
import os

import pandas as pd
from starter import app
from Combine.CombineFeatures import combineFeatures

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files'),

        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'margin-left': '300px',
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-div', children=[]),
    html.Div(id='output-data-upload', children=[]),
])

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))

def display_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

def parse_contents(contents, filename, date):
    if contents is not None:
        content_type, content_string = contents.split(',')
        # change this to a .wav use pipeline to get. test
        decoded = base64.b64decode(content_string)
        try:

            if '.csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            elif '.xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
            elif '.wav' in filename:
                # dcc.send_file(
                #     "assets/audio"),
                # Attempt to make the file playable

                combineFeatures(f'assets/{filename}', "assets/transcript")
                df = pd.read_csv("assets/transcript.csv")
                # df = pd.read_csv("assets/random.csv")
                # os.remove("assets/transcript.csv")
                # html.Audio(id="audio", src='assets/0a15bb21-3993-4496-8672-f3be45769356.wav', controls=True, autoPlay=False)
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        if '.csv' in filename or '.xls' in filename:
            return html.Div([
                html.H5(filename),
                html.H6(datetime.datetime.fromtimestamp(date)),

                dash_table.DataTable(
                    df.to_dict('records'),
                    [{'name': i, 'id': i} for i in df.columns]
                ),

                html.Hr(),  # horizontal line

                # For debugging, display the raw contents provided by the web browser
                html.Div('Raw Content'),
                html.Pre(contents[0:200] + '...', style={
                    'whiteSpace': 'pre-wrap',
                    'wordBreak': 'break-all'
                })
            ], id = 'transcript')
        else:
            return html.Div([
                html.H5(filename),
                html.Button(html.Audio(id="audio", src=f'assets/{filename}', controls=True, autoPlay=False)),
                html.Div(style={'padding': '2rem'}),
                dash_table.DataTable(

                    df.to_dict('records'),
                    [{'name': i, 'id': i} for i in df.columns],
                    css=[{
                        'selector': '.dash-spreadsheet td div',
                        'rule': '''
                        line-height: 15px;
                        max-height: 30px; min-height: 30px; height: 30px;
                        display: block;
                        overflow-y: hidden;
                    '''
                    }],
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'lineHeight': '15px'
                    },
                    style_cell={'textAlign': 'left'},
                    # style_data={'whiteSpace': 'normal',
                    #             'minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
                        # 'max-height': '30px', 'min-height': '30px', 'height': '30px',
                        # 'lineHeight': '15px',
                        #         },
                    style_table={'textAlign': 'center', 'width': '1050px'},
                ),

                html.Hr(),  # horizontal line

                # For debugging, display the raw contents provided by the web browser
                # html.Div('Raw Content'),
                # html.Pre(contents[0:200] + '...', style={
                #     'whiteSpace': 'pre-wrap',
                #     'wordBreak': 'break-all',
                # })
            ],
            style={'margin-left': '300px'}
            )

# @app.callback(Output('stored-data', 'data'),
#               Input('transcript', 'transcript'))
#
# def put_in_data(transcript):
#     print(transcript)

# @app.callback(Output('output-div', 'children'),
#               Input('submit-button','n_clicks'),
#               State('stored-data','data'),
#               State('xaxis-data','value'),
#               State('yaxis-data', 'value'))
# def make_graphs(n, data, x_data, y_data):
#     if n is None:
#         return dash.no_update
#     else:
#         bar_fig = px.bar(data, x=x_data, y=y_data)
#         # print(data)
#         return dcc.Graph(figure=bar_fig)
