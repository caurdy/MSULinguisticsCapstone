import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash import dcc, html, dash_table, callback_context
import os
import pandas as pd
from dash.exceptions import PreventUpdate

from starter import app
from Combine.CombineFeatures import combineFeatures

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

var = tuple()

layout = html.Div([

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.Button(id="UpButt", children=[
                'Drag and Drop or ',
                html.A('Select File'), ], n_clicks=0,
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
                        }, )

        ]),

        # Allow multiple files to be uploaded
        multiple=False
    ),
    # ), n_clicks=0,
    #             style={
    #                 'width': '100%',
    #                 'height': '60px',
    #                 'lineHeight': '60px',
    #                 'borderWidth': '1px',
    #                 'borderStyle': 'dashed',
    #                 'borderRadius': '5px',
    #                 'textAlign': 'center',
    #                 'margin': '10px',
    #                 'margin-left': '300px',
    #             },
    #             ),
    html.Div(id='output-div', children=[]),
    html.Div(id='output-data-upload', children=[]),
    # html.Button("Save Transcript", id='saver', children=var, n_clicks=0),
])


@app.callback(
    Output('stored-data', 'data'),
    Output('click_save', 'data'),
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    Input('UpButt', 'n_clicks'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    State('stored-data', 'data'),
    State('click_save', 'data'),
    prevent_initial_callback=True)
def display_output(list_of_contents, n_clicks, list_of_names, list_of_dates, data, click):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    # print("ID: ", changed_id)

    if list_of_contents is not None:
        click = click or {'clicks': 0}

        click['clicks'] = click['clicks'] + 1
        n_clicks = click['clicks']
        children = [

            parse_contents(list_of_contents, list_of_names, list_of_dates, n_clicks, data)
        ]
        # parse_contents(c, n, d, count) for c, n, d in
        # zip(list_of_contents, list_of_names, list_of_dates)]
        return data, click, children[0][1]
    else:
        return data, click, []



def parse_contents(contents, filename, date, cnt, store):
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

                diarization_time, transcript_time, avg_conf = combineFeatures(f'assets/{filename}',
                                                                              f'assets/transcript_{cnt}')
                df = pd.read_csv(f'assets/transcript_{cnt}.csv')
                # global var
                archive = tuple((filename, f'transcript_{cnt}.csv', datetime.datetime.now().strftime('%m/%d/%Y')))
                store.append(archive)
                # df = pd.read_csv("assets/random.csv")
                # print("COUNT: ", cnt)
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
            ], id='transcript')
        else:
            return store, html.Div([
                html.H5(filename),
                html.Button(html.Audio(id="audio", src=f'assets/{filename}', controls=True, autoPlay=False)),
                html.Div('Transcription Time: ' + str(transcript_time) + " seconds"),
                html.Div("Diarization Time: " + str(diarization_time) + " seconds"),
                html.Div("Average ASR Confidence: " + str(round(avg_conf, 3)) + '%'),
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
#              Input('saver', 'n_clicks'),
#               prevent_inital_callback = True)
# def setupvar(clicks):
#     return html.Div(var)


# def store_data(_):
#     if _ is not None:
#         return [var]
#     else:
#         PreventUpdate

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
