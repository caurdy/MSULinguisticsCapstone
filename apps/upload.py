import base64
import datetime
import io
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, callback_context
import pandas as pd
from starter import app
from Combine.CombineFeatures import combineFeatures
import time

var = tuple()

layout = html.Div([

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.Button(id="upload", children=[
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
                        }, )
        ]),

        multiple=False,
        style={'margin-left': '300px'}
    ),
    html.Div(id='output-div', children=[]),
    html.Div(id='output-data-upload', children=[]),

])


@app.callback(
    Output('stored-data', 'data'),
    Output('click_save', 'data'),
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    Input('upload', 'n_clicks'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    State('stored-data', 'data'),
    State('click_save', 'data'),
    prevent_initial_callback=True)
def display_output(list_of_contents, n_clicks, list_of_names, list_of_dates, data, click):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    if list_of_contents is not None:
        click = click or {'clicks': 0}

        click['clicks'] = click['clicks'] + 1
        n_clicks = click['clicks']
        children = [
            parse_contents(list_of_contents, list_of_names, list_of_dates, n_clicks, data)
        ]
        return data, click, children[0][1]
    else:
        return data, click, []


def parse_contents(contents, filename, date, cnt, store):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        with open(f'assets/{filename}', 'wb') as fp:
            fp.write(decoded)

        try:

            if '.csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            elif '.xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
            elif '.wav' in filename:
                transcript_filepath = f'assets/' + filename.replace('.wav', '.json')
                transcripts, diarization_time, transcript_time, avg_conf = combineFeatures(f'assets/{filename}',
                                                                                           transcript_filepath)
                # global var
                archive = tuple((filename,
                                 transcript_filepath,
                                 datetime.datetime.now().strftime('%m/%d/%Y'),
                                 time.time(),
                                 len(store)))
                store.append(archive)

        except Exception as e:
            print(e)
            return store, html.Div([
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
                html.Div("Average ASR Confidence: " + str(round(avg_conf*100, 2)) + '%'),
                html.Div(style={'padding': '2rem'}),
                dash_table.DataTable(
                    #df.to_dict('records'),
                    #[{'name': i, 'id': i} for i in df.columns],
                    transcripts,
                    [{'name': i, 'id': i} for i in transcripts[0].keys()],
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
                    style_table={'textAlign': 'center', 'width': '1050px'},
                ),

                html.Hr(),  # horizontal line
            ],
                style={'margin-left': '300px'}
            )
