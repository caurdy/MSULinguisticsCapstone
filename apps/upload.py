import base64
import datetime
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, callback_context
from starter import app
import time
import json
from DataScience.TimeAlignment import ASRTimeAligner

var = tuple()

timeAligner = ASRTimeAligner(diarizationModelPath="PyannoteProj/data_preparation/saved_model/model_03_25_2022_10_38_52")

layout = html.Div([
    html.H3("Upload Audio (.wav) Files for Transcription", style={'margin-left': '300px'}),
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
    html.Div(id='output-div', children=[], className='output-loading'),
    html.Div(id='output-data-upload', children=[]),
    html.Div(id='output-data-punc', children=[])
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
            if '.wav' in filename:
                transcript_filepath = f'assets/' + filename.replace('.wav', '.json')
                transcripts, diarization_time, transcript_time, avg_conf = timeAligner.timeAlign(f'assets/{filename}',
                                                                                                 'assets/')
                # global var
                archive = tuple((filename,
                                 transcript_filepath,
                                 datetime.datetime.now().strftime('%m/%d/%Y'),
                                 time.time(),
                                 f'Transcription Time: {str(transcript_time)} seconds',
                                 f"Diarization Time: {str(diarization_time)} seconds",
                                 f"Average ASR Confidence: {str(round(avg_conf * 100, 2))}%",
                                 len(store)))
                store.append(archive)

                return store, html.Div([
                    html.H5(filename),
                    html.Button(html.Audio(id="audio", src=f'assets/{filename}', controls=True, autoPlay=False)),
                    html.Div('Transcription Time: ' + str(transcript_time) + " seconds"),
                    html.Div("Diarization Time: " + str(diarization_time) + " seconds"),
                    html.Div("Average ASR Confidence: " + str(round(avg_conf * 100, 2)) + '%'),
                    html.Div(style={'padding': '2rem'}),
                    dash_table.DataTable(
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
                        editable=True,
                    ),

                    html.Hr(),  # horizontal line
                    html.Button('Restore Punctutation & Generate NER', id='thepunctuator', n_clicks=0),
                ],
                    style={'margin-left': '300px'}
                )

        except Exception as e:
            print(e)
            return store, html.Div([
                'There was an error processing this file.'
            ])


@app.callback(
    Output('output-data-punc', 'children'),
    Input('thepunctuator', 'n_clicks'),
    prevent_initial_callback=True
)
def thePunctuator(clicks):
    if clicks > 0:

        #transcripts, punc_time, ner_time = timeAligner.getEntitiesLastTranscript()
        with open('../assets/AbbottCostelloWhosonFirst.json') as jsonFile:
            transcripts = json.load(jsonFile)

        return html.Div([
            #html.Div('Named Entity Recognition Time: ' + str(round(ner_time, 3)) + " seconds"),
            #html.Div("Punctuation Restoration Time: " + str(round(punc_time, 3)) + " seconds"),
            dash_table.DataTable(
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
            ), ], style={'margin-left': '300px'})
