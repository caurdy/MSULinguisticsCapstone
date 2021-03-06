import base64
import datetime
from scipy.io.wavfile import write
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, callback_context
from starter import app
import time
import json
from DataScience.TimeAlignment import ASRTimeAligner
from DataScience.TimeAlignment import Wav2Vec2ASR
import tensorflow
import torch
import sounddevice as sd

var = tuple()
english = "facebook/wav2vec2-large-960h-lv60-self"
jacob = "caurdy/wav2vec2-large-960h-lv60-self_MIDIARIES_72H_FT"
# current = english
current = jacob
audio_json = ""

timeAligner = ASRTimeAligner(asrModel=current,
                             diarizationModelPath="PyannoteProj/data_preparation/saved_model/model_03_25_2022_10_38_52",
                             useCuda=True)
user_audio = ""


layout = html.Div([
    html.H3("Upload Audio (.wav) Files for Transcription or Record a 10s Audio", style={'margin-left': '300px'}),
    html.Div([
    html.Button("Record Your Own Audio", id='record', n_clicks=0, style={'display':'inline-block'}),
    html.Button('Transcribe Audio', id='self-audio', n_clicks=0, hidden=True, style={'display':'inline-block'}),
    ], style={'margin-left': '300px'}),
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
    #dcc.Interval(id='interval', interval=1 * 1000, n_intervals=0),
    html.Div(id='counter', children=[]),
    html.Hr(),
    dcc.Loading(id="ls-loading", children=[html.Div(id="ls-loading-output")], type="default"),
    html.Div(),
    html.Div(id='output-div', children=[], className='output-loading'),
    html.Div(id='user-output', children=[]),
    html.Div(id='output-data-upload', children=[]),
    html.Div(id='output-data-punc', children=[])
])

# @app.callback(Output('ls-loading', 'children'),
#     [Input('ls-loading', 'value')])
# def update_interval(n):
#     return

@app.callback(
    Output('stored-data', 'data'),
    Output('click_save', 'data'),
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    Input('upload', 'n_clicks'),
    Input('self-audio', 'n_clicks'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    State('stored-data', 'data'),
    State('click_save', 'data'),
    State('language', 'data'),
    prevent_initial_callback=True)
def display_output(list_of_contents, n_clicks, audio_clicks, list_of_names, list_of_dates, data, click, language):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    global current
    # if language == "english":
    #     current = english
    # elif language == "jacob":
    #     current = jacob
    # timeAligner.asrModel = current
    if language != "" and language != "english" and current != jacob:
        current = jacob
        timeAligner.asrModel.loadModel(jacob)
    if language != "" and language != "jacob" and current != english:
        current = english
        timeAligner.asrModel.loadModel(english)

    if 'self-audio' in changed_id and audio_clicks > 0:
        audio = transcribeUser(data)
        layout = audio[1]
        transcript = audio[0]
        click = click or {'clicks': 0}
        click['clicks'] = click['clicks'] + 1
        data.append(transcript)
        global audio_json
        audio_json = transcript[1]
        return data, click, layout

    elif list_of_contents is not None:
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
                global audio_json
                audio_json = archive[1]

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
                        # editable=True,
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
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if clicks > 0 and 'thepunctuator' in changed_id:
        global audio_json
        file = audio_json
        transcripts, punc_time, ner_time = timeAligner.getEntitiesLastTranscript()
        with open(f'{file}') as jsonFile:
            transcripts = json.load(jsonFile)

        return html.Div([
            html.Div('Named Entity Recognition Time: ' + str(round(ner_time, 3)) + " seconds"),
            html.Div("Punctuation Restoration Time: " + str(round(punc_time, 3)) + " seconds"),
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

@app.callback(Output('output-div', 'children'),
              Output('self-audio', 'hidden'),
              Output('record-clicks', 'data'),
                Output('ls-loading', 'children'),
                [Input('ls-loading', 'value')],
              Input('record', 'n_clicks'),
              State('record-clicks', 'data'),)
def recordAudio(value, clicks, recorded_clicks):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'record' in changed_id and clicks > 0:
        recorded_clicks = recorded_clicks or {'clicks': 0}
        recorded_clicks['clicks'] = recorded_clicks['clicks'] + 1
        # filename = f'assets/mywavfile{clicks}.wav'
        fs = 44100
        seconds = 10
        with open(f'assets/mywavfile{clicks}.wav', 'wb') as fp:
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
            sd.wait()  # Wait until recording is finished
            write(f'assets/mywavfile{clicks}.wav', fs, myrecording)  # Save as WAV file
        global user_audio
        user_audio = f'mywavfile{clicks}.wav'
        return html.Div([html.H5("VOICE AUDIO RECORDED"),
                         ],
                        style={'margin-left':'300px'}), False, recorded_clicks, []
    else:
        return [], True, recorded_clicks, []

# @app.callback(Output('user-output', 'children'),
#               Input('self-audio', 'n_clicks'))
def transcribeUser(store):
    filename = user_audio
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
            # store.append(archive)
            # click = click or {'clicks': 0}
            # click['clicks'] = click['clicks'] + 1
            # n_clicks = click['clicks']

            return archive, html.Div([
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
                    style_cell={'textAlign': 'left', 'width':'auto'},
                    style_table={'textAlign': 'center', 'width': '1050px'},
                    # editable=True,
                ),

                html.Hr(),  # horizontal line
                html.Button('Restore Punctutation & Generate NER', id='thepunctuator', n_clicks=0),
            ],
                style={'margin-left': '300px'}
            )

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ]), ""