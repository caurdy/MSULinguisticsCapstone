# Add Model Retraining Page Here
# GET BUTTONS
import base64
import datetime
import time
import webbrowser
from queue import Queue

from dash.dependencies import Output, Input, State
from dash import dcc, html, callback_context
import dash_bootstrap_components as dbc
import os
import json
import dash_uploader as du
from dash.exceptions import PreventUpdate

from DataScience.SpeechToTextHF import Wav2Vec2ASR
from assets.TestPipline import SpeakerDiaImplement
# from assets.database_loader import CreateDatabase
# from DataScience.DashProgress import DashProgress
# from PyannoteProj.data_preparation.saved_model import model_0, model_1
# from DataScience.SpeechToTextHF import

from starter import app

du.configure_upload(app, 'assets/TrainingData/', use_upload_id=False)

progress_queue = Queue(1)
progress_memory = 0
base_model_string = "patrickvonplaten/wav2vec2-base-100h-with-lm"
data_file = "~/Documents/Beta/Data/correctedShort.json"
output_dir = "~/Documents/Beta/Data/asr_models/"
num_epochs = 3
new_diary_model = ""

english = "facebook/wav2vec2-large-960h-lv60-self"
# jacob = "patrickvonplaten/wav2vec2-large-xlsr-53-jacob-with-lm"
jacob = "caurdy/wav2vec2-large-960h-lv60-self_MIDIARIES_72H_FT"
# current_model_asr = "English, trained on 01/21/22"
current_model_asr = ""
current_model_diarization = "Diarization iteration 02, trained on 01/21/22"


# du.configure_upload(app, )
def get_upload(uid):
    return du.upload(id=uid)


layout = html.Div([
    html.H1("Chose which aspect to retrain"),
    # HINT

    html.Hr(),

    dcc.Dropdown(['Speech to Text', 'Diarization'], id='asr-dropdown',
                 style={'display': 'inline-block', 'width': '90%'}),
    html.Button(html.Img(id='img', src='assets/qmark.png', style={'display': 'inline-block',
                                                                  'width': '30px', 'height': '20%',
                                                                  }),
                id='hints', n_clicks=0),

    html.Div(id='hintOutput', children=[]),

    html.Hr(),

    html.Div(id='asr-output', children=[]),

    html.Hr(),

    html.Div(id='model_set', children=[html.Div(f"Current ASR Model: {current_model_asr}", id="asr-button-output"),
                                       ]),

    html.Hr(),

    html.Div(id="col_asr", n_clicks=0, title="Speech Recognition Models",
             children=["Speech Recognition", html.Div(html.Button("Base_Model_untrained_01_21_22", id='model1asr',
                                                                  n_clicks=0)),
                       html.Div(html.Button("Trained_Model_04_01_22", id='model2asr', n_clicks=0))],
             style={'margin': '20px'}),

],
    style={'margin-left': '300px'})


@app.callback(Output('hintOutput', 'children'),
              Input('hints', 'n_clicks'))
def display_help(clicks):
    if clicks % 2 == 1:
        # return html.Embed(src="assets/Information buttons-converted.pdf",width="750",height="400"),
        return html.Div([html.Iframe(src='/assets/Information buttons-converted.pdf', width='150%', height='1500px')])
    else:
        return []


@app.callback(Output('asr-output', 'children'),
              Input('asr-dropdown', 'value'))
def update_output(value):
    if not value:
        return []
    else:
        if value == 'Speech to Text':
            # return asrtrain.layout
            return html.Div([html.H4("Select a Base Model or Provide a Hugging Face Model String"),
                             html.Hr(),
                             dcc.Dropdown(['Base_Model_untrained_01_21_22', 'Trained_Model_04_01_22'], id='speech-dropdown'),
                             html.Hr(),
                             dcc.Input(id="hf_input", type="text", placeholder="Insert a Hugging Face Model"),
                             html.Hr(),
                             dcc.Upload(
                                 id='speech-model',
                                 children=html.Div([
                                     html.Button(id="asrtrainButt", children=[
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
                                                     # 'margin-left': '300px',
                                                 },
                                                 )

                                 ]),

                                 # Allow multiple files to be uploaded
                                 multiple=True,
                                 # style={'margin-left': '300px'}
                             ),
                             # dcc.Interval(id='clock', interval=1000, n_intervals=0, max_intervals=-1),
                             # dbc.Progress(value=0, id="progress_bar"),
                             html.Button("Train", id='start-work', n_clicks=0),
                             html.Div(id='speech-output', children=[]),
                             ])
        else:
            # Read from sample.json to get dropdown list
            f = open('PyannoteProj/data_preparation/saved_model/sample.json')
            data = json.load(f)
            return html.Div(
                [html.H4("Select a Base Model and Upload a Folder containing a LIST, RTTM, UEM, and WAV set"),
                 html.Hr(),
                 dcc.Dropdown(data, id='diary-dropdown'),
                 du.Upload(
                     # id='diary-model',
                     text='Drag and Drop a Training Set Folder',
                     cancel_button=True,
                     upload_id='diary-model',
                     max_file_size=200000000,
                     max_files=800,
                     filetypes=['rttm', 'txt', 'uem', '.wav', 'zip'],
                     id='upload-files',

                 ),
                 html.Button('Start Training the Model', id='diary-train', n_clicks=0),
                 html.Div(id='diary-input', children=[]),
                 html.Div(id='diary-output', children=[]),
                 html.Div(id='true-dtrain', children=[]),
                 ])
    # return f'You have selected {value}'


@app.callback(Output('speech-output', 'children'),
              Output('speech-dropdown', 'options'),
              Input('hf_input', 'value'),
              Input('speech-dropdown', 'value'),
              Input('speech-model', 'contents'),
              Input('start-work', 'n_clicks'),  # Use N_clicks to reDraw and name displayed Models
              State('speech-model', 'filename'),
              State('speech-dropdown', 'options'),  # Use to append item to the dropdown
              prevent_initial_callback=True, )
def selectModel(hf_input, value, contents, clicks, filename, dropdown_options):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    # if contents is not None and value is not None:
    if clicks > 0:
        try:
            if hf_input != "Insert a Hugging Face Model":
                base_model_string = hf_input
            elif value == "Trained_Model_04_01_22":
                base_model_string = jacob
            else:
                base_model_string = english
            asr_model = Wav2Vec2ASR(use_cuda=True)
            asr_model.loadModel(base_model_string)
            asr_model.train(data_file, data_file, output_dir, num_epochs=3)
            asr_model.saveModel("../assets/asr_models/modelExample")
            return [], dropdown_options
            # dropdown_options.append(f'MyModel{clicks}')
            # return html.Div(html.H5(f'Speech Model set to MyModel{clicks}')), dropdown_options
            # Run it through the model
            # save the model

        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.']), dropdown_options
    else:
        return [], dropdown_options


@du.callback(output=Output('diary-input', 'children'),
             id='upload-files',)
def get_files(filenames):
    return filenames

@app.callback(Output('diary-output', 'children'),
              # Output('diary-dropdown', 'options'),
              Input('diary-train', 'n_clicks'),
              Input('diary-dropdown', 'value'),
              Input('diary-input', 'children'),
              State('upload-files', 'isCompleted'),

              # State('diary-dropdown', 'options'), # Use to append item to the dropdown
              prevent_initial_callback=True, )
# ADD TRIGGER FOR RESEARCHER TO START TRAINING VS AUTOMATIC(Once conditions met)
def selectModel(clicks, value, filenames, completed):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'diary-train' in changed_id and clicks > 0:
    # if 'diary-dropdown' in changed_id:Sample text for typing scenario
        if completed == False:
            return []
        if value is not None and filenames is not None:
            try:
                dia_pipeline = SpeakerDiaImplement()
                dia_pipeline.AddPipeline(model_name=f"assets/saved_model/{value}/seg_model.ckpt",
                                         parameter_name=f"assets/saved_model/{value}/hyper_parameter.json")
                old, new, new_model_name = dia_pipeline.TrainData(f'{filenames[0][32:-4]}')
                # old, new, new_model_name = dia_pipeline.TrainData('TrainingData/SampleData')
                # new_model_name = 'test'
                # old = 4
                # new = 3

                global new_diary_model
                new_diary_model = new_model_name
                # os.mkdir(f'PyannoteProj/data_preparation/saved_models/MyModel{clicks}')
                # open(f'PyannoteProj/data_preparation/saved_models/MyModel{clicks}/{filename}', 'w')
                return html.Div([
                    html.H5(
                        f'The Diarization Error Rate was {old} and it is {new} right now. Model Saved as "{new_model_name}"!'),
                    dcc.Input(id="diary-input", type="text", placeholder="Name the new model", n_submit=0),
                    html.Div(id='input-processor', children=[])
                ])
            except Exception as e:
                print(e)
                return html.Div([
                    'There was an error processing this folder.'])
        else:
            return []
    else:
        return []


@app.callback(Output('input-processor', 'children'),
              Output('diary-dropdown', 'options'),
              Input('diary-input', 'value'),
              Input('diary-input', 'n_submit'),
              State('diary-dropdown', 'options'))
def saveDiaryModel(input, submit, options):
    if input and submit > 0:
        # old_path = os.path.abspath(f'assets/saved_model/{new_diary_model}')
        # new_path = os.path.abspath('assets/saved_model')
        os.chdir(os.path.abspath('assets/saved_model/'))
        os.rename(f'{new_diary_model}', f'{input}')
        options.append(f'{input}')
        return f'Model is set to {input}', options
    else:
        return [], options


@app.callback(Output('asr-button-output', 'children'),
              Output('language', 'data'),
              Output('model-save', 'data'),
              Input('model1asr', 'n_clicks'),
              Input('model2asr', 'n_clicks'),
              State('language', 'data'),
              State('model-save', 'data'))
def parse_contents(mod1, mod2, model, selected):
    global current_model_asr
    if selected != "":
        current_model_asr = selected
    else:
        current_model_asr = f"Training iteration 01, trained on 4/1/22"
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'model1asr' in changed_id and mod1 > 0:
        current_model_asr = f"English Base, loaded in on 1/21/22"
        model = 'english'
    elif 'model2asr' in changed_id and mod2 > 0:
        current_model_asr = f"Training iteration 01, trained on 4/1/22"
        model = 'jacob'
    selected = model

    return html.Div([
        html.H5(f"Current ASR Model: {current_model_asr}")
    ]), model, selected
