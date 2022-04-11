# Add Model Retraining Page Here
# GET BUTTONS
import base64

from dash.dependencies import Output, Input, State
from dash import dcc, html, callback_context
import os
import json
import dash_uploader as du
from dash.exceptions import PreventUpdate

from PyannoteProj.TestPipline import SpeakerDiaImplement
from PyannoteProj.database_loader import CreateDatabase
from PyannoteProj.data_preparation.saved_model import model_0, model_1

from starter import app


# du.configure_upload(app, )
def get_upload(uid):
    return du.upload(id=uid)


layout = html.Div([
    html.H1("Chose which aspect to retrain"),
    # HINT

    html.Hr(),

    # dbc.DropdownMenu(
    # label="Menu",
    # children=[
    #     dbc.DropdownMenu(
    #     children=[
    #         dbc.DropdownMenuItem("Item 1"),
    #         dbc.DropdownMenuItem("Item 2"),
    #     ],
    #     ),
    #     dbc.DropdownMenuItem("Item 2"),
    #     dbc.DropdownMenuItem("Item 3"),
    # ],
    # ))

    # dcc.Upload(html.A('Upload File')),

    dcc.Dropdown(['Speech to Text', 'Diarization'], id='asr-dropdown',
                 style={'display': 'inline-block', 'width': '90%'}),
    html.Button(html.Img(id='img', src='assets/qmark.png', style={'display': 'inline-block',
                                                                  'width': '30px', 'height': '20%',
                                                                  }),
                id='hints', n_clicks=0),

    html.Hr(),

    html.Div(id='asr-output', children=[]),

    html.Hr(),

],
    style={'margin-left': '300px'})


@app.callback(Output('asr-output', 'children'),
              Input('asr-dropdown', 'value')
              )
def update_output(value):
    if not value:
        return []
    else:
        if value == 'Speech to Text':
            # return asrtrain.layout
            return html.Div([html.H4("Select a Base Model or Upload a model"),
                             html.Hr(),
                             dcc.Dropdown(['Name-1', 'HuggingFace'], id='speech-dropdown'),
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
                 dcc.Upload(
                     id='diary-model',
                     children=html.Div([
                         html.Button(id="diarytrainButt", children=[
                             'Drag and Drop or ',
                             html.A('Select a Folder'), ], n_clicks=0,
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
                                     }, )

                     ]),

                     # Allow multiple files to be uploaded
                     multiple=True,
                     # style={'margin-left': '300px'}
                 ),
                 html.Div(id='diary-output', children=[]),
                 ])
    # return f'You have selected {value}'


@app.callback(Output('speech-output', 'children'),
              Output('speech-dropdown', 'options'),
              Input('speech-dropdown', 'value'),
              Input('speech-model', 'contents'),
              Input('asrtrainButt', 'n_clicks'),  # Use N_clicks to reDraw and name displayed Models
              State('speech-model', 'filename'),
              State('speech-dropdown', 'options'),  # Use to append item to the dropdown
              prevent_initial_callback=True, )
def selectModel(value, contents, clicks, filename, dropdown_options):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if contents is not None and value is not None:
        try:

                # dropdown_options.append(f'MyModel{clicks}')
                return html.Div(html.H5(f'Speech Model set to MyModel{clicks}')), dropdown_options
            # Run it through the model
            # save the model
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.']), dropdown_options
    else:
        return [], dropdown_options


@app.callback(Output('diary-output', 'children'),
              # Output('diary-dropdown', 'options'),
              Input('diary-dropdown', 'value'),
              Input('diary-model', 'contents'),
              Input('diarytrainButt', 'n_clicks'),  # Use N_clicks to reDraw and name displayed Models
              State('diary-model', 'filename'),
              # State('diary-dropdown', 'options'), # Use to append item to the dropdown
              prevent_initial_callback=True, )
def selectModel(value, contents, clicks, filename):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    # if 'diary-dropdown' in changed_id:
    if value is not None and contents is not None:
        # content_type, content_string = contents.split(',')
        # decoded = base64.b64decode(content_string)
        try:
            # if '.uem' in filename:
            # dropdown_options.append(f'MyModel{clicks}')
            # Add Model to directory

            CreateDatabase("Talkbank")  # This is training data
            dia_pipeline = SpeakerDiaImplement()
            dia_pipeline.AddPipeline(model_name=f"assets/saved_model/{value}/seg_model.ckpt",
                                     parameter_name=f"assets/saved_model/{value}/hyper_parameter.json")
            dia_pipeline.TrainData('assets/Talkbank')
            # os.mkdir(f'PyannoteProj/data_preparation/saved_models/MyModel{clicks}')
            # open(f'PyannoteProj/data_preparation/saved_models/MyModel{clicks}/{filename}', 'w')
            return html.Div(html.H5(f'Speech Model set to MyModel{clicks}'))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this folder.'])
    else:
        return []
