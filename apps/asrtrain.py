# # Add Model Retraining Page Here
# # GET BUTTONS
# from dash.dependencies import Input, Output, State
# from dash import dcc, html, callback_context
# import json
# from PyannoteProj.data_preparation.saved_model import model_0, model_1
# # turn this into a pattern matching list build of the included models
# #
# # f = open('sample.json')
# # data = json.load(f)
#
# from starter import app
#
# current_model_diarization = "Model-1-DIA"
# current_model_asr = "Model-1-ASR"
#
# layout = html.Div([
#     html.H1("Retraining the Models"),
#
#     html.Hr(),
#
#     # dcc.Upload(html.A('Upload File')),
#
#     dcc.Dropdown(['NYC', 'MTL', 'SF'], 'NYC', id='asr-dropdown'),
#     html.Div(id='asr-output-container'),
#
#     html.Hr(),
#
#     dcc.Dropdown(['model1', 'model2'], 'model1', id='diary-dropdown'),
#     html.Div(id='diary-output-container'),
#
#     html.Hr(),
#
#     dcc.Upload(
#         id = "upload-file",
#         children = [
#         'Drag and Drop or ',
#         html.A('Select a Training File (.uem or .json)')
#     ], style={
#         'width': '100%',
#         'height': '60px',
#         'lineHeight': '60px',
#         'borderWidth': '1px',
#         'borderStyle': 'dashed',
#         'borderRadius': '5px',
#         'textAlign': 'center'
#         },
#         multiple= True
#     ),
#
#     html.Hr(),
#
#     html.Div(id='output-data', children=[html.Div(f"Current ASR Model: {current_model_asr}", id="asr-output"),
#                                          html.Div(f"Current Diarization Model: {current_model_diarization}", id="dia-output")],
#              style= {'margin': '20px'}),
#
#     html.Hr(),
#     html.Div(id="col_asr", n_clicks=0, title="Speech Recognition Models",
#              children=["Speech Recognition", html.Div(html.Button("Model 1 ASR", id='model1asr', n_clicks=0)),
#                        html.Div(html.Button("Model 2 ASR", id='model2asr', n_clicks=0))],
#              style={'margin': '20px'}),
#     html.Hr(),
#     html.Div(id="col_di", n_clicks=0, title="Speaker Diarization Models",
#              children=["Speaker Diarization", html.Div(html.Button("Model 1 DIA", id='model1dia', n_clicks=0)),
#                        html.Div(html.Button("Model 2 DIA", id='model2dia', n_clicks=0))],
#              style={'margin': '20px'}),
#     html.Hr()
#
# ],
# style={'margin-left': '300px'})
#
# @app.callback(Output('asr-output', 'children'),
#               Input('upload-file', 'contents'),
#               State('upload-file', 'filename'),
#               Input('model1asr', 'n_clicks'),
#               Input('model2asr', 'n_clicks'))
#
#
# def parse_contents(contents, filename, mod1, mod2):
#     if contents is not None and filename[0][-4:] != ".uem":
#         if filename[0][-4:] != "json":
#             return html.Div([
#                 html.H5(f"File invalid. Please upload a .json file or select a pretrained Model")
#             ])
#
#         return html.Div([
#             html.H5(f"Current ASR model is being trained on: {filename[0]}"),
#         ])
#     else:
#         changed_id = [p['prop_id'] for p in callback_context.triggered][0]
#         if 'model1asr' in changed_id:
#             mod = 'Model-1-ASR'
#             return html.Div([
#                 html.H5(f"Current ASR model is: {mod}"),
#             ])
#         elif 'model2asr' in changed_id:
#             mod = 'Model-2-ASR'
#             return html.Div([
#                 html.H5(f"Current ASR model is: {mod}"),
#             ])
#         else:
#             return html.Div([
#                 html.H5(f"Current ASR model is: {current_model_asr}"),
#             ])
#
#
# @app.callback(Output('dia-output', 'children'),
#               Input('upload-file', 'contents'),
#               State('upload-file', 'filename'),)
#
#
# def parse_contents(contents, filename, mod1, mod2):
#     if contents is not None and filename[0][-4:] != "json":
#         if filename[0][-3:] != "uem":
#             return html.Div([
#                 html.H5(f"File invalid. Please upload a .uem file or select a pretrained Model")
#             ])
#         return html.Div([
#             html.H5(f"Current Diarization model is being trained on: {filename[0]}"),
#         ])
#
#
# @app.callback(
#     Output('diary-output-container', 'children'),
#     Input('diary-dropdown', 'value')
# )
# def update_output(value):
#     if value == 'model1':
#         x = 2
#         # TestPipline.id = 1
#     else:
#         x = 2
#         # TestPipline.id = 0
#     return f'You have selected {value}'