# Add Model Retraining Page Here
# GET BUTTONS
from dash.dependencies import Input, Output, State
from dash import dcc, html, callback_context

from starter import app

current_model_diarization = "Model-1-DIA"
current_model_asr = "Model-1-ASR"

layout = html.Div([
    html.H1("Select or Upload a Model"),
    # dcc.Upload(html.Button('Upload File'), id="upload-file"),

    html.Hr(),

    # dcc.Upload(html.A('Upload File')),

    # html.Hr(),

    dcc.Upload(
        id = "upload-file",
        children = [
        'Drag and Drop or ',
        html.A('Select a Training File (.uem or .json)')
    ], style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center'
        },
        multiple= True
    ),

    html.Hr(),

    html.Div(id='output-data', children=[html.Div(f"Current ASR Model: {current_model_asr}", id="asr-output"),
                                         html.Div(f"Current Diarization Model: {current_model_diarization}", id="dia-output")],
             style= {'margin': '20px'}),

    html.Hr(),
    html.Div(id="col_asr", n_clicks=0, title="Speech Recognition Models",
             children=["Speech Recognition", html.Div(html.Button("Model 1 ASR", id='model1asr', n_clicks=0)),
                       html.Div(html.Button("Model 2 ASR", id='model2asr', n_clicks=0))],
             style={'margin': '20px'}),
    html.Hr(),
    html.Div(id="col_di", n_clicks=0, title="Speaker Diarization Models",
             children=["Speaker Diarization", html.Div(html.Button("Model 1 DIA", id='model1dia', n_clicks=0)),
                       html.Div(html.Button("Model 2 DIA", id='model2dia', n_clicks=0))],
             style={'margin': '20px'}),
    html.Hr()

],
style={'margin-left': '300px'})

@app.callback(Output('asr-output', 'children'),
              Input('upload-file', 'contents'),
              State('upload-file', 'filename'),
              Input('model1asr', 'n_clicks'),
              Input('model2asr', 'n_clicks'))


def parse_contents(contents, filename, mod1, mod2):
    if contents is not None and filename[0][-4:] != ".uem":
        if filename[0][-4:] != "json":
            return html.Div([
                html.H5(f"File invalid. Please upload a .json file or select a pretrained Model")
            ])

        return html.Div([
            html.H5(f"Current ASR model is being trained on: {filename[0]}"),
        ])
    else:
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        if 'model1asr' in changed_id:
            mod = 'Model-1-ASR'
            return html.Div([
                html.H5(f"Current ASR model is: {mod}"),
            ])
        elif 'model2asr' in changed_id:
            mod = 'Model-2-ASR'
            return html.Div([
                html.H5(f"Current ASR model is: {mod}"),
            ])
        else:
            return html.Div([
                html.H5(f"Current ASR model is: {current_model_asr}"),
            ])


@app.callback(Output('dia-output', 'children'),
              Input('upload-file', 'contents'),
              State('upload-file', 'filename'),
              Input('model1dia', 'n_clicks'),
              Input('model2dia', 'n_clicks'))


def parse_contents(contents, filename, mod1, mod2):
    if contents is not None and filename[0][-4:] != "json":
        if filename[0][-3:] != "uem":
            return html.Div([
                html.H5(f"File invalid. Please upload a .uem file or select a pretrained Model")
            ])
        return html.Div([
            html.H5(f"Current Diarization model is being trained on: {filename}"),
        ])
    else:
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        if 'model1dia' in changed_id:
            mod = 'Model-1-DIA'
            return html.Div([
                html.H5(f"Current DIA model is: {mod}"),
            ])
        elif 'model2dia' in changed_id:
            mod = 'Model-2-DIA'
            return html.Div([
                html.H5(f"Current DIA model is: {mod}"),
            ])
        else:
            return html.Div([
                html.H5(f"Current DIA model is: {current_model_diarization}"),
            ])

@app.callback(
    Output('container-button-timestamp', 'children'),
    Input('model1', 'n_clicks'),
    Input('model2', 'n_clicks'),
)
def displayClick(mod1, mod2):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'model1' in changed_id:
        msg = 'Button 1 was most recently clicked'
    elif 'model2' in changed_id:
        msg = 'Button 2 was most recently clicked'
    else:
        msg = 'None of the buttons have been clicked yet'
    return html.Div(msg)

# @app.callback(Output())
#
# def update_output(n_clicks, value):
#     changed_id = [p['prop_id'] for p in callback_context.triggered][0]
#
#     # return 'The input value was "{}" and the button has been clicked {} times'.format(
#     #     value,
#     #     n_clicks
#     # )
#     current_model = "Model-2.json"
#     return layout
# deal with n_clicks on the button, if the button is pushed change name to another value
