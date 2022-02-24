# Add Model Retraining Page Here
# GET BUTTONS
from dash.dependencies import Input, Output, State
from dash import dcc, html, callback_context

from starter import app

current_model = "Model-1.json"

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
        html.A('Select a File')
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

    html.Div(id='output-data', children=[html.Div(f"Current Model: {current_model}")], style= {'margin': '20px'}),

    html.Hr(),

    html.Div(html.Button("Module 1", id='model1', n_clicks=0)),
    html.Div(html.Button("Module 2", id='model2', n_clicks=0)),
],
style={'margin-left': '300px'})

@app.callback(Output('output-data', 'children'),
              Input('upload-file', 'contents'),
              Input('model1', 'n_clicks'),
              Input('model2', 'n_clicks'),
              State('upload-file', 'filename'))


def parse_contents(contents, filename, mod1, mod2):
    if contents is not None:

        return html.Div([
            html.H5(f"Current model is: {filename}"),
        ])
    else:
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        if 'model1' in changed_id:
            mod = 'Model-1.json'
            return html.Div([
                html.H5(f"Current model is: {mod}"),
            ])
        elif 'model2' in changed_id:
            mod = 'Model-2.json'
            return html.Div([
                html.H5(f"Current model is: {mod}"),
            ])
        else:
            return html.Div([
                html.H5(f"Current model is: {current_model}"),
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
