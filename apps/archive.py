# Add Transcript Archive here
# Have wav and associated csv and be able to view full csv
from dash.dependencies import Input, Output
from dash import html, callback_context, dash_table
import datetime
import pandas as pd


from starter import app

button = html.A(html.Button('Show Transcript', id="dispaly"), href="/")

# row =
layout = html.Div([
    html.H1("Transcript Archive"),
html.Div([
    html.Div(id="archive", children=[
        html.Div(html.H5(datetime.datetime.now().strftime('%m/%d/%Y')),
                 style={'display': 'inline-block',
                        'width': '17%',
                        'borderStyle': 'dashed',
                        'borderWidth': '2px',
                        'textAlign': 'center'}),
        html.Div(html.H5("audio_file_02212022.wav"),
                 style={'display': 'inline-block',
                        'width': '34%',
                        'borderStyle': 'dashed',
                        'borderWidth': '2px',
                        'textAlign': 'center'}),
        html.Div(html.H5("transcript_02212022.csv"),
                 style={'display': 'inline-block',
                        'width': '33%',
                        'borderStyle': 'dashed',
                        'borderWidth': '2px',
                        'textAlign': 'center'}),
        html.A(html.Button('Show Transcript', id="display1", n_clicks=0), style={'display': 'inline-block'}),
    ], style= {'display': 'inline-block', 'margin': '10px', 'width': '120%', 'borderStyle': 'dashed',
                        'borderWidth': '1px', 'padding': '1rem'}),

    html.Hr(),
html.Div([
        html.Div(html.H5(datetime.datetime.now().strftime('%m/%d/%Y')),
                 style={'display': 'inline-block',
                        'width': '17%',
                        'borderStyle': 'dashed',
                        'borderWidth': '2px',
                        'textAlign': 'center'}),
        html.Div(html.H5("audio_file_02202022.wav"),
                 style={'display': 'inline-block',
                        'width': '34%',
                        'borderStyle': 'dashed',
                        'borderWidth': '2px',
                        'textAlign': 'center'}),
        html.Div(html.H5("transcript_02172022a.csv"),
                 style={'display': 'inline-block',
                        'width': '33%',
                        'borderStyle': 'dashed',
                        'borderWidth': '2px',
                        'textAlign': 'center'}),
        html.A(html.Button('Show Transcript', id="dispaly2"), href="/", style={'display': 'inline-block'}),
    ], style= {'display': 'inline-block', 'margin': '10px', 'width': '120%', 'borderStyle': 'dashed',
                        'borderWidth': '1px', 'padding': '1rem'}),
    html.Hr(),
html.Div([
        html.Div(html.H5(datetime.datetime.now().strftime('%m/%d/%Y')),
                 style={'display': 'inline-block',
                        'width': '17%',
                        'borderStyle': 'dashed',
                        'borderWidth': '2px',
                        'textAlign': 'center'}),
        html.Div(html.H5("audio_file_02182021.wav"),
                 style={'display': 'inline-block',
                        'width': '34%',
                        'borderStyle': 'dashed',
                        'borderWidth': '2px',
                        'textAlign': 'center'}),
        html.Div(html.H5("transcript_021920223.csv"),
                 style={'display': 'inline-block',
                        'width': '33%',
                        'borderStyle': 'dashed',
                        'borderWidth': '2px',
                        'textAlign': 'center'}),
        html.A(html.Button('Show Transcript', id="dispaly3"), href="/", style={'display': 'inline-block'}),
    ], style= {'display': 'inline-block', 'margin': '10px', 'width': '120%', 'borderStyle': 'dashed',
                        'borderWidth': '1px', 'padding': '1rem'}),

    html.Hr(),
html.Div([
        html.Div(html.H5(datetime.datetime.now().strftime('%m/%d/%Y')),
                 style={'display': 'inline-block',
                        'width': '17%',
                        'borderStyle': 'dashed',
                        'borderWidth': '2px',
                        'textAlign': 'center'}),
        html.Div(html.H5("audio_file_02222022.wav"),
                 style={'display': 'inline-block',
                        'width': '34%',
                        'borderStyle': 'dashed',
                        'borderWidth': '2px',
                        'textAlign': 'center'}),
        html.Div(html.H5("transcript_02222022.csv"),
                 style={'display': 'inline-block',
                        'width': '33%',
                        'borderStyle': 'dashed',
                        'borderWidth': '2px',
                        'textAlign': 'center'}),


        html.A(html.Button('Show Transcript', id="dispaly4"), href="/", style={'display': 'inline-block'}),
    ], style= {'display': 'inline-block', 'margin': '10px', 'width': '120%', 'borderStyle': 'dashed',
                        'borderWidth': '1px', 'padding': '1rem'}),

    html.Hr(),

    ])
],
style={'margin-left': '300px'})

@app.callback(
    Output('archive', 'children'),
    Input('display1', 'n_clicks'),
    # Input('display1', 'n_clicks'),
    # Input('display3', 'n_clicks'),
    # Input('display4', 'n_clicks')
)
def displayClick(n_clicks):
    # changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    # if 'display1' in changed_id:
    if n_clicks % 2 == 1:
        df = pd.read_csv("assets/transcript.csv")
        return html.Div([html.Div([
            html.Div(html.H5(datetime.datetime.now().strftime('%m/%d/%Y')),
                     style={'display': 'inline-block',
                            'width': '17%',
                            'borderStyle': 'dashed',
                            'borderWidth': '2px',
                            'textAlign': 'center'}),
            html.Div(html.H5("audio_file_02212022.wav"),
                     style={'display': 'inline-block',
                            'width': '34%',
                            'borderStyle': 'dashed',
                            'borderWidth': '2px',
                            'textAlign': 'center'}),
            html.Div(html.H5("transcript_02212022.csv"),
                     style={'display': 'inline-block',
                            'width': '33%',
                            'borderStyle': 'dashed',
                            'borderWidth': '2px',
                            'textAlign': 'center'}),
            html.A(html.Button('Show Transcript', id="display1", n_clicks=1), style={'display': 'inline-block'}),
        ]),

        html.Hr(),
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
    ])
    else:
        # return layout
        return html.Div([
            html.Div(html.H5(datetime.datetime.now().strftime('%m/%d/%Y')),
                     style={'display': 'inline-block',
                            'width': '17%',
                            'borderStyle': 'dashed',
                            'borderWidth': '2px',
                            'textAlign': 'center'}),
            html.Div(html.H5("audio_file_02212022.wav"),
                     style={'display': 'inline-block',
                            'width': '34%',
                            'borderStyle': 'dashed',
                            'borderWidth': '2px',
                            'textAlign': 'center'}),
            html.Div(html.H5("transcript_02212022.csv"),
                     style={'display': 'inline-block',
                            'width': '33%',
                            'borderStyle': 'dashed',
                            'borderWidth': '2px',
                            'textAlign': 'center'}),
            html.A(html.Button('Show Transcript', id="display1", n_clicks=0), style={'display': 'inline-block'}),
        ])
    # elif 'display2' in changed_id:
    #     msg = 'Button 2 was most recently clicked'
    # elif 'display3' in changed_id:
    #     msg = 'Button 3 was most recently clicked'
    # elif 'display4' in changed_id:
    #     msg = "Hello World!"
    # return html.Div(msg)

# @app.callback(Output("archive", "children"),
#               [Input("stored_data", "data")])
#
# def display_archive(data):
#     return html.Div([
#         html.H1(data)
#         # dash_table.DataTable(
#         # data.to_dict('records'),
#         # [{'name': i, 'id': i} for i in data.columns]
#         # ),
#         #
#         # html.Hr(),  # horizontal line
#         #
#         # # For debugging, display the raw contents provided by the web browser
#         # html.Div('Raw Content'),
#         # html.Pre(contents[0:200] + '...', style={
#         #     'whiteSpace': 'pre-wrap',
#         #     'wordBreak': 'break-all'
#         # })
#     ])