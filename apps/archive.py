# Add Transcript Archive here
# Have wav and associated csv and be able to view full csv
from dash.dependencies import Input, Output, State
from dash import html, callback_context, dash_table, MATCH, ALL
import datetime
import pandas as pd

from starter import app

button = html.A(html.Button('Show Transcript', id="dispaly"), href="/", style={})

# row =
transcripts = []
btn_nlicks = []

layout = html.Div([
    html.H1("Transcript Archive"),
    html.Button("Get Stored Data", id='loader', n_clicks=0),
    html.Div(id='load', children=[]),
    html.Div(id='chosen', children=[]),
    # print(transcripts),
    #     html.Div([
    #         html.Div(id="archive", children=[
    #             html.Div(html.H5(datetime.datetime.now().strftime('%m/%d/%Y')),
    #                      style={'display': 'inline-block',
    #                             'width': '17%',
    #                             'borderStyle': 'dashed',
    #                             'borderWidth': '2px',
    #                             'textAlign': 'center'}),
    #             html.Div(html.H5("audio_file_02212022.wav"),
    #                      style={'display': 'inline-block',
    #                             'width': '34%',
    #                             'borderStyle': 'dashed',
    #                             'borderWidth': '2px',
    #                             'textAlign': 'center'}),
    #             html.Div(html.H5("transcript_02212022.csv"),
    #                      style={'display': 'inline-block',
    #                             'width': '33%',
    #                             'borderStyle': 'dashed',
    #                             'borderWidth': '2px',
    #                             'textAlign': 'center'}),
    #             html.A(html.Button('Show Transcript', id="display1", n_clicks=0), style={'display': 'inline-block'}),
    #         ], style={'display': 'inline-block', 'margin': '10px', 'width': '120%', 'borderStyle': 'dashed',
    #                   'borderWidth': '1px', 'padding': '1rem'}),
    #
    #         html.Hr(),
    #         html.Div([
    #             html.Div(html.H5(datetime.datetime.now().strftime('%m/%d/%Y')),
    #                      style={'display': 'inline-block',
    #                             'width': '17%',
    #                             'borderStyle': 'dashed',
    #                             'borderWidth': '2px',
    #                             'textAlign': 'center'}),
    #             html.Div(html.H5("audio_file_02202022.wav"),
    #                      style={'display': 'inline-block',
    #                             'width': '34%',
    #                             'borderStyle': 'dashed',
    #                             'borderWidth': '2px',
    #                             'textAlign': 'center'}),
    #             html.Div(html.H5("transcript_02172022a.csv"),
    #                      style={'display': 'inline-block',
    #                             'width': '33%',
    #                             'borderStyle': 'dashed',
    #                             'borderWidth': '2px',
    #                             'textAlign': 'center'}),
    #             html.A(html.Button('Show Transcript', id="display2"), href="/", style={'display': 'inline-block'}),
    #         ], style={'display': 'inline-block', 'margin': '10px', 'width': '120%', 'borderStyle': 'dashed',
    #                   'borderWidth': '1px', 'padding': '1rem'}),
    #         html.Hr(),
    #         html.Div([
    #             html.Div(html.H5(datetime.datetime.now().strftime('%m/%d/%Y')),
    #                      style={'display': 'inline-block',
    #                             'width': '17%',
    #                             'borderStyle': 'dashed',
    #                             'borderWidth': '2px',
    #                             'textAlign': 'center'}),
    #             html.Div(html.H5("audio_file_02182021.wav"),
    #                      style={'display': 'inline-block',
    #                             'width': '34%',
    #                             'borderStyle': 'dashed',
    #                             'borderWidth': '2px',
    #                             'textAlign': 'center'}),
    #             html.Div(html.H5("transcript_021920223.csv"),
    #                      style={'display': 'inline-block',
    #                             'width': '33%',
    #                             'borderStyle': 'dashed',
    #                             'borderWidth': '2px',
    #                             'textAlign': 'center'}),
    #             html.A(html.Button('Show Transcript', id="dispaly3"), href="/", style={'display': 'inline-block'}),
    #         ], style={'display': 'inline-block', 'margin': '10px', 'width': '120%', 'borderStyle': 'dashed',
    #                   'borderWidth': '1px', 'padding': '1rem'}),
    #
    #         html.Hr(),
    #         html.Div([
    #             html.Div(html.H5(datetime.datetime.now().strftime('%m/%d/%Y')),
    #                      style={'display': 'inline-block',
    #                             'width': '17%',
    #                             'borderStyle': 'dashed',
    #                             'borderWidth': '2px',
    #                             'textAlign': 'center'}),
    #             html.Div(html.H5("audio_file_02222022.wav"),
    #                      style={'display': 'inline-block',
    #                             'width': '34%',
    #                             'borderStyle': 'dashed',
    #                             'borderWidth': '2px',
    #                             'textAlign': 'center'}),
    #             html.Div(html.H5("transcript_02222022.csv"),
    #                      style={'display': 'inline-block',
    #                             'width': '33%',
    #                             'borderStyle': 'dashed',
    #                             'borderWidth': '2px',
    #                             'textAlign': 'center'}),
    #
    #             html.A(html.Button('Show Transcript', id="dispaly4"), href="/", style={'display': 'inline-block'}),
    #         ], style={'display': 'inline-block', 'margin': '10px', 'width': '120%', 'borderStyle': 'dashed',
    #                   'borderWidth': '1px', 'padding': '1rem'}),
    #
    #         html.Hr(),
    #
    #     ])
],
    style={'margin-left': '300px'})


# @app.callback(
#     Output('archive', 'children'),
#     Input('display1', 'n_clicks'),
#     # Input('display1', 'n_clicks'),
#     # Input('display3', 'n_clicks'),
#     # Input('display4', 'n_clicks')
# )
# def displayClick(n_clicks):
#     # changed_id = [p['prop_id'] for p in callback_context.triggered][0]
#     # if 'display1' in changed_id:
#     if n_clicks % 2 == 1:
#         df = pd.read_csv("assets/transcript.csv")
#         return html.Div([html.Div([
#             html.Div(html.H5(datetime.datetime.now().strftime('%m/%d/%Y')),
#                      style={'display': 'inline-block',
#                             'width': '17%',
#                             'borderStyle': 'dashed',
#                             'borderWidth': '2px',
#                             'textAlign': 'center'}),
#             html.Div(html.H5("audio_file_02212022.wav"),
#                      style={'display': 'inline-block',
#                             'width': '34%',
#                             'borderStyle': 'dashed',
#                             'borderWidth': '2px',
#                             'textAlign': 'center'}),
#             html.Div(html.H5("transcript_02212022.csv"),
#                      style={'display': 'inline-block',
#                             'width': '33%',
#                             'borderStyle': 'dashed',
#                             'borderWidth': '2px',
#                             'textAlign': 'center'}),
#             html.A(html.Button('Show Transcript', id="display1", n_clicks=1), style={'display': 'inline-block'}),
#         ]),
#
#         html.Hr(),
#         html.Div(style={'padding': '2rem'}),
#         dash_table.DataTable(
#
#             df.to_dict('records'),
#             [{'name': i, 'id': i} for i in df.columns],
#             css=[{
#                 'selector': '.dash-spreadsheet td div',
#                 'rule': '''
#                                     line-height: 15px;
#                                     max-height: 30px; min-height: 30px; height: 30px;
#                                     display: block;
#                                     overflow-y: hidden;
#                                 '''
#             }],
#             style_data={
#                 'whiteSpace': 'normal',
#                 'height': 'auto',
#                 'lineHeight': '15px'
#             },
#             style_cell={'textAlign': 'left'},
#             # style_data={'whiteSpace': 'normal',
#             #             'minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
#             # 'max-height': '30px', 'min-height': '30px', 'height': '30px',
#             # 'lineHeight': '15px',
#             #         },
#             style_table={'textAlign': 'center', 'width': '1050px'},
#         ),
#
#         html.Hr(),  # horizontal line
#     ])
#     else:
#         # return layout
#         return html.Div([
#             html.Div(html.H5(datetime.datetime.now().strftime('%m/%d/%Y')),
#                      style={'display': 'inline-block',
#                             'width': '17%',
#                             'borderStyle': 'dashed',
#                             'borderWidth': '2px',
#                             'textAlign': 'center'}),
#             html.Div(html.H5("audio_file_02212022.wav"),
#                      style={'display': 'inline-block',
#                             'width': '34%',
#                             'borderStyle': 'dashed',
#                             'borderWidth': '2px',
#                             'textAlign': 'center'}),
#             html.Div(html.H5("transcript_02212022.csv"),
#                      style={'display': 'inline-block',
#                             'width': '33%',
#                             'borderStyle': 'dashed',
#                             'borderWidth': '2px',
#                             'textAlign': 'center'}),
#             html.A(html.Button('Show Transcript', id="display1", n_clicks=0), style={'display': 'inline-block'}),
#         ])

@app.callback(
    Output('archive-size', 'data'),
    Output("load", "children"),
    Input("loader", "n_clicks"),
    State("stored-data", "data"),
    State('archive-size', 'data'),
    prevent_initial_call=True,
)
def get_from_store(count, var_store, size):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    size = size or {'clicks': 0}
    size['clicks'] = len(var_store)

    if 'loader' in changed_id and count > 0:
        var_store.sort(reverse=True)
        layout = []
        for i in var_store:
            i = tuple((i[0], i[1], i[2]))
            transcripts.append(i)
            btn = html.Button('Show Transcript', id={'role': 'tshow', 'index' : f'{i[1][-5:-4]}'}, n_clicks=0,
                              style={'display': 'inline-block'})

            btn_nlicks.append(btn)
            layout.append(
                html.Div([
                html.Div([
                    html.Div(html.H5(i[2]),
                             style={'display': 'inline-block',
                                    'width': '17%',
                                    'borderStyle': 'dashed',
                                    'borderWidth': '2px',
                                    'textAlign': 'center'}),
                    html.Div(html.H5(i[0]),
                             style={'display': 'inline-block',
                                    'width': '34%',
                                    'height': '2%',
                                    'borderStyle': 'dashed',
                                    'borderWidth': '2px',
                                    'textAlign': 'center'}),
                    html.Div(html.H5(i[1]),
                             style={'display': 'inline-block',
                                    'width': '33%',
                                    'borderStyle': 'dashed',
                                    'borderWidth': '2px',
                                    'textAlign': 'center'}),

                    btn,
                ], style={'display': 'inline-block', 'margin': '10px', 'width': '120%', 'borderStyle': 'dashed',
                          'borderWidth': '1px', 'padding': '1rem'},
                ),
                html.Div(id={'role': 'display', 'index': f'{i[1][-5:-4]}'}, children=[])]))
        btn_nlicks.append(id('tshow_1.n_clicks'))
        var_store.sort(reverse=True)
        return size, layout
    else:
        return size, []


# def update(ignore):
#     return np.random.uniform()
#
# for i in range(len(transcripts)):
#     app.callback(
#         Output('chosen', 'children'),
#         [Input(f'tshow_{i}', 'n_clicks')]
#     )(update)
@app.callback(
    Output({'role': 'display', 'index': MATCH}, 'children'),
    [Input({'role': 'tshow', 'index': ALL}, 'n_clicks')],
    # Input('transcripts', 'transcripts'),
    # params=list(input(f'tshow_{i}', "n_clicks") for i in range(1, len(transcripts)), property = 'n_clicks'),
    # Input({'type':'mybuttons', 'index':ALL}, 'n_clicks'),
    State('archive-size', 'data'),
    State('stored-data', 'data'),
    # Input('display1', 'n_clicks'),
    # Input('display3', 'n_clicks'),
    # Input('display4', 'n_clicks'),
    prevent_initial_call=True,
)
# n_clicks currently just takes how many times each Input (tshow_) has been clicked
# try to form an object with an id that contains all present n_clicks?
def displayClick(n_clicks, size, archive):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    # if changed_id == 'tshow_2.n_clicks':
    x = transcripts
    if n_clicks != [0*len(transcripts)]:
        index = size['clicks']-int(changed_id[10:11])
        if int(n_clicks[index]) > 0 and 'tshow' in changed_id:
            if n_clicks[index] % 2 == 1:
                # changed_id = int(changed_id[6:-9])
                df = pd.read_csv(f"assets/transcript_{index+1}.csv")
                return html.Div([
                    html.Div([html.Hr(),
                              html.H4(transcripts[index][1]),
                              html.Div(html.Button(
                                  html.Audio(id="audio", src=f'assets/{transcripts[index][0]}', controls=True,
                                             autoPlay=False))),
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
                              ), ]),
                    html.Hr()])
            else:
                return []
    else:
        return []



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
