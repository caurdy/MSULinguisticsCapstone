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
],
    style={'margin-left': '300px'})


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
        var_store.sort(key=lambda x: x[3], reverse=True)
        layout = []
        for i in var_store:
            i = tuple((i[0], i[1], i[2], i[-1]))
            print(i)
            transcripts.append(i)
            btn = html.Button('Show Transcript', id={'role': 'tshow', 'index': f'{i[-1]}'}, n_clicks=0,
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
                html.Div(id={'role': 'display', 'index': f'{i[-1]}'}, children=[])]))
        btn_nlicks.append(id('tshow_1.n_clicks'))
        var_store.sort(reverse=True)
        return size, layout
    else:
        return size, []


@app.callback(
    Output({'role': 'display', 'index': MATCH}, 'children'),
    [Input({'role': 'tshow', 'index': ALL}, 'n_clicks')],
    State('archive-size', 'data'),
    State('stored-data', 'data'),
    prevent_initial_call=True,
)
# n_clicks currently just takes how many times each Input (tshow_) has been clicked
# try to form an object with an id that contains all present n_clicks?
def displayClick(n_clicks, size, archive):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if n_clicks != [0*len(transcripts)]:
        index = int(changed_id[10:11])
        audio_path = archive[index][0]
        transcript_path = archive[index][1]
        if int(n_clicks[index]) > 0 and 'tshow' in changed_id:
            if n_clicks[index] % 2 == 1:
                df = pd.read_json(transcript_path)
                return html.Div([
                    html.Div([html.Hr(),
                              html.H4(transcripts[index][1]),
                              html.Div(html.Button(
                                  html.Audio(id="audio", src=audio_path, controls=True,
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
                                  style_table={'textAlign': 'center', 'width': '1050px'},
                              ), ]),
                    html.Hr()])
            else:
                return []
    else:
        return []