import plotly.express as px
import pandas as pd
import json

with open('predictions.json') as f:
    data = json.load(f)


def reshape_prediction_data(data):
    new_data = []
    for data_point in data:
        predicted = data_point['predicted']
        true = data_point['true']
        size = 100
        new_data.append((true[0], true[1], 1, "true", size))
        new_data.append((predicted[0], predicted[1], 0, "predicted", size))
    return new_data


def reshape_grid_data(data):
    new_data = []
    for data_point in data:
        new_data.append(
            (
                data_point['location'][0],   # lat
                data_point['location'][1],   # lon
                -1,                          # is_true(1: true, 0: predicted, -1: grid)
                data_point['grid_location'], # name
                50                           # size
            )
        )
    return new_data


def data_to_df(data):
    pred_data = reshape_prediction_data(data['prediction_data'])
    grid_data = reshape_grid_data(data['grid'])
    print(grid_data)
    return pd.DataFrame(pred_data + grid_data, columns=['lat', 'lon', 'is_true', 'name', 'size'])


df = data_to_df(data)

fig = px.scatter_mapbox(df,
                        lat='lat',
                        lon='lon',
                        color='is_true',
                        size='size',
                        hover_data=['name'],
                        zoom=16,
                        height=800,
                        width=800)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()
