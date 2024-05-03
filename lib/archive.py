"""
Stuff that was useful at some point but is no longer used.
10/10 version control.
"""
# def dump_n_predictions(n, to_file):
#     prediction_data = []
#     for i in range(n):
#         data_point = list(fun_dl)[i]
    
#         yhat = project_lib.eval(model, data_point[0])
#         ytrue = data_point[1]

#         yhat = yhat.reshape(grid_size * grid_size,).cpu()
#         ytrue = ytrue.reshape(grid_size * grid_size,).cpu()

#         print(torch.argmax(yhat), torch.argmax(ytrue), yhat, torch.argmax(yhat) == torch.argmax(ytrue))
#         prediction_data.append({
#             "predicted": (np.array(test_data.label_to_real_location(yhat))).tolist(),
#             "true": list(test_data.labels.values())[i],
#         })

#     out_data = {
#             "prediction_data": prediction_data,
#             "grid": test_data.get_real_location_grid(),
#         }
#     with open(to_file, 'w') as f:
#         json.dump(out_data, f)
#     return out_data
