g, predict, rule = lv1.Graph.load(entries["best"], weights, info["sizes"]["n_ctx"])
whole = g.entry(predict=predict)
parts = measure(whole)
print(save_entry("whole", whole), parts["fidelity"], parts["reward"])
