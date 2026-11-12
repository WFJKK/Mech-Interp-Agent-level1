x, out, ops = network_outputs()
print("inputs", x.shape, x.dtype, "forward operations", ops, "check", info["check"], "kind", info["data_set"]["kind"], "n_total", info["n_total"])
g = graph()
top = g.entry(interpretation=g.tokens().argmax())
half = g.entry(interpretation=g.tokens().argmin(), region=g.token(0) > 0.0)
for name, e in (("top", top), ("half", half)):
    m = measure(e)
    print(name, save_entry(name, e), round(m["coverage"], 2), m["fidelity_low"] <= m["fidelity"], round(m["cost"], 4))
