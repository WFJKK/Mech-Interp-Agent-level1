def packed(names, scales):
    """The three matrices multiplied once, with those named kept as whole numbers, times a power of two."""
    g = graph()
    w = {}
    for k in ("embed", "hidden", "unembed"):
        x = g.weight(k)
        if k in names:
            x = lv1.cast(lv1.cast(x * scales[k], int), float) * (1.0 / scales[k])
        w[k] = x
    product = w["unembed"] @ (w["hidden"] @ w["embed"])
    return g.entry(interpretation=lv1.take_sum(product, g.tokens(), axis=1).argmax())


scales = {"embed": 2.0**65, "hidden": 2.0**64, "unembed": 2.0**64}
for name, names in (("packed_unembed", ["unembed"]), ("packed_hidden", ["hidden"]), ("packed_embed", ["embed"]),
                    ("packed_two", ["hidden", "unembed"])):
    e = packed(names, scales)
    print(save_entry(name, e), round(measure(e)["fidelity"], 4))
f = graph()
print(save_entry("itself", f.entry(interpretation=model.reference(f, lv1))))
