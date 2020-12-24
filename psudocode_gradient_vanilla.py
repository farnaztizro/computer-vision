while True:
    Wgradient = evaluate_gradient(loss, data, W)
    W += -alpha * Wgradient