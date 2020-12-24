while True:
    batch = next_training_batch(data, 256)
    Wgradient = evaluate_gradient(loss, batch, W)
    W += -alpha *Wgradient