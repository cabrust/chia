def batches_from(samples, batch_size):
    complete_batches = []
    current_batch = []
    for sample in samples:
        current_batch.append(sample)
        if len(current_batch) == batch_size:
            complete_batches.append(current_batch)
            current_batch = []

    if len(current_batch) > 0:
        complete_batches.append(current_batch)

    return complete_batches


def batches_from_pair(Xs, ys, ws, batch_size):
    complete_batches = []

    current_batch_X = []
    current_batch_y = []
    current_batch_w = []

    for (X, y, w) in zip(Xs, ys, ws):
        current_batch_X.append(X)
        current_batch_y.append(y)
        current_batch_w.append(w)
        if len(current_batch_X) == batch_size:
            complete_batches.append((current_batch_X, current_batch_y, current_batch_w))
            current_batch_X = []
            current_batch_y = []
            current_batch_w = []

    if len(current_batch_X) > 0:
        complete_batches.append((current_batch_X, current_batch_y, current_batch_w))

    return complete_batches
