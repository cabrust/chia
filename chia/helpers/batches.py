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


def batches_from_pair(Xs, ys, batch_size):
    complete_batches = []

    current_batch_X = []
    current_batch_y = []

    for (X, y) in zip(Xs, ys):
        current_batch_X.append(X)
        current_batch_y.append(y)
        if len(current_batch_X) == batch_size:
            complete_batches.append((current_batch_X, current_batch_y))
            current_batch_X = []
            current_batch_y = []

    if len(current_batch_X) > 0:
        complete_batches.append((current_batch_X, current_batch_y))

    return complete_batches
