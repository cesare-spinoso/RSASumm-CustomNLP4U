# TODO: Add a utility for truncating text based on model constraints


def batch_data(data: list, batch_size: int, idx: int) -> tuple[list]:
    s = slice(idx * batch_size, (idx + 1) * batch_size)
    batched_data = [elt[s] for elt in data]
    return (batched for batched in batched_data)
