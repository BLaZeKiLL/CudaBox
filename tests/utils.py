import torch


def dtype_str_id(dtype):
    return str(dtype).removeprefix("torch.")
