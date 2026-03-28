"""Helpers to persist FALCO objects in HDF5."""

from __future__ import annotations

from pathlib import Path
import types

import h5py
import numpy as np

import falco


_NODE_TYPE_ATTR = "node_type"


def save_object_to_hdf5(obj, file_path: str | Path) -> None:
    """Save a nested FALCO object tree to an HDF5 file.

    Args:
        obj: Object to serialize.
        file_path: Destination HDF5 path.
    """
    with h5py.File(file_path, "w") as h5_file:
        _write_node(h5_file, "root", obj)


def load_object_from_hdf5(file_path: str | Path):
    """Load an object tree produced by ``save_object_to_hdf5``.

    Args:
        file_path: Source HDF5 path.

    Returns:
        The reconstructed object.
    """
    with h5py.File(file_path, "r") as h5_file:
        return _read_node(h5_file["root"])


def _write_node(parent, name: str, value) -> None:
    if isinstance(value, falco.config.Object):
        group = parent.create_group(name)
        group.attrs[_NODE_TYPE_ATTR] = "falco_object"
        for key, sub_value in value.data.items():
            _write_node(group, str(key), sub_value)
        return

    if isinstance(value, dict):
        group = parent.create_group(name)
        group.attrs[_NODE_TYPE_ATTR] = "dict"
        for key, sub_value in value.items():
            _write_node(group, str(key), sub_value)
        return

    if isinstance(value, (list, tuple)):
        group = parent.create_group(name)
        group.attrs[_NODE_TYPE_ATTR] = "tuple" if isinstance(value, tuple) else "list"
        for index, sub_value in enumerate(value):
            _write_node(group, f"item_{index:06d}", sub_value)
        return

    if isinstance(value, np.ndarray):
        dataset = parent.create_dataset(name, data=value)
        dataset.attrs[_NODE_TYPE_ATTR] = "ndarray"
        return

    if value is None:
        group = parent.create_group(name)
        group.attrs[_NODE_TYPE_ATTR] = "none"
        return

    if isinstance(value, str):
        dataset = parent.create_dataset(
            name, data=np.array(value, dtype=h5py.string_dtype("utf-8"))
        )
        dataset.attrs[_NODE_TYPE_ATTR] = "str"
        return

    if isinstance(value, bytes):
        dataset = parent.create_dataset(name, data=np.frombuffer(value, dtype=np.uint8))
        dataset.attrs[_NODE_TYPE_ATTR] = "bytes"
        return

    if isinstance(value, (np.generic, bool, int, float, complex)):
        dataset = parent.create_dataset(name, data=np.asarray(value))
        dataset.attrs[_NODE_TYPE_ATTR] = "scalar"
        return

    if isinstance(value, types.SimpleNamespace):
        group = parent.create_group(name)
        group.attrs[_NODE_TYPE_ATTR] = "simple_namespace"
        for key, sub_value in vars(value).items():
            _write_node(group, str(key), sub_value)
        return

    raise TypeError(f"Unsupported type for HDF5 serialization: {type(value)}")


def _read_node(node):
    if isinstance(node, h5py.Dataset):
        node_type = node.attrs.get(_NODE_TYPE_ATTR, "dataset")
        value = node[()]

        if node_type == "bytes":
            return bytes(np.asarray(value, dtype=np.uint8).tolist())

        if isinstance(value, np.ndarray):
            if node_type == "str":
                return value.astype(str).item() if value.shape == () else value.astype(str)
            return value

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, bytes):
            return value.decode("utf-8") if node_type == "str" else value

        return value

    node_type = node.attrs.get(_NODE_TYPE_ATTR, "group")

    if node_type == "none":
        return None

    if node_type == "falco_object":
        obj = falco.config.Object()
        for key in node.keys():
            obj[key] = _read_node(node[key])
        return obj

    if node_type == "dict":
        return {key: _read_node(node[key]) for key in node.keys()}

    if node_type in ("list", "tuple"):
        items = [_read_node(node[key]) for key in sorted(node.keys())]
        return tuple(items) if node_type == "tuple" else items

    if node_type == "simple_namespace":
        obj = types.SimpleNamespace()
        for key in node.keys():
            setattr(obj, key, _read_node(node[key]))
        return obj

    return {key: _read_node(node[key]) for key in node.keys()}
