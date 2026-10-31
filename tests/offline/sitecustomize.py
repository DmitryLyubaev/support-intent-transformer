"""Inherited by Python test children; loopback HTTP is allowed, external sockets are not."""

import socket

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex


def _allowed(address):
    return not isinstance(address, tuple) or address[0] in ("127.0.0.1", "::1", "localhost")


def _offline_connect(sock, address):
    if not _allowed(address):
        raise OSError("Offline software check rejected external connection")
    return _connect(sock, address)


def _offline_connect_ex(sock, address):
    if not _allowed(address):
        raise OSError("Offline software check rejected external connection")
    return _connect_ex(sock, address)


socket.socket.connect = _offline_connect
socket.socket.connect_ex = _offline_connect_ex
