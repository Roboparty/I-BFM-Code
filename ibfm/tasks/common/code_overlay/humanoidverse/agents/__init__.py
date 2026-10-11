"""Namespace overlay for checkpoint-compatible agent modules."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
