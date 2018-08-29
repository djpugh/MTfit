"""
extensions.py
******************

Utility functions for handling MTfit extensions.

Simple functions that are used throughout the module
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


import pkg_resources


def get_extensions(group, defaults=False):
    """
    Get the setuptools entrypoint installed extensions for a given entrypoint
    (group). Default values can be set using a dictionary of name:function values.

    Args
        group: str setuptools entrypoint name.

    Keyword Args
        defaults: dict dictionary of name:function pairs for default values.

    Returns
        (list,dict): tuple of extension name list and dictionary of extension name : function pairs.
    """
    names = []
    funcs = {}
    # Defaults
    if type(defaults) == dict:
        for (plugin_name, plugin) in defaults.items():
            if plugin_name not in names:
                funcs[plugin_name] = plugin
                names.append(plugin_name)
    # Entrypoints
    for entrypoint in pkg_resources.iter_entry_points(group=group):
        plugin = entrypoint.load()
        names.append(entrypoint.name.lower())
        funcs[entrypoint.name.lower()] = plugin
    names = list(set(names))
    return (names, funcs)


def evaluate_extensions(group, defaults=False, **kwargs):
    """
    Return the list of results from evaluating all the extensions in a given
    group (used for e.g. documentation entrypoint)

    Args
        group: str setuptools entrypoint name.

    Keyword Args
        defaults: dict dictionary of name:function pairs for default values.

    Returns
        list: List of results from evaluating each extensions function for the entrypoint.
    """
    results = []
    try:
        # Defaults
        if type(defaults) == dict:
            for plugin in defaults.values():
                results.append(plugin(**kwargs))
        # Entrypoints
        for entrypoint in pkg_resources.iter_entry_points(group=group):
            try:
                plugin = entrypoint.load()
                results.append(plugin(**kwargs))
            except Exception:
                pass
    except Exception:
        pass
    return results
