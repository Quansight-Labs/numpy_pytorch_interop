"""
Utility function to facilitate testing.

"""
from __future__ import annotations

import contextlib
import gc
import platform
import re
import sys
import warnings
from functools import wraps
from warnings import WarningMessage

import torch

import torch_np as np
from torch_np import arange, array
from torch_np import asarray as asanyarray
from torch_np import empty, float32, intp, ndarray
from torch_np._normalizations import ArrayLike, normalizer

__all__ = [
    "assert_equal",
    "assert_almost_equal",
    "assert_approx_equal",
    "assert_array_equal",
    "assert_",
    "assert_array_almost_equal",
    "assert_",
    "assert_raises_regex",
    "assert_warns",
    "assert_allclose",
    "suppress_warnings",
]


verbose = 0

IS_WASM = platform.machine() in ["wasm32", "wasm64"]
IS_PYPY = sys.implementation.name == "pypy"
IS_PYSTON = hasattr(sys, "pyston_version_info")
HAS_REFCOUNT = getattr(sys, "getrefcount", None) is not None and not IS_PYSTON


def assert_(val, msg=""):
    """
    Assert that works in release mode.
    Accepts callable msg to allow deferring evaluation until failure.

    The Python built-in ``assert`` does not work when executing code in
    optimized mode (the ``-O`` flag) - no byte-code is generated for it.

    For documentation on usage, refer to the Python documentation.

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    if not val:
        try:
            smsg = msg()
        except TypeError:
            smsg = msg
        raise AssertionError(smsg)


def build_err_msg(
    arrays,
    err_msg,
    header="Items are not equal:",
    verbose=True,
    names=("ACTUAL", "DESIRED"),
    precision=8,
):
    msg = ["\n" + header]
    if err_msg:
        if err_msg.find("\n") == -1 and len(err_msg) < 79 - len(header):
            msg = [msg[0] + " " + err_msg]
        else:
            msg.append(err_msg)
    if verbose:
        for i, a in enumerate(arrays):
            if isinstance(a, ndarray):
                # precision argument is only needed if the objects are ndarrays
                # r_func = partial(array_repr, precision=precision)
                r_func = ndarray.__repr__
            else:
                r_func = repr

            try:
                r = r_func(a)
            except Exception as exc:
                r = f"[repr failed for <{type(a).__name__}>: {exc}]"
            if r.count("\n") > 3:
                r = "\n".join(r.splitlines()[:3])
                r += "..."
            msg.append(f" {names[i]}: {r}")
    return "\n".join(msg)


def assert_equal(actual, desired, err_msg="", verbose=True):
    __tracebackhide__ = True  # Hide traceback for py.test

    num_nones = sum([actual is None, desired is None])
    if num_nones == 1:
        raise AssertionError(f"Not equal: {actual} != {desired}")
    elif num_nones == 2:
        return True
    # else, carry on

    if isinstance(actual, np.DType) or isinstance(desired, np.DType):
        result = actual == desired
        if not result:
            raise AssertionError(f"Not equal: {actual} != {desired}")
        else:
            return True

    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k, i in desired.items():
            if k not in actual:
                raise AssertionError(repr(k))
            assert_equal(actual[k], desired[k], f"key={k!r}\n{err_msg}", verbose)
        return

    from torch_np._normalizations import normalize_array_like

    actual = normalize_array_like(actual)
    desired = normalize_array_like(desired)
    return (actual == desired).all()


assert_array_equal = assert_equal


@normalizer
def assert_almost_equal(
    actual: ArrayLike, desired: ArrayLike, decimal=7, err_msg="", verbose=True
):
    atol = 1.5 * 10.0 ** (-decimal)
    if actual.ndim == 0 or desired.ndim == 0:
        actual, desired = torch.broadcast_tensors(actual, desired)
    return torch.testing.assert_close(
        actual, desired, atol=atol, rtol=0, msg=err_msg, check_dtype=False
    )


assert_array_almost_equal = assert_almost_equal


import unittest


class _Dummy(unittest.TestCase):
    def nop(self):
        pass


_d = _Dummy("nop")


def assert_raises_regex(exception_class, expected_regexp, *args, **kwargs):
    """
    assert_raises_regex(exception_class, expected_regexp, callable, *args,
                        **kwargs)
    assert_raises_regex(exception_class, expected_regexp)

    Fail unless an exception of class exception_class and with message that
    matches expected_regexp is thrown by callable when invoked with arguments
    args and keyword arguments kwargs.

    Alternatively, can be used as a context manager like `assert_raises`.

    Notes
    -----
    .. versionadded:: 1.9.0

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    return _d.assertRaisesRegex(exception_class, expected_regexp, *args, **kwargs)


@normalizer
def assert_allclose(
    actual: ArrayLike,
    desired: ArrayLike,
    rtol=1e-7,
    atol=0,
    equal_nan=True,
    err_msg="",
    verbose=True,
    check_dtype=False,
):
    if actual.ndim == 0 or desired.ndim == 0:
        actual, desired = torch.broadcast_tensors(actual, desired)
    return torch.testing.assert_close(
        actual,
        desired,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        msg=err_msg,
        check_dtype=check_dtype,
    )


@contextlib.contextmanager
def _assert_warns_context(warning_class, name=None):
    __tracebackhide__ = True  # Hide traceback for py.test
    with suppress_warnings() as sup:
        l = sup.record(warning_class)
        yield
        if not len(l) > 0:
            name_str = f" when calling {name}" if name is not None else ""
            raise AssertionError("No warning raised" + name_str)


def assert_warns(warning_class, *args, **kwargs):
    """
    Fail unless the given callable throws the specified warning.

    A warning of class warning_class should be thrown by the callable when
    invoked with arguments args and keyword arguments kwargs.
    If a different type of warning is thrown, it will not be caught.

    If called with all arguments other than the warning class omitted, may be
    used as a context manager:

        with assert_warns(SomeWarning):
            do_something()

    The ability to be used as a context manager is new in NumPy v1.11.0.

    .. versionadded:: 1.4.0

    Parameters
    ----------
    warning_class : class
        The class defining the warning that `func` is expected to throw.
    func : callable, optional
        Callable to test
    *args : Arguments
        Arguments for `func`.
    **kwargs : Kwargs
        Keyword arguments for `func`.

    Returns
    -------
    The value returned by `func`.

    Examples
    --------
    >>> import warnings
    >>> def deprecated_func(num):
    ...     warnings.warn("Please upgrade", DeprecationWarning)
    ...     return num*num
    >>> with np.testing.assert_warns(DeprecationWarning):
    ...     assert deprecated_func(4) == 16
    >>> # or passing a func
    >>> ret = np.testing.assert_warns(DeprecationWarning, deprecated_func, 4)
    >>> assert ret == 16
    """
    if not args:
        return _assert_warns_context(warning_class)

    func = args[0]
    args = args[1:]
    with _assert_warns_context(warning_class, name=func.__name__):
        return func(*args, **kwargs)


@contextlib.contextmanager
def _assert_no_warnings_context(name=None):
    __tracebackhide__ = True  # Hide traceback for py.test
    with warnings.catch_warnings(record=True) as l:
        warnings.simplefilter("always")
        yield
        if len(l) > 0:
            name_str = f" when calling {name}" if name is not None else ""
            raise AssertionError(f"Got warnings{name_str}: {l}")


def assert_no_warnings(*args, **kwargs):
    """
    Fail if the given callable produces any warnings.

    If called with all arguments omitted, may be used as a context manager:

        with assert_no_warnings():
            do_something()

    The ability to be used as a context manager is new in NumPy v1.11.0.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    func : callable
        The callable to test.
    \\*args : Arguments
        Arguments passed to `func`.
    \\*\\*kwargs : Kwargs
        Keyword arguments passed to `func`.

    Returns
    -------
    The value returned by `func`.

    """
    if not args:
        return _assert_no_warnings_context()

    func = args[0]
    args = args[1:]
    with _assert_no_warnings_context(name=func.__name__):
        return func(*args, **kwargs)


def _gen_alignment_data(dtype=float32, type="binary", max_size=24):
    """
    generator producing data with different alignment and offsets
    to test simd vectorization

    Parameters
    ----------
    dtype : dtype
        data type to produce
    type : string
        'unary': create data for unary operations, creates one input
                 and output array
        'binary': create data for unary operations, creates two input
                 and output array
    max_size : integer
        maximum size of data to produce

    Returns
    -------
    if type is 'unary' yields one output, one input array and a message
    containing information on the data
    if type is 'binary' yields one output array, two input array and a message
    containing information on the data

    """
    ufmt = "unary offset=(%d, %d), size=%d, dtype=%r, %s"
    bfmt = "binary offset=(%d, %d, %d), size=%d, dtype=%r, %s"
    for o in range(3):
        for s in range(o + 2, max(o + 3, max_size)):
            if type == "unary":
                inp = lambda: arange(s, dtype=dtype)[o:]
                out = empty((s,), dtype=dtype)[o:]
                yield out, inp(), ufmt % (o, o, s, dtype, "out of place")
                d = inp()
                yield d, d, ufmt % (o, o, s, dtype, "in place")
                yield out[1:], inp()[:-1], ufmt % (
                    o + 1,
                    o,
                    s - 1,
                    dtype,
                    "out of place",
                )
                yield out[:-1], inp()[1:], ufmt % (
                    o,
                    o + 1,
                    s - 1,
                    dtype,
                    "out of place",
                )
                yield inp()[:-1], inp()[1:], ufmt % (o, o + 1, s - 1, dtype, "aliased")
                yield inp()[1:], inp()[:-1], ufmt % (o + 1, o, s - 1, dtype, "aliased")
            if type == "binary":
                inp1 = lambda: arange(s, dtype=dtype)[o:]
                inp2 = lambda: arange(s, dtype=dtype)[o:]
                out = empty((s,), dtype=dtype)[o:]
                yield out, inp1(), inp2(), bfmt % (o, o, o, s, dtype, "out of place")
                d = inp1()
                yield d, d, inp2(), bfmt % (o, o, o, s, dtype, "in place1")
                d = inp2()
                yield d, inp1(), d, bfmt % (o, o, o, s, dtype, "in place2")
                yield out[1:], inp1()[:-1], inp2()[:-1], bfmt % (
                    o + 1,
                    o,
                    o,
                    s - 1,
                    dtype,
                    "out of place",
                )
                yield out[:-1], inp1()[1:], inp2()[:-1], bfmt % (
                    o,
                    o + 1,
                    o,
                    s - 1,
                    dtype,
                    "out of place",
                )
                yield out[:-1], inp1()[:-1], inp2()[1:], bfmt % (
                    o,
                    o,
                    o + 1,
                    s - 1,
                    dtype,
                    "out of place",
                )
                yield inp1()[1:], inp1()[:-1], inp2()[:-1], bfmt % (
                    o + 1,
                    o,
                    o,
                    s - 1,
                    dtype,
                    "aliased",
                )
                yield inp1()[:-1], inp1()[1:], inp2()[:-1], bfmt % (
                    o,
                    o + 1,
                    o,
                    s - 1,
                    dtype,
                    "aliased",
                )
                yield inp1()[:-1], inp1()[:-1], inp2()[1:], bfmt % (
                    o,
                    o,
                    o + 1,
                    s - 1,
                    dtype,
                    "aliased",
                )


class suppress_warnings:
    """
    Context manager and decorator doing much the same as
    ``warnings.catch_warnings``.

    However, it also provides a filter mechanism to work around
    https://bugs.python.org/issue4180.

    This bug causes Python before 3.4 to not reliably show warnings again
    after they have been ignored once (even within catch_warnings). It
    means that no "ignore" filter can be used easily, since following
    tests might need to see the warning. Additionally it allows easier
    specificity for testing warnings and can be nested.

    Parameters
    ----------
    forwarding_rule : str, optional
        One of "always", "once", "module", or "location". Analogous to
        the usual warnings module filter mode, it is useful to reduce
        noise mostly on the outmost level. Unsuppressed and unrecorded
        warnings will be forwarded based on this rule. Defaults to "always".
        "location" is equivalent to the warnings "default", match by exact
        location the warning warning originated from.

    Notes
    -----
    Filters added inside the context manager will be discarded again
    when leaving it. Upon entering all filters defined outside a
    context will be applied automatically.

    When a recording filter is added, matching warnings are stored in the
    ``log`` attribute as well as in the list returned by ``record``.

    If filters are added and the ``module`` keyword is given, the
    warning registry of this module will additionally be cleared when
    applying it, entering the context, or exiting it. This could cause
    warnings to appear a second time after leaving the context if they
    were configured to be printed once (default) and were already
    printed before the context was entered.

    Nesting this context manager will work as expected when the
    forwarding rule is "always" (default). Unfiltered and unrecorded
    warnings will be passed out and be matched by the outer level.
    On the outmost level they will be printed (or caught by another
    warnings context). The forwarding rule argument can modify this
    behaviour.

    Like ``catch_warnings`` this context manager is not threadsafe.

    Examples
    --------

    With a context manager::

        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Some text")
            sup.filter(module=np.ma.core)
            log = sup.record(FutureWarning, "Does this occur?")
            command_giving_warnings()
            # The FutureWarning was given once, the filtered warnings were
            # ignored. All other warnings abide outside settings (may be
            # printed/error)
            assert_(len(log) == 1)
            assert_(len(sup.log) == 1)  # also stored in log attribute

    Or as a decorator::

        sup = np.testing.suppress_warnings()
        sup.filter(module=np.ma.core)  # module must match exactly
        @sup
        def some_function():
            # do something which causes a warning in np.ma.core
            pass
    """

    def __init__(self, forwarding_rule="always"):
        self._entered = False

        # Suppressions are either instance or defined inside one with block:
        self._suppressions = []

        if forwarding_rule not in {"always", "module", "once", "location"}:
            raise ValueError("unsupported forwarding rule.")
        self._forwarding_rule = forwarding_rule

    def _clear_registries(self):
        if hasattr(warnings, "_filters_mutated"):
            # clearing the registry should not be necessary on new pythons,
            # instead the filters should be mutated.
            warnings._filters_mutated()
            return
        # Simply clear the registry, this should normally be harmless,
        # note that on new pythons it would be invalidated anyway.
        for module in self._tmp_modules:
            if hasattr(module, "__warningregistry__"):
                module.__warningregistry__.clear()

    def _filter(self, category=Warning, message="", module=None, record=False):
        if record:
            record = []  # The log where to store warnings
        else:
            record = None
        if self._entered:
            if module is None:
                warnings.filterwarnings("always", category=category, message=message)
            else:
                module_regex = module.__name__.replace(".", r"\.") + "$"
                warnings.filterwarnings(
                    "always", category=category, message=message, module=module_regex
                )
                self._tmp_modules.add(module)
                self._clear_registries()

            self._tmp_suppressions.append(
                (category, message, re.compile(message, re.I), module, record)
            )
        else:
            self._suppressions.append(
                (category, message, re.compile(message, re.I), module, record)
            )

        return record

    def filter(self, category=Warning, message="", module=None):
        """
        Add a new suppressing filter or apply it if the state is entered.

        Parameters
        ----------
        category : class, optional
            Warning class to filter
        message : string, optional
            Regular expression matching the warning message.
        module : module, optional
            Module to filter for. Note that the module (and its file)
            must match exactly and cannot be a submodule. This may make
            it unreliable for external modules.

        Notes
        -----
        When added within a context, filters are only added inside
        the context and will be forgotten when the context is exited.
        """
        self._filter(category=category, message=message, module=module, record=False)

    def record(self, category=Warning, message="", module=None):
        """
        Append a new recording filter or apply it if the state is entered.

        All warnings matching will be appended to the ``log`` attribute.

        Parameters
        ----------
        category : class, optional
            Warning class to filter
        message : string, optional
            Regular expression matching the warning message.
        module : module, optional
            Module to filter for. Note that the module (and its file)
            must match exactly and cannot be a submodule. This may make
            it unreliable for external modules.

        Returns
        -------
        log : list
            A list which will be filled with all matched warnings.

        Notes
        -----
        When added within a context, filters are only added inside
        the context and will be forgotten when the context is exited.
        """
        return self._filter(
            category=category, message=message, module=module, record=True
        )

    def __enter__(self):
        if self._entered:
            raise RuntimeError("cannot enter suppress_warnings twice.")

        self._orig_show = warnings.showwarning
        self._filters = warnings.filters
        warnings.filters = self._filters[:]

        self._entered = True
        self._tmp_suppressions = []
        self._tmp_modules = set()
        self._forwarded = set()

        self.log = []  # reset global log (no need to keep same list)

        for cat, mess, _, mod, log in self._suppressions:
            if log is not None:
                del log[:]  # clear the log
            if mod is None:
                warnings.filterwarnings("always", category=cat, message=mess)
            else:
                module_regex = mod.__name__.replace(".", r"\.") + "$"
                warnings.filterwarnings(
                    "always", category=cat, message=mess, module=module_regex
                )
                self._tmp_modules.add(mod)
        warnings.showwarning = self._showwarning
        self._clear_registries()

        return self

    def __exit__(self, *exc_info):
        warnings.showwarning = self._orig_show
        warnings.filters = self._filters
        self._clear_registries()
        self._entered = False
        del self._orig_show
        del self._filters

    def _showwarning(
        self, message, category, filename, lineno, *args, use_warnmsg=None, **kwargs
    ):
        for cat, _, pattern, mod, rec in (self._suppressions + self._tmp_suppressions)[
            ::-1
        ]:
            if issubclass(category, cat) and pattern.match(message.args[0]) is not None:
                if mod is None:
                    # Message and category match, either recorded or ignored
                    if rec is not None:
                        msg = WarningMessage(
                            message, category, filename, lineno, **kwargs
                        )
                        self.log.append(msg)
                        rec.append(msg)
                    return
                # Use startswith, because warnings strips the c or o from
                # .pyc/.pyo files.
                elif mod.__file__.startswith(filename):
                    # The message and module (filename) match
                    if rec is not None:
                        msg = WarningMessage(
                            message, category, filename, lineno, **kwargs
                        )
                        self.log.append(msg)
                        rec.append(msg)
                    return

        # There is no filter in place, so pass to the outside handler
        # unless we should only pass it once
        if self._forwarding_rule == "always":
            if use_warnmsg is None:
                self._orig_show(message, category, filename, lineno, *args, **kwargs)
            else:
                self._orig_showmsg(use_warnmsg)
            return

        if self._forwarding_rule == "once":
            signature = (message.args, category)
        elif self._forwarding_rule == "module":
            signature = (message.args, category, filename)
        elif self._forwarding_rule == "location":
            signature = (message.args, category, filename, lineno)

        if signature in self._forwarded:
            return
        self._forwarded.add(signature)
        if use_warnmsg is None:
            self._orig_show(message, category, filename, lineno, *args, **kwargs)
        else:
            self._orig_showmsg(use_warnmsg)

    def __call__(self, func):
        """
        Function decorator to apply certain suppressions to a whole
        function.
        """

        @wraps(func)
        def new_func(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return new_func
