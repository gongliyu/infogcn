from pathlib import Path
import os
import typing
import collections.abc

Paths = typing.Union[str, os.PathLike, typing.Sequence[typing.Union[str, os.PathLike]]]


def list_files(paths: Paths,
               extension: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
               recursive: bool = True,
               case_sensitive: bool = False,
               exclude_path: typing.Optional[Paths] = None,
               return_relative_dir: bool = False)-> typing.Sequence[os.PathLike]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = [Path(x).expanduser() for x in paths]
    if extension is None:
        extension = []
    if isinstance(extension, (str, bytes)):
        extension = [str(extension)]
    if not case_sensitive:
        extension = [x.lower() for x in extension]
    extension = [x if x[0] == '.' else '.' + x for x in extension if len(x) > 0]
    extension = set(extension)
    if exclude_path is None:
        exclude_path = []
    if isinstance(exclude_path, (str, os.PathLike)):
        exclude_path = [exclude_path]
    exclude_path = [Path(x).expanduser().absolute() for x in exclude_path]

    flist, relative_dir_list, Q = [], [], collections.deque()
    for p in paths:
        if any(x == p.absolute() or x in p.absolute().parents for x in exclude_path):
            continue
        if p.is_dir():
            Q.append((p, p))
        elif p.is_file():
            flist.append(p)
            if return_relative_dir:
                relative_dir_list.append(Path('.'))
        else:
            raise ValueError(f'Invalid path: {p}')
    while Q:
        (p, src_dir) = Q.popleft()
        for child in p.iterdir():
            if any(x == child.absolute() or x in child.absolute().parents for x in exclude_path):
                continue
            if child.is_dir():
                Q.append((child, src_dir))
            elif child.is_file():
                if extension:
                    ext = child.suffix if case_sensitive else child.suffix.lower()
                    if ext not in extension:
                        continue
                flist.append(child)
                if return_relative_dir:
                    relative_dir_list.append(child.parent.relative_to(src_dir))
    if return_relative_dir:
        assert len(flist) == len(relative_dir_list)
        return flist, relative_dir_list
    else:
        return flist


def split_path(p):
    """Split path into compoenents recursively

    Examples:
    >>> split_path('abc/edf/ghi.xyz')
    ('abc', 'edf', 'ghi.xyz')
    >>> split_path('/abc/def/ghi.xyz')
    ('abc', 'def', 'ghi.xyz')
    """
    rest, tail = os.path.split(p)
    if rest in ('', os.path.sep):
        return tail,
    return split_path(rest) + (tail,)


def find_fist_containing_path_pair(*args, treat_equal_as_contianing=True):
    """Find the first pair of path that one of them is parent of the other

    Examples:
    >>> p1 = 'abc/def/ghi'
    >>> p2 = 'xyz/uvw'
    >>> p3 = 'xyz/uvw/whatever'
    >>> assert find_fist_containing_path_pair(p1, p2) is None
    >>> assert find_fist_containing_path_pair([p1, p2]) is None
    >>> find_fist_containing_path_pair(p1, p2, p3)
    ('xyz/uvw', 'xyz/uvw/whatever')
    >>> find_fist_containing_path_pair([p1, p2, p3])
    ('xyz/uvw', 'xyz/uvw/whatever')
    >>> find_fist_containing_path_pair(p2, p2, treat_equal_as_contianing=True)
    ('xyz/uvw', 'xyz/uvw')
    >>> assert find_fist_containing_path_pair(p2, p2, treat_equal_as_contianing=False) is None
    """
    if not args:
        return
    if len(args) == 1:
        if isinstance(args[0], (str, os.PathLike)):
           return
        elif isinstance(args[0], collections.abc.Iterable):
            paths = list(args[0])
        else:
            raise ValueError(f'{args[0]} is neither a path or a iterable of paths.')
    else:
        paths = list(args)
    paths = args if len(args) > 1 else args[0]
    original_paths = [str(x) for x in paths]
    paths = [Path(x).expanduser().resolve() for x in paths]
    n = len(paths)
    for i, p in enumerate(paths):
        for j in range(i + 1, n):
            p2 = paths[j]
            if p in p2.parents:
                return original_paths[i], original_paths[j]
            if p2 in p.parents:
                return original_paths[j], original_paths[i]
            if treat_equal_as_contianing and p == p2:
                return original_paths[i], original_paths[j]


def assert_paths_disjoint(*args):
    """Assert whether a list of paths are disjoint

    Example:
    >>> p1 = 'abc/def/ghi'
    >>> p2 = 'xyz/uvw'
    >>> p3 = 'xyz/uvw/whatever'
    >>> assert_paths_disjoint(p1, p2)
    >>> assert_paths_disjoint(p1, p2, p3)
    Traceback (most recent call last):
      ...
    ValueError: ...
    """
    path_pair = find_fist_containing_path_pair(*args)
    if path_pair:
        raise ValueError(f'Input paths are not disjoint, {path_pair[0]} is parent of {path_pair[1]}')