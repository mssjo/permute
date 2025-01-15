# permute
Python package for permuations and permutation groups

Import with
```
import permute
```
This package contains two classes: `permutation` and `group`, the latter of which is abstract and has a number of subclasses.

A `permutation` represents the operation of rearranging objects in a sequence, and the objects support various methods for creation, inspection, composition, etc.; see the docstrings in `permute/permutation.py`.

A `group` represents a permutation group and acts as an iterable over the elements of that group, which are permutation objects.
Concrete subclasses represent the symmetric group S_n and various common subgroups, and there are helper classes for combining groups; see the docstrings in `permute/group.py`.
All versions of `group` are implemented as multi-pass iterators, and some cache information so that subsequent run-throughs are more efficient than the first.

DISCLAIMER This package has not been extensively tested, so use with care.
