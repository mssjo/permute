
indent = 0

from abc import abstractmethod
from numpy import cumsum
from math import factorial
from functools import reduce
from copy import deepcopy
import collections.abc
from enum import Enum

from . import Permutation

class Group(collections.abc.Iterator):
    """
    Class for representing permutation groups via their generators.

    Group itself is an abstract class, and its subclasses implement specific groups.
    A Group is an iterator that iterates through the elements of a permutation group in some way,
    returning a Permutation after each iteration.
    By convention, the first permutation returned should be the identity.

    Associated with the Group class is also a set of methods for building larger groups
    through various ways of combining permutation groups.
    To aid in this, a Group should be able to iterate through its group multiple times:
    after raising StopIteration, it should be in the same state it was when it started the iteration,
    allowing the iteration to be repeated.
    """

    def __iter__(self):
        return self
    
    @abstractmethod
    def __next__(self):
        pass

    def __contains__(self, elem):
        return elem in set(self)

    def __len__(self):
        return sum(1 for _ in self)

    def primitives(self):
        """
        Find the primitives of a group.

        The primitives are the minimal set of group elements such that any other element
        is equal to a power of a primitive element.
        This can be found in linear time.
        """
        elems = set()
        prim = []

        for elem in self:
            if elem not in elems:
                prim.append(elem)
                for p in range(1, elem.order()):
                    power = elem ** p
                    elems.add(elem ** p)
                    if power in prim:
                        prim.erase(power)

        return prim

    def generating_set(self):
        """
        Find a generating set of a group.

        This is a minimal set of group elements such that any other element can be expressed
        as a product of elements of the generating set.
        Thus, Span(self.generating_set()) is the same as self, possibly iterated over in
        another order.

        This is done in a brute-force fashion by trying Spans of larger and larger subsets
        until one is found that spans the whole group, and is therefore VERY inefficient.
        """

        elements = set(self)
        candidates = [{element} for element in elements]

        while True:
            for candidate in candidates:
                if set(Span(candidate)) == elements:
                    return candidate

            candidates = [cand | {elem} for cand in candidates for elem in elements if elem not in cand]

    def is_normal_subgroup_of(self, other):
        """
        Check if self is a normal subgroup of other.

        This is done the brute-force way.
        Generating sets could be used, but since finding generating sets is also
        done brute-force I doubt it would help.
        """

        return all(n.conjugate(g) in self for n in self for g in other)

class Trivial(Group):
    """
    The trivial group, consisting of the identity permutation.

    Acts as the identity for the subgroup product and quotient.
    """

    def __init__(self, n):
        self._size = n
        self._done = False

    def __next__(self):
        if self._done:
            self._done = False
            raise StopIteration
        else:
            self._done = True
            return Permutation.identity(self._size)

    def __str__(self):
        return f'Trivial group I_{self._size}'

class Zn(Group):
    """
    The cyclic group.

    Iterates over the cyclic permutations of n elements.
    """

    def __init__(self, n):
        self._size = n
        iter(self)

    def __iter__(self):
        self._step = 0
        return self
            
    def __next__(self):
        if self._step < self._size:
            perm = Permutation( [(i+self._step) % self._size for i in range(self._size)] )
            self._step += 1
            return perm
        else:
            raise StopIteration
    
    def __str__(self):
        return f'Cyclic group Z_{self._size}'
        
        

class SnAlgorithm(Enum):
    HEAP = 1
    PLAIN = 2
    LEXICAL = 3

class Sn(Group):
    """
    The symmetric group.

    Iterates over all permutations of n elements.
    """


    def __init__(self, n, algorithm = SnAlgorithm.LEXICAL):
        self._size = n
        self._algorithm = algorithm
        iter(self)

    def __iter__(self):
        self._perm = list(range(self._size))

        match self._algorithm:
            case SnAlgorithm.HEAP:
                self._stack = [0] * self._size
                self._idx = 0
            case SnAlgorithm.PLAIN:
                self._directions = [0] + [-1]*(self._size-1)
                self._positions  = list(range(self._size))
                self._done = False
            case SnAlgorithm.LEXICAL:
                self._done = False

        return self

    def __next__(self):

        match self._algorithm:

            # Non-recursive Heap's Algorithm, fast and simple
            case SnAlgorithm.HEAP:
                return self.next_Heap()

            # Plain changes, AKA Steinhaus-Johnson-Trotter-Even algorithm
            # Also fast, produces a nicer order that loops back around to the identity.
            # Alternates parity, which makes it useful for generating the alternating group.
            case SnAlgorithm.PLAIN:
                return self.next_plain()

            # Naranyana Pandita's algorithm
            # Generates things in lexical order; slower than the others but still linear-time.
            # Note that Distributions([1,1,...,1], equivalent=False) would also work,
            # but is slow and complicated even if simplified to this special case.
            case SnAlgorithm.LEXICAL:
                return self.next_lexical()


    def _swap(self, i,j):
        self._perm[i], self._perm[j] = self._perm[j], self._perm[i]

    def next_Heap(self):

        if self._idx == 0 and self._stack[0] == 0:
            self._idx = -1
            return Permutation(self._perm)

        self._idx = 0
        while self._idx < len(self._perm):
            if self._stack[self._idx] < self._idx:
                if self._idx % 2 != 0:
                    self._swap(self._stack[self._idx], self._idx)
                else:
                    self._swap(0, self._idx)

                self._stack[self._idx] += 1

                return Permutation(self._perm)
            else:
                self._stack[self._idx] = 0
                self._idx += 1

        raise StopIteration

    def next_plain(self):

        if self._done:
            self._done = False
            raise StopIteration

        # Save the result before updating to ensure the identity comes first
        result = Permutation(self._perm)

        # Find largest element with nonzero direction, if there is none we are done.
        for i in reversed(range(self._size)):
            if self._directions[i] != 0:
                val = i
                break
        else:
            self._done = True
            return result


        # Move it one step in the direction, adjust self._positions accordingly
        pos = self._positions[val]
        self._positions[val] += self._directions[val]
        self._positions[ self._perm[pos + self._directions[val]] ] -= self._directions[val]
        self._swap(pos, pos + self._directions[val])

        # Set direction to zero if we reached the end or a greater element
        if (
            self._positions[val] == 0
            or self._positions[val] == self._size-1
            or self._perm[ self._positions[val] + self._directions[val] ] > val
            ):
            self._directions[val] = 0

        # All larger elements with zero direction start moving towards the last moved element
        for i in range(val+1, self._size):
            assert self._directions[i] == 0
            self._directions[i] = -1 if self._positions[i] > self._positions[val] else +1

        return result

    def next_lexical(self):

        if self._done:
            self._done = False
            raise StopIteration

        result = Permutation(self._perm)

        for i in reversed(range(self._size-1)):
            if self._perm[i] < self._perm[i+1]:
                k = i
                break
        else:
            self._done = True
            return result

        for i in reversed(range(k+1, self._size)):
            if self._perm[k] < self._perm[i]:
                l = i
                break

        self._swap(k, l)

        if k < self._size - 1:
            for i in range(1, 1 + (self._size-k) // 2):
                self._swap(k+i, self._size - i)

        return result




    def __str__(self):
        return f'Symmetric group S_{len(self._perm)}'

class An(Group):
    """
    The alternating group.

    Iterates over the all even permutations of n elements.
    """

    def __init__(self, n):
        self._Sn = Sn(n, SnAlgorithm.PLAIN)

    def __iter__(self):
        self._Sn.__iter__()
        return self

    def __next__(self):
        # Plain changes alternate the parities, so just iterate an extra time to skip the odd ones.
        perm = next(self._Sn)
        next(self._Sn)
        return perm

    def __str__(self):
        return f'Alternating group A_{len(self._perm)}'

class Distributions(Group):
    """
    The group of ways to distribute elements into subsets of a given size,
    with no regard to the order within each subset.
    Counting subsets of equal size as equivalent is optional.
    """

    def __init__(self, subsets, equivalent = True):
        """
        Initialize the group.

        subsets -- a sequence of positive integers representing the lengths of the subsets in order.
            The length of the permutations in the group is the sum of these.
        equivalent -- if True (the default), subsets of equal size are counted as equivalent.
            If False, they are not.
            Alternatively, this may be a sequence of the same length as subsets.
            Then subsets are only counted as equivalent when the corresponding entries in
            equivalent are equal; falsey values are not considered equal even if they are.
        """

        if not all(sub > 0 for sub in subsets):
            raise ValueError("Nonpositive subset size")

        # Convert all versions of equivalent to the array form
        if not isinstance(equivalent, collections.abc.Sequence):
            equivalent = [bool(equivalent)] * len(subsets)
        elif len(equivalent) != len(subsets):
            raise ValueError(f"Equivalence array length ({len(equivalent)}) does not match subset array length ({len(subset)})")

        # Assign each subset an offset pointing to the preceding subset that is equivalent to it
        # (if any, otherwise zero)
        self._equiv = [0] * len(subsets)
        for i in range(len(subsets)):
            for j in reversed(range(i)):
                if subsets[i] == subsets[j] and equivalent[i] and equivalent[j] and equivalent[i] == equivalent[j]:
                    self._equiv[i] = j - i
                    break

        self._subsets = deepcopy(subsets)                   # subset lengths
        self._start_index = [0] + list(cumsum(subsets)[:-1])# start index of each subset
        iter(self)

    def __iter__(self):
        self._fill = [0] * len(self._subsets)           # number of elements currently put in subset
        self._placement = []                            # location (index and subset index)
                                                        #  of each element in order
        self._current = list(range(sum(self._subsets))) # current permutation
        return self


    def __next__(self):

        def try_push(elem, sub):
            nonlocal self
            # Reject placements that put indices in a subset out of order
            if any(self._current[self._start_index[sub] + i] > elem for i in range(self._fill[sub])):
                #print(f"Can't put {elem}: subset {sub} would be out of order")
                return False
            # Reject placements that put equivalent subsets out of order
            # Since indices are placed in order, this is judged as follows:
            # When placing the first element in a set, the previous set in the equivalence sequence
            #  (if any) must not be empty.
            if not self._fill[sub] and self._equiv[sub] != 0 and not self._fill[sub + self._equiv[sub]]:
                #print(f"Can't put {elem}: subset {sub + self._equiv[sub]} is equivalent")
                return False

            index = self._start_index[sub] + self._fill[sub]
            #print(f"Put {elem} at {index} (element {self._fill[sub]} in subset {sub})")

            self._fill[sub] += 1
            self._placement.append( (index, sub) )
            self._current[index] = elem

            return True

        def pop():
            nonlocal self
            (_, oldsub) = self._placement.pop()
            self._fill[oldsub] -= 1
            return oldsub

        # Move the last element to the next valid location.
        # If there is none, remove it altogether and repeat for the second-to-last, and so on.
        if self._placement:

            # Unconditionally remove the last element, since it can never be moved.
            pop()

            update = True
            while update:

                # If we run out of elements, we are done
                if not self._placement:
                    raise StopIteration

                oldsub = pop()
                elem = len(self._placement)

                for sub in range(oldsub+1, len(self._subsets)):
                    if self._fill[sub] < self._subsets[sub] and try_push(elem, sub):
                        update = False
                        break

        # Fill the permutation back up by placing each unassigned element
        #  in its first valid location
        while len(self._placement) < sum(self._subsets):
            elem = len(self._placement)
            for sub in range(len(self._subsets)):
                if self._fill[sub] < self._subsets[sub] and try_push(elem, sub):
                    break
            else:
                assert False, "Unreachable code reached, please reconsider life choices"

        return Permutation(self._current)

class Offset(Group):
    """
    Modified Group producing offset permutations.

    Wraps an existing Group and produces permutations offset by a given amount.
    """
    def __init__(self, group, offs):
        self._group = group
        self._offs = offs
        
    def __iter__(self):
        self._group.__iter__()
        return self

    def __next__(self):
        return next(self._group).offset(self._offs)
    
    def __str__(self):
        return f'{self._group} offset by {self._offs}'
    
class Block(Group):
    """
    Modified Group producing block permutations.

    Wraps an existing Group and produces permutations that act on blocks of a given size.
    """
    def __init__(self, group, block_len):
        self._group = group
        self._block_len = block_len
        
    def __iter__(self):
        self._group.__iter__()
        return self

    def __next__(self):
        return next(self._group).block(self._block_len)
    
    def __str__(self):
        return f'{self._group} on blocks of {self._block_len}'

class DirectProduct(Group):
    """
    Direct product of Groups.

    Wraps a set of Groups to generate the permutations in the direct product of their permutation groups.
    Each iterate is the concatenation of the iterates of all individual Groups,
    and the iteration is done so that one Group goes through a full cycle of iteration before the next
    one is iterated one step.
    """

    def __init__(self, *groups):
                
        self._groups = groups
        iter(self)
        
    def __iter__(self):
        self._done = False
        for g in self._groups:
            iter(g)

        try:
            self._perms = [next(group) for group in self._groups]
        except StopIteration:
            raise ValueError("Product contains invalid group")

        return self
        
    def __next__(self):
        if self._done:
            self._done = False
            raise StopIteration

        result = Permutation.concatenate(*self._perms)

        for i in reversed(range(len(self._groups))):
            try:
                self._perms[i] = next(self._groups[i])
                break
            except StopIteration:
                try:
                    iter(self._groups[i])
                    self._perms[i] = next(self._groups[i])
                except StopIteration:
                    raise ValueError("Product contains invalid group")
        else:
            self._done = True
        
        return result
    
    def __str__(self):
        return f'Direct product of ({") and (".join(str(gen) for gen in self._groups)})'
        
class SubgroupProduct(Group):
    """
    Subgroup product of Groups.

    Wraps a set of Groups to generate the permutations in the subgroup product
    of their permutation groups, which must consist of permutations of the same size.
    Each iterate is the composition of the iterates of all individual Groups,
    and the iteration is done so that one Group goes through a full cycle of iteration
    before the next one is iterated one step.
    """

    def __init__(self, *groups):

        self._groups = groups
        iter(self)

    def __iter__(self):
        self._done = False
        for g in self._groups:
            iter(g)

        try:
            self._perms = [next(group) for group in self._groups]
        except StopIteration:
            raise ValueError("Product contains invalid group")

        return self

    def __next__(self):
        if self._done:
            self._done = False
            raise StopIteration

        result = Permutation.compose(*self._perms)

        for i in reversed(range(len(self._groups))):
            try:
                self._perms[i] = next(self._groups[i])
                break
            except StopIteration:
                try:
                    iter(self._groups[i])
                    self._perms[i] = next(self._groups[i])
                except StopIteration:
                    raise ValueError("Product contains invalid group")
        else:
            self._done = True

        return result
    
    
    def __str__(self):
        global indent
        indent += 1
        
        sep = '\n' + '\t'*indent
        sub = sep.join([str(g) for g in self._groups])
        
        indent -= 1
        
        return f'Subgroup product of ({") and (".join(str(gen) for gen in self._groups)})'

class ZR(SubgroupProduct):
    """
    The flavor structure symmetry group.

    Given a list R of positive integers, which is sorted before continuing,
    it is the subgroup product of two groups:
    one is the direct product of Zn(n) for each n in R,
    the other is the direct product of Sn(m) for each length-m stretch of equal n in R,
    acting on blocks of size n.
    This makes it the symmetry group of a product of traces of matrices under permutation of
    those matrices, where R lists the size of the traces.
    """

    def __init__(self, R:[int]):
        list.sort(R)

        Zns = []
        Sns = []

        count = 0
        for i in range(0, len(R)):
            Zns.append(Zn(R[i]))

            count += 1
            if i == len(R)-1 or R[i] != R[i+1]:

                Sns.append( Block(Sn(count), R[i]) )
                count = 0

        return super().__init__( DirectProduct(*Zns), DirectProduct(*Sns) )

class Span:
    """
    The group spanned by a set of permutations (the generating set, "genset" for short).

    This is not a subclass of Group, since its iterators return tuples consisting
    of a permutation and a description of the product of genset elements used to produce it.
    A subclass is provided that only returns the permutations, making it a proper Group.
    """
    
    def __init__(self, genset):
        """
        Initialize a span.

        genset -- a collection of permutations generating the group.
            It may either be a dict indexing the permutations by custom names,
            or it may be a plain sequence, in which case they are named g1,g2,...
            All genset elements must be permutations of the same size.
        """
        
        if not genset:
            raise ValueError("Empty genset not allowed (can't tell what identity element would be)")
        
        # Allow anonymous generators in a list, name them g1,g2,...
        if not isinstance(genset, collections.abc.Mapping):
            self.__init__({f"g{i}" : p for i,p in enumerate(genset)})
            return
        
        lens = [len(perm) for perm in genset.values()]
        if any(lens[i] != lens[0] for i in range(1, len(genset))):
            raise ValueError("All genset elements must match in size")
        
        identity = Permutation.identity( lens[0] )
        
        self._genset = genset
        
        self._span = {identity : []}
        
        self._to_extend = {perm : [elem] for elem,perm in genset.items()}
        self._span.update(self._to_extend) # Just so the first iteration is in a better order
        
        self._extended = {}
        
        self._genset_iter = iter(self._genset.items())
        self._span_iter  = iter(self._span.items())
        
        try:
            self._elem_to_apply = next( self._genset_iter )
        except StopIteration:
            # Happens if genset is empty.
            self._value_to_extend = None
        else:
            self._value_to_extend = self._to_extend.popitem()

    def __len__(self):
        return len(self.permutations())
    def __iter__(self):
        return self

    def __next__(self) -> (Permutation, list):
        """
        Get the next permutation in the span, and the corresponding product of genset elements.

        The product of genset elements is given as a list of genset element names, in the order
        they are to be multiplied (commutativity is not taken into account).
        This is guaranteed to be the shortest possible such product, and that if it is not unique,
        that it is the lexicographically least one according to the order in which the genset elements
        were originally provided.

        The order of iteration is first by length of products, then lexicographically by genset element
        according to the order in which they were originally provided.

        This works by trying all words constructed from the alphabet given by the genset,
        and keeping track of the permutations thus produced.
        The words are constructed letter by letter, and only words that produce a novel
        permutation are included and considered for further extension.
        This guarantees that the iteration will terminate after visiting each element in the span
        exactly once, as well as the minimality and ordering stated above.
        The downside is that all iterates have to be stored, rather than generating them on the fly.
        After the first round of iterations, the stored iterates are just iterated through
        rather than using the genset to generate them again.
        """
        
        # This bit does double duty.
        # The first time around, it iterates through the basis before extending.
        # Subsequent times, it iterates through the saved span.
        if self._span_iter:
            try:
                return next(self._span_iter)
            except StopIteration:
                self._span_iter = None

                if not self._value_to_extend:
                    raise StopIteration
        
        # Iterate over all values to extend
        while self._value_to_extend:
            perm, prod = self._value_to_extend
            
            # Try all ways of extending this value
            while self._genset_iter:
                
                name, elem = self._elem_to_apply
                new_perm = perm * elem
                
                # Continue iteration over genset
                try:
                    self._elem_to_apply = next(self._genset_iter)
                except StopIteration:
                    self._genset_iter = None
                
                # Only accept if this produces a new permutation
                if new_perm not in self._span and new_perm not in self._extended:
                    new_prod = prod + [name]
                    
                    self._extended[new_perm] = new_prod
                    
                    return new_perm, new_prod
                
            # Reset genset iteration
            self._genset_iter = iter(self._genset.items())
            try:
                self._elem_to_apply = next(self._genset_iter)
            except StopIteration:
                break
            
            # Fetch next value
            if self._to_extend:
                self._value_to_extend = self._to_extend.popitem() 
                
            else:
                break
            
        # This competes one cycle of extensions.
        # Save the new values and prepare for the next cycle.
        self._span.update(self._extended)
        self._to_extend, self._extended = self._extended, self._to_extend
        
        if self._to_extend:
            self._value_to_extend = self._to_extend.popitem()
        
            # Continue recursively because who can be arsed to wrap everything in another while loop?
            return next(self)
        
        # If no more extensions are possible, we are done!
        # For subsequent iterations, just set up an iterator to saved values
        else:
            self._span_iter  = iter(self._span.items())
            raise StopIteration
            
    def permutations(self):
        """ Obtain the Group of permutations in the span, ignoring the genset element products. """
        class Permutations(Group):
            def __init__(self, span):
                self._span = span
                
            def __next__(self):
                perm, _ = next(self._span)
                return perm

            def __str__(self):
                return str(self._span)

        return Permutations(self)
        
        
    def genset_products(self):
        """ Obtain an iterator over the genset element products in the span, ignoring the permutations. """
        class BasisProducts:
            def __init__(self, span):
                self._span = span
                
            def __next__(self):
                _, prod = next(self._span)
                return prod

            def __str__(self):
                return str(self._span)

        return BasisProducts(self)

    def __str__(self):
        return f"Span of {', '.join(str(gen) for gen in self._genset)}"

class Quotient(Group):
    """
    Quotient group.

    Given two groups G and N, the quotient group G/N consists of one representative from each
    equivalence class in the G under (right) composition with N.
    This is only a true group if N is a normal subgroup of G.
    The representative is simply the first member of the equivalence class that appears
    when iterating through G.
    The first time G/N is iterated through, representatives are found in a brute-force manner
    by composing successive elements of G with N and skipping those that are in the same
    equivalence class as one that has already been returned.
    In subsequent iterations, the list of representatives is simply iterated through.
    """

    def __init__(self, G, N):
        self._num = G
        self._den = N
        self._reps = set()
        self._rep_iter = None

    def __next__(self):

        try:
            if self._rep_iter:
                return next(self._rep_iter)

            while True:
                n = next(self._num)

                for r in self._reps:
                    assert r in self._reps

                for d in self._den:
                    if n*d in self._reps:
                        break
                else:
                    self._reps.add(n)
                    return n

        except StopIteration:
            self._rep_iter = iter(self._reps)
            raise StopIteration

    def __str__(self):
        return f'Quotient of ({self._num}) and ({self._den})'


