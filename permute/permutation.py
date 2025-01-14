from math import gcd
from functools import reduce
from copy import deepcopy
from re import findall

from bidict import bidict

class Permutation:
    """
    Class for representing permutations.

    A Permutation object represents a reordering of the elements of a list or similar container,
    and can be applied to any list of the same length as the permutation.
    This class implements a set of common operations on permutation, such as composition, inversion,
    inspection of order, parity and cycle type, and so on.
    """
    
#-- Magic methods --#

    def __init__(self, perm = None, size = None, check = True):
        """
        Create a permutation that permutes the elements of lists.

        perm -- the permutation moves the element at index index perm[i] to index i.
        size -- if perm is omitted, create the identity permutation of size elements.
                    If perm is provided, size is only used for verification.
        check -- if True, verify that perm is actually a permutation and that
                    len(perm) == size (if size is given).
        """

        if perm is None:
            if size is None:
                raise ValueError("Insufficient information to create a permutation")
            if size < 0:
                raise ValueError("Permutation size must be positive")
            self._map = tuple(range(size))
        else:
            self._map = tuple(perm)

        if check:
            if size is not None and len(self._map) != size:
                raise ValueError(f"Permutation array size ({len(self._map)}) does not match the given size ({size})")
            if not self.is_valid():
                raise ValueError(f"Permutation array {self._map} is not a permutation")
            
    def __getitem__(self, idx):
        """ Obtain the image of element idx under the permutation. """
        return self._map[idx]
    
    def __call__(self, array):
        """ Apply the permutation to the array given as argument. """
        return self.permute(array)
    
    def __len__(self):
        """ Get the size of a permutation, matching the size of lists it permutes. """
        return len(self._map)
    
    def __iter__(self):
        """ Get an iterator to the underlying index map. """
        return iter(self._map)
    
    def __str__(self):
        """ Represent a permutation as (i1 i2 i3 ...) where iN is the image of index N. """
        return self.oneline_string()
    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return hash(self._map)

    def __add__(self, other):
        """ Concatenate permutations. """

        return self.append(other)
        
    def __mul__(self, other):
        """
        Compose permutations.

        p.permute(q.permute(list)) is equivalent to (q*p).permute(list); note the order.
        """
        if len(self) != len(other):
            raise ValueError("Attempring to multiply permutations of different length")

        return Permutation(other.permute(self))
    
    def __pow__(self, power):
        """
        Take a power of a permutation.

        p**power performs the equivalent of power repeated applications of p.
        If power is negative, this gives the inverse permutation raised to abs(power).
        It is linear-time in len(self) and constant-time in pow (or rather, it is as
        expensive as computing power % len(self)).
        """

        perm = list(range(len(self)))
        for cycle in self.cycles(canonical = False):
            for pos, index in enumerate(cycle):
                perm[index] = cycle[(pos+power) % len(cycle)]

        return Permutation(perm)


    def __mod__(self, other):
        """
        Take one permutation modulo another.

        Each permutation divides all other permutations into equivalence classes
        under composition with powers of this permutation. The modulo operation
        maps each permutation to a unique representative of the equivalence class it
        is in.

        The choice of representative is made by lexicographically minimising the
        internal representation of the permutation. This gives a simple and
        consistent choice, and guarantees that the identity permutation is returned
        whenever possible.

        The implementation is rather brute-force and not very efficient.

        Return  a permutation q such that q * other**n == self
                 for some n . It is chosen such that
                 p1 % p2 == p3 % p2 whenever p3 == p1 * p2**m for some m .
        """

        least = self

        for power in range(1, other.order()):
            perm = (other ** -power) * self

            if perm < least:
                least = perm
            elif perm == least:
                break

        return least
    
    def __eq__(self, other):
        """
        Compare two permutations for equality.

        The comparison self == 1 is also valid, checking if the permutation is the identity.
        No other comparisons to integers are allowed.
        """
        if other == 1:
            return self.is_identity()
        return self._map == other._map
    
    def __lt__(self, other):
        """ Compare two permutations lexicographically by their index maps. """
        return self._map < other._map

#-- Generator methods --#

    @staticmethod
    def identity(size):
        """ Generate the identity permutation. """
        return Permutation( size=size )
    
    @staticmethod
    def cyclic(size, offs):
        """ Generate the cyclic permutation that maps element 0 to element offs. """
        return Permutation( [(i + offs) % size for i in range(size)] )

    def mapping(self, objects=None):
        """
        Obtain the permutation as the mapping {i : self[i]}.

        If objects is provided, it will instead be {objects[i] : objects[self[i]]},
        requiring objects to have the appropriate length for that.
        """

        if objects is None:
            return {i : self[i] for i in range(len(self))}
        else:
            return {objects[i] : objects[self[i]] for i in range(len(self))}
            
    def reversed(self):
        """
        Obtain the reverse of a permutation.

        For example, identity(size).reversed().permute(list) == list.reversed().
        """
        return Permutation( list(reversed(self._map)) )
        
    def inverse(self):
        """ Obtain the inverse of a permutation, such that self * self.inverse() == 1. """

        inv = [0] * len(self)
        for i in range(len(self)):
            inv[ self._map[i] ] = i
            
        return Permutation(inv)

    def conjugate(self, other):
        """ Equivalent to other * self * other.inverse() """

        if len(self) != len(other):
            raise ValueError("Attempring to multiply permutations of different length")

        conj = [0] * len(self)
        for i in range(len(self)):
            conj[ other[i] ] = other[ self[i] ]

        return Permutation(conj)
    
    def offset(self, offset):
        """
        Obtain a permutation such that
            self.offset(offset).permute(list) == self.permute(list, offset = offset)
        """
        return Permutation( [i for i in range(offset)] + [i + offset for i in self._map] )
        
    def block(self, block_len):
        """
        Obtain a permutation such that
            self.block(block_len).permute(list) == self.permute(list, block_len = block_len)
        """
        return Permutation( [i*block_len + j for i in self._map for j in range(block_len)] )
    
    def append(self, perm):
        """
        Concatenate two permutations so that they act on subsequent portions of a list.
        """
        return Permutation( list(self._map) + [i + len(self) for i in perm._map] )

    def on_range(self, begin, end):
        """
        Restrict a permutation to a range of indices.

        begin -- the start index of the range (inclusive)
        end -- the end index of the range (exclusive)

        The result is a permutation of length (end - begin) that acts the same way that
        self does on the specified range of elements.
        The range of elements must be closed under self, i.e., elements within it must
        map to other elements within it.

        Thus,
        self.permute(array)[begin:end]
        is equal to
        self.on_range(begin, end).permute(array[begin:end])
        in valid cases.
        """

        if begin < 0:
            begin = len(self) + begin
        if end < 0:
            end = len(self) + end

        if begin < 0 or begin >= len(self) or end <= 0 or end > len(self) or begin > end:
            raise ValueError(f"Range [{begin},{end}) out of bounds in {len(self)}-element permutation")

        restrict = [self[i+begin]-begin for i in range(end-begin)]

        if any(i < 0 or i >= (end-begin) for i in restrict):
            raise ValueError("Target range is not closed under the permutation")

        #print(f"{restrict=}")

        return Permutation(restrict)

    def on_subset(self, subset):
        """
        Restrict a permutation to a subset of the objects it permutes.
        Works like self.on_range(begin, end) but allows a non-contiguous range.

        subset -- an array of the same length as self.
            self must map its truthy elements to other truthy elements, and falsy to falsy.
            The projected permutation has length equal to the number of truthy
            elements of subset, and permutes them the same way the original
            permutation permutes the full set of objects.

        Thus,
        [elem for elem in self.permute(array) if condition(elem)]
        is equal to
        [elem for elem in self.on_subset(subset).permute(e for e in array if condition(e))]
        but self.on_subset(subset) can of course be applied to other arrays than subset.
        """

        if len(subset) != len(self):
            raise ValueError("Target array size mismatch")

        restrict = bidict({})
        j = 0
        for i, elem in enumerate(subset):
            if bool(elem) != bool(subset[self[i]]):
                raise ValueError("Target subset is not closed under the permutation")
            if bool(elem):
                restrict[i] = j
                j += 1

        return Permutation( [restrict[self[restrict.inverse[i]]] for i in range(j)] )

    
    @staticmethod
    def sorting_permutation(array, key = lambda x: x, reverse = False):
        """
        Obtain the permutation that sorts a given array.

        array -- the given array. Is not modified by this method.
        key -- key used for sorting; see array.sort()
        reverse -- if True, the permutation reverse-sorts the array.
        """
        sort = list(range(len(array)))
        sort.sort(reverse = reverse, key = lambda i: key(array[i]))
        return Permutation(sort)

    @staticmethod
    def parse_cycles(size, string, sep=' ', base=0):
        """
        Read a permutation specified in cycle notation.

        Thus, parse_cycles(self.cycle_string(sep), sep) == self.

        size -- the size of the permutation.
        string -- a sequence of cycles expressed as parenthesis-enclosed sep-separated lists
        of base-based indices.
            The method does not care what the string contains outside the parentheses,
            or if cycles of length 1 are omitted or not.
            May also be "id" (possibly surrounded by whitespace), in which case
            the identity permutation is returned.
        base -- offset applied to all indices to allow 1-based indices (default: 0)
        """
        if string.strip() == 'id':
            return Permutation.identity(size)

        perm = list(range(size))
        for cycle_str in findall(r'\(([^)]+)\)', string):
            cycle = [int(elem)-base for elem in cycle_str.split(sep)]
            for i in range(len(cycle)):
                perm[ cycle[i] ] = cycle[(i+1) % len(cycle)]

        #print(f"Parse cycles: {string} -> {Permutation(perm)}")
        return Permutation(perm)

    @staticmethod
    def concatenate(*perms):
        """ Concatenate a list of permutations, as if by repeated application of +. """

        if len(perms) == 1:
            return perms[0]

        #print(f"Concatenating {','.join(str(p) for p in perms)}")

        cat = []
        for perm in perms:
            cat += [p + len(cat) for p in perm]

        return Permutation(cat)

    @staticmethod
    def compose(*perms):
        """
        Compose a list of equal-length permutations, as if by repeated application of *.

        Note that composition acts right-to-left.
        """

        if len(perms) == 1:
            return perms[0]

        #print(f"Composing {','.join(str(p) for p in perms)}")

        comp = list(range(len(perms[0])))
        for perm in reversed(perms):
            perm.permute_in_place(comp)

        return Permutation(comp)
    
#-- Application of permuations --#

    def permute(self, array, allow_oversize = False):
        """
        Return a permuted copy of an array as a list.

        array -- the array to be permuted.
        allow_oversize -- if True (default: False) array is allowed to be longer than self.
            The extra elements are left unpermuted.
        """

        if len(array) < len(self):
            raise ValueError("Array too short for permutation")
        if len(array) > len(self):
            if not allow_oversize:
                raise ValueError("Permuting array of mismatched size")
            else:
                return self.permute(array[:len(self)]) + list(array[len(self):])

        return [ array[i] for i in self ]
    

    def permute_in_place(self, array, allow_oversize = False):
        """
        Apply a permutation in-place to an array.

        array -- the mutable array to be permuted.
        allow_oversize -- if True (default: False) array is allowed to be longer than self.
            The extra elements are left unpermuted.
        """

        if len(array) < len(self):
            raise ValueError("Array too short for permutation")
        if len(array) > len(self) and not allow_oversize:
            raise ValueError("Permuting array of mismatched size")

        for dst in range(len(self)):

            # Here's the clever bit.
            # It works fine to move elements from higher indices to lower,
            # but when moving them from lower to higher we face the problem
            # that they will be moved again by a later swap.
            # But the permutation tells us where that will be, so we just follow it until
            # we find a higher index!
            src = self._map[dst]
            while src < dst:
                src = self._map[src]

            if src != dst:
                array[src], array[dst] = array[dst], array[src]

        return array

    def permute_bits(self, bits):
        """
        Apply a permutation to the bits of an integer.

        bits -- the integer to be permuted.
        """
        
        res = 0
        
        for i in self._map:
            res |= (bits & 1) << i
            bits >>= 1
            
        return res
            
    
#-- Properties of permutations --#

    @staticmethod
    def is_permutation(array):
        """ Check if a array of indices represents a permutation of range(len(array)). """
        return (
            min(array) == 0
            and max(array) == len(array)-1
            and len(array) == len(set(array))
            )
    def is_valid(self):
        """ Check that a permutation actually represents a permutation. """
        return self.is_permutation(self._map)

    def is_identity(self):
        """ Check if a permutation is the identity permutation. """
        return all( i == m for i,m in enumerate(self._map) )
    
    def cycles(self, canonical = True):
        """
        Obtain the list of cycles  of a permutation.

        A cycle is a list containing the full orbit of a given index under repeated
        application of the permutation.
        By their nature, cycles are non-overlapping, but the choice of starting index
        is not unique.
        Cycles of length 1, i.e., indices that are mapped to themselves, are omitted.

        canonical -- if True, the cycles are rotated so that they start with their largest
                index, and are sorted by starting index.
        """
        cycles = []
        visited = [False] * len(self)
        
        # Canonical notation has cycles rotated so that their largest element comes first
        def canonicalise(cycl):
            max_idx = 0
            max_val = cycl[max_idx]
            
            for i in range(1,len(cycl)):
                if cycl[i] > max_val:
                    max_val = cycl[i]
                    max_idx = i
                    
            return cycl[max_idx:] + cycl[:max_idx]
    
        for i in range(len(self)):
            if visited[i] or self[i] == i:
                continue
            
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = self[j]
                
            if canonical:
                cycles.append( canonicalise(cycle) )
            else:
                cycles.append(cycle)
            
        # Canonical notation has cycles sorted by their first element
        if(canonical):
            cycles.sort(key = lambda c: c[0])
            
        return cycles
    
    def cycle_type(self):
        """
        Obtain the cycle type of a permutation.

        The cycle type is the sorted list of lengths of cycles of a permutaion,
        including length-1 cycles.
        """
        
        decomp = []
        visited = [False] * len(self)
        
        for i in range(len(self)):
            if visited[i]:
                continue
            
            cyc_len = 0
            j = i
            while not visited[j]:
                visited[j] = True
                cyc_len += 1
                j = self[j]
                
            decomp.append(cyc_len)
            
        return sorted(decomp)
    
    def oneline_string(self, sep=' ', base=0):
        """
        Write the permutation in one-line form.

        sep -- a separator character (default: space).
        base -- offset applied to all indices to allow 1-based indices (default: 0)

        The permutation is represented as a parentheses-enclosed sep-separated list of the
        image of each element in order.
        With sep == " ", this is the default string representation.
        """

        return f"({sep.join(str(base+i) for i in self._map)})"

    def twoline_string(self, sep=' ', base=0, topline = None):
        """
        Write the permutation in two-line form.

        sep -- a separator character (default: space).
        base -- offset applied to all indices to allow 1-based indices (default: 0)
        topline -- a permutation of the same size as self, or a list of integers such that
                Permutation(topline) is such a permutation; may be omitted.

        Two strings are returned, each formatted like self.oneline_string(sep).
        The first is the one-line representation of topline, or of the identity permutation
        if topline is omitted.
        The second contains the respective images of each element of the first line,
        so if topline is omitted, this is equal to self.oneline_string(self, sep).
        The elements of both strings are padded so that they align when printed
        under one another.
        """

        if topline is not None:
            if len(topline) != len(self) or not self.is_permutation(topline):
                raise ValueError("Invalid top line given")
        if len(self) == 0:
            return "()", "()"

        top = [str(base+     topline[i] ) + sep for i in range(len(self) - 1)] + [str(base+     topline[-1] )]
        bot = [str(base+self[topline[i]]) + sep for i in range(len(self) - 1)] + [str(base+self[topline[-1]])]

        top = [top[i].ljust(len(bot[i])) for i in range(len(self))]
        bot = [bot[i].ljust(len(top[i])) for i in range(len(self))]

        return f"({''.join(top)})", f"({''.join(bot)})"


    def cycle_string(self, sep=' ', base=0):
        """
        Write the permutation in cycle form.

        sep -- a separator character (default: space).
        base -- offset applied to all indices to allow 1-based indices (default: 0)

        Each cycle enclosed in parentheses and has its elements separated by sep.
        Nothing is written outside the parentheses.
        The identity permutation is represented as "id".
        """
        cycles = self.cycles()
        if len(cycles) == 0:
            return 'id'
        else:
            return ''.join([f'({sep.join(str(base+c) for c in cycle)})' for cycle in cycles])
    
    def fixed_points(self):
        """ Obtain the list of fixed points of the permutation, i.e., elements that map to themselves. """
        return [i for i in range(len(self)) if self._map[i] == i]
    
    def order(self):
        """
        Obtain the smallest power to which self must be raised in order to return the identity.

        This is found in linear time by utilizing a property of the cycle type.
        """
        return reduce(lambda a,b: (a*b)//gcd(a,b), self.cycle_type())
    
    def parity(self):
        """
        Obtain the parity of a permutation: True if odd, False if even, in analogy with i % 2 for integers.

        All permutations can be expressed as a composition of two-element swaps (transpositions).
        The parity of the permutation is the parity (even/oddness) of the number of transpositions needed.

        This is obtained in linear time by utilizing a property of cycles.
        """
        n_evens = 0
        for cyc_len in self.cycle_type:
            if cyc_len % 2 != 0:
                n_evens += 1
        
        return n_evens % 2
