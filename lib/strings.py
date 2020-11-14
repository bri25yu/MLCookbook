"""
@author bri25yu

A library dedicated to algorithms relating to strings!
"""


class StringAlgs:
    
    @staticmethod
    def longest_proper_prefix_suffix(s: str) -> list:
        """
        Parameters
        ----------
        s: str

        Returns
        -------
        lps: list
            A list where the value at each index i is the length of the longest proper prefix
            of s[0..i] which is also a suffix of s[0..i].

        >>> f = StringAlgs.longest_proper_prefix_suffix
        >>> f("AAAA")
        [0, 1, 2, 3]
        >>> f("ABCDE")
        [0, 0, 0, 0, 0]
        >>> f("AABAACAABAA")
        [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]
        >>> f("AAACAAAAAC")
        [0, 1, 2, 0, 1, 2, 3, 3, 3, 4]
        >>> f("AAABAAA")
        [0, 1, 2, 0, 1, 2, 3]bb

        """
        lps = [0] * len(s)
        i, p = 1, 0  # i is the current index, p is the length of the previous longest prefix suffix
        while i < len(s):
            if s[i] == s[p]:
                # If the current character matches the current character of the prefix
                # then we set the lps to be the p+1
                p += 1
                lps[i] = p
                i += 1
            elif p > 0:
                # If the current character doesn't match and there could still be a non-zero prefix length,
                # the previous longest length can only be lps[p-1], but we still aren't guaranteed
                # to have a match, so we don't increment i.
                p = lps[p - 1]
            else:
                lps[i] = 0
                i += 1
        return lps

    @staticmethod
    def pattern_search_kmp(s: str, p: str) -> list:
        """
        Parameters
        ----------
        s: str
            The string to search for the input pattern p.
        p: str
            The pattern to search for in the input string s.

        Returns
        -------
        indices: list
            A list containing all indices that match the pattern p.

        >>> f = StringAlgs.pattern_search_kmp
        >>> f("THIS IS A TEST TEXT", "TEST")
        [10]
        >>> f("AABAACAADAABAABA", "AABA")
        [0, 9, 12]
        >>> f("AAAAAAAAAAAAAAAAAB", "AAAAB")
        [13]
        >>> f("ABABABCABABABCABABABC", "ABABAC")
        []
        >>> f("AAAAABAAABA", "AAAA")
        [0, 1]

        """
        lps = StringAlgs.longest_proper_prefix_suffix(p)
        matches = []
        i, prev = 0, 0
        while i < len(s):
            if s[i] == p[prev]:
                i += 1
                prev += 1
            if prev == len(p):
                # If we're at the end of the pattern string,
                # it means that we've found an instance of our pattern.
                matches.append(i - prev)
                prev = lps[prev - 1]
            elif i < len(s) and s[i] != p[prev]:
                if prev != 0:
                    prev = lps[prev - 1]
                else:
                    i += 1
        return matches


if __name__ == "__main__":
    import doctest
    doctest.testmod()
