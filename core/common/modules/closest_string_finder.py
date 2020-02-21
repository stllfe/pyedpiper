import re
import numpy as np
from typing import List, Tuple, Iterable


class ClosestStringFinder:
    def __init__(self):
        self.regex = re.compile('([a-zA-Z]*)')

    def _casefold_cleanup(self, string: str):
        string = string.lower().strip()
        return ''.join(self.regex.findall(string))

    @staticmethod
    def _convert_to_numpy(string: str):
        return np.array(list(string))

    def normalized_distance(self, left: str, right: str) -> float:
        """Calculates both Jaccard and Levenshtein metrics and returns their normalized combination.
        :param left: first string
        :param right: second string
        :return: (1 - jaccard_index) * (levenshtein_distance / len(right))
        """
        j = self.jaccard_index(left, right)
        l = self.levenshtein_distance(left, right) / len(right)
        return (1 - j) * l

    def jaccard_index(self, left: str, right: str) -> float:
        """Calculates the Jaccard index between two strings.
        :param left: first string
        :param right: second string
        :return: float index
        """
        left = self._casefold_cleanup(left)
        right = self._casefold_cleanup(right)

        def coefficient(s1, s2):
            num_matches = sum(s1 == s2)
            return num_matches / (len(s1) + len(s2) - num_matches)

        np_strings = [self._convert_to_numpy(left), self._convert_to_numpy(right)]
        lengths = map(len, np_strings)

        min_idx = np.argmin(lengths)[0]
        np_strings[min_idx] = np.resize(np_strings[min_idx], max(lengths))
        return coefficient(*np_strings)

    def levenshtein_distance(self, left: str, right: str) -> int:
        """Calculates the Levenshtein distance between two strings.
        :param left: first string
        :param right: second string
        :return: int distance
        """
        left = self._casefold_cleanup(left)
        right = self._casefold_cleanup(right)

        left_len, right_len = len(left), len(right)
        if left_len > right_len:
            # Make sure left <= right, to use O(min(left_len, right_len)) space
            left, right = right, left
            left_len, right_len = right_len, left_len

        # Keep current and previous row, not entire matrix
        current_row = range(left_len + 1)

        for i in range(1, right_len + 1):
            previous_row, current_row = current_row, [i] + [0] * left_len
            for j in range(1, left_len + 1):
                add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
                if left[j - 1] != right[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        return current_row[left_len]

    def find_closest(self, needle: str, collection: Iterable[str], precision=4) -> Tuple[List[str], float]:
        """Finds the closes string from collection.
        :param needle: String to measure the distance against
        :param collection: Any iterable containing strings
        :param precision: Floating point precision for normalized distances
        :return: Tuple: List of closest strings from collection and their `normalized_distance()`s.
        """
        distances = {
            string: round(self.normalized_distance(needle, string), precision)
            for string in collection
        }
        distances = {
            string: distance
            for string, distance in sorted(distances.items(), key=lambda item: item[1])
        }

        results = []
        closest_distance = min(distances.values())
        closest_strings = filter(lambda item: item[1] == closest_distance, distances.items())
        for string, _ in closest_strings:
            results.append(string)

        return results, closest_distance
