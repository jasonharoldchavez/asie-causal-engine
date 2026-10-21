import unittest

import numpy as np

from asie_engine.matrix import edge_to_matrix, matrix_to_edges


class CanonicalMatrixOrientationTests(unittest.TestCase):
    def test_forward_and_reverse_edges_are_not_transposed(self):
        forward = edge_to_matrix([(0, 1)], 2)
        reverse = edge_to_matrix([(1, 0)], 2)
        self.assertEqual(forward[1, 0], 1.0)
        self.assertEqual(forward[0, 1], 0.0)
        self.assertEqual(reverse[0, 1], 1.0)
        self.assertEqual(reverse[1, 0], 0.0)

    def test_chain_orientation(self):
        matrix = edge_to_matrix([(0, 1), (1, 2)], 3)
        self.assertEqual(matrix[1, 0], 1.0)  # A -> B
        self.assertEqual(matrix[2, 1], 1.0)  # B -> C
        self.assertEqual(matrix[0, 1], 0.0)
        self.assertEqual(matrix[1, 2], 0.0)

    def test_common_cause_orientation(self):
        matrix = edge_to_matrix([(0, 1), (0, 2)], 3)
        self.assertEqual(matrix[1, 0], 1.0)  # A -> B
        self.assertEqual(matrix[2, 0], 1.0)  # A -> C
        self.assertEqual(matrix[0, 1], 0.0)
        self.assertEqual(matrix[0, 2], 0.0)

    def test_unrelated_nodes_have_no_edge(self):
        matrix = edge_to_matrix([(0, 1)], 3)
        self.assertEqual(matrix[2, 0], 0.0)
        self.assertEqual(matrix[0, 2], 0.0)

    def test_weighted_round_trip(self):
        expected = [(0, 1, 0.7), (2, 1, -0.4)]
        matrix = edge_to_matrix(expected, 3)
        self.assertEqual(matrix_to_edges(matrix), expected)

    def test_threshold_and_diagonal_handling(self):
        matrix = np.array([[9.0, 0.1], [0.8, 7.0]])
        self.assertEqual(matrix_to_edges(matrix, threshold=0.2), [(0, 1, 0.8)])
        self.assertEqual(
            matrix_to_edges(matrix, threshold=0.2, include_self=True),
            [(0, 0, 9.0), (0, 1, 0.8), (1, 1, 7.0)],
        )

    def test_invalid_inputs_fail_closed(self):
        for edges, size in [([(0, 3)], 3), ([(0, 0)], 2), ([(0, 1), (0, 1)], 2),
                            ([(0.5, 1)], 2), ([(0, 1, float("nan"))], 2)]:
            with self.subTest(edges=edges, size=size):
                with self.assertRaises(ValueError):
                    edge_to_matrix(edges, size)
        for matrix in ([[1, 2, 3]], [[1, float("nan")], [0, 1]], [["x"]]):
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    matrix_to_edges(matrix)


if __name__ == "__main__":
    unittest.main()
