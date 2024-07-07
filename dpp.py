"""
DPP (Determinantal Point Processes) Greedy Algorithm Implementation

This module provides two implementations of a greedy algorithm for DPP:
1. A standard version (`dpp`) that selects a diverse subset of items.
2. A sliding window version (`dpp_sw`) that maintains a fixed-size window of selected items.

Variable Names Explanation:
- L: A 2-dimensional array representing the Cholesky factor of the kernel matrix.
     In the context of DPP, the Cholesky factor is used for intermediate computations
     to maintain orthogonality and ensure numerical stability during the selection process.

- Y_diag: A 1-dimensional array representing the diagonal elements of the kernel matrix.
          These diagonal elements are used as initial diversity scores for each item.
          During the selection process, this array is updated to reflect the remaining
          diversity contributions of the unselected items.

- kernel_row: A 1-dimensional array representing a row of the kernel matrix corresponding
              to the currently selected item. It is used to update the diversity scores
              and the Cholesky factor.

- L_optimal: A 1-dimensional array representing the optimal Cholesky factor components
             for the currently selected item. It is used to update the intermediate
             computations (L) during each iteration of the algorithm.

- L_diag: A scalar representing the diagonal component of the Cholesky factor for the
          currently selected item. It is used to normalize the updates to the Cholesky
          factor and the diversity scores.

- E: A 1-dimensional array representing the updated vector for the currently selected item.
     It is calculated during each iteration and used to update the Cholesky factor (L) and
     the diversity scores (Y_diag).

- V: A 2-dimensional array representing the orthogonal basis vectors in the sliding window
     version of the DPP algorithm. It is used to maintain a fixed-size window of selected
     items, ensuring that the diversity scores are updated correctly within the window.

Functions:
- initialize_dpp(kernel_matrix): Initializes the variables required for the DPP algorithm,
  including the Cholesky factor (L), the initial diversity scores (Y_diag), and the list
  of selected items.

- update_dpp(kernel_matrix, L, Y_diag, selected_item, k): Updates the intermediate variables
  after selecting an item, including the Cholesky factor (L) and the diversity scores (Y_diag).

- select_next_item(Y_diag, epsilon): Selects the next item based on the highest diversity score
  and checks if the selection process should stop based on the epsilon threshold.

- dpp(kernel_matrix, max_length, epsilon=1E-10): Implements the standard greedy algorithm for DPP,
  using the helper functions to select a diverse subset of items.

- update_window(V, L, window_start, k): Updates the sliding window for the DPP algorithm,
  including the orthogonal basis vectors (V) and the Cholesky factor (L).

- dpp_sw(kernel_matrix, window_size, max_length, epsilon=1E-10): Implements the sliding window
  version of the greedy algorithm for DPP, using the update_window function to manage the windowed updates.
"""
import numpy as np
import math


def initialize_dpp(kernel_matrix: np.ndarray) -> tuple:
    """
    Initialize variables for the DPP algorithm.
    :param kernel_matrix: 2-d array
    :return: initial values of L, Y_diag, selected_items, selected_item
    """
    num_items = kernel_matrix.shape[0]
    L = np.zeros((num_items, num_items))  # Cholesky factor matrix
    Y_diag = np.copy(np.diag(kernel_matrix))  # Diagonal elements of the kernel matrix
    selected_item = np.argmax(Y_diag)  # Index of the item with the highest initial diversity score
    selected_items = [selected_item]  # List of selected items initialized with the first item
    return L, Y_diag, selected_items, selected_item


def update_dpp(kernel_matrix: np.ndarray, L: np.ndarray, Y_diag: np.ndarray, selected_item: int, k: int) -> tuple:
    """
    Update the variables for the next iteration of the DPP algorithm.
    :param kernel_matrix: 2-d array
    :param L: 2-d array of intermediate computations (Cholesky factor)
    :param Y_diag: 1-d array of diversity scores (diagonal of the kernel matrix)
    :param selected_item: int
    :param k: the index of the current iteration
    :return: updated values of L, Y_diag
    """
    if k > 0:
        L_diag = math.sqrt(Y_diag[selected_item])
        L_optimal = L[:k, selected_item]
        kernel_row = kernel_matrix[selected_item, :]
        E = (kernel_row - np.dot(L_optimal, L[:k, :])) / L_diag
        L[k, :] = E
        Y_diag -= np.square(E)
    Y_diag[selected_item] = -np.inf
    return L, Y_diag


def select_next_item(Y_diag: np.ndarray, epsilon: float) -> tuple:
    """
    Select the next item with the highest diversity score.
    :param Y_diag: 1-d array of diversity scores
    :param epsilon: small positive scalar
    :return: next selected item, boolean indicating if selection should stop
    """
    selected_item = np.argmax(Y_diag)
    max_value = Y_diag[selected_item]
    return selected_item, max_value < epsilon


def dpp(kernel_matrix: np.ndarray, max_length: int, epsilon: float = 1E-10) -> list:
    """
    Fast implementation of the greedy algorithm for DPP.
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list of selected items
    """
    L, Y_diag, selected_items, selected_item = initialize_dpp(kernel_matrix)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        L, Y_diag = update_dpp(kernel_matrix, L, Y_diag, selected_item, k)
        selected_item, stop = select_next_item(Y_diag, epsilon)
        if stop:
            break
        selected_items.append(selected_item)
    return selected_items


def update_window(V: np.ndarray, L: np.ndarray, window_start: int, k: int) -> tuple:
    """
    Update the sliding window for the DPP algorithm.
    :param V: 2-d array of orthogonal basis vectors
    :param L: 2-d array of intermediate computations
    :param window_start: int
    :param k: int
    :return: updated values of V, L
    """
    for i in range(window_start, k + 1):
        t = math.sqrt(V[i, i] ** 2 + V[i, window_start - 1] ** 2)
        c = t / V[i, i]
        s = V[i, window_start - 1] / V[i, i]
        V[i, i] = t
        V[i + 1:k + 1, i] += s * V[i + 1:k + 1, window_start - 1]
        V[i + 1:k + 1, i] /= c
        V[i + 1:k + 1, window_start - 1] *= c
        V[i + 1:k + 1, window_start - 1] -= s * V[i + 1:k + 1, i]
        L[i, :] += s * L[window_start - 1, :]
        L[i, :] /= c
        L[window_start - 1, :] *= c
        L[window_start - 1, :] -= s * L[i, :]
    return V, L


def dpp_sw(kernel_matrix: np.ndarray, window_size: int, max_length: int, epsilon: float = 1E-10) -> list:
    """
    Sliding window version of the greedy algorithm for DPP.
    :param kernel_matrix: 2-d array
    :param window_size: positive int
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list of selected items
    """
    L, Y_diag, selected_items, selected_item = initialize_dpp(kernel_matrix)
    V = np.zeros((max_length, max_length))
    window_start = 0

    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        L_optimal = L[window_start:k, selected_item]
        L_diag = math.sqrt(Y_diag[selected_item])
        V[k, window_start:k] = L_optimal
        V[k, k] = L_diag
        kernel_row = kernel_matrix[selected_item, :]
        E = (kernel_row - np.dot(L_optimal, L[window_start:k, :])) / L_diag
        L[k, :] = E
        Y_diag -= np.square(E)
        if len(selected_items) >= window_size:
            window_start += 1
            V, L = update_window(V, L, window_start, k)
            Y_diag += np.square(L[window_start - 1, :])
        Y_diag[selected_item] = -np.inf
        selected_item, stop = select_next_item(Y_diag, epsilon)
        if stop:
            break
        selected_items.append(selected_item)
    return selected_items
