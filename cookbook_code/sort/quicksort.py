def partition(arr, first, last):
    pivot = first
    for pos in range(first, last):
        if arr[pos] < arr[last]:
            arr[pos], arr[pivot] = arr[pivot], arr[pos]
            pivot += 1
    arr[pivot], arr[last] = arr[last], arr[pivot]
    return pivot

def qucik_sort(arr, first, last):
    if first < last:
        pi = partition(arr, first, last)
        qucik_sort(arr, first, pi-1)
        qucik_sort(arr, pi+1, last)

A = [534, 246, 933, 127, 277, 321, 454, 565, 220]
qucik_sort(A, 0, len(A) - 1)
print(A)