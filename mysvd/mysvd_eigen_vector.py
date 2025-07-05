import numpy as np

# A = P * D * P⁻¹


A = np.array([[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]])

ei = np.linalg.eig(A)

print(ei)


print(ei[1] @ np.diag(ei[0]) @ np.linalg.inv(ei[1]))



print(ei[1] @ np.diag(ei[0] ** 2) @ np.linalg.inv(ei[1]))
print(A @ A)
