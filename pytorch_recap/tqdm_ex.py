from tqdm import tqdm
import time

# tqdm With Iterables:
data = range(50)

for item in tqdm(data, desc="Iterating"):
    time.sleep(0.01)  # Simulate processing each item

# Using tqdm with map
def work(x):
    time.sleep(0.1)  # Simulate a task
    return x ** 2

data = range(20)
results = list(tqdm(map(work, data), desc="Mapping", total=len(data)))

# Nested Progress Bars:
for i in tqdm(range(3), desc="Outer Loop"):
    for j in tqdm(range(5), desc="Inner Loop", leave=False):
        time.sleep(0.2)  # Simulate work