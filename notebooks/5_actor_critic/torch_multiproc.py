import torch.multiprocessing as mp

def worker(data_chunk, results, index):
    """
    Worker function to compute the sum of a chunk of data.
    Args:
        data_chunk (list): A chunk of data to process.
        results (list): A shared list to store results.
        index (int): Index to store the result in the shared list.
    """
    results[index] = sum(data_chunk)

def main():
    data = list(range(1, 101))  # Large array
    num_processes = 4  # Number of processes to use
    chunk_size = len(data) // num_processes

    # Split data into chunks
    data_chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]

    # Shared memory for results
    results = mp.Array('d', [0] * num_processes)  # 'd' is for double (floating-point numbers)

    # Create and start processes
    processes = []
    for i, chunk in enumerate(data_chunks):
        p = mp.Process(target=worker, args=(chunk, results, i))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Sum the partial results
    total_sum = sum(results)
    print(f"Total Sum: {total_sum}")

if __name__ == "__main__":
    mp.set_start_method("spawn")  # Set start method for compatibility
    main()