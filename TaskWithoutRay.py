import time


def sumOfSquares(sizeOfTask): # function calculatin the sum of the squares of the numbers till the sizeOfTask
    return sum(i * i for i in range(1, sizeOfTask + 1))

# Number of Tasks
noOfTasks = 10
# Size of each task
sizeOfTask = 10**5

start_time = time.time() # noting the start time of the process

results = [] # initializing an empty list for later use
for _ in range(noOfTasks):
    results.append(sumOfSquares(sizeOfTask)) # appending the results

end_time = time.time() # noting the end time when process has been done

print(f"Results: {results}")
print(f"Time taken without Ray: {end_time - start_time} seconds") # subtracting to get the time consumption or time complexity