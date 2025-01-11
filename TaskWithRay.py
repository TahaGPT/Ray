import time
import ray
from ray.exceptions import TaskCancelledError


ray.init() # initializing a ray for the program

@ray.remote # this remote function
def sumOfSquares(sizeOfTask): # function calculatin the sum of the squares of the numbers till the sizeOfTask
    return sum(i * i for i in range(1, sizeOfTask + 1))

# Number of tasks
noOfTasks = 10
# Size of each task
sizeOfTask = 10**5

start_time = time.time() # noting the start time of the process

# we did not initialized an empty list here as the get function will return a reference an the list will be automatically created or downloaded form the clustered worker node frmo wherever it was sent

for _ in range(noOfTasks):
    try:
        results = ray.get([sumOfSquares.remote(sizeOfTask)]) # receiving the reference from the remote worker
    except TaskCancelledError: # added an exception in case the get function does not return the desired output
        print("Object reference was cancelled.")
end_time = time.time() # noting the end time when process has been done

ray.shutdown() # shutting down the initialized ray

print(f"Results: {results}")
print(f"Time taken with Ray: {end_time - start_time} seconds") # subtracting to get the time consumption or time complexity




