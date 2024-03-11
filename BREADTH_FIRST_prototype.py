import queue
import shapely
from shapely.geometry import LineString


def breadth_first(crop, start,end):
    
    # Initialize the queue with the start position
    q = queue.Queue()
    q.put(start)
    
    # Initialize the distances matrix with -1 (unvisited cells)
    distances = []
    for rij in range(len(crop)):
        distances.append([-1]*len(crop[0]))
        
    distances[start[0]][start[1]] = 0  # The distance to the start position is 0
    
    # Mogelijke stappen: R/D/L/U
    movements = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while not q.empty():
        x, y = q.get()
        
        # If we have reached the end position, we can stop
        if (x, y) == end:
            break
        
        for dx, dy in movements:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is inside the crop and is a path (0)
            if 0 <= nx < len(crop) and 0 <= ny < len(crop[0]) and crop[nx][ny] == 0 and distances[nx][ny] == -1: #check of binnen de crop (horizontaal en verticaal) EN geen muur EN onbezocht
                distances[nx][ny] = distances[x][y] + 1
                q.put((nx, ny))
    
    return distances

def print_shortest_path(distances, start, end):
    for row in distances:
        row_as_strings = []
        for cell in row:
            row_as_strings.append(str(cell))
        # print(' '.join(row_as_strings)) 
        # Dit was er eerst om de distances matrix te printen (nu niet meer nodig)
    x, y = end
    path = [(x, y)]
    
    # The possible movements from a cell
    movements = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while (x, y) != start:
        for dx, dy in movements:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is inside the distances matrix and is closer to the start
            if 0 <= nx < len(distances) and 0 <= ny < len(distances[0]):
                if distances[nx][ny] == distances[x][y] - 1:
                    path.append((nx, ny))
                    x, y = nx, ny
                    break
    
    # Reverse the path so it goes from start to end
    path.reverse()
    # def simplify_path(path, tolerance=1.0):
    #     # Convert path to LineString object
    #     line = LineString(path)
        
    #     # Simplify the line
    #     simplified_line = line.simplify(tolerance, preserve_topology=False)
        
    #     # Convert simplified line back to list of tuples
    #     simplified_path = list(simplified_line.coords)
        
    #     return simplified_path


    # simplified_path = simplify_path(path)
    # return simplified_path
    return path

        
    # for x, y in path:
    #     print(f"({x}, {y})")

# Usage
# crop = [
#     [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#     [1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0, 1, 0, 1, 1],
#     [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
#     [1, 1, 1, 1, 1, 0, 0, 0, 0, 1d],
#     [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#     [1, 0, 1, 1, 1, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
# ]
# start = (0, 7)
# end = (9, 8)
# distances = breadth_first(crop, start)
# print_shortest_path(distances, start, end)