import random
import csv

# Define the number of data points to generate
num_points = 1000

# Define the ranges for the values for each variable
temp_range = (70, 110)
humid_range = (20, 70)
precip_range = (0, 2)
wind_range = (0, 40)
pressure_range = (29.70, 30.20)

# Generate the data points and add them to the csv file
with open('data.csv', mode='a', newline='') as file:
    for i in range(num_points):
        temp = random.randint(*temp_range)
        humid = random.randint(*humid_range)
        precip = round(random.uniform(*precip_range), 1)
        wind = random.randint(*wind_range)
        pressure = round(random.uniform(*pressure_range), 2)
        wildfire = random.randint(0, 1)
        # Writes data
        writer = csv.writer(file)
        writer.writerow([temp, humid, precip, wind, pressure, wildfire])