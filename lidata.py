import socket
import math
import csv

def hex_to_decimal(hex_str, factor=1):
    # Swap bytes for little endian format
    swapped_bytes = ''.join([hex_str[i:i + 2] for i in range(0, len(hex_str), 2)][::-1])
    # Convert hex to decimal and apply the factor
    return int(swapped_bytes, 16) * factor

def polar_to_cartesian(azimuth, distance):
    # Convert azimuth from degrees to radians
    theta_rad = math.radians(azimuth)
    # Compute x and y coordinates
    x = distance * math.cos(theta_rad) / 1000
    y = distance * math.sin(theta_rad) / 1000
    return x, y
def decode_data_packet(data_packet, csvfile):
    # Ensure the data packet has the expected length
    if len(data_packet) != 2412:
        raise ValueError('Invalid data packet length')
     # Remove the last 12 characters as they are useless
    data_packet = data_packet[:-12]

    # Split the data packet into blocks of 200 characters
    data_blocks = [data_packet[i:i + 200] for i in range(0, len(data_packet), 200)]

    

    for i, block in enumerate(data_blocks):
        # Check the flag byte
        if block[:4] != 'ffee':
            print('Invalid block flag')
            continue
        # Extract and decode the azimuth data
        azimuth_hex = block[4:8]
        azimuth = hex_to_decimal(azimuth_hex, factor=0.01)

        # Extract and decode the distance data
        distance_hex = block[8:12]
        distance = hex_to_decimal(distance_hex, factor=2)  # Factor of 2 to get actual distance

        # Convert polar to Cartesian coordinates
        x, y = polar_to_cartesian(azimuth, distance)

        # Write the Cartesian coordinates to the CSV file
        csvfile.writerow({'x': x, 'y': y})

def receive_and_decode():
    # Define the UDP socket parameters
    UDP_IP = '192.168.1.125'  # Replace with the IP address of the Ethernet port
    UDP_PORT = 2368       # Replace with the desired UDP port number

    # Create the UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    # Create a CSV file for output
    with open('output.csv', mode='w', newline='') as file:
        fieldnames = ['x', 'y']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        count = 0  # Initialize a counter for the number of rows written
        while count < 1248:
            # Receive data from the socket
            data, addr = sock.recvfrom(1248)  # Buffer size is 1248 bytes

            # Convert the data to a hexadecimal string
            data_packet = data.hex()

            # Decode the data packet and write to CSV
            decode_data_packet(data_packet, writer)

            count += 1  # Increment the counter

if __name__ == "__main__":
    receive_and_decode()
