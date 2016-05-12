input_filename = "data/matrix.output"
output_filename = "data/matrix.reorder"

input_file = open(input_filename, "r");
output_file = open(output_filename, "w");

line = input_file.read()
output_file.write(line + '\n')


line = input_file.read()
token = line.split()

while line != '':
    i = int(token[0])
    j = int(token[1])
    
    output_file.write('{0} {1} {2}\n'.format((i-1)/8660 + 3 * ((i-1) % 8660) + 1, (j-1)/8660 + 3 * ((j-1) % 8660) + 1, token[2]))
    
    line = input_file.read()
    token = line.split()

input_file.close()
output_file.close()
