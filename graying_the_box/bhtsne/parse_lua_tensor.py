import numpy as np

def parse_lua_tensor(file, dim):

    k = 0
    read_data = []
    with open(file, 'r') as f:
        for line in f:
            if 'Columns' in line or line == '\n':
                continue
            else:
                read_data.append(line)
            k += 1

    read_data = read_data[2:-1]
    samples = []
    for line in read_data:
        line = line[0:-1] # remove end of line mark
        line_ = line.split(' ')
        line_ = filter(None,line_)

        samples.append(line_)

    z = np.array(samples)
    num_cols = dim # DEBUG HERE!!!
    num_lines = z.shape[0] * z.shape[1] / num_cols
    my_mat = np.zeros(shape=(num_lines,num_cols), dtype='float32')

    num_blocks = z.shape[0] / num_lines

    for i in range(num_blocks):
        my_mat[:,i*z.shape[1]:(i+1)*z.shape[1]] = z[num_lines*i : (i+1)*num_lines,:]

    return my_mat