import sys

path = "/home/ubuntu/tensorflow/tensorflow/core/user_ops/varmatmul/"

def omit_2D_TNS(infile, outfile):
    seen = dict()
    with open(path + infile, 'r') as file_1, \
         open(path + outfile, 'w') as file_2:

            for line in file_1.readlines():
                split = line.split()
                dropped = [split[i] for i in [0, 1, -1]]
                coords_tuple = tuple([split[i] for i in [0, 1]])
                # print coords_tuple
                if coords_tuple in seen:
                    continue
                else:
                    seen[coords_tuple] = True
                output = " ".join(dropped)
                file_2.write(output + "\n")


if __name__ == '__main__':
    omit_2D_TNS(sys.argv[1], sys.argv[2])
