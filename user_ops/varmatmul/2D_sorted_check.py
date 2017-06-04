import sys

path = "/home/ubuntu/tensorflow/tensorflow/core/user_ops/varmatmul/"

def two_D_sorted_check(infile):
    seen = dict()
    with open(path + infile, 'r') as file_1:

            prv_coords_tuple = None
            for line in file_1.readlines():
                split = line.split()
                coords_tuple = tuple([int(i) for i in split[:-1]])
                
#                print prv_coords_tuple, coords_tuple
                if prv_coords_tuple != None and coords_tuple < prv_coords_tuple:
                    print("encountered a bad ordering at ", coords_tuple)

                prv_coords_tuple = coords_tuple
                


if __name__ == '__main__':
    two_D_sorted_check(sys.argv[1])
