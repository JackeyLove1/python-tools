'''
references: https://docs.python.org/3/howto/argparse.html
action
Specify how an argument should be handled
'store', 'store_const', 'store_true', 'append', 'append_const', 'count', 'help', 'version'

choices
Limit values to a specific set of choices
['foo', 'bar'], range(1, 10), or Container instance

const
Store a constant value

default
Default value used when an argument is not provided
Defaults to None

dest
Specify the attribute name used in the result namespace
help
Help message for an argument
metavar
Alternate display name for the argument as shown in help
nargs
Number of times the argument can be used
int, '?', '*', '+', or argparse.REMAINDER
required
Indicate whether an argument is required or optional
True or False
type
Automatically convert an argument to the given type
int, float, argparse.FileType('w'), or callable function
'''
import argparse
parser = argparse.ArgumentParser(description='this is a args-parser')
'''
summary
dest: store the value in the name 
type: data type
default: the default value

'''
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))
parser.add_argument('-a','--address',dest='Address',type=str,default=None,action='store',help='Input the HTTP/HTTPS address of video page.')